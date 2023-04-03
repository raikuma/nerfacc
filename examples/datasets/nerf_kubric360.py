import glob
import json
import os

import cv2
import numpy as np
import torch
from pyquaternion import Quaternion
from torch.utils.data import Dataset

from datasets.utils import get_rays, logging_time, Rays


def parseKubric360(root_path):
    metadata_path = os.path.join(root_path, "metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # intrinsic
    resolution = np.array(metadata["metadata"]["resolution"])
    width, height = resolution
    focal_length = np.array(metadata["camera"]["focal_length"])  # mm
    sensor_width = np.array(metadata["camera"]["sensor_width"])  # mm
    fx = focal_length / sensor_width * resolution[0]  # pixel
    fy = fx
    cx = width / 2  # pixel
    cy = height / 2  # pixel
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # extrinsic
    positions = np.array(metadata["camera"]["positions"])
    positions[:, 2] -= 10.0  # optimize
    quaterions = np.array(metadata["camera"]["quaternions"])  # [w, x, y, z]
    rotation_matrices = np.array([Quaternion(q).rotation_matrix for q in quaterions])
    camtoworlds = np.concatenate(
        [rotation_matrices, positions[..., np.newaxis]], axis=2
    )

    img_paths = sorted(glob.glob(os.path.join(root_path, "rgba*.png")))

    return {
        "resolution": resolution,  # [w, h]
        "K": K,  # [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        "camtoworlds": camtoworlds,  # (N, 3, 4)
        "img_paths": img_paths,  # (N,)
    }


class SubjectLoader(Dataset):
    """
    NeRF dataset
    """

    def __init__(self, subject_id, root_fp, split, num_rays, device, **kwargs):
        """
        root_path (str): path to the root directory of the dataset
        split (str): ["train"|"test"]. train gives ray unit batch and test gives image unit batch
        near (float): near clipping plane
        far (float): far clipping plane
        """
        self.root_path = os.path.join(root_fp, subject_id)
        self.split = split
        self.factor = kwargs['factor']
        self.num_rays = num_rays
        self.device = device

        self.debug = False
        if self.debug:
            self.factor = 16

        self.parse()

        if self.split in ["train", "val"]:
            self.rgb = self.rgb.reshape(-1, self.rgb.shape[-1])
            self.ray_o = self.ray_o.reshape(-1, self.ray_o.shape[-1])
            self.ray_d = self.ray_d.reshape(-1, self.ray_d.shape[-1])

    def parse(self):
        data = parseKubric360(self.root_path)
        self.resolution = data["resolution"] // self.factor
        self.K = data["K"]
        self.K[0, 2] //= self.factor  # cx
        self.K[1, 2] //= self.factor  # cy

        # Split: https://www.notion.so/NeRF-Dataset-ff3440b58a954105ae1f8674bb0113a8?pvs=4#d46fc4d1778549a6a2868a8e8fe0be3c
        if self.split == "train":
            idx = list(range(15, 76, 1))
        # elif self.split == "val":
        #     idx = list(range(16, 75, 2))
        elif self.split == "test":
            idx = list(range(9, 15+1)) + list(range(76-1, 82))
        # elif self.split == "all":
        #     idx = list(range(9, 82))
        else:
            raise ValueError("Invalid split")
        self.camtoworlds = data["camtoworlds"][idx]
        self.img_paths = [data["img_paths"][i] for i in idx]

        if self.debug:
            self.img_paths = self.img_paths[:2]
            self.camtoworlds = self.camtoworlds[:2]

        print(f"[Dataset] {self.split} {len(self.img_paths)} images")

        self.width, self.height = self.resolution
        self.rgb = self.get_imgs()
        self.ray_o, self.ray_d = self.get_rays()

    @logging_time
    def get_imgs(self):
        def load_img(path):
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.width, self.height))
            img = img / 255.0
            return img

        imgs = np.array([load_img(path) for path in self.img_paths])
        return torch.tensor(imgs, dtype=torch.float32)

    @logging_time
    def get_rays(self):
        ray_o_world, ray_d_world = get_rays(
            K=self.K, camtoworlds=self.camtoworlds, resolution=self.resolution
        )
        return (
            torch.tensor(ray_o_world, dtype=torch.float32),
            torch.tensor(ray_d_world, dtype=torch.float32),
        )

    def __len__(self):
        if self.num_rays is None:
            return len(self.rgb)
        return len(self.rgb) // self.num_rays

    @torch.no_grad()
    def __getitem__(self, idx):
        if self.num_rays is None:
            return {
                "pixels": self.rgb[idx].to(self.device),
                "rays": Rays(origins=self.ray_o[idx].to(self.device), viewdirs=self.ray_d[idx].to(self.device)),
                "color_bkgd": torch.rand(3, device=self.device),
            }
        return {
            "pixels": self.rgb[idx*self.num_rays:(idx+1)*self.num_rays].to(self.device),
            "rays": Rays(origins=self.ray_o[idx*self.num_rays:(idx+1)*self.num_rays].to(self.device),
                         viewdirs=self.ray_d[idx*self.num_rays:(idx+1)*self.num_rays].to(self.device)),
            "color_bkgd": torch.rand(3, device=self.device),
        }
    
    def update_num_rays(self, num_rays):
        self.num_rays = num_rays
