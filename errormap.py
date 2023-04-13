import numpy as np
import imageio.v2 as imageio

TH = 0.15

def error(img1, img2):
    return np.array(np.linalg.norm(img1 - img2, axis=-1) > TH, dtype=np.uint8)

def readimg(path):
    img = imageio.imread(path)
    return np.array(img, dtype=np.float32) / 255.0

def writeimg(path, img):
    imageio.imwrite(path, (img * 255).astype(np.uint8))

def main():
    test_idx = list(range(9, 15+1)) + list(range(76-1, 82))
    for scene in range(101, 110+1):
        for i, idx in enumerate(test_idx):
            img_pred = readimg(f"/scratch/woongohcho/output/{scene}/{i}_rgb_test.png")
            img_gt = readimg(f"/scratch/woongohcho/output/{scene}/{i}_rgb_gt.png")
            errimg = error(img_pred, img_gt)
            writeimg(f"/scratch/woongohcho/output/{scene}/{i}_mask.png", errimg)

if __name__ == "__main__":
    main()