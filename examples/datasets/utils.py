"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import collections
import time

import numpy as np

Rays = collections.namedtuple("Rays", ("origins", "viewdirs"))


def namedtuple_map(fn, tup):
    """Apply `fn` to each element of `tup` and cast to `tup`'s namedtuple."""
    return type(tup)(*(None if x is None else fn(x) for x in tup))

def get_rays(K, camtoworlds, resolution):
    width, height = resolution
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # Camera view: [+x is right, +y is up, +z is back away from camera]
    x, y = np.meshgrid(
        np.linspace(0, width - 1, width),
        np.linspace(0, height - 1, height),
        indexing="xy",
    )
    dx, dy = x - cx, y - cy
    ray_d_cam = np.stack(
        [
            dx / fx,
            -dy / fy,
            -np.ones_like(dx),
        ],
        axis=-1,
    )
    ray_d_cam = ray_d_cam / np.linalg.norm(ray_d_cam, axis=-1, keepdims=True)

    # World view: [+z is up, xy plane is parallel to the ground]
    # (N, 3, 3) @ (H, W, 3) -> (N, H, W, 3)
    ray_d_world = np.einsum("nij,hwj->nhwi", camtoworlds[:, :3, :3], ray_d_cam)
    ray_o_world = (
        camtoworlds[:, :3, 3]
        .reshape(-1, 1, 1, 3)
        .repeat(height, axis=1)
        .repeat(width, axis=2)
    )
    return ray_o_world, ray_d_world

def logging_time(original_fn):
    def wrapper_fn(*args, **kwargs):
        start_time = time.time()
        result = original_fn(*args, **kwargs)
        end_time = time.time()
        print(
            "WorkingTime[{}]: {:.4f} sec".format(
                original_fn.__name__, end_time - start_time
            )
        )
        return result

    return wrapper_fn
