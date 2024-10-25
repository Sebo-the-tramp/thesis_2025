import random
import time
from pathlib import Path
from typing import List

import imageio.v3 as iio
import numpy as np
import tyro
from tqdm.auto import tqdm
import trimesh
import pygltflib

import numpy as np
import pandas as pd

from pyntcloud import PyntCloud

import viser
import viser.transforms as tf
from viser.extras.colmap import (
    read_cameras_binary,
    read_images_binary,
    read_points3d_binary,
)
from scipy.spatial.transform import Rotation as R

def main(
    colmap_path: Path = Path(__file__).parent / "_assets/0",
    images_path: Path = Path(__file__).parent / "_assets/images_8",
    downsample_factor: int = 2,
) -> None:
    """Visualize COLMAP sparse reconstruction outputs.

    Args:
        colmap_path: Path to the COLMAP reconstruction directory.
        images_path: Path to the COLMAP images directory.
        downsample_factor: Downsample factor for the images.
    """
    server = viser.ViserServer()
    # server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")    

    cloud = PyntCloud.from_file("./_assets/sparse_pc.ply")
    print(cloud.points)

    colors = cloud.points[["red", "green", "blue"]].values / 255
    points = cloud.points[["x", "y", "z"]].values

    point_cloud = server.scene.add_point_cloud(
        name="/colmap/pcd",
        points=points,
        colors=colors,
        point_size=0.005,
        wxyz=R.from_euler("xyz", [180, 0, -90], degrees=True).as_quat(),
    )    

    # ##########################
    # #         COLMAP         #
    # ##########################

    # points = np.array([points3d[p_id].xyz for p_id in points3d])
    # colors = np.array([points3d[p_id].rgb for p_id in points3d])

    # point_mask = np.random.choice(points.shape[0], gui_points.value, replace=False)
    # point_cloud = server.scene.add_point_cloud(
    #     name="/colmap/pcd",
    #     points=points[point_mask],
    #     colors=colors[point_mask],
    #     point_size=gui_point_size.value,
    #     wxyz=R.from_euler("xyz", [180, 0, -90], degrees=True).as_quat(),
    # )
    # frames: List[viser.FrameHandle] = []    

    # need_update = True    

   

    while True:

        time.sleep(1e-3)


if __name__ == "__main__":
    tyro.cli(main)