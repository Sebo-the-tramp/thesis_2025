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
import json

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

    # ##########################
    # #         COLMAP         #
    # ##########################

    cloud = PyntCloud.from_file("./sparse_pc.ply")

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
    # #         JSON           #
    # ##########################

    # Load the JSON file
    json_path = "./_assets/transforms.json"

    with open(json_path, "r") as f:
        data = json.load(f)
        # print(data["frames"])

        for frame in data["frames"]:
            matrix = np.array(frame["transform_matrix"]).reshape(4, 4)
            
            rotation_matrix = matrix[:3, :3]
            translation = matrix[:3, 3]

            print(frame["colmap_im_id"], translation, rotation_matrix)

            server.scene.add_camera_frustum(
                name=str(frame["colmap_im_id"]),
                aspect=1.7,
                fov=60,
                color=[random.random(), random.random(), random.random()],
                scale=0.1,
                position=translation,
                wxyz=R.from_matrix(rotation_matrix).as_quat(scalar_first=True),
            )


   

    while True:

        time.sleep(1e-3)


if __name__ == "__main__":
    tyro.cli(main)