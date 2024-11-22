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

    # Load the colmap info.
    cameras = read_cameras_binary(colmap_path / "cameras.bin")
    images = read_images_binary(colmap_path / "images.bin")
    points3d = read_points3d_binary(colmap_path / "points3D.bin")
    gui_reset_up = server.gui.add_button(
        "Reset up direction",
        hint="Set the camera control 'up' direction to the current camera's 'up'.",
    )

    ##########################
    #         GUI            #
    ##########################

    @gui_reset_up.on_click
    def _(event: viser.GuiEvent) -> None:
        client = event.client
        assert client is not None
        client.camera.up_direction = tf.SO3(client.camera.wxyz) @ np.array(
            [0.0, -1.0, 0.0]
        )
        
    gui_points = server.gui.add_slider(
        "Max points",
        min=1,
        max=len(points3d),
        step=1,
        initial_value=min(len(points3d), 50_000),
    )
    gui_frames = server.gui.add_slider(
        "Max frames",
        min=1,
        max=len(images),
        step=1,
        initial_value=min(len(images), 200),
    )
    gui_point_size = server.gui.add_slider(
        "Point size", min=0.01, max=0.1, step=0.001, initial_value=0.01
    )

    gui_donwload_button = server.gui.add_button(
        "Download",
        hint="Download the current scene",
    )   

    def visualize_frames() -> None:
        """Send all COLMAP elements to viser for visualization. This could be optimized
        a ton!"""

         ##########################
        #         COLMAP         #
        ##########################

        points = np.array([points3d[p_id].xyz for p_id in points3d])
        colors = np.array([points3d[p_id].rgb for p_id in points3d])    

        # there is a weird conversion from point cloud and camera positions
        # therefore we need this kind of addon to make everything works
        points = points @ R.from_euler("xyz", [90, 0, 0], degrees=True).as_matrix()

        point_mask = np.random.choice(points.shape[0], gui_points.value, replace=False)
        point_cloud = server.scene.add_point_cloud(
            name="/colmap/pcd",
            points=points[point_mask],
            colors=colors[point_mask],
            point_size=gui_point_size.value,        
        )
        frames: List[viser.FrameHandle] = []

        T_world_camera_metadata = {}

        N = 0  # Replace with the desired size
        denser_points = points
        denser_colors = colors

        # Remove existing image frames.
        for frame in frames:
            frame.remove()
        frames.clear()

        # Interpret the images and cameras.
        img_ids = [im.id for im in images.values()]
        # random.shuffle(img_ids)
        img_ids = sorted(img_ids[: gui_frames.value])

        def attach_callback(
            frustum: viser.CameraFrustumHandle, frame: viser.FrameHandle
        ) -> None:
            @frustum.on_click
            def _(_) -> None:
                for client in server.get_clients().values():
                    client.camera.wxyz = frame.wxyz
                    client.camera.position = frame.position        
        
        for img_id in tqdm(img_ids):
            color = (0.0, 0.0, 0.0)
            img = images[img_id]
            cam = cameras[img.camera_id]

            # Skip images that don't exist.
            image_filename = images_path / f"frame_{img.name[1:]}"  
            # print(image_filename)          
            if not image_filename.exists():
                print("not exists", image_filename)
                continue            

            final_rotation = R.from_euler("xyz", [0, 0, 90], degrees=True).as_matrix() @ R.from_quat(img.qvec).as_matrix()
            final_rotation_quad = R.from_matrix(final_rotation).as_quat()

            T_world_camera = tf.SE3.from_rotation_and_translation(
                tf.SO3(final_rotation_quad), img.tvec
            ).inverse()
            
            # print(f"Image {img.name[1:-4]}: {T_world_camera}")
            # saving the rotation and translation info to be used later
            T_world_camera_metadata[img.name[1:-4]] = T_world_camera             

            if(img.name[1:-4] == "00002" or img.name[1:-4] == "00100"):
                color = (1.0, 0.0, 1.0)
                print(T_world_camera.translation())

            frame = server.scene.add_frame(
                f"/colmap/frame_{img_id}",
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=0.1,
                axes_radius=0.005,
            )
            frames.append(frame)

            H, W = cam.height, cam.width
            fy = cam.params[1]
            image = iio.imread(image_filename)
            image = image[::downsample_factor, ::downsample_factor]            
            frustum = server.scene.add_camera_frustum(
                f"/colmap/frame_{img_id}/frustum",
                fov=2 * np.arctan2(H / 2, fy),
                aspect=W / H,
                scale=0.15,
                image=image,
                color=color,
            )
            attach_callback(frustum, frame)


        ##########################
        #     DENSER clouds      #
        ##########################        

        with open("./_assets/image_pairs.txt") as f:
            lines = f.readlines()
            pairs = [line.strip().split(",") for line in lines]

        for pair in pairs:
            file_path = f"./_assets/{pair[0]}-{pair[1]}.glb"
            
            scene = trimesh.load_mesh(file_path, process=False)
            gltf = pygltflib.GLTF2().load(file_path)
    
            # Decode buffer data if needed
            if not hasattr(gltf.buffers[0], 'data'):
                print("Buffer needs decoding...")
                gltf.buffers[0].data = gltf._glb_data
              
            color_view = gltf.bufferViews[gltf.meshes[0].primitives[0].attributes.COLOR_0]

            # Extract raw color data
            start = color_view.byteOffset
            end = start + color_view.byteLength
            
            raw_colors = np.frombuffer(
                gltf.buffers[0].data[start:end],
                dtype=np.uint8
            ).reshape(-1, 4)[:, :3]  # Reshape to RGBA
            
            # Convert to float32 and normalize to 0-1 range
            colors_float = raw_colors.astype(np.float32) / 255.0            
            # only add RGB values
            denser_colors = np.concatenate((denser_colors[:, :3], colors_float), axis=0)            

            # creating the shift from the origin to reset the pivot point to the camera origin
            # this is to make everything easier to rotate with respect to the world origin
            frustum_points = np.array(scene.geometry["geometry_2"].vertices)                

            origin_frustum = frustum_points[1]            

            points = np.array(scene.geometry["geometry_0"].vertices) - origin_frustum
            frustum_points = np.array(scene.geometry["geometry_2"].vertices) - origin_frustum

            # creating a best approximation of 4 points in the image polane of the frustum
            # TODO analytically calculate the real values from fov and fx, fy
            aspect = W / H
            best_approximation = np.array([[0.19,-0.19/aspect,0.19],[-0.19,-0.19/aspect,0.19], [-0.19,0.19/aspect,0.19],[0.19,0.19/aspect,0.19]])    

            mean_points = np.vstack((frustum_points[:1], frustum_points[2:5]))            

            frustum_mine_3 = server.scene.add_camera_frustum(
                name=f"mast3r/{pair[0]}/camera_frustum",
                fov=2 * np.arctan2(H / 2, fy),
                aspect=W / H,
                scale=0.11,
                color=(1.0, 0.0, 1.0),
                position=T_world_camera_metadata[pair[0]].translation(),
                wxyz=T_world_camera_metadata[pair[0]].rotation().wxyz                
            )

            frustum_mine_0 = server.scene.add_point_cloud(
                name=f"mast3r/{pair[0]}/frustum",
                points = mean_points,
                colors = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0]]),
                point_size = gui_point_size.value,                
            )            

            rot, _, _ = R.align_vectors(mean_points, best_approximation, return_sensitivity=True)            

            new_points_after = mean_points @ rot.as_matrix()
            frustum_after = server.scene.add_point_cloud(
                name=f"mast3r/{pair[0]}/best_approx_after_rotation",
                points = new_points_after,
                colors = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0]]),
                point_size = gui_point_size.value,
                wxyz=T_world_camera_metadata[pair[0]].rotation().wxyz
            )

            new_dense_cloud_after = points @ rot.as_matrix()
            # applying rotation and translation and scaling on the points themselves so to make it easier to export
            new_dense_cloud_after = ((new_dense_cloud_after@ R.from_quat(T_world_camera_metadata[pair[0]].rotation().wxyz, scalar_first=True).as_matrix().T) * float(pair[2])) + T_world_camera_metadata[pair[0]].translation()
            denser_points = np.concatenate((denser_points, new_dense_cloud_after), axis=0)

            denser_pc_2_after = server.scene.add_point_cloud(
                name=f"mast3r/{pair[0]}/no rotation",
                points = new_dense_cloud_after,
                colors = colors_float,
                point_size = gui_point_size.value,                
            )   

        @gui_donwload_button.on_click
        def _(event: viser.GuiEvent) -> None:

            print("Downloading...")

            try:
                client = event.client
                assert client is not None
                
                denser_colors_int = denser_colors * 255

                df = pd.DataFrame()
                df["x"] = denser_points[:,0]
                df["y"] = denser_points[:,1]
                df["z"] = denser_points[:,2]
                df["red"] = denser_colors_int[:,0].astype(np.uint8)
                df["green"] = denser_colors_int[:,1].astype(np.uint8)
                df["blue"] = denser_colors_int[:,2].astype(np.uint8)

                cloud = PyntCloud(df)

                cloud.to_file("sparse_pc.ply", as_text=True)

                client.add_notification(
                    title="Download complete",
                    body="Check your folders!",
                    loading=False,
                    with_close_button=True,
                    auto_close=5000,
                )
            
            except Exception as e:
                print(e)
                client.add_notification(
                    title="Download failed",
                    body="Something went wrong!",
                    loading=False,
                    with_close_button=True,
                    auto_close=5000,
                )                

    need_update = True

    @gui_points.on_update
    def _(_) -> None:        
        point_mask = np.random.choice(points.shape[0], gui_points.value, replace=False)
        point_cloud.points = points[point_mask]
        point_cloud.colors = colors[point_mask]

    @gui_frames.on_update
    def _(_) -> None:
        nonlocal need_update
        need_update = True

    @gui_point_size.on_update
    def _(_) -> None:
        point_cloud.point_size = gui_point_size.value

   

    while True:
        if need_update:
            need_update = False
            visualize_frames()

        time.sleep(1e-3)


if __name__ == "__main__":
    tyro.cli(main)