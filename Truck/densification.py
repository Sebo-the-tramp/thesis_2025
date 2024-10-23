import random
import time
from pathlib import Path
from typing import List

import imageio.v3 as iio
import numpy as np
import tyro
from tqdm.auto import tqdm
import trimesh

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
        initial_value=min(len(images), 100),
    )
    gui_point_size = server.gui.add_slider(
        "Point size", min=0.01, max=0.1, step=0.001, initial_value=0.01
    )

    gui_shift_size = server.gui.add_slider(
        "Shift", min=-10, max=10, step=0.1, initial_value=0
    )

    points = np.array([points3d[p_id].xyz for p_id in points3d])
    colors = np.array([points3d[p_id].rgb for p_id in points3d])

    point_mask = np.random.choice(points.shape[0], gui_points.value, replace=False)
    point_cloud = server.scene.add_point_cloud(
        name="/colmap/pcd",
        points=points[point_mask],
        colors=colors[point_mask],
        point_size=gui_point_size.value,
        wxyz=R.from_euler("xyz", [180, 0, -90], degrees=True).as_quat(),
    )
    frames: List[viser.FrameHandle] = []

    def visualize_frames() -> None:
        """Send all COLMAP elements to viser for visualization. This could be optimized
        a ton!"""

        target_rot = None
        target_trans = None

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

            # print(img.qvec, img.tvec)

            final_rotation = R.from_euler("xyz", [0, 0, 90], degrees=True).as_matrix() @ R.from_quat(img.qvec).as_matrix()
            final_rotation_quad = R.from_matrix(final_rotation).as_quat()

            T_world_camera = tf.SE3.from_rotation_and_translation(
                tf.SO3(final_rotation_quad), img.tvec
            ).inverse()

            if(img.name[1:] == "00002.jpg"):                
                target_rot = T_world_camera.rotation().wxyz
                target_trans = T_world_camera.translation()                 
                color = (1.0, 0.0, 1.0)    

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


        # adding new cloudpoint       
        scene = trimesh.load_mesh("./_assets/tmpw3y8h0ai_scene.glb", process=False)
        
        # adding new origin to shift the rotation point to the frustum origin point
        new_origin = np.array([-0.24831064,  0.37235096,  2.46506047])       
        points = np.array(scene.geometry["geometry_0"].vertices) - new_origin        

        frustum_points = np.array(scene.geometry["geometry_2"].vertices) - new_origin             


        # computing rotation of the frustum of the pc
        mean_points = np.vstack((frustum_points[:1], frustum_points[2:5]))        

        aspect = W / H                
        best_approximation = np.array([[0.19,-0.19/aspect,0.19],[-0.19,-0.19/aspect,0.19], [-0.19,0.19/aspect,0.19],[0.19,0.19/aspect,0.19]])    
        # best_approximation = best_approximation @ R.from_quat(target_rot, scalar_first=True).as_matrix()    

        denser_pc_2 = server.scene.add_point_cloud(
            name="mast3r/image center",
            # points = np.array([[0.18, 0.2, 0.18/aspect],[0.18, 0.2, -0.18/aspect], [-0.18, 0.2, 0.18/aspect], [-0.18, 0.2, -0.18/aspect]]),
            points = best_approximation, 
            colors = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0]]),
            point_size = 0.01,
            wxyz=R.from_quat(target_rot, scalar_first=True).as_quat(scalar_first=True) 
            # wxyz=R.from_matrix(result_rotation).as_quat(scalar_first=True)
        )   

        frustum_mine_3 = server.scene.add_camera_frustum(
            name="Camera Frustum Mine Target 3",
            fov=2 * np.arctan2(H / 2, fy),
            aspect=W / H,
            scale=0.11,
            color=(1.0, 0.0, 1.0),
            wxyz=R.from_quat(target_rot, scalar_first=True).as_quat(scalar_first=True)
            # wxyz=np.array([0, 0, 0, 1]),
        )

        frustum_mine_0 = server.scene.add_point_cloud(
            name="mast3r/frustum",
            points = mean_points,
            colors = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0]]),
            point_size = gui_point_size.value,            
            # wxyz=R.from_matrix(result_rotation).as_quat(scalar_first=True)
        )     

        rot, _, _ = R.align_vectors(mean_points, best_approximation, return_sensitivity=True)        

        new_points_after = mean_points @ rot.as_matrix()
        denser_pc_2_after = server.scene.add_point_cloud(
            name="mast3r/best_approx_after_rotation",
            points = new_points_after,
            colors = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0]]),
            point_size = gui_point_size.value,
            wxyz=R.from_quat(target_rot, scalar_first=True).as_quat(scalar_first=True)  
            # wxyz=R.from_matrix(rot.as_matrix()).as_quat(scalar_first=True),
            # position=new_origin
        )

        new_dense_cloud_after = points @ rot.as_matrix()
        denser_pc_2_after = server.scene.add_point_cloud(
            name="mast3r/scene_after_rotation",
            points = new_dense_cloud_after*1.7,
            colors = np.ones_like(points)*0.5,
            point_size = gui_point_size.value,             
            wxyz=R.from_quat(target_rot, scalar_first=True).as_quat(scalar_first=True), 
            position=target_trans
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

    @gui_shift_size.on_update
    def _(_) -> None:
        shift = np.array([gui_shift_size.value, 0, 0])
        frustum_pc.points = frustum_points + shift

    


    while True:
        if need_update:
            need_update = False
            visualize_frames()

        time.sleep(1e-3)


if __name__ == "__main__":
    tyro.cli(main)