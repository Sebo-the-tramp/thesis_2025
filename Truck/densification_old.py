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
    colmap_path: Path = Path(__file__).parent / "0",
    images_path: Path = Path(__file__).parent / "images_8",
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

    theta = 0
    gui_shift_theta = server.gui.add_slider(
        "thate", min=-180, max=180, step=1, initial_value=0
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
                print("HEREEEE")           
                # print(T_world_camera.rotation())
                # print(T_world_camera.translation())
                target_rot = T_world_camera.rotation().wxyz
                target_trans = T_world_camera.translation()
                # point_origin_frustum = scene.geometry["geometry_2"].vertices[1]                
                # diff = T_world_camera.translation() - point_origin_frustum                
                # first_matrix = T_world_camera.rotation().wxyz                
                # print("diff", diff)        
                color = (1.0, 0.0, 0.0)    

            frame = server.scene.add_frame(
                f"/colmap/frame_{img_id}",
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=0.1,
                axes_radius=0.005,
            )
            frames.append(frame)

            # For pinhole cameras, cam.params will be (fx, fy, cx, cy).
            # if cam.model != "PINHOLE":
            #     print(f"Expected pinhole camera, but got {cam.model}")

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
        scene = trimesh.load_mesh("./tmpw3y8h0ai_scene.glb", process=False)
        pc_denser_rotation = R.from_euler("xyz", [0,-90, 90], degrees=True).as_matrix()
        new_origin = np.array([-0.24831064,  0.37235096,  2.46506047])       
        points = np.array(scene.geometry["geometry_0"].vertices) - new_origin
        # points = points @ pc_denser_rotation

        frustum_points = np.array(scene.geometry["geometry_2"].vertices) - new_origin     
        # frustum_points = frustum_points @ pc_denser_rotation

        frustum_mine_1 = server.scene.add_camera_frustum(
            name="mast3r/camera",
            fov=2 * np.arctan2(H / 2, fy),
            aspect=W / H,
            scale=0.11,
            color=(1.0, 0.0, 0.0),              
        )

        # computing rotation of the frustum of the pc
        mean_points = np.vstack((frustum_points[:1], frustum_points[2:5]))        
        c_plane = np.mean(mean_points, axis=0).reshape(1, 3)

        forward = c_plane - frustum_points[1]

        print("forward", forward)
        
        # project forward to the xy plane
        z = np.array([0, 0, 1])
        v_proj_z = forward - np.dot(forward, z) * z
        z_unit = v_proj_z / np.linalg.norm(v_proj_z)
        cos_theta = np.dot(z_unit, np.array([0, 1, 0]))
        theta = np.arccos(cos_theta)
        z_theta_deg = np.degrees(theta)
        print("thetaZ", z_theta_deg)

        # project to the xz plane
        y = np.array([0, 1, 0])
        v_proj_y = forward - np.dot(forward, y) * y
        print("v_proj_y", v_proj_y)
        y_unit = v_proj_y / np.linalg.norm(v_proj_y)
        cos_theta = np.dot(y_unit, np.array([0, 0, 1]))
        theta = np.arccos(cos_theta)
        y_theta_deg = np.degrees(theta)
        print("thetaY", y_theta_deg)

        # project to the yz plane
        x = np.array([1, 0, 0])
        v_proj_x = forward - np.dot(forward, x) * x
        print("v_proj_x", v_proj_x)
        x_unit = v_proj_x / np.linalg.norm(v_proj_x)
        cos_theta = np.dot(x_unit, np.array([0, 0, 1]))
        theta = np.arccos(cos_theta)
        x_theta_deg = np.degrees(theta)
        print("thetaX", x_theta_deg)

        
        new_points = frustum_points[:5]
        new_points = np.concatenate((new_points, c_plane), axis=0)

        src_matrix = R.from_euler("xyz", [(180-x_theta_deg[0]),y_theta_deg[0],-5], degrees=True).as_matrix()
        # src_matrix = R.from_euler("xyz", [-x_theta_deg[0], 180-y_theta_deg[0], 175], degrees=True).as_matrix()
        src_quat = R.from_matrix(src_matrix).as_quat(scalar_first=True)

        fov = 2 * np.arctan2(H / 2, fy),
        print("fov", fov)

        frustum_mine_1 = server.scene.add_camera_frustum(
            name="Camera Frustum Mine SRC",
            fov=fov,
            aspect=W / H,
            scale=0.11,
            color=(1.0, 0.0, 0.0),
            wxyz=src_quat,
        )

        focal_length = W/2 / np.tan(fov[0]/2)
        print("focal_length", focal_length)

        print("Aspect", W / H)
        aspect = W / H

        print("FY", fy) 
        
        best_approximation = np.array([[0.19,-0.19/aspect,0.19],[-0.19,-0.19/aspect,0.19], [-0.19,0.19/aspect,0.19],[0.19,0.19/aspect,0.19]])        

        denser_pc_2 = server.scene.add_point_cloud(
            name="mast3r/image center",
            # points = np.array([[0.18, 0.2, 0.18/aspect],[0.18, 0.2, -0.18/aspect], [-0.18, 0.2, 0.18/aspect], [-0.18, 0.2, -0.18/aspect]]),
            points = best_approximation, 
            colors = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0]]),
            point_size = 0.01,
            wxyz=R.from_quat(target_rot, scalar_first=True).as_quat(scalar_first=True) 
            # wxyz=R.from_matrix(result_rotation).as_quat(scalar_first=True)
        )   

        print("SRC", R.from_matrix(src_matrix).as_euler("xyz", degrees=True))
        print("TARGETT", R.from_quat(target_rot,scalar_first=True).as_euler("xyz", degrees=True))
        
        target_rot_matrix = R.from_quat(target_rot, scalar_first=True).as_matrix()
        print("TARGETT", R.from_matrix(target_rot_matrix).as_euler("xyz", degrees=True))

        rot_quaternion = R.from_matrix(target_rot_matrix).as_quat(scalar_first=True)
        print("ROT QUAT", R.from_quat(rot_quaternion, scalar_first=True).as_euler("xyz", degrees=True))

        src_target_matrix = np.dot(target_rot_matrix, src_matrix.T)
        print("SRC TARGET", R.from_matrix(src_target_matrix).as_euler("xyz", degrees=True))

        result_rotation = src_target_matrix @ src_matrix

        print("RESULT", R.from_matrix(result_rotation).as_euler("xyz", degrees=True))

        # frustum_mine_2 = server.scene.add_camera_frustum(
        #     name="Camera Frustum Mine Target",
        #     fov=2 * np.arctan2(H / 2, fy),
        #     aspect=W / H,
        #     scale=0.11,
        #     color=(1.0, 0.0, 0.0),
        #     wxyz=R.from_matrix(result_rotation).as_quat(scalar_first=True)
        # )

        # print("test", R.from_quat([0, 0, 0, 1], scalar_first=True).as_euler("xyz", degrees=True))

        print("TARGET", R.from_quat(target_rot, scalar_first=True).as_euler("xyz", degrees=True))

        frustum_mine_3 = server.scene.add_camera_frustum(
            name="Camera Frustum Mine Target 3",
            fov=2 * np.arctan2(H / 2, fy),
            aspect=W / H,
            scale=0.11,
            color=(1.0, 0.0, 1.0),
            wxyz=R.from_quat(target_rot, scalar_first=True).as_quat(scalar_first=True)
            # wxyz=np.array([0, 0, 0, 1]),
        )


        denser_pc_2 = server.scene.add_point_cloud(
            name="mast3r/scene",
            points = points,
            colors = np.ones_like(points)*0.5,
            point_size = gui_point_size.value,             
            # wxyz=R.from_matrix(result_rotation).as_quat(scalar_first=True)
        )   

        frustum_mine_0 = server.scene.add_point_cloud(
            name="mast3r/frustum",
            points = mean_points,
            colors = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0]]),
            point_size = gui_point_size.value,            
            # wxyz=R.from_matrix(result_rotation).as_quat(scalar_first=True)
        )

        # add rotation 180 degree
        rotation_x_180 = R.from_euler("xyz", [180, 180, 0], degrees=True).as_matrix()
        new_result_rotation = rotation_x_180 @ result_rotation        


        denser_pc_2_after = server.scene.add_point_cloud(
            name="mast3r/scene_after_rotation",
            points = points*1.7,
            colors = np.ones_like(points)*0.5,
            point_size = gui_point_size.value,             
            wxyz=R.from_matrix(new_result_rotation).as_quat(scalar_first=True),
            position=target_trans
        )   

 

        # translation_diff = frustum_points[1] - target_trans
        # src_target_matrix = np.dot(R.from_quat(target_rot).as_matrix(), src_rotation.T)

        # frustum_mine_2 = server.scene.add_camera_frustum(
        #     name="Camera Frustum Mine Target",
        #     fov=2 * np.arctan2(H / 2, fy),
        #     aspect=W / H,
        #     scale=0.11,
        #     color=(1.0, 0.0, 0.0),  
        #     position=frustum_points[1],
        #     wxyz=R.from_matrix(src_rotation).as_quat()
        # )

        # frustum_mine_3 = server.scene.add_camera_frustum(
        #     name="Camera Frustum Mine SRC->TARGET",
        #     fov=2 * np.arctan2(H / 2, fy),
        #     aspect=W / H,
        #     scale=0.11,
        #     color=(1.0, 0.0, 1.0),  
        #     position=frustum_points[1],
        #     wxyz=R.from_matrix(src_target_matrix).as_quat()
        # )        
       

        # @gui_shift_theta.on_update
        # def _(_) -> None:
        #     theta = gui_shift_theta.value
        #     print("E")
        #     new_rotation_matrix = R.from_euler("xyz", [0, 0, theta], degrees=True).as_matrix() @ rotation_matrix
        #     print(R.from_matrix(new_rotation_matrix).as_quat())
        #     frustum_mine_3.wxyz = R.from_matrix(new_rotation_matrix).as_quat()

        #     frustum_mine_1.wxyz = R.from_euler("xyz", [0, 0, theta], degrees=True).as_quat()

        #     denser_pc_2.wxyz = R.from_euler("zyx", [0, 0, theta], degrees=True).as_quat()


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