import random
import time
from pathlib import Path
from typing import List

import imageio.v3 as iio
import numpy as np
import tyro
from tqdm.auto import tqdm

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
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
    

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
        initial_value=10_000,
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
    
    gui_pointX = server.gui.add_slider(
        "X",
        min=-10,
        max=10,
        step=0.1,
        initial_value=2.7,
    )
    
    gui_pointY = server.gui.add_slider(
        "Y",
        min=-10,
        max=10,
        step=0.1,
        initial_value=0.5,
    )
    
    gui_pointZ = server.gui.add_slider(
        "Z",
        min=-10,
        max=10,
        step=0.1,
        initial_value=-5.9,
    )
    
    scale = server.gui.add_slider(
        "Scale",
        min=0.1,
        max=10,
        step=0.1,
        initial_value=1,
    )
    
    angle_theta = server.gui.add_slider(
        "Theta",
        min=-180,
        max=180,
        step=1,
        initial_value=8,
    )

    points = np.array([points3d[p_id].xyz for p_id in points3d])
    colors = np.array([points3d[p_id].rgb for p_id in points3d])

    point_mask = np.random.choice(points.shape[0], gui_points.value, replace=False)
    point_cloud = server.scene.add_point_cloud(
        name="/colmap/pcd",
        points=points[point_mask],
        colors=colors[point_mask],
        point_size=gui_point_size.value,
    )
    frames: List[viser.FrameHandle] = []
    
    glb_binary = open("tmpw3y8h0ai_scene.glb", "rb").read()
    # glb_binary = open("tmpw3y8h0ai_scene.gltf", "rb").read()
    theta = angle_theta.value  # Replace with your desired angle for Z-axis rotation
    rotation = R.from_euler('xy', [0, theta], degrees=True)        
    
    import trimesh
    scene = trimesh.load("tmpw3y8h0ai_scene.glb")
    
    # Check if the loaded object is a Scene
    if isinstance(scene, trimesh.Scene):
        # Combine all geometries in the scene into a single mesh
        mesh = trimesh.util.concatenate([geometry for geometry in scene.geometry.values()])
    else:
        # If it's already a mesh, just use it
        mesh = scene    

    print(point_cloud)
    
    scene = trimesh.load_mesh("./tmpw3y8h0ai_scene.glb", process=False)

    # maybe iamge, frustum, image, fustum, pointcloud
    print(scene.geometry.keys())
    print(scene.geometry["geometry_0"])
    
    points = np.array(scene.geometry["geometry_0"].vertices)
    print(points)
    
    # denser_pc = server.scene.add_glb(
    #     name="Denser Point Cloud",
    #     glb_data = glb_binary,
    #     position = np.array([gui_pointX.value, gui_pointY.value, gui_pointZ.value]),
    #     wxyz = rotation.as_quat(),
    #     scale = scale.value,
    # )


    
    # frustum_1 = server.scene.add_camera_frustum(
    #     name="Camera Frustum 1",
    #     fov=60,
    #     aspect=16/9,
    #     scale=0.15,
    #     color=(1.0, 0.0, 0.0),
    #     position=scene.geometry["geometry_2"].vertices[1],
    #     wxyz=(0.0, 0.0, 0.0, 1.0),
    # )
    
    frustum_points = np.array(scene.geometry["geometry_2"].vertices)
    print(frustum_points)                   
    
    
    print(server.scene)        
    
    print("added denser point cloud")

    def visualize_frames() -> None:
        """Send all COLMAP elements to viser for visualization. This could be optimized
        a ton!"""

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

        point_origin_frustum = None
        diff = None
        first_matrix = None

        for img_id in tqdm(img_ids):
            img = images[img_id]
            cam = cameras[img.camera_id]

            # Skip images that don't exist.
            image_filename = images_path / f"frame_{img.name[1:]}"            
            if not image_filename.exists():
                print("not exists", image_filename)
                continue

            T_world_camera = tf.SE3.from_rotation_and_translation(
                tf.SO3(img.qvec), img.tvec
            ).inverse()
            
            color = (0.0, 0.0, 0.0)
                        
            if(img.name[1:] == "00002.jpg"):
                print("HEREEEE")           
                print(T_world_camera.rotation())
                print(T_world_camera.translation())
                color = (1.0, 0.0, 0.0)
                
                point_origin_frustum = scene.geometry["geometry_2"].vertices[1]
                
                diff = T_world_camera.translation() - point_origin_frustum
                
                first_matrix = T_world_camera.rotation().wxyz
                
                print("diff", diff)                                

                
            if(img.name[1:] == "00013.jpg"):
                print("HEREEEE")           
                print(T_world_camera.rotation())
                print(T_world_camera.translation())
                color = (0.0, 1.0, 0.0)
            
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
            
            
        # computing rotation of the frustum of the pc
        mean_points = np.vstack((frustum_points[:1], frustum_points[2:5]))        
        c_plane = np.mean(mean_points, axis=0).reshape(1, 3)
        
        new_points = frustum_points[:5]
        new_points = np.concatenate((new_points, c_plane), axis=0)
        
        forward_vector = frustum_points[1] - c_plane 
        forward_vector = forward_vector / np.linalg.norm(forward_vector)
        print("f", forward_vector)
        
        right_vector = [frustum_points[0]] - np.array(frustum_points[2])
        right_vector = right_vector / np.linalg.norm(right_vector)
        print("r", right_vector)
        
        up_vector = np.cross(right_vector, forward_vector)
        up_vector = up_vector / np.linalg.norm(up_vector)
        print("u", up_vector)
        
        rotation_matrix = np.array([right_vector[0], up_vector[0], -forward_vector[0]]).T
        print("r", rotation_matrix)
        
        # print(R.from_matrix(rotation_matrix).as_quat())
        
        denser_pc = server.scene.add_point_cloud(
            name="frustum_points",
            points = frustum_points, 
            colors = np.array([[1.0, 0.0, 0.0]]).repeat(frustum_points.shape[0], axis=0),
            point_size = gui_point_size.value,                     
        )
    
        
        server.scene.add_camera_frustum(
            name="Camera Frustum",
            fov=2 * np.arctan2(H / 2, fy),
            aspect=W / H,
            scale=0.11,
            color=(1.0, 0.0, 0.0),
            position=frustum_points[1]+diff,
            wxyz=R.from_matrix(rotation_matrix).as_quat("wxzy"),
        )
    
        print(first_matrix)
        

        
        server.scene.add_camera_frustum(
            name="Camera Frustum B",
            fov=2 * np.arctan2(H / 2, fy),
            aspect=W / H,
            scale=0.11,
            color=(1.0, 0.0, 1.0),
            position=frustum_points[1]+diff,
            wxyz=first_matrix,
            # wxyz=R.from_matrix(to_rotate_matrix).as_quat("wxzy"),
        ) 

       
        # first matrix is the rotation of the first camera that needs to be reached        
        first_matrix_euler = R.from_quat(first_matrix).as_euler("xyz", degrees=True)
        print("first_matrix", first_matrix)
        
        rotation_matrix_euler = R.from_matrix(rotation_matrix).as_euler("xyz", degrees=True)
        print("rotation_matrix", rotation_matrix)
        
        first_matrix_euler_norm = first_matrix_euler / np.linalg.norm(first_matrix_euler)
        rotation_matrix_euler_norm = rotation_matrix_euler / np.linalg.norm(rotation_matrix_euler)
        
        rotation_axis = np.cross(first_matrix_euler_norm, rotation_matrix_euler_norm)
        sin_theta = np.linalg.norm(rotation_axis)
        cos_theta = np.dot(first_matrix_euler_norm, rotation_matrix_euler_norm)
        
        #check for edge case
        
        if sin_theta == 0:
            if cos_theta == 1:
                rotation = R.from_euler("xyz", [0, 0, 0], degrees=True)
            elif cos_theta == -1:
                rotation = R.from_euler("xyz", [180, 0, 0], degrees=True)
                
        rotation_axis = rotation_axis / sin_theta
        
        skew_matrix = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                                [rotation_axis[2], 0, -rotation_axis[0]],
                                [-rotation_axis[1], rotation_axis[0], 0]])
        
        to_rotate_matrix = np.eye(3) + sin_theta * skew_matrix + (1 - cos_theta) * np.dot(skew_matrix, skew_matrix)
        
        print("to_rotate_matrix", R.from_matrix(to_rotate_matrix).as_euler("xyz", degrees=True))                
        
        denser_pc_2 = server.scene.add_point_cloud(
            name="Denser Point Cloud",
            points = points,
            colors = np.ones_like(points)*0.5,
            point_size = gui_point_size.value,
            position=diff,
            # wxyz=first_matrix,

            wxyz= R.from_matrix(to_rotate_matrix).as_quat("wxzy"),
            # wxyz=R.from_matrix(rotation_matrix).as_quat("wxzy"),
            
        )        
        
        # server.scene.add_camera_frustum(
        #     name="Camera Frustum B",
        #     fov=2 * np.arctan2(H / 2, fy),
        #     aspect=W / H,
        #     scale=0.11,
        #     color=(1.0, 0.0, 1.0),
        #     position=frustum_points[1]+diff,
        #     wxyz=first_matrix,
        #     # wxyz=R.from_matrix(to_rotate_matrix).as_quat("wxzy"),
        # ) 


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
        denser_pc.point_size = gui_point_size.value
        
    @gui_pointX.on_update
    def _(_) -> None:
        denser_pc.position = np.array([gui_pointX.value, 0, gui_pointZ.value])
        
    @gui_pointY.on_update
    def _(_) -> None:
        denser_pc.position = np.array([gui_pointX.value, gui_pointY.value, gui_pointZ.value])
        
    @gui_pointZ.on_update
    def _(_) -> None:
        denser_pc.position = np.array([gui_pointX.value, 0, gui_pointZ.value])
    
    @angle_theta.on_update
    def _(_) -> None:
        theta = angle_theta.value  # Replace with your desired angle for Z-axis rotation
        rotation = R.from_euler('xy', [0, theta], degrees=True)
        denser_pc.wxyz = rotation.as_quat()
        
    @scale.on_update
    def _(_) -> None:
        denser_pc.scale = scale.value        

    while True:
        if need_update:
            need_update = False
            visualize_frames()

        time.sleep(1e-3)


if __name__ == "__main__":
    tyro.cli(main)