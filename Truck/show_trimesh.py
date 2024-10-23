import numpy as np

import trimesh

scene = trimesh.load("./tmpw3y8h0ai_scene.glb", force="scene")
# scene = trimesh.load("./tmpw3y8h0ai_scene.gltf", force="scene")

print(scene)

scene.show()

# # maybe iamge, frustum, image, fustum, pointcloud
# print(scene.geometry)
print(scene.geometry["geometry_0"])
print(scene.geometry["geometry_0"].metadata)
print(scene.geometry["geometry_1"].vertices)

# scene.geometry["geometry_0"].show()
# # 
# scene.geometry["geometry_1"].show() # image
scene.geometry["geometry_2"] # frsutum

print(scene.geometry["geometry_2"].vertices)

# scene.geometry["geometry_2"].verticse = scene.geometry["geometry_2"][:2]
scene.geometry["geometry_2"].show()

## PROJECT TH IMAGE TO THE POINT CLOUD
