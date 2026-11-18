import os
import cv2
import numpy as np
import mujoco

xml_path = "unitree_rl_gym/resources/robots/g1_description/scene_tactical_house.xml"
m = mujoco.MjModel.from_xml_path(xml_path)
d = mujoco.MjData(m)

renderer = mujoco.Renderer(m, 720, 1280)

# 1. Perspective 1: Overview of the House exterior facade and open entrance door
d.qpos[0] = 0.85
d.qpos[1] = 0.0
d.qpos[2] = 0.77
mujoco.mj_forward(m, d)

cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.lookat = [2.2, 0.0, 0.8]
cam.distance = 5.2
cam.elevation = -22.0
cam.azimuth = 140.0

renderer.update_scene(d, cam)
rgb_facade = renderer.render()
cv2.imwrite("preview_scene_tactical_house_facade.png", cv2.cvtColor(rgb_facade, cv2.COLOR_RGB2BGR))
print("Saved preview_scene_tactical_house_facade.png")

# 2. Perspective 2: Robot ascending the marble staircase inside the house
d.qpos[0] = 2.85
d.qpos[1] = 0.02
d.qpos[2] = 0.86
mujoco.mj_forward(m, d)

cam.lookat = [3.0, 0.0, 0.9]
cam.distance = 3.2
cam.elevation = -18.0
cam.azimuth = 155.0

renderer.update_scene(d, cam)
rgb_stairs = renderer.render()
cv2.imwrite("preview_scene_tactical_house_stairs.png", cv2.cvtColor(rgb_stairs, cv2.COLOR_RGB2BGR))
print("Saved preview_scene_tactical_house_stairs.png")

# 3. Perspective 3: Robot head camera FPV inside the vestibule looking at the staircase
head_cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
renderer.update_scene(d, camera="head_camera")
rgb_fpv = renderer.render()
cv2.imwrite("preview_scene_tactical_house_fpv.png", cv2.cvtColor(rgb_fpv, cv2.COLOR_RGB2BGR))
print("Saved preview_scene_tactical_house_fpv.png")

print("All house previews generated successfully.")
