import cv2
import numpy as np
import mujoco

xml_path = "unitree_rl_gym/resources/robots/g1_description/scene_tactical_house.xml"
m = mujoco.MjModel.from_xml_path(xml_path)
d = mujoco.MjData(m)
renderer = mujoco.Renderer(m, 720, 1280)

# Position robot at entrance porch (X=0.9m)
d.qpos[0] = 0.90
d.qpos[1] = 0.0
d.qpos[2] = 0.77
mujoco.mj_forward(m, d)

cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE

# Test 1: From behind robot looking through the front door into the house
cam.lookat = [1.5, 0.0, 0.8]
cam.distance = 3.2
cam.elevation = -12.0
cam.azimuth = 180.0  # directly facing +X

renderer.update_scene(d, cam)
cv2.imwrite("cam_test_entrance_back.png", cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))

# Test 2: Angle 3/4 front isometric
cam.lookat = [1.3, 0.0, 0.7]
cam.distance = 3.8
cam.elevation = -18.0
cam.azimuth = -140.0

renderer.update_scene(d, cam)
cv2.imwrite("cam_test_entrance_34.png", cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))

# Test 3: Cutaway / elevated architectural view showing exterior + interior corridor + stairs
cam.lookat = [2.2, 0.0, 0.9]
cam.distance = 4.8
cam.elevation = -42.0
cam.azimuth = -125.0

renderer.update_scene(d, cam)
cv2.imwrite("cam_test_overview_top.png", cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))

print("Rendered 3 camera tests.")
