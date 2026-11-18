import cv2
import numpy as np
import mujoco

xml_path = "unitree_rl_gym/resources/robots/g1_description/scene_tactical_house.xml"
m = mujoco.MjModel.from_xml_path(xml_path)
d = mujoco.MjData(m)

renderer = mujoco.Renderer(m, 720, 1280)

# 1. Perspective 1: Vue d'ensemble de la Villa 3D réelle téléchargée d'Internet avec le robot devant l'entrée
d.qpos[0] = 0.50
d.qpos[1] = 0.00
d.qpos[2] = 0.77
mujoco.mj_forward(m, d)

cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.lookat = [3.5, 0.0, 2.2]
cam.distance = 15.0
cam.elevation = -18.0
cam.azimuth = -30.0

renderer.update_scene(d, cam)
cv2.imwrite("preview_scene_tactical_house_facade.png", cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
print("Saved preview_scene_tactical_house_facade.png (Exterior Real 3D Villa)")

# 2. Perspective 2: Vue de suivi 3/4 du robot entrant et montant l'escalier en marbre
d.qpos[0] = 2.85
d.qpos[1] = 0.02
d.qpos[2] = 0.86
mujoco.mj_forward(m, d)

cam.lookat = [3.2, 0.0, 0.95]
cam.distance = 2.6
cam.elevation = -12.0
cam.azimuth = 15.0

renderer.update_scene(d, cam)
cv2.imwrite("preview_scene_tactical_house_stairs.png", cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
print("Saved preview_scene_tactical_house_stairs.png (Interior Marble Stairs)")

# 3. Perspective 3: Robot head camera FPV
renderer.update_scene(d, camera="head_camera")
cv2.imwrite("preview_scene_tactical_house_fpv.png", cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
print("Saved preview_scene_tactical_house_fpv.png (FPV Inside Real Villa)")

print("All real villa previews generated successfully.")
