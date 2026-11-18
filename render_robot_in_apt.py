import cv2
import mujoco

xml_path = "unitree_rl_gym/resources/robots/g1_description/test_apt_sim.xml"
m = mujoco.MjModel.from_xml_path(xml_path)
d = mujoco.MjData(m)

# Place robot in the dining room
d.qpos[0] = 3.8
d.qpos[1] = -5.6
d.qpos[2] = 0.77
mujoco.mj_forward(m, d)

renderer = mujoco.Renderer(m, 720, 1280)

# Camera 1: Looking at the robot beside the dining table in the real apartment
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.lookat = [3.8, -5.0, 0.8]
cam.distance = 3.6
cam.elevation = -18.0
cam.azimuth = -140.0

renderer.update_scene(d, cam)
cv2.imwrite("preview_robot_in_real_apartment.png", cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
print("Saved preview_robot_in_real_apartment.png")

# Camera 2: Robot passing through the real apartment door
d.qpos[0] = 2.6
d.qpos[1] = -3.5
d.qpos[2] = 0.77
mujoco.mj_forward(m, d)

cam.lookat = [2.6, -3.2, 0.8]
cam.distance = 2.8
cam.elevation = -15.0
cam.azimuth = 60.0

renderer.update_scene(d, cam)
cv2.imwrite("preview_robot_entering_door.png", cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
print("Saved preview_robot_entering_door.png")
