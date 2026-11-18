import cv2
import mujoco

xml_path = "unitree_rl_gym/resources/robots/g1_description/test_castle.xml"
m = mujoco.MjModel.from_xml_path(xml_path)
d = mujoco.MjData(m)

# Place robot on the stairs at X = -3.0, Y = -22.5
# Let's find the floor height at this spot
d.qpos[0] = -3.0
d.qpos[1] = -22.5
d.qpos[2] = 1.65 # mid-stair height
mujoco.mj_forward(m, d)

renderer = mujoco.Renderer(m, 720, 1280)
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.lookat = [-3.0, -22.5, 1.8]
cam.distance = 3.5
cam.elevation = -12.0
cam.azimuth = 60.0

renderer.update_scene(d, cam)
cv2.imwrite("preview_robot_on_castle_stairs.png", cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
print("Saved preview_robot_on_castle_stairs.png")
