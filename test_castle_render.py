import cv2
import mujoco

xml = """<mujoco model="skokloster_robot_test">
  <include file="g1_12dof.xml"/>
  <visual>
    <quality shadowsize="4096"/>
    <headlight diffuse="0.8 0.8 0.8" ambient="0.4 0.4 0.4"/>
    <global offwidth="1280" offheight="720"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.35 0.45 0.65" rgb2="0.1 0.15 0.25" width="512" height="512"/>
    <texture type="2d" name="tex_castle" file="skokloster_texture_2k.png"/>
    <material name="mat_castle" texture="tex_castle" reflectance="0.1"/>
    <mesh name="castle" file="skokloster_castle.obj" inertia="shell"/>
  </asset>
  <worldbody>
    <light pos="0 -14 4.5" dir="0 0 -1" diffuse="1 0.95 0.9" castshadow="true"/>
    <geom type="mesh" mesh="castle" pos="0 0 0" material="mat_castle" contype="0" conaffinity="0"/>
    <geom name="castle_floor" type="plane" size="25 25 0.05" pos="0 -14 0.02" rgba="0 0 0 0"/>
  </worldbody>
</mujoco>"""

with open("unitree_rl_gym/resources/robots/g1_description/test_castle.xml", "w") as f:
    f.write(xml)

m = mujoco.MjModel.from_xml_path("unitree_rl_gym/resources/robots/g1_description/test_castle.xml")
d = mujoco.MjData(m)

# Place robot walking in front of the table
d.qpos[0] = 0.0
d.qpos[1] = -11.0
d.qpos[2] = 0.80
mujoco.mj_forward(m, d)

renderer = mujoco.Renderer(m, 720, 1280)
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.lookat = [0.0, -13.5, 0.9]
cam.distance = 4.5
cam.elevation = -12.0
cam.azimuth = 180.0

renderer.update_scene(d, cam)
cv2.imwrite("preview_robot_in_real_castle.png", cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
print("Saved preview_robot_in_real_castle.png")
