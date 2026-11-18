import cv2
import mujoco

xml = """<mujoco model="apt_steps_test">
  <visual>
    <quality shadowsize="4096"/>
    <headlight diffuse="0.8 0.8 0.8" ambient="0.4 0.4 0.4"/>
    <global offwidth="1280" offheight="720"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.35 0.45 0.65" rgb2="0.1 0.15 0.25" width="512" height="512"/>
    <texture type="2d" name="tex_apt" file="unitree_rl_gym/resources/robots/g1_description/apartment_1_texture.png"/>
    <material name="mat_apt" texture="tex_apt" reflectance="0.1"/>
    <mesh name="apt" file="unitree_rl_gym/resources/robots/g1_description/meshes/real_apartment_scan.obj" inertia="shell"/>
  </asset>
  <worldbody>
    <light pos="4.5 -4.5 3" dir="0 0 -1" diffuse="1 0.95 0.9"/>
    <geom type="mesh" mesh="apt" pos="0 0 0" material="mat_apt"/>
  </worldbody>
</mujoco>"""

m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
renderer = mujoco.Renderer(m, 720, 1280)

cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.lookat = [4.8, -4.8, 0.6]
cam.distance = 3.5
cam.elevation = -20.0
cam.azimuth = -135.0

renderer.update_scene(d, cam)
cv2.imwrite("preview_apt_stairs.png", cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
print("Saved preview_apt_stairs.png")
