import cv2
import mujoco

xml = """<mujoco model="villa_preview">
  <statistic center="0 0 3" extent="20"/>
  <visual>
    <quality shadowsize="4096"/>
    <headlight diffuse="0.8 0.8 0.8" ambient="0.3 0.3 0.3"/>
    <global offwidth="1280" offheight="720"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.35 0.45 0.65" rgb2="0.1 0.15 0.25" width="512" height="512"/>
    <texture type="2d" name="tex_apt" file="unitree_rl_gym/resources/robots/g1_description/apartment_1_texture.png"/>
    <material name="mat_apt" texture="tex_apt" reflectance="0.1"/>
    <mesh name="apt" file="unitree_rl_gym/resources/robots/g1_description/meshes/real_apartment_scan.obj" inertia="shell"/>
  </asset>
  <worldbody>
    <light pos="2 -2 3" dir="0 0 -1" diffuse="1 0.95 0.9" specular="0.2 0.2 0.2"/>
    <light pos="4 -5 3" dir="0 0 -1" diffuse="0.8 0.8 0.8" specular="0.1 0.1 0.1"/>
    <geom type="mesh" mesh="apt" pos="0 0 0" material="mat_apt"/>
  </worldbody>
</mujoco>"""

m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)

renderer = mujoco.Renderer(m, 720, 1280)
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.lookat = [3.0, -3.0, 0.8]
cam.distance = 6.0
cam.elevation = -20.0
cam.azimuth = 120.0

renderer.update_scene(d, cam)
cv2.imwrite("preview_real_villa_internet.png", cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
print("Saved preview_real_villa_internet.png")
