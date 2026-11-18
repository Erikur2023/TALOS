import cv2
import mujoco

xml = """<mujoco model="skokloster_stairs_find">
  <visual>
    <quality shadowsize="4096"/>
    <headlight diffuse="0.8 0.8 0.8" ambient="0.4 0.4 0.4"/>
    <global offwidth="1280" offheight="720"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.35 0.45 0.65" rgb2="0.1 0.15 0.25" width="512" height="512"/>
    <texture type="2d" name="tex_castle" file="unitree_rl_gym/resources/robots/g1_description/skokloster_texture_2k.png"/>
    <material name="mat_castle" texture="tex_castle" reflectance="0.1"/>
    <mesh name="castle" file="unitree_rl_gym/resources/robots/g1_description/meshes/skokloster_castle.obj" inertia="shell"/>
  </asset>
  <worldbody>
    <light pos="-3 -23 5" dir="0 0 -1" diffuse="1 0.95 0.9"/>
    <geom type="mesh" mesh="castle" pos="0 0 0" material="mat_castle"/>
  </worldbody>
</mujoco>"""

m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
renderer = mujoco.Renderer(m, 720, 1280)

cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.lookat = [-3.0, -22.0, 1.8]
cam.distance = 5.0
cam.elevation = -10.0
cam.azimuth = 45.0

renderer.update_scene(d, cam)
cv2.imwrite("preview_castle_staircase.png", cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
print("Saved preview_castle_staircase.png")
