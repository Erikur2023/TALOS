import cv2
import mujoco

xml = """<mujoco model="skokloster_preview">
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
    <light pos="0 -10 5" dir="0 0.5 -1" diffuse="1 0.95 0.9"/>
    <light pos="0 -18 5" dir="0 0.5 -1" diffuse="0.9 0.9 0.9"/>
    <geom type="mesh" mesh="castle" pos="0 0 0" material="mat_castle"/>
  </worldbody>
</mujoco>"""

m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
renderer = mujoco.Renderer(m, 720, 1280)

cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE

views = [
    # Look at staircase from ground floor hall
    {"lookat": [0.0, -14.0, 1.5], "dist": 6.5, "elev": 5.0, "azim": 90.0, "name": "staircase_look_up"},
    {"lookat": [0.0, -14.0, 2.5], "dist": 9.0, "elev": -15.0, "azim": 0.0, "name": "grand_hall"},
    {"lookat": [0.0, -12.0, 3.0], "dist": 14.0, "elev": -35.0, "azim": 145.0, "name": "overview"}
]

for v in views:
    cam.lookat = v["lookat"]
    cam.distance = v["dist"]
    cam.elevation = v["elev"]
    cam.azimuth = v["azim"]
    renderer.update_scene(d, cam)
    cv2.imwrite(f"castle_view_{v['name']}.png", cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
    print(f"Saved castle_view_{v['name']}.png")
