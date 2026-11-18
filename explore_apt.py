import cv2
import numpy as np
import mujoco

xml = """<mujoco model="apt_explorer">
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
    <light pos="2 -2 3" dir="0 0 -1" diffuse="1 0.95 0.9"/>
    <light pos="4 -5 3" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
    <light pos="0 -6 3" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
    <geom type="mesh" mesh="apt" pos="0 0 0" material="mat_apt"/>
  </worldbody>
</mujoco>"""

m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
renderer = mujoco.Renderer(m, 720, 1280)

cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE

views = [
    # 1. Hallway / corridor through the open door
    {"lookat": [2.5, -1.0, 0.8], "dist": 4.5, "elev": -15.0, "azim": 45.0, "name": "corridor"},
    # 2. Living room / stairs / entrance area
    {"lookat": [1.0, -4.5, 0.8], "dist": 5.0, "elev": -18.0, "azim": -135.0, "name": "living_stairs"},
    # 3. Bedroom / stairs
    {"lookat": [4.0, -5.5, 0.8], "dist": 5.0, "elev": -20.0, "azim": 90.0, "name": "room3"},
    # 4. Top-down cutaway plan of the entire real apartment
    {"lookat": [3.0, -3.5, 0.5], "dist": 11.0, "elev": -89.0, "azim": 90.0, "name": "floorplan"}
]

for v in views:
    cam.lookat = v["lookat"]
    cam.distance = v["dist"]
    cam.elevation = v["elev"]
    cam.azimuth = v["azim"]
    renderer.update_scene(d, cam)
    cv2.imwrite(f"apt_view_{v['name']}.png", cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
    print(f"Saved apt_view_{v['name']}.png")
