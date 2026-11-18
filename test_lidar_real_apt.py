import mujoco
import numpy as np
from run_sim import scan_lidar_points

xml = """<mujoco model="apt_lidar_test">
  <include file="g1_12dof.xml"/>
  <visual>
    <quality shadowsize="4096"/>
    <headlight diffuse="0.8 0.8 0.8" ambient="0.4 0.4 0.4"/>
    <global offwidth="1280" offheight="720"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.35 0.45 0.65" rgb2="0.1 0.15 0.25" width="512" height="512"/>
    <texture type="2d" name="tex_apt" file="apartment_1_texture.png"/>
    <material name="mat_apt" texture="tex_apt" reflectance="0.1"/>
    <mesh name="apt" file="real_apartment_scan.obj" inertia="shell"/>
  </asset>
  <worldbody>
    <light pos="3 -4 2.8" dir="0 0 -1" diffuse="1 0.95 0.9"/>
    <!-- The REAL 3D scanned apartment mesh with its 4K texture -->
    <geom name="real_apartment" type="mesh" mesh="apt" pos="0 0 0" material="mat_apt"/>
    <geom name="floor_plane" type="plane" size="10 10 0.05" pos="0 0 0.00" rgba="0 0 0 0"/>
  </worldbody>
</mujoco>"""

with open("unitree_rl_gym/resources/robots/g1_description/test_apt_sim.xml", "w") as f:
    f.write(xml)

m = mujoco.MjModel.from_xml_path("unitree_rl_gym/resources/robots/g1_description/test_apt_sim.xml")
d = mujoco.MjData(m)

# Place robot in the clear walkway of the dining room facing the door
d.qpos[0] = 3.5
d.qpos[1] = -5.5
d.qpos[2] = 0.77
# Face towards the doorway (towards -Y, cap ~ 270 deg)
# quat for yaw = -pi/2
d.qpos[3] = np.cos(-np.pi/4)
d.qpos[4] = 0.0
d.qpos[5] = 0.0
d.qpos[6] = np.sin(-np.pi/4)
mujoco.mj_forward(m, d)

geomid = np.zeros(1, dtype=np.int32)
ranges, points = scan_lidar_points(m, d, geomid, -np.pi/2)

print("LiDAR Scan in Real Apartment:")
print("Rays hit:", len(points))
print("Min range:", np.min(ranges), "meters")
for px, py, dist, a in points[:10]:
    print(f"  Ray angle={np.degrees(a):+5.1f}° | Dist={dist:.2f}m")
