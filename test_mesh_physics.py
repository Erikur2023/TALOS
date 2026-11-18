import mujoco

import mujoco
import numpy as np

xml = """<mujoco>
  <asset>
    <mesh name="apt" file="unitree_rl_gym/resources/robots/g1_description/meshes/real_apartment_scan.obj" inertia="shell"/>
  </asset>
  <worldbody>
    <geom name="real_apartment" type="mesh" mesh="apt" pos="0 0 0" contype="0" conaffinity="0"/>
  </worldbody>
</mujoco>"""

m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
geomid = np.zeros(1, dtype=np.int32)
orig = np.array([3.5, -5.5, 0.45])
ray = np.array([0.0, 1.0, 0.0]) # towards the doorway
dist = mujoco.mj_ray(m, d, orig, ray, None, 1, 1, geomid)
print("Ray hit dist with contype=0:", dist, "Geom:", mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, geomid[0]))
