import os
import mujoco
import numpy as np

mesh_dir = "unitree_rl_gym/resources/robots/g1_description/meshes"
for f in os.listdir(mesh_dir):
    if f.endswith(".stl") or f.endswith(".STL"):
        if any(k in f.lower() for k in ["ruin", "base", "build", "house", "concrete"]):
            path = os.path.join(mesh_dir, f)
            xml = f'''<mujoco><compiler angle="radian"/><asset><mesh name="m" file="{path}"/></asset><worldbody/></mujoco>'''
            try:
                m = mujoco.MjModel.from_xml_string(xml)
                verts = m.mesh_vert
                print(f"{f}: {len(verts)} verts, min={np.min(verts, axis=0)}, max={np.max(verts, axis=0)}")
            except Exception as e:
                print(f"Error loading {f}: {e}")
