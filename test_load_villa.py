import mujoco

xml = """<mujoco>
  <asset>
    <mesh name="villa" file="unitree_rl_gym/resources/robots/g1_description/meshes/real_apartment_scan.obj" inertia="shell"/>
  </asset>
  <worldbody>
    <geom type="mesh" mesh="villa"/>
  </worldbody>
</mujoco>"""

try:
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_step(m, d)
    print("SUCCESS: MuJoCo loaded seaside_villa.obj!")
    print(f"ngeom: {m.ngeom}, nmesh: {m.nmesh}, verts: {len(m.mesh_vert)}, tris: {len(m.mesh_face)}")
except Exception as e:
    print(f"MuJoCo error: {e}")
