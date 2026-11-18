import mujoco

xml = """<mujoco>
  <asset>
    <mesh name="real_house" file="real_angel_house.obj" scale="0.1 0.1 0.1" inertia="shell"/>
  </asset>
  <worldbody>
    <geom type="mesh" mesh="real_house"/>
  </worldbody>
</mujoco>"""

try:
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_step(m, d)
    print("SUCCESS: MuJoCo loaded real_angel_house.obj!")
    print(f"ngeom: {m.ngeom}, nmesh: {m.nmesh}")
except Exception as e:
    print(f"MuJoCo error: {e}")
