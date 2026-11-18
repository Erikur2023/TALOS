import mujoco

meshes = [
    'house_front_door.stl',
    'house_console_table.stl',
    'house_stair_banister.stl',
    'house_ceiling_lamp.stl'
]

for f in meshes:
    xml = f"""<mujoco>
      <asset>
        <mesh name="m" file="unitree_rl_gym/resources/robots/g1_description/meshes/{f}"/>
      </asset>
      <worldbody>
        <geom type="mesh" mesh="m"/>
      </worldbody>
    </mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_step(m, d)
    print(f"Loaded {f} successfully: ngeom={m.ngeom}")

print("ALL HOUSE STL MESHES FULLY VALIDATED IN MUJOCO!")
