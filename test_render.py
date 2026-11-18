import mujoco
from PIL import Image

xml = """<mujoco>
  <asset>
    <mesh name="b1" file="unitree_rl_gym/resources/robots/g1_description/meshes/boulder_large.stl"/>
    <material name="m1" rgba="0.65 0.62 0.58 1.0" roughness="0.9"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" size="5 5 0.1" type="plane" rgba="0.3 0.3 0.3 1"/>
    <geom type="mesh" mesh="b1" pos="0 0 0" material="m1"/>
  </worldbody>
</mujoco>"""

m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
mujoco.mj_step(m, d)
renderer = mujoco.Renderer(m, 480, 640)
cam = mujoco.MjvCamera()
cam.distance = 2.0
cam.lookat = [0, 0, 0.2]
renderer.update_scene(d, camera=cam)
Image.fromarray(renderer.render()).save("test_boulder_render.png")
print("Rendered test_boulder_render.png!")
