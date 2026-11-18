import mujoco
from PIL import Image

xml = """<mujoco>
  <asset>
    <texture type="2d" name="t1" file="unitree_rl_gym/resources/robots/g1_description/tactical_ground.png"/>
    <material name="m1" texture="t1" texrepeat="2 2" reflectance="0.1"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" size="5 5 0.05" type="plane" material="m1"/>
  </worldbody>
</mujoco>"""

m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
mujoco.mj_step(m, d)
renderer = mujoco.Renderer(m, 480, 640)
cam = mujoco.MjvCamera()
cam.distance = 4.0
renderer.update_scene(d, camera=cam)
Image.fromarray(renderer.render()).save("test_plane_tex.png")
print("Rendered test_plane_tex.png!")
