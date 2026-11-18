import mujoco
import cv2

xml = """<mujoco model="pro_lighting_test">
  <include file="unitree_rl_gym/resources/robots/g1_description/g1_12dof.xml"/>
  <visual>
    <quality shadowsize="4096" offsamples="8"/>
    <map shadowclip="0.5" shadowscale="0.6"/>
    <headlight diffuse="0.05 0.05 0.05" ambient="0.15 0.15 0.15" specular="0 0 0"/>
    <global offwidth="1280" offheight="720" azimuth="-135" elevation="-20"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.18 0.24 0.35" rgb2="0.05 0.07 0.12" width="512" height="3072"/>
    <texture type="2d" name="ground_grid" builtin="checker" mark="edge" rgb1="0.22 0.22 0.20" rgb2="0.16 0.16 0.15" markrgb="0.38 0.38 0.35" width="300" height="300"/>
    <material name="mat_ground" texture="ground_grid" texuniform="true" texrepeat="5 5" reflectance="0.08"/>
    <material name="mat_concrete" rgba="0.5 0.48 0.45 1" roughness="0.9"/>
  </asset>
  <worldbody>
    <light pos="1.5 -2.5 5.0" dir="0.2 0.5 -1" castshadow="true" diffuse="0.95 0.92 0.85" specular="0.35 0.35 0.3"/>
    <light pos="3.0 2.5 3.5" dir="-0.2 -0.5 -1" castshadow="false" diffuse="0.25 0.30 0.40" specular="0.05 0.05 0.05"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="mat_ground"/>
    <geom name="block1" type="box" size="0.3 0.2 0.3" pos="1.5 -0.5 0.3" material="mat_concrete"/>
    <geom name="block2" type="box" size="0.3 0.2 0.4" pos="2.2 0.5 0.4" material="mat_concrete"/>
  </worldbody>
</mujoco>"""

m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)

renderer = mujoco.Renderer(m, 720, 1280)
renderer.update_scene(d)
img = renderer.render()
bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imwrite("preview_pro_lighting.png", bgr)
print("Generated preview_pro_lighting.png successfully!")
