import os
import numpy as np

def generate_rocky_scene():
    xml_path = os.path.join("unitree_rl_gym", "resources", "robots", "g1_description", "scene_tactical_rocky.xml")
    
    np.random.seed(42)
    
    # 85 pierres, galets arrondis et dalles inégales (relief non-plat réaliste)
    stones = []
    materials = ['mat_stone_grey', 'mat_stone_dark', 'mat_stone_sand']
    
    for i in range(85):
        x = np.random.uniform(0.5, 4.3)
        y = np.random.uniform(-1.0, 1.0)
        # Rayons de la pierre (ellipsoïde de 12 à 25 cm de large)
        rx = np.random.uniform(0.08, 0.16)
        ry = np.random.uniform(0.07, 0.14)
        # Épaisseur verticale entre 1.5 cm et 3.2 cm (déclenche des inclinaisons d'appui réelles)
        rz = np.random.uniform(0.012, 0.028)
        z = rz * 0.45
        # ANGLES EN RADIANS (car g1_12dof.xml utilise compiler angle="radian")
        pitch_rad = np.radians(np.random.uniform(-5.0, 5.0))
        roll_rad = np.radians(np.random.uniform(-5.0, 5.0))
        yaw_rad = np.radians(np.random.uniform(-180.0, 180.0))
        mat = np.random.choice(materials)
        stones.append(f'    <geom name="rock_stone_{i}" type="ellipsoid" size="{rx:.3f} {ry:.3f} {rz:.3f}" pos="{x:.3f} {y:.3f} {z:.3f}" euler="{pitch_rad:.3f} {roll_rad:.3f} {yaw_rad:.3f}" material="{mat}"/>')

    stones_str = "\n".join(stones)

    xml_content = f'''<mujoco model="talos_tactical_rocky_terrain">
  <include file="g1_12dof.xml"/>

  <statistic center="2.5 0.0 0.8" extent="3.5"/>

  <visual>
    <quality shadowsize="4096" offsamples="8"/>
    <map shadowclip="0.5" shadowscale="0.6"/>
    <headlight diffuse="0.08 0.08 0.08" ambient="0.18 0.18 0.18" specular="0 0 0"/>
    <rgba haze="0.18 0.16 0.14 1"/>
    <global offwidth="1280" offheight="720" azimuth="-135" elevation="-22"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.28 0.22 0.18" rgb2="0.08 0.06 0.05" width="512" height="3072"/>
    
    <!-- Sol rocheux tout-terrain / Canyon désertique -->
    <texture type="2d" name="ground_rock" builtin="checker" mark="edge" rgb1="0.28 0.24 0.20" rgb2="0.22 0.18 0.15" markrgb="0.38 0.32 0.26" width="300" height="300"/>
    <material name="mat_ground_terrain" texture="ground_rock" texuniform="true" texrepeat="6 6" reflectance="0.05"/>

    <texture type="2d" name="tex_concrete" file="concrete_diffuse.png"/>
    <material name="mat_concrete" texture="tex_concrete" texuniform="true" texrepeat="1 1" reflectance="0.04" roughness="0.9"/>

    <texture type="2d" name="tex_rust" file="rust_diffuse.png"/>
    <material name="mat_rust" texture="tex_rust" texuniform="true" texrepeat="1 1" reflectance="0.12" roughness="0.6"/>

    <!-- Matériaux géologiques rocheux -->
    <material name="mat_stone_grey" rgba="0.48 0.45 0.42 1.0" reflectance="0.02" roughness="0.95"/>
    <material name="mat_stone_dark" rgba="0.32 0.30 0.28 1.0" reflectance="0.02" roughness="0.98"/>
    <material name="mat_stone_sand" rgba="0.55 0.48 0.38 1.0" reflectance="0.03" roughness="0.90"/>
    <material name="mat_canyon_cliff" rgba="0.42 0.34 0.26 1.0" reflectance="0.02" roughness="0.98"/>
    <material name="target_zone" rgba="0.10 0.85 0.30 1.0"/>

    <!-- VRAIS MESHES 3D DE ROCHERS (KENNEY / GITHUB CQTERRAIN) -->
    <mesh name="m_boulder_large" file="boulder_large.stl"/>
    <mesh name="m_boulder_medium" file="boulder_medium.stl"/>
    <mesh name="m_boulder_small" file="boulder_small.stl"/>
    <mesh name="m_rock_debris_1" file="rock_debris_1.stl"/>
    <mesh name="m_rock_debris_2" file="rock_debris_2.stl"/>
    <mesh name="m_rock_debris_3" file="rock_debris_3.stl"/>
  </asset>

  <worldbody>
    <light pos="1.5 -2.5 5.5" dir="0.25 0.5 -1" castshadow="true" diffuse="0.98 0.92 0.82" specular="0.4 0.4 0.35"/>
    <light pos="3.0 2.5 4.0" dir="-0.2 -0.5 -1" castshadow="false" diffuse="0.28 0.30 0.38" specular="0.05 0.05 0.05"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="mat_ground_terrain"/>

    <!-- PAROIS DU CANYON TACTIQUE (MURS ROCHEUX NATURELS) -->
    <geom name="cliff_wall_left_1" type="box" size="1.3 0.20 0.75" pos="1.6 1.50 0.75" material="mat_canyon_cliff"/>
    <geom name="cliff_wall_left_2" type="box" size="1.0 0.20 0.55" pos="3.8 1.45 0.55" material="mat_canyon_cliff"/>
    <geom name="cliff_wall_right_1" type="box" size="1.5 0.20 0.70" pos="1.8 -1.50 0.70" material="mat_canyon_cliff"/>
    <geom name="cliff_wall_right_2" type="box" size="0.9 0.20 0.50" pos="3.9 -1.45 0.50" material="mat_canyon_cliff"/>

    <!-- ZONE DE CAILLOUX ET SURFACES NON-PLATES (LIT DE 85 PIERRES & GALETS) -->
{stones_str}

    <!-- RAMPES NATURELLES & SURFACES EN PENTE DOUCE (SECTIONS NON-PLATES 4° à 6°) -->
    <geom name="rock_ramp_1" type="box" size="0.25 0.30 0.020" pos="1.10 -0.10 0.015" euler="0.08 0.02 0.0" material="mat_stone_grey"/>
    <geom name="rock_ramp_2" type="box" size="0.30 0.35 0.022" pos="2.65 0.15 0.018" euler="-0.07 0.04 0.2" material="mat_stone_sand"/>
    <geom name="rock_ramp_3" type="box" size="0.25 0.30 0.020" pos="3.55 -0.15 0.016" euler="0.07 -0.03 -0.15" material="mat_stone_dark"/>

    <!-- CHICANE DE ROCHERS MASSIFS (OBSTACLES NATURELS 3D) -->

    <!-- Amas Rocheux 1 (X=1.50m, Flanc Droit) : Grand Rocher d'effondrement -->
    <geom name="boulder_rock_1" type="mesh" mesh="m_boulder_large" pos="1.50 -0.75 0.0" euler="0 0 0.25" material="mat_stone_dark"/>
    <geom name="rock_clutter_1" type="mesh" mesh="m_boulder_small" pos="1.75 -0.60 0.0" euler="0.1 0.05 -0.3" material="mat_stone_grey"/>

    <!-- Amas Rocheux 2 (X=2.30m, Flanc Gauche) : Éperon rocheux massif -->
    <geom name="boulder_rock_2" type="mesh" mesh="m_rock_debris_2" pos="2.30 0.75 0.0" euler="0 0 -0.5" material="mat_canyon_cliff"/>
    <geom name="rock_clutter_2" type="mesh" mesh="m_boulder_medium" pos="2.05 0.60 0.0" euler="0.05 -0.1 0.4" material="mat_stone_sand"/>

    <!-- Amas Rocheux 3 (X=3.10m, Flanc Droit) : Bloc de falaise tombé -->
    <geom name="boulder_rock_3" type="mesh" mesh="m_rock_debris_1" pos="3.10 -0.75 0.0" euler="0 0 -0.25" material="mat_canyon_cliff"/>
    <geom name="rock_clutter_3" type="mesh" mesh="m_boulder_small" pos="2.85 -0.60 0.0" euler="-0.1 0.1 0.6" material="mat_stone_dark"/>

    <!-- Amas Rocheux 4 (X=3.85m, Flanc Gauche) : Pylône naturel et rochers -->
    <geom name="boulder_rock_4" type="mesh" mesh="m_boulder_large" pos="3.85 0.70 0.0" euler="0 0 0.7" material="mat_stone_grey"/>
    <geom name="rock_clutter_4" type="mesh" mesh="m_boulder_medium" pos="3.60 0.80 0.0" euler="0.1 -0.05 -0.2" material="mat_stone_sand"/>

    <!-- Cible d'extraction tactique (X=4.6m, Y=0.0m) -->
    <geom name="extraction_goal" type="cylinder" size="0.25 0.04" pos="4.6 0.0 0.04" material="target_zone"/>
  </worldbody>
</mujoco>'''

    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(xml_content)
    print(f"Scène tout-terrain rocheuse générée : {xml_path}")

if __name__ == "__main__":
    generate_rocky_scene()
