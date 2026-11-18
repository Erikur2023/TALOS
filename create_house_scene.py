import os

XML_CONTENT = """<mujoco model="talos_tactical_house">
  <include file="g1_12dof.xml"/>

  <statistic center="2.5 0.0 0.8" extent="3.6"/>

  <visual>
    <quality shadowsize="4096" offsamples="8"/>
    <map shadowclip="0.5" shadowscale="0.6"/>
    <headlight diffuse="0.10 0.10 0.10" ambient="0.18 0.18 0.18" specular="0 0 0"/>
    <rgba haze="0.25 0.26 0.28 1"/>
    <global offwidth="1280" offheight="720" azimuth="-130" elevation="-20"/>
  </visual>

  <asset>
    <!-- Skybox & Ambiance architecturale -->
    <texture type="skybox" builtin="gradient" rgb1="0.30 0.35 0.45" rgb2="0.10 0.12 0.15" width="512" height="3072"/>

    <!-- Textures PBR Réalistes -->
    <texture type="2d" name="tex_flagstone" file="flagstone_path_diffuse.png"/>
    <material name="mat_flagstone" texture="tex_flagstone" texuniform="true" texrepeat="8 8" reflectance="0.08" roughness="0.8"/>

    <texture type="2d" name="tex_brick" file="brick_facade_diffuse.png"/>
    <material name="mat_brick_facade" texture="tex_brick" texuniform="true" texrepeat="3 3" reflectance="0.05" roughness="0.9"/>

    <texture type="2d" name="tex_parquet" file="wood_parquet_diffuse.png"/>
    <material name="mat_wood_parquet" texture="tex_parquet" texuniform="true" texrepeat="4 4" reflectance="0.25" roughness="0.45"/>

    <texture type="2d" name="tex_marble" file="marble_stairs_diffuse.png"/>
    <material name="mat_marble_stairs" texture="tex_marble" texuniform="true" texrepeat="2 2" reflectance="0.35" roughness="0.35"/>

    <texture type="2d" name="tex_interior_wall" file="interior_wall_diffuse.png"/>
    <material name="mat_interior_wall" texture="tex_interior_wall" texuniform="true" texrepeat="4 2" reflectance="0.08" roughness="0.8"/>

    <!-- Matériaux secondaires -->
    <material name="mat_real_villa" rgba="0.94 0.92 0.88 1.0" reflectance="0.2"/>
    <material name="mat_mahogany" rgba="0.35 0.16 0.08 1.0" reflectance="0.25" roughness="0.5"/>
    <material name="mat_brass" rgba="0.85 0.72 0.25 1.0" reflectance="0.70" roughness="0.2"/>
    <material name="mat_lawn" rgba="0.18 0.32 0.12 1.0" reflectance="0.04" roughness="0.95"/>
    <material name="mat_roof_trim" rgba="0.25 0.25 0.28 1.0" reflectance="0.1" roughness="0.7"/>
    <material name="mat_target_gold" rgba="0.95 0.78 0.15 1.0"/>

    <!-- Meshes 3D Réels téléchargés d'Internet (Seaside Villa & Accessoires) -->
    <mesh name="m_real_villa" file="real_villa_architecture.obj" inertia="shell"/>
    <mesh name="m_front_door" file="house_front_door.stl"/>
    <mesh name="m_console_table" file="house_console_table.stl"/>
    <mesh name="m_stair_banister" file="house_stair_banister.stl"/>
    <mesh name="m_ceiling_lamp" file="house_ceiling_lamp.stl"/>
  </asset>

  <worldbody>
    <!-- Éclairage extérieur (Soleil de fin d'après-midi) -->
    <light pos="0.5 -3.5 6.0" dir="0.3 0.6 -1" castshadow="true" diffuse="0.95 0.90 0.82" specular="0.3 0.28 0.2"/>
    
    <!-- Éclairage intérieur de la maison (Lustre vestibule et mezzanine chaud 2700K) -->
    <light pos="2.0 0.0 2.2" dir="0 0 -1" castshadow="true" diffuse="1.0 0.88 0.68" specular="0.2 0.18 0.1"/>
    <light pos="3.8 0.0 2.5" dir="0 0 -1" castshadow="false" diffuse="0.85 0.75 0.60" specular="0.1 0.1 0.05"/>

    <!-- SOL PRINCIPAL FLUIDE ET CONTINU (Z = 0) -->
    <geom name="floor" size="15 15 0.05" type="plane" material="mat_flagstone"/>
    <geom name="lawn_left" type="box" size="4.0 3.0 0.001" pos="0.0 5.0 0.001" material="mat_lawn"/>
    <geom name="lawn_right" type="box" size="4.0 3.0 0.001" pos="0.0 -5.0 0.001" material="mat_lawn"/>

    <!-- ENVELOPPE ARCHITECTURALE RÉELLE : VILLA 3D COMPLÈTE (14 574 FACES TÉLÉCHARGÉE D'INTERNET) -->
    <geom name="villa_exterior_real" type="mesh" mesh="m_real_villa" pos="0.5 0.0 0.0" material="mat_real_villa" contype="0" conaffinity="0"/>

    <!-- FAÇADE EXTÉRIEURE EN BRIQUES ROUGES (X = 1.35m) -->
    <!-- Mur gauche de la façade (porte de 1.10m de large entre Y=-0.55 et Y=+0.55) -->
    <geom name="house_facade_L" type="box" size="0.10 1.05 1.25" pos="1.35 1.60 1.25" material="mat_brick_facade"/>
    <!-- Mur droit de la façade -->
    <geom name="house_facade_R" type="box" size="0.10 1.05 1.25" pos="1.35 -1.60 1.25" material="mat_brick_facade"/>
    <!-- Linteau supérieur au-dessus de la porte (dégagement sous plafond 2.15m) -->
    <geom name="house_facade_lintel" type="box" size="0.10 0.55 0.20" pos="1.35 0.0 2.30" material="mat_brick_facade"/>
    <!-- Corniche décorative de toit -->
    <geom name="house_cornice" type="box" size="0.18 2.80 0.06" pos="1.35 0.0 2.53" material="mat_roof_trim"/>

    <!-- PORTE D'ENTRÉE EN CHÊNE MASSIF OUVERTE (Pivotée contre le mur du vestibule à Y = -0.58m) -->
    <geom name="house_door_open" type="mesh" mesh="m_front_door" pos="1.45 -0.58 0.0" euler="0 0 -1.5708" material="mat_mahogany"/>

    <!-- VESTIBULE / COULOIR INTÉRIEUR (X = 1.35m à 2.35m) -->
    <!-- Murs latéraux intérieurs avec moulures -->
    <geom name="hallway_wall_L" type="box" size="0.55 0.08 1.25" pos="1.85 1.15 1.25" material="mat_interior_wall"/>
    <geom name="hallway_wall_R" type="box" size="0.55 0.08 1.25" pos="1.85 -1.15 1.25" material="mat_interior_wall"/>

    <!-- Console d'entrée et vase décoratif contre le mur droit -->
    <geom name="hallway_console" type="mesh" mesh="m_console_table" pos="1.95 -0.95 0.0" euler="0 0 1.5708" material="mat_mahogany"/>

    <!-- Suspension / Lustre du vestibule -->
    <geom name="hallway_chandelier" type="mesh" mesh="m_ceiling_lamp" pos="1.90 0.0 2.10" material="mat_brass"/>

    <!-- ESCALIER INTÉRIEUR EN MARBRE (5 Marches chanfreinées de 3.8 cm de dénivelé) -->
    <!-- Marche 1 (X = 2.48m, élévation 0.038m) -->
    <geom name="stair_step_1" type="box" size="0.13 0.65 0.019" pos="2.48 0.0 0.019" material="mat_marble_stairs"/>
    <geom name="stair_lip_1" type="capsule" size="0.019 0.65" pos="2.35 0.0 0.019" euler="1.5708 0 0" material="mat_marble_stairs"/>

    <!-- Marche 2 (X = 2.74m, élévation 0.076m) -->
    <geom name="stair_step_2" type="box" size="0.13 0.65 0.038" pos="2.74 0.0 0.038" material="mat_marble_stairs"/>
    <geom name="stair_lip_2" type="capsule" size="0.019 0.65" pos="2.61 0.0 0.038" euler="1.5708 0 0" material="mat_marble_stairs"/>

    <!-- Marche 3 (X = 3.00m, élévation 0.114m) -->
    <geom name="stair_step_3" type="box" size="0.13 0.65 0.057" pos="3.00 0.0 0.057" material="mat_marble_stairs"/>
    <geom name="stair_lip_3" type="capsule" size="0.019 0.65" pos="2.87 0.0 0.057" euler="1.5708 0 0" material="mat_marble_stairs"/>

    <!-- Marche 4 (X = 3.26m, élévation 0.152m) -->
    <geom name="stair_step_4" type="box" size="0.13 0.65 0.076" pos="3.26 0.0 0.076" material="mat_marble_stairs"/>
    <geom name="stair_lip_4" type="capsule" size="0.019 0.65" pos="3.13 0.0 0.076" euler="1.5708 0 0" material="mat_marble_stairs"/>

    <!-- Marche 5 (X = 3.52m, élévation 0.190m) -->
    <geom name="stair_step_5" type="box" size="0.13 0.65 0.095" pos="3.52 0.0 0.095" material="mat_marble_stairs"/>
    <geom name="stair_lip_5" type="capsule" size="0.019 0.65" pos="3.39 0.0 0.095" euler="1.5708 0 0" material="mat_marble_stairs"/>

    <!-- Rampes & Balustrades d'escalier sculptées -->
    <geom name="stair_banister_L" type="mesh" mesh="m_stair_banister" pos="2.35 0.68 0.0" material="mat_mahogany"/>
    <geom name="stair_banister_R" type="mesh" mesh="m_stair_banister" pos="2.35 -0.68 0.0" material="mat_mahogany"/>

    <!-- Murs d'escalier latéraux -->
    <geom name="stair_wall_L" type="box" size="0.65 0.08 1.25" pos="3.05 1.15 1.25" material="mat_interior_wall"/>
    <geom name="stair_wall_R" type="box" size="0.65 0.08 1.25" pos="3.05 -1.15 1.25" material="mat_interior_wall"/>

    <!-- PALIER SUPÉRIEUR / MEZZANINE (1er Étage, élévation Z = 0.190m) -->
    <geom name="landing_floor" type="box" size="0.55 0.95 0.095" pos="4.20 0.0 0.095" material="mat_wood_parquet"/>
    <!-- Mur du fond du palier -->
    <geom name="landing_wall_back" type="box" size="0.08 1.05 1.25" pos="4.80 0.0 1.25" material="mat_interior_wall"/>
    <geom name="landing_wall_L" type="box" size="0.55 0.08 1.25" pos="4.20 1.05 1.25" material="mat_interior_wall"/>
    <geom name="landing_wall_R" type="box" size="0.55 0.08 1.25" pos="4.20 -1.05 1.25" material="mat_interior_wall"/>

    <!-- Cible d'extraction finale / Balise dorée au sommet des escaliers (X = 4.5m, Z = 0.195m) -->
    <geom name="house_target" type="cylinder" size="0.28 0.005" pos="4.50 0.0 0.195" material="mat_target_gold" contype="0" conaffinity="0"/>
  </worldbody>
</mujoco>
"""

dest_path = "unitree_rl_gym/resources/robots/g1_description/scene_tactical_house.xml"
with open(dest_path, "w", encoding="utf-8") as f:
    f.write(XML_CONTENT)

print(f"Generated {dest_path} successfully.")
