#!/usr/bin/env python3
"""
Générateur des nouvelles scènes de simulation pour TALOS #225 :
1. scene_tactical_trench.xml      : Tranchée de combat fortifiée & parapets de sacs de sable
2. scene_tactical_industrial.xml  : Complexe industriel endommagé & cuves chimiques NRBC
3. scene_tactical_night.xml       : Infiltration nocturne tactique & projecteurs de sécurité
"""

import os
import mujoco
import cv2
import numpy as np

ROBOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unitree_rl_gym", "resources", "robots", "g1_description")

# =============================================================================
# SCÈNE 1 : TRANCHÉE DE COMBAT & PARAPETS FORTIFIÉS (TRENCH)
# =============================================================================
trench_xml = """<mujoco model="talos_tactical_trench">
  <include file="g1_12dof.xml"/>

  <statistic center="2.5 0.0 0.8" extent="3.5"/>

  <visual>
    <quality shadowsize="4096" offsamples="8"/>
    <map shadowclip="0.5" shadowscale="0.6"/>
    <headlight diffuse="0.08 0.08 0.08" ambient="0.18 0.18 0.18" specular="0 0 0"/>
    <rgba haze="0.30 0.26 0.20 1"/>
    <global offwidth="1280" offheight="720" azimuth="-135" elevation="-22"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.35 0.32 0.28" rgb2="0.12 0.10 0.08" width="512" height="3072"/>
    <texture type="2d" name="tex_trench_earth" file="earth_trench_diffuse.png"/>
    <material name="mat_trench_earth" texture="tex_trench_earth" texuniform="true" texrepeat="6 6" reflectance="0.02" roughness="0.95"/>
    
    <material name="mat_sandbag" rgba="0.68 0.60 0.44 1.0" reflectance="0.03" roughness="0.92"/>
    <material name="mat_sandbag_dark" rgba="0.52 0.46 0.34 1.0" reflectance="0.02" roughness="0.95"/>
    <material name="mat_wood_plank" rgba="0.45 0.30 0.18 1.0" reflectance="0.05" roughness="0.85"/>
    <material name="mat_ammo_crate" rgba="0.22 0.32 0.18 1.0" reflectance="0.08" roughness="0.75"/>
    <material name="mat_steel_barb" rgba="0.35 0.36 0.38 1.0" reflectance="0.25" roughness="0.5"/>
    <material name="target_zone" rgba="0.10 0.85 0.30 1.0"/>

    <mesh name="m_solid_girder" file="solid_girder_1.stl"/>
    <mesh name="m_solid_slab_1" file="solid_slab_1.stl"/>
    <mesh name="m_barrel_photoreal" file="photoreal_barrel.stl"/>
  </asset>

  <worldbody>
    <light pos="2.0 -3.0 5.5" dir="0.2 0.5 -1" castshadow="true" diffuse="0.95 0.90 0.78" specular="0.3 0.25 0.2"/>
    <light pos="3.5 2.5 4.0" dir="-0.2 -0.5 -1" castshadow="false" diffuse="0.30 0.28 0.25" specular="0.05 0.05 0.05"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="mat_trench_earth"/>

    <!-- Parapets de tranchée fortifiés en terre et sacs de sable -->
    <geom name="trench_parapet_L" type="box" size="2.5 0.12 0.55" pos="2.5 1.65 0.55" material="mat_sandbag"/>
    <geom name="trench_parapet_L_top" type="capsule" size="0.14 2.4" pos="2.5 1.60 0.85" euler="0 1.5708 0" material="mat_sandbag_dark"/>

    <geom name="trench_parapet_R" type="box" size="2.5 0.12 0.55" pos="2.5 -1.65 0.55" material="mat_sandbag"/>
    <geom name="trench_parapet_R_top" type="capsule" size="0.14 2.4" pos="2.5 -1.60 0.85" euler="0 1.5708 0" material="mat_sandbag_dark"/>

    <!-- OBSTACLES TACTIQUES DE TRANCHÉE EN CHICANE (Disposition optimale validée) -->
    <!-- Obstacle 1 (X=1.80m, Droit) : Caisse de munitions lourdes -->
    <geom name="trench_ammo_box_1" type="box" size="0.20 0.20 0.35" pos="1.80 -0.32 0.35" material="mat_ammo_crate"/>
    <geom name="trench_beam_1" type="cylinder" size="0.06 0.25" pos="1.80 -0.32 0.75" euler="0 0.5 0.2" material="mat_wood_plank"/>

    <!-- Obstacle 2 (X=2.85m, Gauche) : Barricade de sacs de sable fortifiée -->
    <geom name="trench_sandbag_2" type="box" size="0.20 0.22 0.45" pos="2.85 0.35 0.45" material="mat_sandbag"/>
    <geom name="trench_sandbag_top_2" type="capsule" size="0.14 0.20" pos="2.85 0.35 0.95" euler="0 1.5708 0" material="mat_sandbag_dark"/>

    <!-- Cible d'extraction sécurisée (X=4.5m, Y=0.0m) -->
    <geom name="trench_extraction" type="cylinder" size="0.26 0.04" pos="4.5 0.0 0.04" material="target_zone"/>
  </worldbody>
</mujoco>"""

# =============================================================================
# SCÈNE 2 : COMPLEXE INDUSTRIEL ENDOMMAGÉ & FUITE CHIMIQUE (INDUSTRIAL)
# =============================================================================
industrial_xml = """<mujoco model="talos_tactical_industrial">
  <include file="g1_12dof.xml"/>

  <statistic center="2.5 0.0 0.8" extent="3.5"/>

  <visual>
    <quality shadowsize="4096" offsamples="8"/>
    <map shadowclip="0.5" shadowscale="0.6"/>
    <headlight diffuse="0.06 0.06 0.06" ambient="0.15 0.15 0.15" specular="0 0 0"/>
    <rgba haze="0.20 0.22 0.25 1"/>
    <global offwidth="1280" offheight="720" azimuth="-135" elevation="-22"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.15 0.18 0.22" rgb2="0.05 0.06 0.08" width="512" height="3072"/>
    <texture type="2d" name="tex_grate" file="steel_grate_diffuse.png"/>
    <material name="mat_steel_grate" texture="tex_grate" texuniform="true" texrepeat="8 8" reflectance="0.12" roughness="0.6"/>

    <texture type="2d" name="tex_rust" file="rust_diffuse.png"/>
    <material name="mat_industrial_rust" texture="tex_rust" texuniform="true" texrepeat="2 2" reflectance="0.10" roughness="0.7"/>

    <material name="mat_silo" rgba="0.55 0.58 0.62 1.0" reflectance="0.3" roughness="0.4"/>
    <material name="mat_pipe" rgba="0.82 0.35 0.15 1.0" reflectance="0.2" roughness="0.5"/>
    <material name="mat_hazard_yellow" rgba="0.92 0.72 0.08 1.0" reflectance="0.25" roughness="0.45"/>
    <material name="mat_concrete_pillar" rgba="0.42 0.42 0.40 1.0" reflectance="0.04" roughness="0.9"/>
    <material name="target_zone" rgba="0.10 0.85 0.30 1.0"/>

    <mesh name="m_solid_girder" file="solid_girder_1.stl"/>
    <mesh name="m_barrel_photoreal" file="photoreal_barrel.stl"/>
  </asset>

  <worldbody>
    <!-- Éclairage industriel au sodium (Teinte ambrée et ombres denses) -->
    <light pos="2.0 -2.5 5.5" dir="0.2 0.5 -1" castshadow="true" diffuse="0.98 0.85 0.55" specular="0.4 0.35 0.2"/>
    <light pos="3.5 2.5 4.0" dir="-0.2 -0.5 -1" castshadow="false" diffuse="0.25 0.28 0.35" specular="0.05 0.05 0.05"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="mat_steel_grate"/>

    <!-- Murs d'usine latéraux avec tuyauteries industrielles -->
    <geom name="ind_wall_L1" type="box" size="2.5 0.12 0.85" pos="2.5 1.65 0.85" material="mat_concrete_pillar"/>
    <geom name="ind_pipe_L_top" type="cylinder" size="0.07 2.4" pos="2.5 1.50 1.35" euler="0 1.5708 0" material="mat_pipe"/>

    <geom name="ind_wall_R1" type="box" size="2.5 0.12 0.85" pos="2.5 -1.65 0.85" material="mat_concrete_pillar"/>
    <geom name="ind_pipe_R_top" type="cylinder" size="0.07 2.4" pos="2.5 -1.50 1.35" euler="0 1.5708 0" material="mat_pipe"/>

    <!-- OBSTACLES INDUSTRIELS NRBC EN CHICANE (Disposition optimale validée) -->
    <!-- Obstacle 1 (X=1.80m, Droit) : Fût de matière dangereuse toxique -->
    <geom name="ind_barrel_1" type="mesh" mesh="m_barrel_photoreal" pos="1.80 -0.30 0.0" euler="0 0 0.2" material="mat_hazard_yellow"/>

    <!-- Obstacle 2 (X=2.85m, Gauche) : Cuve cylindrique de stockage chimique -->
    <geom name="ind_silo_2" type="cylinder" size="0.20 0.65" pos="2.85 0.32 0.65" material="mat_silo"/>

    <!-- Cible d'extraction / Vanne de purge (X=4.5m, Y=0.0m) -->
    <geom name="ind_extraction" type="cylinder" size="0.26 0.04" pos="4.5 0.0 0.04" material="target_zone"/>
  </worldbody>
</mujoco>"""

# =============================================================================
# SCÈNE 3 : OPÉRATION SPÉCIALE NOCTURNE & INFILTRATION (NIGHT)
# =============================================================================
night_xml = """<mujoco model="talos_tactical_night">
  <include file="g1_12dof.xml"/>

  <statistic center="2.5 0.0 0.8" extent="3.5"/>

  <visual>
    <quality shadowsize="4096" offsamples="8"/>
    <map shadowclip="0.5" shadowscale="0.6"/>
    <!-- Éclairage d'ambiance nocturne froid -->
    <headlight diffuse="0.02 0.03 0.04" ambient="0.08 0.10 0.14" specular="0 0 0"/>
    <rgba haze="0.04 0.06 0.10 1"/>
    <global offwidth="1280" offheight="720" azimuth="-135" elevation="-22"/>
  </visual>

  <asset>
    <!-- Ciel de nuit profonde étoilé -->
    <texture type="skybox" builtin="gradient" rgb1="0.04 0.06 0.12" rgb2="0.01 0.02 0.04" width="512" height="3072"/>
    <texture type="2d" name="ground_asphalt" builtin="checker" mark="edge" rgb1="0.10 0.10 0.12" rgb2="0.07 0.07 0.08" markrgb="0.18 0.18 0.20" width="300" height="300"/>
    <material name="mat_night_asphalt" texture="ground_asphalt" texuniform="true" texrepeat="5 5" reflectance="0.04"/>

    <texture type="2d" name="tex_concrete" file="concrete_diffuse.png"/>
    <material name="mat_night_concrete" texture="tex_concrete" texuniform="true" texrepeat="1 1" reflectance="0.02" roughness="0.9"/>

    <material name="mat_checkpoint_barrier" rgba="0.30 0.32 0.35 1.0" reflectance="0.08" roughness="0.7"/>
    <material name="mat_hazard_stripes" rgba="0.80 0.60 0.10 1.0" reflectance="0.2" roughness="0.5"/>
    <material name="target_zone" rgba="0.10 0.85 0.30 1.0"/>

    <mesh name="m_jersey_barrier" file="jersey_barrier.stl"/>
    <mesh name="m_jersey_cracked" file="jersey_barrier_cracked.stl"/>
    <mesh name="m_solid_girder" file="solid_girder_1.stl"/>
    <mesh name="m_barrel_photoreal" file="photoreal_barrel.stl"/>
  </asset>

  <worldbody>
    <!-- Projecteur de sécurité nocturne principal (Cône de lumière directive) -->
    <light pos="2.5 -2.0 5.0" dir="0.1 0.4 -1" castshadow="true" diffuse="0.75 0.82 0.95" specular="0.3 0.35 0.4"/>
    <!-- Lumière lunaire bleue diffuse -->
    <light pos="1.0 2.5 4.5" dir="-0.1 -0.4 -1" castshadow="false" diffuse="0.12 0.16 0.25" specular="0.02 0.02 0.05"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="mat_night_asphalt"/>

    <!-- Murs de barrage routier & périmètre de sécurité militaire -->
    <geom name="night_wall_L" type="box" size="2.5 0.12 0.60" pos="2.5 1.65 0.60" material="mat_night_concrete"/>
    <geom name="night_wall_R" type="box" size="2.5 0.12 0.60" pos="2.5 -1.65 0.60" material="mat_night_concrete"/>

    <!-- OBSTACLES DU CHECKPOINT NOCTURNE EN CHICANE (Disposition optimale validée) -->
    <!-- Obstacle 1 (X=1.80m, Droit) : Barrière Jersey de barrage militaire -->
    <geom name="night_jersey_1" type="mesh" mesh="m_jersey_barrier" pos="1.80 -0.30 0.0" euler="0 0 0.1" material="mat_checkpoint_barrier"/>

    <!-- Obstacle 2 (X=2.85m, Gauche) : Barrière Jersey fissurée -->
    <geom name="night_jersey_2" type="mesh" mesh="m_jersey_cracked" pos="2.85 0.32 0.0" euler="0 0 -0.2" material="mat_checkpoint_barrier"/>

    <!-- Cible d'extraction sous surveillance (X=4.5m, Y=0.0m) -->
    <geom name="night_extraction" type="cylinder" size="0.26 0.04" pos="4.5 0.0 0.04" material="target_zone"/>
  </worldbody>
</mujoco>"""

# Écriture des 3 fichiers XML
files = {
    "scene_tactical_trench.xml": trench_xml,
    "scene_tactical_industrial.xml": industrial_xml,
    "scene_tactical_night.xml": night_xml
}

for filename, content in files.items():
    filepath = os.path.join(ROBOT_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content.strip())
    print(f"[OK] Fichier de scène généré : {filepath}")
    
    # Validation MuJoCo immédiate
    m = mujoco.MjModel.from_xml_path(filepath)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    
    # Rendu d'une image de prévisualisation HD
    renderer = mujoco.Renderer(m, 720, 1280)
    renderer.update_scene(d)
    bgr = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
    preview_name = f"preview_{filename.replace('.xml', '')}.png"
    cv2.imwrite(preview_name, bgr)
    print(f"  -> Rendu HD validé : {preview_name} ({m.ngeom} geoms)")

print("\n[SUCCÈS] Toutes les 3 nouvelles scènes ont été générées et validées sans aucune erreur !")
