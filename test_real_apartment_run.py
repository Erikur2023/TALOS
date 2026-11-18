import os
import time
import numpy as np
import torch
import mujoco

from run_sim import (
    KPS, KDS, DEFAULT_ANGLES, DOF_POS_SCALE, DOF_VEL_SCALE, ANG_VEL_SCALE,
    ACTION_SCALE, CMD_SCALE, get_gravity_orientation, SIMULATION_DT, CONTROL_DECIMATION,
    vfh_autonomous_navigation, scan_lidar_points
)

xml_content = """<mujoco model="talos_real_apartment">
  <include file="g1_12dof.xml"/>

  <statistic center="2.8 -3.5 0.8" extent="4.0"/>

  <visual>
    <quality shadowsize="4096" offsamples="8"/>
    <map shadowclip="0.5" shadowscale="0.6"/>
    <headlight diffuse="0.85 0.85 0.85" ambient="0.45 0.45 0.45" specular="0 0 0"/>
    <rgba haze="0.25 0.26 0.28 1"/>
    <global offwidth="1280" offheight="720" azimuth="-135" elevation="-22"/>
  </visual>

  <asset>
    <!-- Skybox & Ambiance lumineuse d'intérieur -->
    <texture type="skybox" builtin="gradient" rgb1="0.35 0.45 0.65" rgb2="0.1 0.15 0.25" width="512" height="512"/>

    <!-- VÉRITABLE TEXTURE 4K PHOTORÉALISTE DU SCAN 3D (Meta AI Habitat) -->
    <texture type="2d" name="tex_apt" file="apartment_1_texture.png"/>
    <material name="mat_apt" texture="tex_apt" reflectance="0.1" roughness="0.8"/>

    <!-- VÉRITABLE SCAN 3D COMPLET DE L'APPARTEMENT (381 673 FACES) -->
    <mesh name="m_real_apartment" file="real_apartment_scan.obj" inertia="shell"/>

    <!-- Cible tactique dorée -->
    <material name="mat_target_gold" rgba="0.95 0.78 0.15 1.0"/>
  </asset>

  <worldbody>
    <!-- Éclairage architectural d'intérieur -->
    <light pos="3.5 -5.0 2.6" dir="0 0 -1" diffuse="1.0 0.95 0.88" specular="0.2 0.2 0.2"/>
    <light pos="2.5 -2.0 2.6" dir="0 0 -1" diffuse="0.95 0.95 0.95" specular="0.1 0.1 0.1"/>

    <!-- L'APPARTEMENT 3D RÉEL : MURS, PORTES, TABLE, CHAISES, MEUBLES, SALON -->
    <geom name="real_apartment_mesh" type="mesh" mesh="m_real_apartment" pos="0 0 0" material="mat_apt" contype="0" conaffinity="0"/>

    <!-- Sol physique de l'appartement -->
    <geom name="floor_plane" type="plane" size="10 10 0.05" pos="0 0 -0.06" rgba="0 0 0 0"/>

    <!-- Cible d'extraction au débouché de la porte dans le corridor (X = 2.5m, Y = -2.5m) -->
    <geom name="extraction_target" type="cylinder" size="0.28 0.02" pos="2.5 -2.5 0.02" material="mat_target_gold"/>
  </worldbody>
</mujoco>
"""

scene_path = "unitree_rl_gym/resources/robots/g1_description/scene_real_apartment.xml"
with open(scene_path, "w", encoding="utf-8") as f:
    f.write(xml_content)
print(f"Generated {scene_path} successfully.")

m = mujoco.MjModel.from_xml_path(scene_path)
d = mujoco.MjData(m)
policy = torch.jit.load("unitree_rl_gym/deploy/pre_train/g1/motion.pt").eval()

# Start position: in the dining room
d.qpos[0] = 3.6
d.qpos[1] = -5.5
d.qpos[2] = 0.72

yaw_init = np.radians(105)
d.qpos[3] = np.cos(yaw_init / 2.0)
d.qpos[4] = 0.0
d.qpos[5] = 0.0
d.qpos[6] = np.sin(yaw_init / 2.0)

mujoco.mj_forward(m, d)

goal_pos = np.array([2.5, -2.5])
action = np.zeros(12, dtype=np.float32)
target_dof_pos = DEFAULT_ANGLES.copy()
obs = np.zeros(47, dtype=np.float32)
current_vx = 0.35
geomid_buffer = np.zeros(1, dtype=np.int32)

sim_time = 0.0
total_steps = int(14.0 / SIMULATION_DT)
contacts = 0
falls = 0
mission_done = False

print("=" * 70)
print("TEST RUN: G1 AUTONOMOUS NAVIGATION INSIDE REAL SCANNED APARTMENT")
print("=" * 70)

for s in range(total_steps):
    tau = (target_dof_pos - d.qpos[7:]) * KPS - d.qvel[6:] * KDS
    d.ctrl[:] = tau
    mujoco.mj_step(m, d)
    sim_time += SIMULATION_DT

    if d.qpos[2] < 0.30:
        falls += 1
        print(f"FALL AT t={sim_time:.2f}s! X={d.qpos[0]:.2f}, Y={d.qpos[1]:.2f}, Z={d.qpos[2]:.2f}")
        break

    dist_goal = np.linalg.norm(goal_pos - d.qpos[:2])
    if not mission_done and dist_goal <= 0.35:
        mission_done = True
        print(f"\n--> MISSION SUCCESS! REACHED TARGET IN LIVING ROOM AT t={sim_time:.2f}s! (X={d.qpos[0]:.2f}m, Y={d.qpos[1]:.2f}m, Z={d.qpos[2]:.2f}m)\n")
        break

    if s % CONTROL_DECIMATION == 0:
        desired_vx, desired_vy, desired_vyaw, phase, best_angle, ranges = vfh_autonomous_navigation(
            m, d, goal_pos, geomid_buffer, mission_done
        )
        current_vx = np.clip(desired_vx, current_vx - 0.015, current_vx + 0.015)
        cmd = np.array([current_vx, desired_vy, desired_vyaw], dtype=np.float32)

        qj = (d.qpos[7:] - DEFAULT_ANGLES) * DOF_POS_SCALE
        dqj = d.qvel[6:] * DOF_VEL_SCALE
        quat = d.qpos[3:7]
        omega = d.qvel[3:6] * ANG_VEL_SCALE
        grav = get_gravity_orientation(quat)

        period = 0.8
        phase_clock = (sim_time % period) / period
        obs[:3] = omega
        obs[3:6] = grav
        obs[6:9] = cmd * CMD_SCALE
        obs[9:21] = qj
        obs[21:33] = dqj
        obs[33:45] = action
        obs[45:47] = np.array([np.sin(2*np.pi*phase_clock), np.cos(2*np.pi*phase_clock)], dtype=np.float32)

        with torch.no_grad():
            action = policy(torch.from_numpy(obs).unsqueeze(0)).numpy().squeeze()

        if mission_done:
            target_dof_pos = DEFAULT_ANGLES.copy()
        else:
            target_dof_pos = action * ACTION_SCALE + DEFAULT_ANGLES
            pitch_err = grav[0]
            roll_err = grav[1]
            if abs(pitch_err) > 0.035:
                target_dof_pos[0] -= pitch_err * 0.35
                target_dof_pos[4] -= pitch_err * 0.25
                target_dof_pos[6] -= pitch_err * 0.35
                target_dof_pos[10] -= pitch_err * 0.25
            if abs(roll_err) > 0.035:
                target_dof_pos[1] += roll_err * 0.28
                target_dof_pos[5] += roll_err * 0.16
                target_dof_pos[7] += roll_err * 0.28
                target_dof_pos[11] += roll_err * 0.16

    if s % 250 == 0:
        print(f"t={sim_time:4.1f}s | POS: X={d.qpos[0]:.2f}m, Y={d.qpos[1]:.2f}m, Z={d.qpos[2]:.2f}m | Dist Goal={dist_goal:.2f}m | {phase}")

print(f"\nFinal Result: X={d.qpos[0]:.2f}m, Y={d.qpos[1]:.2f}m, Z={d.qpos[2]:.2f}m, Falls={falls}")
