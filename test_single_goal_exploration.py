import os
import time
import numpy as np
import torch
import mujoco

from run_sim import (
    KPS, KDS, DEFAULT_ANGLES, DOF_POS_SCALE, DOF_VEL_SCALE, ANG_VEL_SCALE,
    ACTION_SCALE, CMD_SCALE, get_gravity_orientation, SIMULATION_DT, CONTROL_DECIMATION,
    scan_lidar_points, LIDAR_ANGLES, LIDAR_MAX_RANGE, NUM_LIDAR_RAYS,
    TacticalSlam2D
)

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
SCENE_XML = os.path.join(WORKSPACE_DIR, "unitree_rl_gym", "resources", "robots", "g1_description", "scene_real_apartment.xml")
POLICY_PT = os.path.join(WORKSPACE_DIR, "unitree_rl_gym", "deploy", "pre_train", "g1", "motion.pt")

def autonomous_explorer_navigation(m, d, goal_pos, geomid_buffer, mission_done=False):
    """
    Contrôleur 100% autonome de découverte d'intérieur inconnu (Zero Waypoint) :
    - Projection de pas virtuel le long de chaque faisceau laser sans collision.
    - Sélection de l'ouverture qui minimise la distance euclidienne à la cible unique.
    - Découverte des pièces, portes et couloirs en temps réel.
    """
    if mission_done:
        return 0.0, 0.0, 0.0, "CIBLE ATTEINTE (SENTINELLE)", 0.0, np.ones(NUM_LIDAR_RAYS) * LIDAR_MAX_RANGE

    robot_pos = d.qpos[:2]
    to_goal = goal_pos - robot_pos
    dist_goal = np.linalg.norm(to_goal)

    if dist_goal <= 0.38:
        return 0.0, 0.0, 0.0, "CIBLE ATTEINTE (SENTINELLE)", 0.0, np.ones(NUM_LIDAR_RAYS) * LIDAR_MAX_RANGE

    quat = d.qpos[3:7]
    qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
    yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

    ranges, points = scan_lidar_points(m, d, geomid_buffer, yaw)

    R_robot = 0.32
    lookahead = 2.2
    best_angle = 0.0
    min_cost = 999.0

    for a in LIDAR_ANGLES:
        cos_a = np.cos(a)
        sin_a = np.sin(a)
        blocked = False
        min_obs_dist = lookahead

        for px, py, d_orig, ray_a in points:
            d_par = px * cos_a + py * sin_a
            d_perp = abs(-px * sin_a + py * cos_a)
            if 0 < d_par < lookahead and d_perp < R_robot:
                blocked = True
                break
            if 0 < d_par < lookahead:
                min_obs_dist = min(min_obs_dist, d_orig)

        if not blocked and min_obs_dist > 0.45:
            # Projection virtuelle le long du rayon libre
            step_len = min(min_obs_dist, 1.4)
            world_ray_angle = yaw + a
            cand_x = robot_pos[0] + step_len * np.cos(world_ray_angle)
            cand_y = robot_pos[1] + step_len * np.sin(world_ray_angle)
            cand_dist = np.linalg.norm(goal_pos - np.array([cand_x, cand_y]))

            cost = cand_dist + 0.20 * abs(a) - 0.25 * (min_obs_dist / lookahead)
            if cost < min_cost:
                min_cost = cost
                best_angle = a

    if min_cost > 900.0 and points:
        best_angle = max(points, key=lambda p: p[2])[3]

    min_front_obs = min([p[2] for p in points if abs(p[3]) < np.radians(16)] + [LIDAR_MAX_RANGE])

    if min_front_obs < 0.42:
        # Mur frontal immédiat : rotation douce vers l'ouverture sélectionnée
        desired_vx = 0.10
        desired_vyaw = float(np.clip(best_angle * 1.6, -0.75, 0.75))
        desired_vy = float(np.clip(0.08 * np.sin(best_angle), -0.08, 0.08))
        phase_label = f"CONTOURNEMENT // CAP [{np.degrees(best_angle):+4.1f}°]"
    else:
        obstacle_slowdown = min(1.0, max(0.20, (min_front_obs - 0.42) / 0.70))
        speed_scale = min(1.0, max(0.35, dist_goal / 0.8)) * obstacle_slowdown
        max_vx = 0.50
        desired_vx = float(np.clip(max_vx * max(0.18, np.cos(best_angle)) * speed_scale, 0.15, max_vx))
        desired_vy = float(np.clip(0.06 * np.sin(best_angle), -0.06, 0.06))
        desired_vyaw = float(np.clip(best_angle * 1.45, -0.70, 0.70))
        phase_label = f"DÉCOUVERTE AUTONOME [CAP {np.degrees(best_angle):+4.1f}°]"

    return desired_vx, desired_vy, desired_vyaw, phase_label, best_angle, ranges

m = mujoco.MjModel.from_xml_path(SCENE_XML)
d = mujoco.MjData(m)
policy = torch.jit.load(POLICY_PT).eval()

d.qpos[0] = 3.6
d.qpos[1] = -5.5
d.qpos[2] = 0.72
yaw_init = 1.8326
d.qpos[3] = np.cos(yaw_init / 2.0)
d.qpos[4] = 0.0
d.qpos[5] = 0.0
d.qpos[6] = np.sin(yaw_init / 2.0)
mujoco.mj_forward(m, d)

# CIBLE UNIQUE DANS LE GRAND SALON : X=1.5m, Y=-0.5m (ZÉRO WAYPOINT)
single_goal = np.array([1.5, -0.5])
print("=" * 80)
print(f"LANCEMENT DÉCOUVERTE AUTONOME (ZÉRO WAYPOINT) -> CIBLE SALON : {single_goal}")
print("LE ROBOT NE CONNAÎT PAS LA MAISON NI LES PORTES : NAVIGATION PAR OUVERTURES LIDAR")
print("=" * 80)

slam = TacticalSlam2D()
action = np.zeros(12, dtype=np.float32)
target_dof_pos = DEFAULT_ANGLES.copy()
obs = np.zeros(47, dtype=np.float32)
current_vx = 0.40
geomid_buffer = np.zeros(1, dtype=np.int32)

sim_time = 0.0
total_steps = int(28.0 / SIMULATION_DT)
mission_done = False

for s in range(total_steps):
    dist_goal = np.linalg.norm(single_goal - d.qpos[:2])

    if not mission_done and dist_goal <= 0.38:
        mission_done = True
        print(f"\n" + "="*75)
        print(f"[TALOS C2] EXTRACTION FINALE RÉUSSIE À t={sim_time:.1f}s ! (X={d.qpos[0]:.2f}m, Y={d.qpos[1]:.2f}m) Dist={dist_goal:.2f}m")
        print("="*75 + "\n")

    if s % CONTROL_DECIMATION == 0:
        if mission_done:
            current_vx = max(0.0, current_vx - 0.025)
            desired_vy = 0.0
            desired_vyaw = 0.0
            phase = "OBJECTIF ATTEINT : SENTINELLE"
            ranges = np.ones(NUM_LIDAR_RAYS) * LIDAR_MAX_RANGE
            best_angle = 0.0
        else:
            desired_vx, desired_vy, desired_vyaw, phase, best_angle, ranges = autonomous_explorer_navigation(
                m, d, single_goal, geomid_buffer, mission_done
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
        sin_phase = np.sin(2.0 * np.pi * phase_clock)
        cos_phase = np.cos(2.0 * np.pi * phase_clock)

        obs[:3] = omega
        obs[3:6] = grav
        obs[6:9] = cmd * CMD_SCALE
        obs[9:21] = qj
        obs[21:33] = dqj
        obs[33:45] = action
        obs[45:47] = np.array([sin_phase, cos_phase], dtype=np.float32)

        with torch.no_grad():
            action = policy(torch.from_numpy(obs).unsqueeze(0)).numpy().squeeze()

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

        if s % 20 == 0:
            qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
            robot_yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
            slam.update(d.qpos[0], d.qpos[1], robot_yaw, ranges, LIDAR_ANGLES)

    tau = (target_dof_pos - d.qpos[7:]) * KPS - d.qvel[6:] * KDS
    d.ctrl[:] = tau
    mujoco.mj_step(m, d)
    sim_time += SIMULATION_DT

    if s % 250 == 0:
        print(f"t={sim_time:4.1f}s | POS: ({d.qpos[0]:+.2f}, {d.qpos[1]:+.2f}) Z={d.qpos[2]:.2f}m | Dist Goal={dist_goal:.2f}m | {phase}")

    if mission_done and sim_time > 26.0:
        break

print(f"\nResult: X={d.qpos[0]:.2f}m, Y={d.qpos[1]:.2f}m, Z={d.qpos[2]:.2f}m, Dist to Goal={dist_goal:.2f}m")
