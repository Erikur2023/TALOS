import os
import time
import numpy as np
import cv2
import torch
import mujoco

from run_sim import (
    KPS, KDS, DEFAULT_ANGLES, DOF_POS_SCALE, DOF_VEL_SCALE, ANG_VEL_SCALE,
    ACTION_SCALE, CMD_SCALE, get_gravity_orientation, SIMULATION_DT, CONTROL_DECIMATION,
    vfh_autonomous_navigation, scan_lidar_points
)

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
SCENE_XML = os.path.join(WORKSPACE_DIR, "unitree_rl_gym", "resources", "robots", "g1_description", "scene_real_apartment.xml")
POLICY_PT = os.path.join(WORKSPACE_DIR, "unitree_rl_gym", "deploy", "pre_train", "g1", "motion.pt")

class TacticalSlam2D:
    def __init__(self, x_bounds=(-2.5, 7.5), y_bounds=(-9.0, 3.0), res=0.035):
        self.min_x, self.max_x = x_bounds
        self.min_y, self.max_y = y_bounds
        self.res = res
        self.w = int((self.max_x - self.min_x) / res)
        self.h = int((self.max_y - self.min_y) / res)
        self.grid = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.grid[:] = (18, 22, 28) # Fond tactique sombre
        self.trajectory = []

    def world_to_map(self, x, y):
        px = int((x - self.min_x) / self.res)
        py = int((self.max_y - y) / self.res)
        return np.clip(px, 0, self.w - 1), np.clip(py, 0, self.h - 1)

    def update(self, rx, ry, yaw, ranges, lidar_angles):
        r_px, r_py = self.world_to_map(rx, ry)
        self.trajectory.append((r_px, r_py))

        for dist, angle in zip(ranges, lidar_angles):
            global_angle = yaw + angle
            hx = rx + dist * np.cos(global_angle)
            hy = ry + dist * np.sin(global_angle)
            h_px, h_py = self.world_to_map(hx, hy)

            # Espace libre (tracé laser exploré)
            cv2.line(self.grid, (r_px, r_py), (h_px, h_py), (45, 55, 70), 1)

            # Obstacle / Mur détecté
            if dist < 4.9:
                cv2.circle(self.grid, (h_px, h_py), 1, (0, 215, 255), -1)

    def render(self, rx, ry, yaw, waypoints=None, current_wp_idx=0):
        disp = self.grid.copy()

        # Trajectoire parcourue
        if len(self.trajectory) > 1:
            pts = np.array(self.trajectory, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(disp, [pts], False, (255, 200, 0), 2)

        # Waypoints
        if waypoints is not None:
            for idx, wp in enumerate(waypoints):
                w_px, w_py = self.world_to_map(wp[0], wp[1])
                color = (0, 255, 120) if idx == current_wp_idx else (150, 150, 150)
                cv2.circle(disp, (w_px, w_py), 5, color, -1)
                cv2.putText(disp, f"WP{idx+1}", (w_px + 8, w_py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

        # Robot et vecteur de cap
        r_px, r_py = self.world_to_map(rx, ry)
        cv2.circle(disp, (r_px, r_py), 6, (0, 255, 255), -1)
        tip_x = int(r_px + 16 * np.cos(-yaw + np.pi/2)) # rotation convention
        tip_y = int(r_py + 16 * np.sin(-yaw + np.pi/2))
        cv2.line(disp, (r_px, r_py), (tip_x, tip_y), (0, 255, 255), 2)

        # Légende HUD
        cv2.putText(disp, "TALOS C2 // SLAM CARTOGRAPHY 2D (LIVOX MID-360)", (12, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(disp, f"SURFACE CARTOGRAPHIEE : {len(self.trajectory)*0.15:.1f} m2", (12, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 220, 255), 1)
        return disp

def run_test():
    m = mujoco.MjModel.from_xml_path(SCENE_XML)
    d = mujoco.MjData(m)
    policy = torch.jit.load(POLICY_PT).eval()

    # Position initiale dans la salle à manger
    d.qpos[0] = 3.6
    d.qpos[1] = -5.5
    d.qpos[2] = 0.72
    yaw_init = 1.8326
    d.qpos[3] = np.cos(yaw_init / 2.0)
    d.qpos[4] = 0.0
    d.qpos[5] = 0.0
    d.qpos[6] = np.sin(yaw_init / 2.0)
    mujoco.mj_forward(m, d)

    slam = TacticalSlam2D()
    waypoints = [
        np.array([3.8, -3.4]),  # WP1: Sortie salle à manger / Porte
        np.array([2.5, -2.7]),  # WP2: Couloir principal
        np.array([0.6, -2.6]),  # WP3: Carrefour couloir
        np.array([0.1, -1.6]),  # WP4: Virage entrée salon
        np.array([1.2, -1.2])   # WP5: Objectif extraction dans le salon
    ]
    current_wp_idx = 0
    mission_done = False

    action = np.zeros(12, dtype=np.float32)
    target_dof_pos = DEFAULT_ANGLES.copy()
    obs = np.zeros(47, dtype=np.float32)
    current_vx = 0.40
    geomid_buffer = np.zeros(1, dtype=np.int32)

    sim_time = 0.0
    total_steps = int(25.0 / SIMULATION_DT)
    from run_sim import LIDAR_ANGLES

    print("=" * 80)
    print("TEST MISSION RESIDENCE : PATROUILLE MULTI-PIECES & SLAM CARTOGRAPHY")
    print("=" * 80)

    for s in range(total_steps):
        current_target = waypoints[current_wp_idx]
        dist_to_wp = np.linalg.norm(current_target - d.qpos[:2])

        if not mission_done and dist_to_wp <= 0.36:
            print(f"[TALOS C2] WP{current_wp_idx+1} ATTEINT a t={sim_time:.1f}s ! (X={d.qpos[0]:.2f}m, Y={d.qpos[1]:.2f}m)")
            if current_wp_idx < len(waypoints) - 1:
                current_wp_idx += 1
                print(f"  -> BASCULE VERS WP{current_wp_idx+1} : {waypoints[current_wp_idx]}")
            else:
                mission_done = True
                print("\n" + "="*70)
                print(f"[TALOS C2] EXTRACTION FINALE ATTEINTE A t={sim_time:.1f}s ! MODE SENTINELLE STABILISE.")
                print("="*70 + "\n")

        # Contrôleur
        if s % CONTROL_DECIMATION == 0:
            if mission_done:
                # Mode Sentinelle : arrêt complet & stabilisation parfaite debout
                target_dof_pos = DEFAULT_ANGLES.copy()
                grav = get_gravity_orientation(d.qpos[3:7])
                pitch_err = grav[0]
                roll_err = grav[1]
                target_dof_pos[0] -= pitch_err * 0.45
                target_dof_pos[4] -= pitch_err * 0.35
                target_dof_pos[6] -= pitch_err * 0.45
                target_dof_pos[10] -= pitch_err * 0.35
                target_dof_pos[1] += roll_err * 0.35
                target_dof_pos[5] += roll_err * 0.20
                target_dof_pos[7] += roll_err * 0.35
                target_dof_pos[11] += roll_err * 0.20
            else:
                desired_vx, desired_vy, desired_vyaw, phase, best_angle, ranges = vfh_autonomous_navigation(
                    m, d, current_target, geomid_buffer, False
                )
                current_vx = np.clip(desired_vx, current_vx - 0.02, current_vx + 0.02)
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

                # Cartographie SLAM à 25 Hz
                if s % 20 == 0:
                    quat = d.qpos[3:7]
                    qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
                    robot_yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
                    slam.update(d.qpos[0], d.qpos[1], robot_yaw, ranges, LIDAR_ANGLES)

        tau = (target_dof_pos - d.qpos[7:]) * KPS - d.qvel[6:] * KDS
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
        sim_time += SIMULATION_DT

        if s % 250 == 0:
            print(f"T+{sim_time:4.1f}s | POS: ({d.qpos[0]:+.2f}, {d.qpos[1]:+.2f}) Z={d.qpos[2]:.2f}m | WP{current_wp_idx+1} dist={dist_to_wp:.2f}m")

        if mission_done and (sim_time > 2.0): # laisser 2s de stabilisation
            break

    # Rendu et export de la carte SLAM finale
    quat = d.qpos[3:7]
    qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
    final_yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    slam_img = slam.render(d.qpos[0], d.qpos[1], final_yaw, waypoints, current_wp_idx)

    slam_path = os.path.join(WORKSPACE_DIR, "cartographie_slam_house.png")
    cv2.imwrite(slam_path, slam_img)
    print(f"[OK] Carte SLAM sauvegardee sous : {slam_path}")
    print(f"Final Pos: X={d.qpos[0]:.2f}m, Y={d.qpos[1]:.2f}m, Z={d.qpos[2]:.2f}m, Mission Done: {mission_done}")

if __name__ == "__main__":
    run_test()
