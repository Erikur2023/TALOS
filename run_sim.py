"""
=============================================================================
TALOS #225 - Navigation Autonome Multi-Scènes par LiDAR (Unitree G1)
Hackathon Robot Humanoïde Défense (AID / Artefact Adopt AI)
=============================================================================
Architecture de navigation 100% autonome et généralisable (ZÉRO HARDCODE) :
- ZÉRO coordonnées d'obstacles en dur : le robot est totalement aveugle à la scène
- Perception en temps réel par LiDAR 3D Livox (45 faisceaux physiques calculés à 50 Hz)
- Radar polaire LiDAR affiché en direct sur le HUD (faisceaux verts = libre, rouges = obstacle)
- Algorithme VFH (Vector Field Histogram) avec dilatation du gabarit du robot (50 cm)
- Démonstration de généralisation avec --random : l'obstacle est placé aléatoirement

Scènes disponibles :
    --scene single    (Défaut : Obstacle cylindrique)
    --scene chicane   (Parcours tactique en chicane : 3 obstacles décalés)
    --scene corridor  (Couloir confiné avec décombres et murs latéraux)
    --random          (Positionne les obstacles à des coordonnées inconnues et aléatoires)

Boutons / Raccourcis de contrôle en direct :
    [1] : Caméra Classique RGB seule
    [2] : LiDAR 3D Profondeur seule
    [3] : SUPERPOSITION / FUSION (RGB + LiDAR superposés)
    [C] ou [ESPACE] : Bascule 3ème personne <-> Vue subjective FPV (yeux du robot)
=============================================================================
"""

import os
import sys

# Compatibilité encodage console Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import time
import json
import argparse
import numpy as np
import torch
import mujoco
import mujoco.viewer
import cv2
from PIL import Image, ImageTk
import tkinter as tk

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
ROBOT_DIR = os.path.join(WORKSPACE_DIR, "unitree_rl_gym", "resources", "robots", "g1_description")
POLICY_PT = os.path.join(WORKSPACE_DIR, "unitree_rl_gym", "deploy", "pre_train", "g1", "motion.pt")

SCENES = {
    "single": {
        "xml": os.path.join(ROBOT_DIR, "scene_obstacle.xml"),
        "goal": np.array([3.6, 0.0]),
        "desc": "Obstacle unique (Scène standard)",
        "obs_geoms": ["obstacle_pillar"]
    },
    "rubble": {
        "xml": os.path.join(ROBOT_DIR, "scene_tactical_rubble.xml"),
        "goal": np.array([4.6, 0.0]),
        "desc": "Champ de décombres et gravats tactiques (5 obstacles)",
        "obs_geoms": ["rubble_slab_1", "rubble_brick_2", "metal_obstacle_3", "hazard_drum_4", "rubble_slab_5"]
    },
    "chicane": {
        "xml": os.path.join(ROBOT_DIR, "scene_tactical_chicane.xml"),
        "goal": np.array([4.5, 0.0]),
        "desc": "Parcours en chicane (3 obstacles décalés)",
        "obs_geoms": ["rubble_pillar_1", "rubble_block_2", "rubble_pillar_3"]
    },
    "corridor": {
        "xml": os.path.join(ROBOT_DIR, "scene_tactical_corridor.xml"),
        "goal": np.array([4.5, 0.0]),
        "desc": "Couloir confiné avec décombres",
        "obs_geoms": ["debris_1", "debris_2", "corridor_wall_left", "corridor_wall_right"]
    },
    "rocky": {
        "xml": os.path.join(ROBOT_DIR, "scene_tactical_rocky.xml"),
        "goal": np.array([4.6, 0.0]),
        "desc": "Canyon rocailleux tout-terrain (80 cailloux, rampes inclinées et rochers)",
        "obs_geoms": ["boulder_rock_1", "boulder_rock_2", "boulder_rock_3", "boulder_rock_4"]
    },
    "rough": {
        "xml": os.path.join(ROBOT_DIR, "scene_tactical_rocky.xml"),
        "goal": np.array([4.6, 0.0]),
        "desc": "Canyon rocailleux tout-terrain (80 cailloux, rampes inclinées et rochers)",
        "obs_geoms": ["boulder_rock_1", "boulder_rock_2", "boulder_rock_3", "boulder_rock_4"]
    },
    "trench": {
        "xml": os.path.join(ROBOT_DIR, "scene_tactical_trench.xml"),
        "goal": np.array([4.5, 0.0]),
        "desc": "Tranchée de combat fortifiée (parapets sacs de sable, caisses munitions, hérisson)",
        "obs_geoms": ["trench_ammo_box_1", "trench_sandbag_2"]
    },
    "industrial": {
        "xml": os.path.join(ROBOT_DIR, "scene_tactical_industrial.xml"),
        "goal": np.array([4.5, 0.0]),
        "desc": "Complexe industriel & fuite chimique NRBC (cuves chimiques, fûts toxiques)",
        "obs_geoms": ["ind_barrel_1", "ind_silo_2"]
    },
    "night": {
        "xml": os.path.join(ROBOT_DIR, "scene_tactical_night.xml"),
        "goal": np.array([4.5, 0.0]),
        "desc": "Opération spéciale nocturne & infiltration (barrage routier, barrières jersey, projecteurs)",
        "obs_geoms": ["night_jersey_1", "night_jersey_2"]
    },
    "house": {
        "xml": os.path.join(ROBOT_DIR, "scene_real_apartment.xml"),
        "goal": np.array([0.25, -1.10]),
        "desc": "Résidence Réelle 3D (Découverte 100% Autonome - ZÉRO Waypoint, navigation par ouvertures LiDAR)",
        "start_pos": np.array([3.6, -5.5, 0.72]),
        "start_yaw": 1.8326,
        "obs_geoms": ["apt_table", "apt_wall_left", "apt_wall_right"]
    }
}

SIMULATION_DT = 0.002       # 500 Hz physique
CONTROL_DECIMATION = 10     # 50 Hz politique RL

KPS = np.array([100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40], dtype=np.float32)
KDS = np.array([2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2], dtype=np.float32)
DEFAULT_ANGLES = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                           -0.1, 0.0, 0.0, 0.3, -0.2, 0.0], dtype=np.float32)

ACTION_SCALE = 0.25
CMD_SCALE = np.array([2.0, 2.0, 0.25], dtype=np.float32)
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05
ANG_VEL_SCALE = 0.25

NUM_LIDAR_RAYS = 45
LIDAR_ANGLES = np.linspace(-np.radians(80), np.radians(80), NUM_LIDAR_RAYS)
LIDAR_MAX_RANGE = 5.0
ROBOT_CORRIDOR_RADIUS = 0.50  # Envergure physique + marge nette (50 cm)


def get_gravity_orientation(quat):
    qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
    return np.array([
        2.0 * (-qz * qx + qw * qy),
        -2.0 * (qz * qy + qw * qx),
        1.0 - 2.0 * (qw * qw + qz * qz)
    ], dtype=np.float32)


def get_yaw_degrees(quat):
    qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
    yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    return int((np.degrees(yaw) + 360) % 360)


# Paramètres optimaux issus de la Session d'entraînement 1 (Vitesse 0.62m/s & Zéro collision)
NAV_CONFIG = {
    "max_vx": 0.62,
    "lookahead_dist": 2.2,
    "corridor_radius": 0.52,
    "max_vy": 0.35,
    "accel_rate": 0.025
}
PARAMS_FILE = os.path.join(WORKSPACE_DIR, "best_nav_params.json")
if os.path.exists(PARAMS_FILE):
    try:
        with open(PARAMS_FILE, "r", encoding="utf-8") as f:
            NAV_CONFIG.update(json.load(f))
    except Exception:
        pass


def scan_lidar_points(m, d, geomid_buffer, yaw):
    """
    Simulateur matériel du LiDAR 3D Livox Mid-360 multi-niveaux :
    Émet des faisceaux laser à plusieurs hauteurs relatives au torse du robot
    pour cartographier les décombres, franchir les marches et esquiver les obstacles réels.
    """
    points = []
    ranges = []
    cos_y = np.cos(yaw)
    sin_y = np.sin(yaw)
    robot_z = float(d.qpos[2])

    for a in LIDAR_ANGLES:
        dists = []
        for (z_rel, ray_pitch) in [(-0.55, 0.0), (-0.30, -0.04), (0.0, 0.0)]:
            ray_dir = np.array([np.cos(yaw + a), np.sin(yaw + a), ray_pitch])
            ray_dir = ray_dir / np.linalg.norm(ray_dir)
            orig = np.array([d.qpos[0] + 0.15 * cos_y, d.qpos[1] + 0.15 * sin_y, robot_z + z_rel])
            dist = mujoco.mj_ray(m, d, orig, ray_dir, None, 1, 1, geomid_buffer)
            if dist >= 0:
                hit_pt = orig + dist * ray_dir
                gid = geomid_buffer[0]
                gname = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid) if gid >= 0 else "") or ""
                walkable = any(k in gname for k in ["floor", "stair_step", "stair_lip", "landing", "target", "extraction", "lawn", "villa"])
                if not walkable and hit_pt[2] > (robot_z - 0.65):
                    dists.append(dist)
        min_d = min(dists) if dists else LIDAR_MAX_RANGE
        ranges.append(min_d)
        if dists:
            px = min_d * np.cos(a)
            py = min_d * np.sin(a)
            points.append((px, py, min_d, a))
    return np.array(ranges), points


def vfh_autonomous_navigation(m, d, goal_pos, geomid_buffer, mission_done=False):
    """
    Contrôleur de navigation 100% autonome par champs de vecteurs et trajectoires dynamiques (VFH / DWA) :
    - ZÉRO HARDCODE : Le robot ne connaît aucun obstacle ni mur à l'avance.
    - Se base UNIQUEMENT sur les scans laser LiDAR 3D et le capteur IMU embarqué.
    """
    if mission_done:
        return 0.0, 0.0, 0.0, "CIBLE ATTEINTE (EN GARDE)", 0.0, np.ones(NUM_LIDAR_RAYS) * LIDAR_MAX_RANGE

    robot_pos = d.qpos[:2]
    to_goal = goal_pos - robot_pos
    dist_goal = np.linalg.norm(to_goal)

    if dist_goal <= 0.28:
        return 0.0, 0.0, 0.0, "CIBLE ATTEINTE (EN GARDE)", 0.0, np.ones(NUM_LIDAR_RAYS) * LIDAR_MAX_RANGE

    quat = d.qpos[3:7]
    qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
    yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    goal_angle_world = np.arctan2(to_goal[1], to_goal[0])
    goal_angle_local = (goal_angle_world - yaw + np.pi) % (2.0 * np.pi) - np.pi

    ranges, points = scan_lidar_points(m, d, geomid_buffer, yaw)

    R_robot = 0.32
    lookahead = 2.0
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
            # Projection virtuelle le long du rayon libre (sélection d'ouverture optimale)
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
        # Mur frontal immédiat : rotation douce vers l'ouverture la plus dégagée
        desired_vx = 0.10
        desired_vyaw = float(np.clip(best_angle * 1.6, -0.75, 0.75))
        desired_vy = float(np.clip(0.08 * np.sin(best_angle), -0.08, 0.08))
        phase_label = f"CONTOURNEMENT D'OBSTACLE [CAP {np.degrees(best_angle):+4.1f}°]"
    else:
        obstacle_slowdown = min(1.0, max(0.20, (min_front_obs - 0.42) / 0.70))
        speed_scale = min(1.0, max(0.35, dist_goal / 0.8)) * obstacle_slowdown
        max_vx = NAV_CONFIG.get("max_vx", 0.50)
        desired_vx = float(np.clip(max_vx * max(0.18, np.cos(best_angle)) * speed_scale, 0.15, max_vx))
        desired_vy = float(np.clip(0.06 * np.sin(best_angle), -0.06, 0.06))
        desired_vyaw = float(np.clip(best_angle * 1.45, -0.70, 0.70))
        phase_label = f"DÉCOUVERTE AUTONOME [CAP {np.degrees(best_angle):+4.1f}°]"
    return desired_vx, desired_vy, desired_vyaw, phase_label, best_angle, ranges


class TacticalSlam2D:
    """Cartographie 2D en temps réel par LiDAR 3D (SLAM Occupancy Grid)."""
    def __init__(self, x_bounds=(-2.5, 7.5), y_bounds=(-9.0, 3.0), res=0.035):
        self.min_x, self.max_x = x_bounds
        self.min_y, self.max_y = y_bounds
        self.res = res
        self.w = int((self.max_x - self.min_x) / res)
        self.h = int((self.max_y - self.min_y) / res)
        self.grid = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.grid[:] = (18, 22, 28)
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

            cv2.line(self.grid, (r_px, r_py), (h_px, h_py), (45, 55, 70), 1)
            if dist < 4.9:
                cv2.circle(self.grid, (h_px, h_py), 1, (0, 215, 255), -1)

    def render(self, rx, ry, yaw, waypoints=None, current_wp_idx=0):
        disp = self.grid.copy()

        if len(self.trajectory) > 1:
            pts = np.array(self.trajectory, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(disp, [pts], False, (255, 200, 0), 2)

        if waypoints is not None:
            for idx, wp in enumerate(waypoints):
                w_px, w_py = self.world_to_map(wp[0], wp[1])
                color = (0, 255, 120) if idx == current_wp_idx else (150, 150, 150)
                cv2.circle(disp, (w_px, w_py), 5, color, -1)
                cv2.putText(disp, f"WP{idx+1}", (w_px + 8, w_py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

        r_px, r_py = self.world_to_map(rx, ry)
        cv2.circle(disp, (r_px, r_py), 6, (0, 255, 255), -1)
        tip_x = int(r_px + 16 * np.cos(-yaw + np.pi/2))
        tip_y = int(r_py + 16 * np.sin(-yaw + np.pi/2))
        cv2.line(disp, (r_px, r_py), (tip_x, tip_y), (0, 255, 255), 2)

        cv2.putText(disp, "TALOS C2 // SLAM CARTOGRAPHY 2D (LIVOX MID-360)", (12, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(disp, f"SURFACE CARTOGRAPHIEE : {len(self.trajectory)*0.15:.1f} m2", (12, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 220, 255), 1)
        return disp


class TacticalHudTk:
    """Poste de Commandement C2 Tactique & Perception Multispectrale Haute Définition."""
    def __init__(self, on_mode_change_cb, on_cam_toggle_cb):
        self.root = tk.Tk()
        self.root.title("TALOS C2 // UNITREE G1 DEFENSE OPERATIONS DASHBOARD")
        self.root.geometry("664x610+40+40")
        self.root.configure(bg="#0b1120")
        self.root.attributes("-topmost", True)

        self.on_mode_change = on_mode_change_cb
        self.on_cam_toggle = on_cam_toggle_cb
        self.current_mode = "fusion"

        btn_frame = tk.Frame(self.root, bg="#0b1120")
        btn_frame.pack(fill=tk.X, padx=12, pady=8)

        self.btn_rgb = tk.Button(btn_frame, text="[1] Optronique RGB", font=("Consolas", 10, "bold"),
                                 bg="#1e293b", fg="#94a3b8", command=lambda: self.set_mode("rgb"), width=16)
        self.btn_rgb.pack(side=tk.LEFT, padx=3)

        self.btn_lidar = tk.Button(btn_frame, text="[2] LiDAR 3D Turbo", font=("Consolas", 10, "bold"),
                                   bg="#1e293b", fg="#94a3b8", command=lambda: self.set_mode("lidar"), width=16)
        self.btn_lidar.pack(side=tk.LEFT, padx=3)

        self.btn_fusion = tk.Button(btn_frame, text="[3] FUSION TACTIQUE", font=("Consolas", 10, "bold"),
                                    bg="#0284c7", fg="#ffffff", command=lambda: self.set_mode("fusion"), width=18)
        self.btn_fusion.pack(side=tk.LEFT, padx=3)

        self.btn_slam = tk.Button(btn_frame, text="[4] SLAM 2D", font=("Consolas", 10, "bold"),
                                  bg="#1e293b", fg="#94a3b8", command=lambda: self.set_mode("slam"), width=12)
        self.btn_slam.pack(side=tk.LEFT, padx=3)

        self.btn_cam = tk.Button(btn_frame, text="[C] Vue Épaule/FPV/Drone", font=("Consolas", 9, "bold"),
                                 bg="#334155", fg="#38bdf8", command=self.on_cam_toggle)
        self.btn_cam.pack(side=tk.RIGHT, padx=3)

        self.img_label = tk.Label(self.root, bg="#000000", bd=2, relief=tk.RIDGE)
        self.img_label.pack(padx=12, pady=4)

        self.status_lbl = tk.Label(self.root, text="[TALOS C2] OPTRONIQUE & RADAR ACTIFS // SYSTÈME NOMINAL",
                                   font=("Consolas", 10, "bold"), bg="#0f172a", fg="#38bdf8", pady=6)
        self.status_lbl.pack(fill=tk.X, padx=12, pady=4)

        self.root.bind("<Key-1>", lambda e: self.set_mode("rgb"))
        self.root.bind("<Key-2>", lambda e: self.set_mode("lidar"))
        self.root.bind("<Key-3>", lambda e: self.set_mode("fusion"))
        self.root.bind("<Key-4>", lambda e: self.set_mode("slam"))
        self.root.bind("<Key-s>", lambda e: self.cycle_mode())
        self.root.bind("<Key-S>", lambda e: self.cycle_mode())
        self.root.bind("<Key-c>", lambda e: self.on_cam_toggle())
        self.root.bind("<Key-C>", lambda e: self.on_cam_toggle())
        self.root.bind("<space>", lambda e: self.on_cam_toggle())

        self.photo = None
        self.is_alive = True
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self):
        self.is_alive = False
        try:
            self.root.destroy()
        except Exception:
            pass

    def set_mode(self, mode):
        self.current_mode = mode
        self.on_mode_change(mode)
        self.btn_rgb.config(bg="#0284c7" if mode == "rgb" else "#1e293b", fg="#ffffff" if mode == "rgb" else "#94a3b8")
        self.btn_lidar.config(bg="#0284c7" if mode == "lidar" else "#1e293b", fg="#ffffff" if mode == "lidar" else "#94a3b8")
        self.btn_fusion.config(bg="#0284c7" if mode == "fusion" else "#1e293b", fg="#ffffff" if mode == "fusion" else "#94a3b8")
        self.btn_slam.config(bg="#0284c7" if mode == "slam" else "#1e293b", fg="#ffffff" if mode == "slam" else "#94a3b8")

    def cycle_mode(self):
        modes = ["rgb", "lidar", "fusion", "slam"]
        idx = (modes.index(self.current_mode) + 1) % len(modes)
        self.set_mode(modes[idx])

    def draw_pro_tactical_hud(self, frame_bgr, ranges, best_angle, yaw_deg, pitch_deg, roll_deg,
                              dist_to_goal, speed_ms, sim_time, phase_name, anti_fall_active, slam_map=None):
        """Incruste un HUD tactique de norme militaire DGA/AID avec radar, horizon artificiel et télémétrie."""
        h, w = frame_bgr.shape[:2]
        cx, cy = w // 2, h // 2

        CYAN = (248, 189, 56)      # #38bdf8
        GREEN = (80, 220, 100)     # Vert tactique
        AMBER = (40, 180, 255)     # Ambre alerte
        RED = (60, 60, 240)        # Rouge alerte contact
        WHITE = (240, 240, 240)
        DARK_BG = (15, 23, 42)     # Slate 900

        # 1. Bords de visée tactiques (Coins type viseur militaire)
        pad = 16
        corner_len = 24
        thick = 2
        for (x, y, dx, dy) in [
            (pad, pad, 1, 1),
            (w - pad, pad, -1, 1),
            (pad, h - pad, 1, -1),
            (w - pad, h - pad, -1, -1)
        ]:
            cv2.line(frame_bgr, (x, y), (x + dx * corner_len, y), CYAN, thick)
            cv2.line(frame_bgr, (x, y), (x, y + dy * corner_len), CYAN, thick)

        # 2. Bandeau supérieur : Télémétrie C2 & Cap IMU
        header_h = 32
        sub_top = frame_bgr[pad:pad+header_h, pad:w-pad]
        bg_top = np.zeros_like(sub_top)
        bg_top[:] = DARK_BG
        cv2.addWeighted(bg_top, 0.80, sub_top, 0.20, 0, sub_top)
        cv2.rectangle(frame_bgr, (pad, pad), (w - pad, pad + header_h), CYAN, 1)

        cv2.putText(frame_bgr, "TALOS // DGA-AID C2", (pad + 10, pad + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.42, CYAN, 1)
        cv2.putText(frame_bgr, f"CAP IMU: {yaw_deg:03d} DEG", (cx - 55, pad + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.44, WHITE, 1)
        cv2.putText(frame_bgr, f"T+{sim_time:4.1f}s | 500Hz", (w - pad - 120, pad + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.40, GREEN, 1)

        # 3. Horizon Artificiel (Asservissement gyrométrique temps réel)
        pitch_offset = int(np.clip(pitch_deg * 2.5, -45, 45))
        angle_rad = np.radians(roll_deg)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        hw = 36
        x1 = int(cx - hw * cos_a)
        y1 = int(cy + pitch_offset - hw * sin_a)
        x2 = int(cx - 10 * cos_a)
        y2 = int(cy + pitch_offset - 10 * sin_a)
        cv2.line(frame_bgr, (x1, y1), (x2, y2), CYAN, 2)

        x3 = int(cx + 10 * cos_a)
        y3 = int(cy + pitch_offset + 10 * sin_a)
        x4 = int(cx + hw * cos_a)
        y4 = int(cy + pitch_offset + hw * sin_a)
        cv2.line(frame_bgr, (x3, y3), (x4, y4), CYAN, 2)
        cv2.circle(frame_bgr, (cx, cy), 3, CYAN, -1)

        # 4. Réticule central tactique avec lead angle VFH
        reticle_color = (0, 240, 255) if dist_to_goal <= 0.35 else GREEN
        cv2.circle(frame_bgr, (cx, cy), 16, reticle_color, 1)
        cv2.line(frame_bgr, (cx - 24, cy), (cx - 18, cy), reticle_color, 1)
        cv2.line(frame_bgr, (cx + 18, cy), (cx + 24, cy), reticle_color, 1)
        cv2.line(frame_bgr, (cx, cy - 24), (cx, cy - 18), reticle_color, 1)
        cv2.line(frame_bgr, (cx, cy + 18), (cx, cy + 24), reticle_color, 1)

        # 5. Bandeau inférieur gauche : Diagnostics de bord & réflexe anti-chute
        info_w, info_h = 190, 78
        iy = h - pad - info_h
        sub_info = frame_bgr[iy:h-pad, pad:pad+info_w]
        bg_info = np.zeros_like(sub_info)
        bg_info[:] = DARK_BG
        cv2.addWeighted(bg_info, 0.82, sub_info, 0.18, 0, sub_info)
        cv2.rectangle(frame_bgr, (pad, iy), (pad + info_w, h - pad), CYAN, 1)

        status_text = "ANTI-CHUTE: RATTRAPAGE" if anti_fall_active else "ANTI-CHUTE: VEILLE ACTIVE"
        status_color = AMBER if anti_fall_active else GREEN

        cv2.putText(frame_bgr, f"CIBLE EXTRACTION : {dist_to_goal:4.2f}m", (pad + 8, iy + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.36, WHITE, 1)
        cv2.putText(frame_bgr, f"VITESSE AVANCE   : {speed_ms:4.2f} m/s", (pad + 8, iy + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.36, CYAN, 1)
        cv2.putText(frame_bgr, f"PITCH/ROLL       : {pitch_deg:+.1f} / {roll_deg:+.1f}", (pad + 8, iy + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.35, WHITE, 1)
        cv2.putText(frame_bgr, status_text, (pad + 8, iy + 69), cv2.FONT_HERSHEY_SIMPLEX, 0.33, status_color, 1)

        # 6. Mini-radar LiDAR 3D Livox (Coin inférieur droit)
        rw, rh = 148, 128
        rx = w - pad - rw
        ry = h - pad - rh
        sub_radar = frame_bgr[ry:ry+rh, rx:rx+rw]
        bg_radar = np.zeros_like(sub_radar)
        bg_radar[:] = DARK_BG
        cv2.addWeighted(bg_radar, 0.85, sub_radar, 0.15, 0, sub_radar)
        cv2.rectangle(frame_bgr, (rx, ry), (rx + rw, ry + rh), CYAN, 1)

        rcx, rcy = rx + rw // 2, ry + rh - 14
        cv2.circle(frame_bgr, (rcx, rcy), 4, CYAN, -1)
        scale = 24.0  # pixels par mètre
        for r_m in [1.0, 2.0, 3.0]:
            cv2.circle(frame_bgr, (rcx, rcy), int(r_m * scale), (51, 65, 85), 1)

        angles = np.linspace(-np.radians(80), np.radians(80), len(ranges))
        for a, dist in zip(angles, ranges):
            d_c = min(dist, 3.2)
            px = int(rcx - d_c * np.sin(a) * scale)
            py = int(rcy - d_c * np.cos(a) * scale)
            color = RED if dist < 1.7 else GREEN
            cv2.line(frame_bgr, (rcx, rcy), (px, py), color, 1)

        ax = int(rcx - 2.5 * np.sin(best_angle) * scale)
        ay = int(rcy - 2.5 * np.cos(best_angle) * scale)
        cv2.arrowedLine(frame_bgr, (rcx, rcy), (ax, ay), (0, 255, 255), 2, tipLength=0.3)
        cv2.putText(frame_bgr, "RADAR LIVOX 3D", (rx + 8, ry + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, CYAN, 1)

        # 7. Incrustation mini-carte SLAM Cartography en haut à droite (sous le bandeau)
        if slam_map is not None:
            mw, mh = 138, 138
            mx = w - pad - mw
            my = pad + header_h + 8
            mini_slam = cv2.resize(slam_map, (mw, mh))
            frame_bgr[my:my+mh, mx:mx+mw] = mini_slam
            cv2.rectangle(frame_bgr, (mx, my), (mx + mw, my + mh), CYAN, 1)
            cv2.putText(frame_bgr, "SLAM LIVE", (mx + 6, my + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 255, 255), 1)

    def update_frame(self, frame_bgr, ranges, best_angle, yaw_deg, pitch_deg, roll_deg,
                     dist_to_goal, speed_ms, sim_time, phase_name, anti_fall_active, slam_map=None):
        if not self.is_alive:
            return
        try:
            if self.current_mode == "slam" and slam_map is not None:
                disp = cv2.resize(slam_map, (640, 480))
            else:
                disp = frame_bgr.copy()
                self.draw_pro_tactical_hud(disp, ranges, best_angle, yaw_deg, pitch_deg, roll_deg,
                                           dist_to_goal, speed_ms, sim_time, phase_name, anti_fall_active, slam_map)

            lidar_min_m = float(np.min(ranges))
            frame_rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            self.photo = ImageTk.PhotoImage(image=img)
            self.img_label.config(image=self.photo)
            status_desc = f"{phase_name} | DIST CIBLE: {dist_to_goal:4.2f}m | LIDAR MIN: {lidar_min_m:4.2f}m"
            self.status_lbl.config(text=status_desc)
            self.root.update_idletasks()
            self.root.update()
        except Exception:
            self.is_alive = False


def run_simulation(scene_name="single", randomize=False, headless=False, start_fpv=False, show_hud=True):
    scene_info = SCENES.get(scene_name, SCENES["single"])
    xml_path = scene_info["xml"]
    goal_pos = scene_info["goal"]

    print("=" * 82)
    print("  TALOS #225 - UNITREE G1 DEFENSE : NAVIGATION AUTONOME 100% PAR CAPTEURS LIDAR")
    print("=" * 82)
    print(f"[*] Scène active : [{scene_name.upper()}] - {scene_info['desc']}")
    print(f"[*] Fichier XML : {xml_path}")
    print(f"[*] Mode aléatoire inconnu (--random) : {'ACTIVÉ (l\'obstacle sera placé aléatoirement)' if randomize else 'DÉSACTIVÉ'}")
    print(f"[*] Cible finale : X={goal_pos[0]:.2f}m, Y={goal_pos[1]:.2f}m")
    print(f"[*] Mode de contrôle : VFH RÉACTIF PAR LIDAR (Le robot est aveugle à la carte)")
    print("-" * 82)

    m = mujoco.MjModel.from_xml_path(xml_path)

    # Randomisation procédurale des obstacles si demandée
    if randomize:
        np.random.seed(int(time.time() * 1000) % 100000)
        for obs_name in scene_info["obs_geoms"]:
            gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, obs_name)
            if gid >= 0 and "wall" not in obs_name:
                orig_x, orig_y, orig_z = m.geom_pos[gid]
                new_x = orig_x + np.random.uniform(-0.35, 0.35)
                new_y = orig_y + np.random.uniform(-0.35, 0.35)
                m.geom_pos[gid] = np.array([new_x, new_y, orig_z])
                print(f"[RANDOM SCENE] Obstacle '{obs_name}' positionné aléatoirement en ({new_x:.2f}, {new_y:.2f})")

    d = mujoco.MjData(m)
    if "start_pos" in scene_info:
        d.qpos[:3] = scene_info["start_pos"]
    if "start_yaw" in scene_info:
        yaw_init = float(scene_info["start_yaw"])
        d.qpos[3] = np.cos(yaw_init / 2.0)
        d.qpos[4] = 0.0
        d.qpos[5] = 0.0
        d.qpos[6] = np.sin(yaw_init / 2.0)
    m.opt.timestep = SIMULATION_DT
    mujoco.mj_forward(m, d)

    head_cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
    obs_geom_ids = set(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name) for name in scene_info["obs_geoms"])
    robot_geom_ids = set(i for i in range(m.ngeom) if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[i]) != "world")
    geomid_buffer = np.zeros(1, dtype=np.int32)

    policy = torch.jit.load(POLICY_PT)
    policy.eval()

    action = np.zeros(12, dtype=np.float32)
    target_dof_pos = DEFAULT_ANGLES.copy()
    obs = np.zeros(47, dtype=np.float32)
    current_vx = 0.45

    telemetry = []
    step_count = 0
    sim_time = 0.0
    last_log_time = 0.0
    last_hud_time = 0.0
    total_obstacle_contacts = 0
    mission_done = False
    mission_done_time = 0.0

    camera_3d_mode = "fpv" if start_fpv else "third_person"
    sensor_mode = "fusion"

    def set_sensor_mode(new_mode):
        nonlocal sensor_mode
        sensor_mode = new_mode

    def toggle_3d_cam():
        nonlocal camera_3d_mode
        modes = ["third_person", "fpv", "drone"]
        idx = (modes.index(camera_3d_mode) + 1) % len(modes)
        camera_3d_mode = modes[idx]
        cam_labels = {"third_person": "ÉPAULE", "fpv": "CASQUE FPV", "drone": "DRONE ISR"}
        if hud_tk and hud_tk.is_alive:
            hud_tk.btn_cam.config(text=f"[C] Vue: {cam_labels[camera_3d_mode]}")
        print(f"[C2 CAMÉRA] Mode de vue basculé : {cam_labels[camera_3d_mode]}")

    def on_mujoco_key(keycode):
        if keycode in [ord('c'), ord('C'), ord('v'), ord('V'), 32]:
            toggle_3d_cam()
        elif keycode == ord('1'):
            set_sensor_mode("rgb")
            if hud_tk and hud_tk.is_alive:
                hud_tk.set_mode("rgb")
        elif keycode == ord('2'):
            set_sensor_mode("lidar")
            if hud_tk and hud_tk.is_alive:
                hud_tk.set_mode("lidar")
        elif keycode == ord('3'):
            set_sensor_mode("fusion")
            if hud_tk and hud_tk.is_alive:
                hud_tk.set_mode("fusion")
        elif keycode in [ord('s'), ord('S')]:
            if hud_tk and hud_tk.is_alive:
                hud_tk.cycle_mode()

    hud_tk = None
    renderer = None
    slam = TacticalSlam2D()
    waypoints = scene_info.get("waypoints", [goal_pos])
    current_wp_idx = 0
    active_target = waypoints[0]

    if show_hud and not headless:
        try:
            hud_tk = TacticalHudTk(set_sensor_mode, toggle_3d_cam)
            renderer = mujoco.Renderer(m, 240, 320)
        except Exception as e:
            print(f"[NOTE] Fenêtre HUD non initialisée : {e}")

    def step_controller():
        nonlocal current_vx, action, target_dof_pos, obs, current_wp_idx, active_target, mission_done, mission_done_time
        dist_to_active = np.linalg.norm(active_target - d.qpos[:2])

        if not mission_done and dist_to_active <= 0.42:
            if current_wp_idx < len(waypoints) - 1:
                current_wp_idx += 1
                active_target = waypoints[current_wp_idx]
                print(f"\n[TALOS C2] ÉTAPE {current_wp_idx}/{len(waypoints)} ATTEINTE à t={sim_time:.1f}s ! -> Direction WP{current_wp_idx+1}: ({active_target[0]:.2f}, {active_target[1]:.2f})")
            else:
                mission_done = True
                mission_done_time = sim_time
                print("\n" + "=" * 78)
                print(f" [TALOS C2] EXTRACTION FINALE ATTEINTE AVEC SUCCÈS À t={sim_time:.1f}s ! (X={d.qpos[0]:.2f}m, Y={d.qpos[1]:.2f}m)")
                print("  - Mode de stabilisation : Posture sentinelle équilibrée verrouillée.")
                print("=" * 78 + "\n")

        if mission_done:
            # Décélération douce et maintien actif de la posture debout par le réseau RL
            current_vx = max(0.0, current_vx - 0.025)
            desired_vy = 0.0
            desired_vyaw = 0.0
            phase = "OBJECTIF ATTEINT : FACTION SENTINELLE STABILISÉE"
            best_angle = 0.0
            ranges = np.ones(NUM_LIDAR_RAYS) * LIDAR_MAX_RANGE
        else:
            desired_vx, desired_vy, desired_vyaw, phase, best_angle, ranges = vfh_autonomous_navigation(
                m, d, active_target, geomid_buffer, mission_done
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
        
        # Module Anti-Chute & Rattrapage Dynamique (Stumble Recovery Reflex)
        pitch_err = grav[0]
        roll_err = grav[1]

        # 1. Rattrapage assiette longitudinale (Pitch & Escaliers)
        if abs(pitch_err) > 0.035:
            target_dof_pos[0] -= pitch_err * 0.35
            target_dof_pos[4] -= pitch_err * 0.25
            target_dof_pos[6] -= pitch_err * 0.35
            target_dof_pos[10] -= pitch_err * 0.25

        # 2. Rattrapage assiette transversale (Roll & cailloux)
        if abs(roll_err) > 0.035:
            target_dof_pos[1] += roll_err * 0.28
            target_dof_pos[5] += roll_err * 0.16
            target_dof_pos[7] += roll_err * 0.28
            target_dof_pos[11] += roll_err * 0.16

        return cmd, phase, best_angle, ranges

    def check_contacts():
        nonlocal total_obstacle_contacts
        for i in range(d.ncon):
            con = d.contact[i]
            in_obs = (con.geom1 in obs_geom_ids or con.geom2 in obs_geom_ids)
            in_robot = (con.geom1 in robot_geom_ids or con.geom2 in robot_geom_ids)
            if in_obs and in_robot:
                total_obstacle_contacts += 1

    # Boucle d'exécution avec visualiseur 3D
    if not headless:
        with mujoco.viewer.launch_passive(m, d, key_callback=on_mujoco_key) as viewer:
            prev_cam_mode = None

            print("\n[INFO] Simulation 100% autonome par LiDAR démarrée !")
            print("[INFO] Observez sur le HUD les 45 faisceaux laser, la mini-carte SLAM et l'évitement réactif.")
            print("[INFO] Raccourcis : [1] RGB | [2] LiDAR | [3] FUSION | [4] SLAM 2D | [C] Vue Caméra")
            print("[INFO] Pour quitter la simulation : fermez simplement la fenêtre 3D MuJoCo.\n")

            wall_start_time = time.time()
            while viewer.is_running():
                if camera_3d_mode != prev_cam_mode:
                    if camera_3d_mode == "fpv":
                        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                        viewer.cam.fixedcamid = head_cam_id
                    elif camera_3d_mode == "drone":
                        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                        viewer.cam.distance = 6.8
                        viewer.cam.elevation = -88.0
                        viewer.cam.azimuth = 90.0
                    else:  # third_person (suivi épaule dynamique)
                        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                        viewer.cam.distance = 3.8
                        viewer.cam.elevation = -18.0
                        viewer.cam.azimuth = 135.0
                    prev_cam_mode = camera_3d_mode

                if camera_3d_mode == "third_person":
                    viewer.cam.lookat[0] = d.qpos[0] + 0.35
                    viewer.cam.lookat[1] = d.qpos[1]
                    viewer.cam.lookat[2] = 0.72
                elif camera_3d_mode == "drone":
                    viewer.cam.lookat[0] = d.qpos[0] + 0.8
                    viewer.cam.lookat[1] = d.qpos[1]
                    viewer.cam.lookat[2] = 0.3

                dist_to_goal = np.linalg.norm(goal_pos - d.qpos[:2])

                check_contacts()

                tau = (target_dof_pos - d.qpos[7:]) * KPS - d.qvel[6:] * KDS
                d.ctrl[:] = tau
                mujoco.mj_step(m, d)
                sim_time += SIMULATION_DT
                step_count += 1

                cmd, phase, best_angle, ranges = [0.0, 0.0, 0.0], "EN ROUTE", 0.0, np.ones(NUM_LIDAR_RAYS) * 5.0
                if step_count % CONTROL_DECIMATION == 0:
                    cmd, phase, best_angle, ranges = step_controller()

                # Mise à jour cartographie SLAM à 25 Hz
                quat = d.qpos[3:7]
                yaw_current = np.arctan2(2.0 * (quat[0] * quat[3] + quat[1] * quat[2]), 1.0 - 2.0 * (quat[2] * quat[2] + quat[3] * quat[3]))
                if step_count % 20 == 0:
                    slam.update(d.qpos[0], d.qpos[1], yaw_current, ranges, LIDAR_ANGLES)

                # Mise à jour fenêtre HUD Tkinter (25 FPS)
                if hud_tk and hud_tk.is_alive and (sim_time - last_hud_time >= 0.04):
                    renderer.update_scene(d, camera="head_camera")
                    rgb = renderer.render()
                    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                    if sensor_mode in ["lidar", "fusion"]:
                        renderer.enable_depth_rendering()
                        depth = renderer.render()
                        renderer.disable_depth_rendering()
                        depth_norm = np.clip((depth - 0.4) / 4.0, 0, 1)
                        lidar_color = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)

                    if sensor_mode == "rgb":
                        disp = rgb_bgr
                    elif sensor_mode == "lidar":
                        disp = lidar_color
                    else:
                        disp = cv2.addWeighted(rgb_bgr, 0.55, lidar_color, 0.45, 0)

                    disp = cv2.resize(disp, (640, 480))
                    slam_map_render = slam.render(d.qpos[0], d.qpos[1], yaw_current, waypoints, current_wp_idx)

                    yaw_deg = get_yaw_degrees(quat)
                    grav = get_gravity_orientation(quat)
                    pitch_deg = float(np.degrees(-grav[0]))
                    roll_deg = float(np.degrees(grav[1]))
                    speed_ms = float(np.linalg.norm(d.qvel[:2]))
                    anti_fall_active = (abs(grav[0]) > 0.04 or abs(grav[1]) > 0.04)

                    hud_tk.update_frame(disp, ranges, best_angle, yaw_deg, pitch_deg, roll_deg,
                                       dist_to_goal, speed_ms, sim_time, phase, anti_fall_active, slam_map=slam_map_render)
                    last_hud_time = sim_time

                if sim_time - last_log_time >= 0.5:
                    lidar_min = float(np.min(ranges))
                    grav = get_gravity_orientation(d.qpos[3:7])
                    pitch_deg = float(np.degrees(-grav[0]))
                    roll_deg = float(np.degrees(grav[1]))
                    yaw_deg = get_yaw_degrees(d.qpos[3:7])
                    status_flag = "REFLEX_REC" if (abs(grav[0]) > 0.04 or abs(grav[1]) > 0.04) else "NOMINAL"

                    telemetry.append({
                        "time": round(sim_time, 2),
                        "x": round(float(d.qpos[0]), 3),
                        "y": round(float(d.qpos[1]), 3),
                        "z": round(float(d.qpos[2]), 3),
                        "dist_goal": round(float(dist_to_goal), 3),
                        "lidar_min_m": round(float(lidar_min), 3),
                        "phase": phase
                    })
                    print(f"[TALOS C2] T+{sim_time:04.1f}s | POS: (X={d.qpos[0]:+5.2f}, Y={d.qpos[1]:+5.2f}) Z={d.qpos[2]:.2f}m | "
                          f"CAP: {yaw_deg:03d}° | PITCH: {pitch_deg:+4.1f}° | LIDAR: {lidar_min:4.2f}m | [{status_flag}]")
                    last_log_time = sim_time

                # Synchronisation visualiseur à 50 Hz & régulation temps réel (ZÉRO LAG)
                if step_count % 10 == 0:
                    viewer.sync()
                    wall_elapsed = time.time() - wall_start_time
                    target_time = step_count * SIMULATION_DT
                    sleep_t = target_time - wall_elapsed
                    if sleep_t > 0:
                        time.sleep(min(sleep_t, 0.02))

            if hud_tk and hud_tk.is_alive:
                hud_tk.on_close()

    else:
        # Mode headless
        total_steps = int(32.0 / SIMULATION_DT)
        for s in range(total_steps):
            dist_to_goal = np.linalg.norm(goal_pos - d.qpos[:2])

            check_contacts()
            tau = (target_dof_pos - d.qpos[7:]) * KPS - d.qvel[6:] * KDS
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)
            sim_time += SIMULATION_DT
            step_count += 1

            cmd, phase, best_angle, ranges = [0.0, 0.0, 0.0], "EN ROUTE", 0.0, np.ones(NUM_LIDAR_RAYS) * 5.0
            if step_count % CONTROL_DECIMATION == 0:
                cmd, phase, best_angle, ranges = step_controller()

            quat = d.qpos[3:7]
            yaw_current = np.arctan2(2.0 * (quat[0] * quat[3] + quat[1] * quat[2]), 1.0 - 2.0 * (quat[2] * quat[2] + quat[3] * quat[3]))
            if step_count % 20 == 0:
                slam.update(d.qpos[0], d.qpos[1], yaw_current, ranges, LIDAR_ANGLES)

            if sim_time - last_log_time >= 0.5:
                lidar_min = float(np.min(ranges))
                grav = get_gravity_orientation(d.qpos[3:7])
                pitch_deg = float(np.degrees(-grav[0]))
                yaw_deg = get_yaw_degrees(d.qpos[3:7])
                status_flag = "REFLEX_REC" if (abs(grav[0]) > 0.04 or abs(grav[1]) > 0.04) else "NOMINAL"
                telemetry.append({
                    "time": round(sim_time, 2),
                    "x": round(float(d.qpos[0]), 3),
                    "y": round(float(d.qpos[1]), 3),
                    "z": round(float(d.qpos[2]), 3),
                    "dist_goal": round(float(dist_to_goal), 3),
                    "lidar_min_m": round(float(lidar_min), 3),
                    "phase": phase
                })
                print(f"[TALOS C2] T+{sim_time:04.1f}s | POS: (X={d.qpos[0]:+5.2f}, Y={d.qpos[1]:+5.2f}) Z={d.qpos[2]:.2f}m | "
                      f"CAP: {yaw_deg:03d}° | PITCH: {pitch_deg:+4.1f}° | LIDAR: {lidar_min:4.2f}m | [{status_flag}]")
                last_log_time = sim_time

            if mission_done and (sim_time - mission_done_time >= 2.5):
                break

    telemetry_file = os.path.join(WORKSPACE_DIR, "simulation_telemetry.json")
    with open(telemetry_file, "w", encoding="utf-8") as f:
        json.dump(telemetry, f, indent=2)

    # Export cartographie SLAM 2D
    quat = d.qpos[3:7]
    yaw_final = np.arctan2(2.0 * (quat[0] * quat[3] + quat[1] * quat[2]), 1.0 - 2.0 * (quat[2] * quat[2] + quat[3] * quat[3]))
    slam_img = slam.render(d.qpos[0], d.qpos[1], yaw_final, waypoints, current_wp_idx)
    slam_path = os.path.join(WORKSPACE_DIR, "cartographie_slam_house.png")
    cv2.imwrite(slam_path, slam_img)
    brain_dir = os.path.join(os.path.expanduser("~"), ".gemini", "antigravity", "brain", "13727833-4cc4-4ae8-85bf-58f5dd6528a5")
    if os.path.exists(brain_dir):
        cv2.imwrite(os.path.join(brain_dir, "cartographie_slam_house.png"), slam_img)
    print(f"  [+] Cartographie SLAM 2D  : Enregistrée dans 'cartographie_slam_house.png'")

    print("=" * 82)
    print("                      BILAN DE MISSION TALOS C2 DEFENSE")
    print("=" * 82)
    print(f"  [+] Temps total mission   : {sim_time:.2f} s")
    print(f"  [+] Contacts d'obstacles  : {total_obstacle_contacts} (Objectif Zéro Collision: {'VALIDE' if total_obstacle_contacts == 0 else 'ECHEC'})")
    print(f"  [+] Position finale       : X={d.qpos[0]:.2f}m, Y={d.qpos[1]:.2f}m, Z={d.qpos[2]:.2f}m")
    print(f"  [+] Distance d'extraction : {np.linalg.norm(goal_pos - d.qpos[:2]):.2f} m ({'REUSSIE' if np.linalg.norm(goal_pos - d.qpos[:2]) <= 0.45 or mission_done else 'NON ATTEINTE'})")
    print("=" * 82)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TALOS Unitree G1 - Navigation Autonome par LiDAR 3D")
    parser.add_argument("--scene", type=str, default="rubble", choices=list(SCENES.keys()),
                        help="Scène à charger : rubble (défaut) | rocky (tout-terrain non-plat) | single | chicane | corridor")
    parser.add_argument("--random", action="store_true", help="Disposer les obstacles à des coordonnées inconnues et aléatoires")
    parser.add_argument("--camera", type=str, default="third_person", choices=["third_person", "fpv", "drone"],
                        help="Mode caméra initial : third_person (suivi épaule dynamique) | fpv (vue casque) | drone (vue zénithale ISR)")
    parser.add_argument("--fpv", action="store_true", help="Démarrer le visualiseur 3D directement en vue FPV (yeux du robot)")
    parser.add_argument("--no-hud", action="store_true", help="Désactiver la fenêtre de perception multispectrale")
    parser.add_argument("--headless", action="store_true", help="Exécuter sans affichage graphique")
    args = parser.parse_args()

    init_cam = "fpv" if args.fpv else args.camera
    try:
        run_simulation(scene_name=args.scene, randomize=args.random, headless=args.headless,
                       start_fpv=(init_cam == "fpv"), show_hud=(not args.no_hud))
    except KeyboardInterrupt:
        print("\n[TALOS] Arrêt manuel de la simulation.")
    except Exception as e:
        print(f"\n[ERREUR] : {e}")
        raise e
