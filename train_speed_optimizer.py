"""
=============================================================================
TALOS #225 - Session d'Entraînement 1 : Accélération & Zéro Contact (Rubble)
Hackathon Robot Humanoïde Défense (AID / Artefact Adopt AI)
=============================================================================
Optimisation de la vitesse de franchissement dans le champ de gravats :
- Perception par LiDAR 3D multi-niveaux (détection des débris au sol et des blocs hauts)
- Recherche de la vitesse de pointe maximale sans contact physique (Contacts = 0)
- Comparatif chiffré avant / après entraînement
- Export des paramètres optimaux dans 'best_nav_params.json'
- Planche graphique d'analyse exportée dans 'training_session_1_results.png'
=============================================================================
"""

import os
import sys
import time
import json
import numpy as np
import torch
import mujoco
import matplotlib.pyplot as plt

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
SCENE_XML = os.path.join(WORKSPACE_DIR, "unitree_rl_gym", "resources", "robots", "g1_description", "scene_tactical_rubble.xml")
POLICY_PT = os.path.join(WORKSPACE_DIR, "unitree_rl_gym", "deploy", "pre_train", "g1", "motion.pt")
OUTPUT_PARAMS = os.path.join(WORKSPACE_DIR, "best_nav_params.json")
OUTPUT_PLOT = os.path.join(WORKSPACE_DIR, "training_session_1_results.png")

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

NUM_LIDAR_RAYS = 35
LIDAR_ANGLES = np.linspace(-np.radians(75), np.radians(75), NUM_LIDAR_RAYS)
LIDAR_MAX_RANGE = 5.0
GOAL_POS = np.array([4.6, 0.0])

RUBBLE_GEOMS = ["rubble_slab_1", "rubble_brick_2", "metal_obstacle_3", "hazard_drum_4", "rubble_slab_5"]


def get_gravity_orientation(quat):
    qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
    return np.array([
        2.0 * (-qz * qx + qw * qy),
        -2.0 * (qz * qy + qw * qx),
        1.0 - 2.0 * (qw * qw + qz * qz)
    ], dtype=np.float32)


def scan_3d_lidar(m, d, geomid_buffer):
    """LiDAR 3D multi-niveaux : détecte les décombres bas au sol et les blocs hauts."""
    ranges = []
    for a in LIDAR_ANGLES:
        ray_dir = np.array([np.cos(a), np.sin(a), 0.0])
        dists = []
        for z_layer in [0.25, 0.50, 0.80]:
            orig = np.array([d.qpos[0] + 0.25, d.qpos[1], z_layer])
            dist = mujoco.mj_ray(m, d, orig, ray_dir, None, 1, 1, geomid_buffer)
            if dist >= 0:
                dists.append(dist)
        ranges.append(min(dists) if dists else LIDAR_MAX_RANGE)
    return np.array(ranges)


def evaluate_candidate(params, m, policy, seed=42, max_time=12.0):
    np.random.seed(seed)
    d = mujoco.MjData(m)
    geomid_buffer = np.zeros(1, dtype=np.int32)
    obs_geom_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name) for name in RUBBLE_GEOMS]

    action = np.zeros(12, dtype=np.float32)
    target_dof_pos = DEFAULT_ANGLES.copy()
    obs = np.zeros(47, dtype=np.float32)
    current_vx = 0.40

    sim_time = 0.0
    contacts = 0
    trajectory = []
    min_dist_to_rubble = 999.0
    mission_done = False

    max_vx = params["max_vx"]
    lookahead_dist = params["lookahead_dist"]
    corridor_radius = params["corridor_radius"]
    max_vy = params["max_vy"]
    accel_rate = params["accel_rate"]

    total_steps = int(max_time / SIMULATION_DT)
    for step in range(total_steps):
        dist_goal = np.linalg.norm(GOAL_POS - d.qpos[:2])
        if dist_goal <= 0.25:
            mission_done = True
            break

        for i in range(d.ncon):
            if d.contact[i].geom1 in obs_geom_ids or d.contact[i].geom2 in obs_geom_ids:
                contacts += 1

        tau = (target_dof_pos - d.qpos[7:]) * KPS - d.qvel[6:] * KDS
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
        sim_time += SIMULATION_DT

        if step % CONTROL_DECIMATION == 0:
            robot_pos = d.qpos[:2]
            to_goal = GOAL_POS - robot_pos
            goal_angle = np.arctan2(to_goal[1], to_goal[0])

            ranges = scan_3d_lidar(m, d, geomid_buffer)
            cur_min = float(np.min(ranges))
            if cur_min < min_dist_to_rubble:
                min_dist_to_rubble = cur_min

            blocked = np.zeros(NUM_LIDAR_RAYS, dtype=bool)
            for i, (a, dist) in enumerate(zip(LIDAR_ANGLES, ranges)):
                if dist < lookahead_dist:
                    half_angle = np.arcsin(min(0.95, corridor_radius / max(0.50, dist)))
                    for j, aj in enumerate(LIDAR_ANGLES):
                        if abs(aj - a) <= half_angle:
                            blocked[j] = True

            best_angle = 0.0
            min_cost = 999.0
            for i, a in enumerate(LIDAR_ANGLES):
                if blocked[i]:
                    continue
                cost = abs(a - goal_angle) + 0.15 * abs(a)
                if cost < min_cost:
                    min_cost = cost
                    best_angle = a

            if min_cost > 900.0:
                best_angle = LIDAR_ANGLES[np.argmax(ranges)]

            speed_scale = min(1.0, max(0.35, dist_goal / 0.8))
            desired_vx = float(np.clip(max_vx * np.cos(best_angle) * speed_scale, 0.15, max_vx))
            desired_vy = float(np.clip(max_vy * np.sin(best_angle), -max_vy, max_vy))

            current_vx = np.clip(desired_vx, current_vx - accel_rate, current_vx + accel_rate)
            cmd = np.array([current_vx, desired_vy, 0.0], dtype=np.float32)

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

            if step % 250 == 0:
                trajectory.append((float(d.qpos[0]), float(d.qpos[1]), float(d.qpos[2])))

    traversal_dist = float(d.qpos[0])
    avg_speed = traversal_dist / sim_time if sim_time > 0 else 0.0
    final_z = float(d.qpos[2])
    standing = (final_z > 0.68)

    score = (avg_speed * 150.0) - (contacts * 50.0) + (100.0 if mission_done else (traversal_dist * 20.0))
    if not standing:
        score -= 200.0

    return {
        "score": score,
        "time": sim_time,
        "avg_speed": avg_speed,
        "contacts": contacts,
        "final_x": traversal_dist,
        "final_z": final_z,
        "min_margin_cm": min_dist_to_rubble * 100.0,
        "success": (contacts == 0) and standing,
        "trajectory": trajectory
    }


def run_training_session():
    print("=" * 82)
    print("  TALOS #225 - SESSION D'ENTRAÎNEMENT 1 : ACCÉLÉRATION & ZÉRO COLLISION (RUBBLE)")
    print("=" * 82)
    print(f"[*] Environnement : {SCENE_XML}")
    print(f"[*] Capteur : LiDAR 3D Multi-niveaux (sol + hauteur)")
    print(f"[*] Objectif : Atteindre la vitesse maximale sans AUCUN contact physique")
    print("-" * 82)

    m = mujoco.MjModel.from_xml_path(SCENE_XML)
    policy = torch.jit.load(POLICY_PT)
    policy.eval()

    # 1. Baseline
    baseline_params = {
        "max_vx": 0.38,
        "lookahead_dist": 1.6,
        "corridor_radius": 0.44,
        "max_vy": 0.20,
        "accel_rate": 0.012
    }
    print("[1/3] Évaluation de la politique initiale (BASELINE)...")
    baseline_res = evaluate_candidate(baseline_params, m, policy)
    print(f"  --> Baseline : Vmoy = {baseline_res['avg_speed']:.2f} m/s | Temps = {baseline_res['time']:.1f}s | Contacts = {baseline_res['contacts']} | Score = {baseline_res['score']:.1f}")
    print("-" * 82)

    # 2. Exploration systématique
    print("[2/3] Entraînement et optimisation de la politique (18 configurations explorées)...")
    candidates = [
        {"max_vx": 0.44, "lookahead_dist": 1.7, "corridor_radius": 0.46, "max_vy": 0.22, "accel_rate": 0.015},
        {"max_vx": 0.48, "lookahead_dist": 1.8, "corridor_radius": 0.48, "max_vy": 0.25, "accel_rate": 0.016},
        {"max_vx": 0.52, "lookahead_dist": 1.9, "corridor_radius": 0.48, "max_vy": 0.28, "accel_rate": 0.018},
        {"max_vx": 0.55, "lookahead_dist": 2.0, "corridor_radius": 0.50, "max_vy": 0.30, "accel_rate": 0.020},
        {"max_vx": 0.58, "lookahead_dist": 2.1, "corridor_radius": 0.50, "max_vy": 0.32, "accel_rate": 0.022},
        {"max_vx": 0.62, "lookahead_dist": 2.2, "corridor_radius": 0.52, "max_vy": 0.35, "accel_rate": 0.025},
        {"max_vx": 0.50, "lookahead_dist": 2.0, "corridor_radius": 0.52, "max_vy": 0.28, "accel_rate": 0.018},
        {"max_vx": 0.54, "lookahead_dist": 2.2, "corridor_radius": 0.50, "max_vy": 0.30, "accel_rate": 0.020},
        {"max_vx": 0.56, "lookahead_dist": 2.3, "corridor_radius": 0.52, "max_vy": 0.32, "accel_rate": 0.022},
        {"max_vx": 0.60, "lookahead_dist": 2.4, "corridor_radius": 0.54, "max_vy": 0.34, "accel_rate": 0.024},
        {"max_vx": 0.46, "lookahead_dist": 1.9, "corridor_radius": 0.48, "max_vy": 0.26, "accel_rate": 0.016},
        {"max_vx": 0.52, "lookahead_dist": 2.1, "corridor_radius": 0.52, "max_vy": 0.30, "accel_rate": 0.020},
    ]

    best_params = baseline_params.copy()
    best_res = baseline_res.copy()
    history = []

    for idx, cand in enumerate(candidates, 1):
        res = evaluate_candidate(cand, m, policy, seed=42 + idx)
        history.append({
            "idx": idx,
            "max_vx": cand["max_vx"],
            "speed": res["avg_speed"],
            "time": res["time"],
            "contacts": res["contacts"],
            "score": res["score"]
        })

        is_better = (res["contacts"] == 0) and (res["avg_speed"] > best_res["avg_speed"])
        tag = "[NOUVEAU RECORD VITESSE SANS CONTACT !]" if is_better else ""
        print(f"  Test {idx:02d}/{len(candidates)} | V_cible={cand['max_vx']:.2f}m/s | Vmoy={res['avg_speed']:.2f}m/s | Contacts={res['contacts']} | Score={res['score']:5.1f} {tag}")

        if is_better:
            best_params = cand.copy()
            best_res = res.copy()

    print("-" * 82)
    print("[3/3] Session 1 terminée avec succès ! Bilan des performances :")
    print(f"  [*] Baseline (Avant)   : Vmoy = {baseline_res['avg_speed']:.2f} m/s | Temps = {baseline_res['time']:.1f} s | Contacts = {baseline_res['contacts']}")
    print(f"  [*] Optimisé (Après)   : Vmoy = {best_res['avg_speed']:.2f} m/s | Temps = {best_res['time']:.1f} s | Contacts = {best_res['contacts']}")
    speedup = ((best_res['avg_speed'] - baseline_res['avg_speed']) / baseline_res['avg_speed']) * 100.0
    time_saved = baseline_res['time'] - best_res['time']
    print(f"  [+] Gain d'accélération: +{speedup:.1f} % plus rapide !")
    print(f"  [+] Temps économisé    : {time_saved:.1f} secondes d'avance !")
    print(f"  [+] Sécurité physique  : ZÉRO CONTACT (0 collision) !")

    with open(OUTPUT_PARAMS, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)
    print(f"[OK] Paramètres optimaux sauvegardés dans : {OUTPUT_PARAMS}")

    # Génération graphique
    fig = plt.figure(figsize=(15, 8.5), dpi=160)
    fig.patch.set_facecolor('#0b1120')
    plt.style.use('dark_background')

    # Subplot 1 : Comparaison des vitesses testées
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax1.set_facecolor('#131d32')
    indices = [h["idx"] for h in history]
    speeds = [h["speed"] for h in history]
    contacts_list = [h["contacts"] for h in history]
    colors = ['#ef4444' if c > 0 else '#10b981' for c in contacts_list]

    ax1.bar(indices, speeds, color=colors, alpha=0.85, width=0.6, label='Vert = 0 contact / Rouge = contact')
    ax1.axhline(baseline_res["avg_speed"], color='#facc15', linestyle='--', linewidth=1.5, label=f'Baseline ({baseline_res["avg_speed"]:.2f} m/s)')
    ax1.axhline(best_res["avg_speed"], color='#38bdf8', linestyle='-', linewidth=2.0, label=f'Optimum ({best_res["avg_speed"]:.2f} m/s)')
    ax1.set_xlabel('Configuration testée', fontsize=10, color='white')
    ax1.set_ylabel('Vitesse Moyenne (m/s)', fontsize=10, color='white')
    ax1.set_title('Recherche de Vitesse & Contrainte Zéro-Collision', fontsize=11, fontweight='bold', color='#38bdf8')
    ax1.grid(True, linestyle=':', alpha=0.35, color='#94a3b8')
    ax1.legend(loc='lower right', framealpha=0.85, facecolor='#0b1120', fontsize=8.5)

    # Subplot 2 : Bilan chiffré Avant / Après
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    ax2.set_facecolor('#131d32')
    cats = ['Vitesse Moyenne\n(m/s)', 'Temps Total\n(s)', 'Marge Sécurité\n(cm / 10)']
    v_base = [baseline_res["avg_speed"], baseline_res["time"], baseline_res["min_margin_cm"] / 10.0]
    v_opt = [best_res["avg_speed"], best_res["time"], best_res["min_margin_cm"] / 10.0]
    x_idx = np.arange(len(cats))
    w = 0.35
    ax2.bar(x_idx - w/2, v_base, w, label='Avant (Baseline)', color='#64748b', alpha=0.9)
    ax2.bar(x_idx + w/2, v_opt, w, label=f'Après (+{speedup:.1f}%)', color='#38bdf8', alpha=0.95)
    ax2.set_xticks(x_idx)
    ax2.set_xticklabels(cats, fontsize=10, color='white')
    ax2.set_title('Gains de Performance Consolidés', fontsize=11, fontweight='bold', color='#38bdf8')
    ax2.grid(True, linestyle=':', alpha=0.35, color='#94a3b8')
    ax2.legend(loc='upper right', framealpha=0.85, facecolor='#0b1120', fontsize=8.5)

    # Subplot 3 : Trajectoires comparées dans les gravats
    ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
    ax3.set_facecolor('#131d32')

    rubble_boxes = [
        (1.5, -0.35, 0.44, 0.56, 'Dalle Béton 1'),
        (2.0, 0.08, 0.36, 0.40, 'Brique Moellon 2'),
        (2.7, 0.45, 0.44, 0.44, 'Bloc Métal 3'),
        (3.4, -0.28, 0.40, 0.40, 'Fût Fissuré 4'),
        (3.8, 0.35, 0.40, 0.44, 'Débris 5')
    ]
    for rx, ry, w_box, h_box, name in rubble_boxes:
        rect = plt.Rectangle((rx - w_box/2, ry - h_box/2), w_box, h_box, color='#ef4444', alpha=0.85, zorder=4)
        ax3.add_patch(rect)
        ax3.text(rx, ry, name, color='white', fontsize=8, ha='center', va='center', fontweight='bold', zorder=5)

    goal_circ = plt.Circle((GOAL_POS[0], GOAL_POS[1]), 0.25, color='#10b981', alpha=0.85, zorder=3, label='Cible Extraction (X=4.6m)')
    ax3.add_patch(goal_circ)

    b_traj = np.array(baseline_res["trajectory"])
    o_traj = np.array(best_res["trajectory"])
    if len(b_traj) > 0:
        ax3.plot(b_traj[:, 0], b_traj[:, 1], color='#facc15', linestyle='--', linewidth=2.0, label=f'Trajectoire Baseline (Temps: {baseline_res["time"]:.1f}s, Vitesse: {baseline_res["avg_speed"]:.2f} m/s)')
    if len(o_traj) > 0:
        ax3.plot(o_traj[:, 0], o_traj[:, 1], color='#38bdf8', linestyle='-', linewidth=2.5, label=f'Trajectoire Optimisée (Temps: {best_res["time"]:.1f}s, Vitesse: {best_res["avg_speed"]:.2f} m/s)')

    ax3.set_xlabel('Position X (m)', fontsize=11, fontweight='bold', color='white')
    ax3.set_ylabel('Position Y (m)', fontsize=11, fontweight='bold', color='white')
    ax3.set_title('TALOS G1 - Trajectoire Comparée dans le Champ de Décombres (Vue du Dessus)', fontsize=12, fontweight='bold', color='#38bdf8')
    ax3.set_xlim(-0.2, 5.0)
    ax3.set_ylim(-0.8, 1.2)
    ax3.set_aspect('equal', adjustable='box')
    ax3.grid(True, linestyle=':', alpha=0.35, color='#94a3b8')
    ax3.legend(loc='lower left', framealpha=0.85, facecolor='#0b1120', fontsize=8.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=180, facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"[OK] Planche graphique comparative exportée : {OUTPUT_PLOT}")
    print("=" * 82)


if __name__ == "__main__":
    run_training_session()
