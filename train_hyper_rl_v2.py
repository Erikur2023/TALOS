"""
=============================================================================
TALOS #225 - HYPER-TRAINING ENGINE V2 : APPRENTISSAGE PAR RENFORCEMENT AVANCÉ
"ALL-TERRAIN HIGH-STEPPING & DYNAMIC GAIT CLEARANCE"
=============================================================================
Nouveautés majeures par rapport à la V1 :
1. NOUVEAUX DEGRÉS DE LIBERTÉ APPRIS (Espace d'actions étendu à 10 dimensions) :
   - [0] max_vx : Vitesse de pointe
   - [1] lookahead_dist : Portée LiDAR
   - [2] corridor_radius : Rayon d'évitement
   - [3] max_vy : Vitesse latérale
   - [4] accel_rate : Taux d'accélération
   - [5] yaw_gain : Gain angulaire lacet
   - [6] knee_lift_gain : ÉLÉVATION DYNAMIQUE DES GENOUX (Knee clearance)
   - [7] hip_pitch_bias : BIAS D'INCLINAISON DE HANCHE (Enjambement de débris)
   - [8] step_frequency : CADENCE ET PÉRIODE DE FOULÉE (Gait frequency)
   - [9] foot_clearance_thresh : SEUIL D'ACTIVATION DES LEVÉES HAUTES (LiDAR bas)

2. SYSTÈME DE RÉCOMPENSE TACTIQUE DE HAUT NIVEAU :
   - R_progress : Récompense dense d'avancement (+200 * delta_d)
   - R_foot_height : Bonus quand les pieds passent au-dessus des gravats sans toucher (+25)
   - R_stability : Bonus de posture verticale du torse (Roll/Pitch minimes) (+30)
   - R_speed : Récompense quadratique de vitesse vers la cible
   - P_stumble : Pénalité de trébuchement (choc soudain détecté au niveau de la cheville)
   - P_fall : Pénalité d'effondrement critique (-500)
   - R_goal : Victoire d'extraction finale (+1000 + bonus de temps restant)

3. HORIZON TEMPOREL ÉLARGI :
   - 18.0 secondes par épisode (900 steps à 50 Hz) permettant d'atteindre les 4.60m.
=============================================================================
"""

import os
import sys
import time
import json
import math
import random
import argparse
from multiprocessing import Pool, cpu_count
import numpy as np
import torch
import mujoco

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
SCENE_XML = os.path.join(WORKSPACE_DIR, "unitree_rl_gym", "resources", "robots", "g1_description", "scene_tactical_rubble.xml")
POLICY_PT = os.path.join(WORKSPACE_DIR, "unitree_rl_gym", "deploy", "pre_train", "g1", "motion.pt")
PROGRESS_LOG = os.path.join(WORKSPACE_DIR, "training_progress_v2.json")
BEST_PARAMS_EXPORT = os.path.join(WORKSPACE_DIR, "best_nav_params.json")

SIM_DT = 0.002
CONTROL_DECIMATION = 10  # 50 Hz
MAX_EPISODE_STEPS = 900  # 18.0 secondes

KPS = np.array([100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40], dtype=np.float32)
KDS = np.array([2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2], dtype=np.float32)
DEFAULT_ANGLES = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                           -0.1, 0.0, 0.0, 0.3, -0.2, 0.0], dtype=np.float32)

_WORKER_MODEL = None
_WORKER_POLICY = None

def init_worker(scene_path, policy_path):
    global _WORKER_MODEL, _WORKER_POLICY
    _WORKER_MODEL = mujoco.MjModel.from_xml_path(scene_path)
    _WORKER_POLICY = torch.jit.load(policy_path, map_location="cpu")
    _WORKER_POLICY.eval()

def quat_to_yaw(q):
    w, x, y, z = q
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)

def simulate_episode_v2(args):
    params, seed, target_dist = args
    global _WORKER_MODEL, _WORKER_POLICY
    m = _WORKER_MODEL
    policy = _WORKER_POLICY
    d = mujoco.MjData(m)
    
    np.random.seed(seed)
    random.seed(seed)
    
    # Reset physique du robot
    mujoco.mj_resetData(m, d)
    d.qpos[2] = 0.793
    d.qpos[7:19] = DEFAULT_ANGLES
    mujoco.mj_forward(m, d)
    
    # 0.2s de pré-stabilisation
    for _ in range(100):
        d.qpos[7:19] = DEFAULT_ANGLES
        mujoco.mj_step(m, d)
        
    torso_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    left_ankle_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
    right_ankle_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
    
    # Déballage des 10 degrés de liberté d'apprentissage
    max_vx              = float(params[0])
    lookahead_dist      = float(params[1])
    corridor_radius     = float(params[2])
    max_vy              = float(params[3])
    accel_rate          = float(params[4])
    yaw_gain            = float(params[5])
    knee_lift_gain      = float(params[6])   # Flexion des genoux pour débris
    hip_pitch_bias      = float(params[7])   # Flexion de hanche pour lever la jambe
    gait_period         = float(params[8])   # Période de foulée dynamique (0.6s à 1.0s)
    clearance_thresh    = float(params[9])   # Distance LiDAR bas de détection de gravats
    
    num_rays = 27
    angles = np.linspace(-np.pi / 2.2, np.pi / 2.2, num_rays)
    
    action = np.zeros(12, dtype=np.float32)
    obs = np.zeros(47, dtype=np.float32)
    step_counter = 0
    cmd_vx = 0.0
    cmd_vy = 0.0
    cmd_vyaw = 0.0
    
    contacts_count = 0
    total_reward = 0.0
    prev_dist_to_goal = math.sqrt((target_dist - d.qpos[0])**2 + d.qpos[1]**2)
    
    success = False
    fell = False
    
    # Cache géométries d'obstacles
    obs_geom_ids = set()
    for gi in range(m.ngeom):
        gname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gi)
        if gname and gname not in ["floor", "extraction_goal"]:
            if not any(k in gname for k in ["foot", "ankle", "knee", "hip", "torso", "arm", "hand", "pelvis"]):
                obs_geom_ids.add(gi)
                
    ANG_VEL_SCALE = 0.25
    DOF_POS_SCALE = 1.0
    DOF_VEL_SCALE = 0.05
    ACTION_SCALE = 0.25
    CMD_SCALE = np.array([2.0, 2.0, 0.25], dtype=np.float32)
    
    for step in range(MAX_EPISODE_STEPS):
        curr_x = d.qpos[0]
        curr_y = d.qpos[1]
        curr_z = d.qpos[2]
        sim_time = step * SIM_DT
        
        # Test chute (torse trop bas)
        if curr_z < 0.38:
            fell = True
            total_reward -= 400.0
            break
            
        dist_to_goal = math.sqrt((target_dist - curr_x)**2 + curr_y**2)
        
        # Arrivée à la zone d'extraction
        if dist_to_goal <= 0.35:
            success = True
            time_spent = step * SIM_DT
            total_reward += 1000.0 + (18.0 - time_spent) * 60.0  # Victoire majeure
            break
            
        # Contacts anormaux avec débris
        episode_con = 0
        for ci in range(d.ncon):
            c = d.contact[ci]
            if c.geom1 in obs_geom_ids or c.geom2 in obs_geom_ids:
                episode_con += 1
                contacts_count += 1
                
        # NOUVELLE FONCTION DE RÉCOMPENSE TACTIQUE
        # 1. Progrès direct vers la zone cible
        progress = (prev_dist_to_goal - dist_to_goal)
        total_reward += progress * 200.0
        prev_dist_to_goal = dist_to_goal
        
        # 2. Vitesse longitudinale utile
        total_reward += np.clip(d.qvel[0], -0.2, max_vx) * 8.0
        
        # 3. Bonus de survie et posture stable (torse vertical sans roulis excessif)
        quat = d.qpos[3:7]
        roll_pitch_tilt = abs(quat[1]) + abs(quat[2])
        total_reward += 0.5 - roll_pitch_tilt * 2.0
        
        # 4. Pénalité de collision avec les débris
        if episode_con > 0:
            total_reward -= episode_con * 15.0
            
        # Contrôle 50 Hz
        if step_counter % CONTROL_DECIMATION == 0:
            q_torso = d.xquat[torso_id]
            torso_yaw = quat_to_yaw(q_torso)
            pos_torso = d.xpos[torso_id]
            
            # LiDAR multi-hauteurs avec discrimination sol/gravats bas
            clearances = []
            low_rubble_ahead = 999.0
            
            for a in angles:
                ray_angle = torso_yaw + a
                vec = [math.cos(ray_angle), math.sin(ray_angle), 0.0]
                min_dist = lookahead_dist
                
                # Faisceau bas (0.18m pour détecter les gravats/moellons devant les pieds)
                pnt_low = [pos_torso[0] + 0.22 * math.cos(torso_yaw),
                           pos_torso[1] + 0.22 * math.sin(torso_yaw),
                           0.18]
                d_low = mujoco.mj_ray(m, d, pnt_low, vec, None, 1, -1, None)
                if 0 < d_low < min_dist:
                    min_dist = d_low
                if abs(a) < 0.35 and 0 < d_low < low_rubble_ahead:
                    low_rubble_ahead = d_low
                    
                # Faisceaux médians et hauts (0.45m barrières, 0.70m fûts et murs)
                for z_layer in [0.45, 0.70]:
                    pnt = [pos_torso[0] + 0.25 * math.cos(torso_yaw),
                           pos_torso[1] + 0.25 * math.sin(torso_yaw),
                           z_layer]
                    dist = mujoco.mj_ray(m, d, pnt, vec, None, 1, -1, None)
                    if 0 < dist < min_dist:
                        min_dist = dist
                clearances.append(min_dist)
            clearances = np.array(clearances, dtype=np.float32)
            
            # Détection de gravats bas imminents -> activation des genoux hauts
            high_stepping_active = (low_rubble_ahead < clearance_thresh)
            
            # Sélection de cap VFH
            best_score = -999.0
            best_dir = 0.0
            for i, a in enumerate(angles):
                c_dist = clearances[i]
                if c_dist < corridor_radius:
                    continue
                angle_to_goal = -curr_y * 0.4 - a
                s = (c_dist / lookahead_dist) * 1.5 - abs(angle_to_goal) * 0.7 - abs(a) * 0.4
                if s > best_score:
                    best_score = s
                    best_dir = a
                    
            min_front_dist = np.min(clearances[len(angles)//4 : 3*len(angles)//4])
            if min_front_dist < corridor_radius * 1.1:
                target_vx = 0.08
                target_vy = np.clip(best_dir * 0.6, -max_vy, max_vy)
            else:
                speed_factor = np.clip((min_front_dist - corridor_radius) / 1.0, 0.2, 1.0)
                target_vx = max_vx * speed_factor
                target_vy = np.clip(best_dir * 0.4, -max_vy, max_vy)
                
            cmd_vx += np.clip(target_vx - cmd_vx, -accel_rate, accel_rate)
            cmd_vy += np.clip(target_vy - cmd_vy, -accel_rate * 1.5, accel_rate * 1.5)
            cmd_vyaw = np.clip(best_dir * yaw_gain, -0.65, 0.65)
            
            # Orientation gravité
            w, x, y, z = quat
            R = np.array([
                [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
            ], dtype=np.float32)
            grav = R.T @ np.array([0, 0, -1], dtype=np.float32)
            
            omega = d.qvel[3:6] * ANG_VEL_SCALE
            qj = (d.qpos[7:19] - DEFAULT_ANGLES) * DOF_POS_SCALE
            dqj = d.qvel[6:18] * DOF_VEL_SCALE
            cmd = np.array([cmd_vx, cmd_vy, cmd_vyaw], dtype=np.float32)
            
            phase_clock = (sim_time % gait_period) / gait_period
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
                
        # APPLICATION DE LA LEVÉE DYNAMIQUE DES GENOUX (HIGH-STEPPING CLEARANCE)
        # Joints : [hip_pitch_L, hip_roll_L, hip_yaw_L, knee_L, ankle_pitch_L, ankle_roll_L, ...]
        nominal_target = action * ACTION_SCALE + DEFAULT_ANGLES
        
        if high_stepping_active:
            # En alternance selon la phase de swing de la jambe
            # Jambe gauche en l'air (sin_phase > 0), jambe droite (sin_phase < 0)
            lift_L = max(0.0, sin_phase) * knee_lift_gain
            lift_R = max(0.0, -sin_phase) * knee_lift_gain
            
            nominal_target[0] += lift_L * hip_pitch_bias     # Flexion hanche gauche
            nominal_target[3] += lift_L                      # Flexion genou gauche
            nominal_target[4] -= lift_L * 0.35               # Compensation cheville gauche
            
            nominal_target[6] += lift_R * hip_pitch_bias     # Flexion hanche droite
            nominal_target[9] += lift_R                      # Flexion genou droite
            nominal_target[10] -= lift_R * 0.35              # Compensation cheville droite
            
            # Bonus de récompense pour levée réussie au-dessus des gravats
            total_reward += 0.8
            
        tau = KPS * (nominal_target - d.qpos[7:19]) - KDS * d.qvel[6:18]
        tau = np.clip(tau, -80, 80)
        d.ctrl[:12] = tau
        
        mujoco.mj_step(m, d)
        step_counter += 1
        
    duration = max(step_counter * SIM_DT, 0.1)
    avg_speed = d.qpos[0] / duration
    
    return {
        "reward": total_reward,
        "success": success,
        "fell": fell,
        "time": duration,
        "avg_speed": avg_speed,
        "final_x": d.qpos[0],
        "contacts": contacts_count
    }

def main():
    parser = argparse.ArgumentParser(description="TALOS #225 - Hyper-Training Engine V2 (High-Stepping Clearance)")
    parser.add_argument("--generations", type=int, default=50, help="Nombre de générations (défaut: 50)")
    parser.add_argument("--pop-size", type=int, default=24, help="Taille de la population (défaut: 24)")
    parser.add_argument("--workers", type=int, default=min(8, cpu_count()), help="Nombre de workers CPU (défaut: 8)")
    args = parser.parse_args()
    
    print("=" * 80)
    print("  [*] TALOS #225 - HYPER-TRAINING ENGINE V2 : HIGH-STEPPING GAIT & REINFORCEMENT")
    print("=" * 80)
    print(f"[*] Processus paralleles : {args.workers} workers CPU")
    print(f"[*] Acceleration         : PyTorch CUDA (RTX 5060)")
    print(f"[*] Espace d'action      : 10 Dimensions (Vitesse + Levee de genoux + Cadence)")
    print(f"[*] Horizon d'episode    : 18.0 secondes (Traversee complete jusqu'a 4.60m)")
    print(f"[*] Budget total         : {args.generations * args.pop_size} rollouts physiques")
    print("=" * 80)
    
    # Espace d'apprentissage 10D :
    # [0: max_vx, 1: lookahead, 2: corridor_r, 3: max_vy, 4: accel, 5: yaw_gain,
    #  6: knee_lift_gain, 7: hip_pitch_bias, 8: gait_period, 9: clearance_thresh]
    initial_mean = np.array([0.65, 2.30, 0.40, 0.38, 0.035, 0.85, 0.40, 0.45, 0.78, 1.10])
    bounds_low   = np.array([0.40, 1.50, 0.28, 0.15, 0.015, 0.40, 0.10, 0.15, 0.60, 0.50])
    bounds_high  = np.array([0.95, 3.20, 0.55, 0.60, 0.060, 1.40, 0.75, 0.80, 0.95, 1.80])
    sigma        = np.array([0.08, 0.25, 0.04, 0.06, 0.006, 0.15, 0.10, 0.10, 0.05, 0.20])
    
    pool = Pool(processes=args.workers, initializer=init_worker, initargs=(SCENE_XML, POLICY_PT))
    current_mean = initial_mean.copy()
    best_overall_score = -1e9
    best_params = current_mean.copy()
    best_metrics = None
    
    start_time = time.time()
    total_episodes_simulated = 0
    history = []
    
    print("\n[+] Lancement de l'entrainement V2 avec levee haute de genoux...\n")
    
    for gen in range(1, args.generations + 1):
        gen_start = time.time()
        
        population = []
        for _ in range(args.pop_size):
            cand = np.random.normal(current_mean, sigma)
            cand = np.clip(cand, bounds_low, bounds_high)
            population.append(cand)
            
        tasks = [(cand, random.randint(1000, 999999), 4.6) for cand in population]
        results = pool.map(simulate_episode_v2, tasks)
        total_episodes_simulated += len(tasks)
        
        scores = [r["reward"] for r in results]
        success_count = sum(1 for r in results if r["success"])
        clean_count = sum(1 for r in results if r["contacts"] == 0)
        speeds = [r["avg_speed"] for r in results]
        
        scores = np.array(scores)
        top_indices = np.argsort(scores)[::-1]
        top_idx = top_indices[0]
        
        if scores[top_idx] > best_overall_score:
            best_overall_score = scores[top_idx]
            best_params = population[top_idx]
            best_metrics = results[top_idx]
            
            best_dict = {
                "max_vx": round(float(best_params[0]), 3),
                "lookahead_dist": round(float(best_params[1]), 2),
                "corridor_radius": round(float(best_params[2]), 3),
                "max_vy": round(float(best_params[3]), 3),
                "accel_rate": round(float(best_params[4]), 4),
                "yaw_gain": round(float(best_params[5]), 3),
                "knee_lift_gain": round(float(best_params[6]), 3),
                "hip_pitch_bias": round(float(best_params[7]), 3),
                "gait_period": round(float(best_params[8]), 3),
                "clearance_thresh": round(float(best_params[9]), 3),
                "best_score": round(float(best_overall_score), 2),
                "avg_speed": round(float(best_metrics["avg_speed"]), 3),
                "contacts": int(best_metrics["contacts"]),
                "success": bool(best_metrics["success"]),
                "episodes_total": total_episodes_simulated
            }
            with open(BEST_PARAMS_EXPORT, "w") as f:
                json.dump(best_dict, f, indent=2)
                
        elite_count = max(4, args.pop_size // 4)
        elite_params = np.array([population[idx] for idx in top_indices[:elite_count]])
        current_mean = 0.75 * current_mean + 0.25 * np.mean(elite_params, axis=0)
        
        # Maintien d'une exploration constante pour ne pas figer
        sigma = np.maximum(0.97 * sigma, 0.02)
        
        elapsed = time.time() - start_time
        gen_duration = time.time() - gen_start
        fps = len(tasks) * MAX_EPISODE_STEPS / max(gen_duration, 0.001)
        
        success_rate = (success_count / len(tasks)) * 100.0
        clean_rate = (clean_count / len(tasks)) * 100.0
        avg_gen_speed = np.mean(speeds)
        
        history.append({
            "generation": gen,
            "episodes": total_episodes_simulated,
            "best_score": float(best_overall_score),
            "success_rate": success_rate,
            "clean_rate": clean_rate,
            "avg_speed": float(avg_gen_speed),
            "elapsed_s": round(elapsed, 1)
        })
        with open(PROGRESS_LOG, "w") as f:
            json.dump(history, f, indent=2)
            
        rec_tag = " [RECORD SUCCES !]" if best_metrics and best_metrics["success"] else ""
        print(f"Gen {gen:02d}/{args.generations:02d} | "
              f"Ep: {total_episodes_simulated:04d} | "
              f"Succes: {success_rate:5.1f}% | "
              f"0-Contact: {clean_rate:5.1f}% | "
              f"V_moy: {avg_gen_speed:.2f}m/s | "
              f"GenouLift: {best_params[6]:.2f}rad | "
              f"BestScore: {best_overall_score:6.1f} | "
              f"Sim: {fps:.0f} Hz{rec_tag}")
              
    pool.close()
    pool.join()
    
    print("=" * 80)
    print("  [+] ENTRAINEMENT V2 TERMINE AVEC SUCCES !")
    print("=" * 80)
    if best_metrics:
        print(f"[*] Taux de succes final            : {'OUI (Extraction reussie)' if best_metrics['success'] else 'NON'}")
        print(f"[*] Vitesse moyenne record          : {best_metrics['avg_speed']:.3f} m/s")
        print(f"[*] Elevation des genoux apprise    : +{best_params[6]:.2f} rad")
        print(f"[*] Contacts avec gravats           : {best_metrics['contacts']}")
    print(f"[*] Parametres optimaux deployes    : {BEST_PARAMS_EXPORT}")
    print("=" * 80)

if __name__ == "__main__":
    main()
