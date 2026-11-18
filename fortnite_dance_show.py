"""
=============================================================================
TALOS #225 - FORTNITE DANCE SHOW 🎮🤖💃
Mode démo auto : le G1 enchaîne toutes les danses Fortnite sur une scène
style boîte de nuit avec lumières qui clignotent au rythme.

Usage :
    python fortnite_dance_show.py              # show auto en boucle
    python fortnite_dance_show.py --headless   # test sans affichage
    python fortnite_dance_show.py --export     # exporte les trajectoires JSON
=============================================================================
"""

import os, sys, math, time, json, argparse
import numpy as np
import mujoco
import mujoco.viewer

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WORKSPACE)
from fortnite_dance import (G1_XML, NU, KP, KD, TAU_MAX, build_scene, base_posture,
                            DANCES, DANCE_ORDER, sample_dance, ElasticHarness)

# ─── Séquence du show : chaque danse dure N boucles ───
SHOW_SEQUENCE = [
    ("idle", 1),          # intro
    ("default", 3),       # Default Dance x3
    ("floss", 4),         # Floss x4 (rapide)
    ("orangejustice", 4), # Orange Justice x4
    ("bestmates", 3),     # Best Mates x3
    ("takethel", 2),      # Take the L x2 (final)
    ("idle", 1),          # outro
]

# Couleurs néon pour les lumières (RGB)
NEON_COLORS = [
    [0.2, 0.6, 1.0],   # bleu
    [1.0, 0.3, 0.8],   # magenta
    [0.3, 1.0, 0.5],   # vert
    [1.0, 0.6, 0.2],   # orange
    [0.8, 0.3, 1.0],   # violet
]


def make_nightclub_scene():
    """Scène boîte de nuit : sol réfléchissant sombre + lumières colorées."""
    model = mujoco.MjModel.from_xml_path(G1_XML)
    data = mujoco.MjData(model)

    # Sol grand, sombre, réfléchissant
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id >= 0:
        model.geom_type[floor_id] = mujoco.mjtGeom.mjGEOM_PLANE
        model.geom_size[floor_id] = [20, 20, 0.1]
        model.geom_friction[floor_id] = [1.0, 0.5, 0.01]
        model.geom_rgba[floor_id] = [0.05, 0.03, 0.12, 1.0]
        model.geom_matid[floor_id] = -1  # pas de material, couleur unie

    model.opt.timestep = 0.002
    model.vis.global_.offwidth = 1920
    model.vis.global_.offheight = 1080

    # Activer les ombres
    model.vis.quality.shadowsize = 2048

    return model, data


def beat_color(t, bpm=120):
    """Couleur qui change au rythme du beat."""
    beat = (t * bpm / 60.0) % len(NEON_COLORS)
    idx = int(beat)
    frac = beat - idx
    c0 = np.array(NEON_COLORS[idx])
    c1 = np.array(NEON_COLORS[(idx + 1) % len(NEON_COLORS)])
    return c0 + frac * (c1 - c0)


def export_trajectories():
    """Exporte toutes les trajectoires de danse en JSON pour déploiement réel."""
    out = {}
    for name, (label, fn) in DANCES.items():
        kf, loop_T = fn()
        # Échantillonner à 50 Hz
        dt_sample = 0.02
        n = int(loop_T / dt_sample)
        frames = []
        for i in range(n):
            t = i * dt_sample
            q = sample_dance(kf, loop_T, t)
            frames.append({"t": round(t, 4), "q": [round(float(x), 5) for x in q]})
        out[name] = {
            "label": label,
            "loop_duration_s": loop_T,
            "sample_rate_hz": 50,
            "num_joints": NU,
            "joint_names": [
                "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
                "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
                "waist_yaw", "waist_roll", "waist_pitch",
                "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
                "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
            ],
            "frames": frames,
        }
        print(f"  📦 {label}: {n} frames @ 50Hz, boucle {loop_T}s")

    path = os.path.join(WORKSPACE, "fortnite_dances_trajectories.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n✅ Trajectoires exportées → {path}")
    print(f"   Prêt pour déploiement sur G1 réel (harnais élastique Unitree requis)")


def run_show(headless=False, loop_forever=True):
    model, data = make_nightclub_scene()
    harness = ElasticHarness()

    dance_data = {name: fn() for name, (_, fn) in DANCES.items()}

    # Init
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0., 0., 0.793]
    data.qpos[3:7] = [1., 0., 0., 0.]
    qi = base_posture()
    for i in range(NU):
        data.qpos[7 + i] = qi[i]
    mujoco.mj_forward(model, data)

    dt = model.opt.timestep
    sim_t = 0.0
    seq_idx = 0
    seq_t = 0.0
    current_dance, current_loops = SHOW_SEQUENCE[0]

    print("=" * 64)
    print("  🎮 TALOS #225 — FORTNITE DANCE SHOW")
    print("  Enchaînement auto :", " → ".join(f"{DANCES[d][0]}" for d, _ in SHOW_SEQUENCE))
    print("=" * 64)

    def advance_sequence():
        nonlocal seq_idx, seq_t, current_dance, current_loops
        seq_idx += 1
        if seq_idx >= len(SHOW_SEQUENCE):
            if loop_forever:
                seq_idx = 0
            else:
                return False
        current_dance, current_loops = SHOW_SEQUENCE[seq_idx]
        seq_t = 0.0
        kf, lT = dance_data[current_dance]
        print(f"  ▶ {DANCES[current_dance][0]}  ({current_loops} boucle{'s' if current_loops>1 else ''})")
        return True

    print(f"  ▶ {DANCES[current_dance][0]}  ({current_loops} boucle{'s' if current_loops>1 else ''})")

    if headless:
        n = int(60.0 / dt)  # 60s de test
        pid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        chutes = 0
        for step in range(n):
            sim_t += dt
            seq_t += dt
            kf, lT = dance_data[current_dance]
            if seq_t >= lT * current_loops:
                if not advance_sequence():
                    break
            q_des = sample_dance(kf, lT, seq_t)
            q = data.qpos[7:7+NU]; dq = data.qvel[6:6+NU]
            tau = np.clip(KP*(q_des-q)+KD*(0.-dq), -TAU_MAX, TAU_MAX)
            data.ctrl[:] = tau
            harness.apply(model, data)
            mujoco.mj_step(model, data)
            harness.anchor(data)
            mujoco.mj_forward(model, data)
            if data.xpos[pid][2] < 0.3:
                chutes += 1
        print(f"\n✅ Show headless 60s terminé — chutes: {chutes}")
        return

    # Mode viewer avec lumières néon qui changent
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Config caméra
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.distance = 3.5
        viewer.cam.elevation = -15
        viewer.cam.azimuth = 110
        viewer.cam.lookat[:] = [0.2, 0, 0.7]

        while viewer.is_running():
            t0 = time.time()
            sim_t += dt
            seq_t += dt

            kf, lT = dance_data[current_dance]
            if seq_t >= lT * current_loops:
                if not advance_sequence():
                    break
                kf, lT = dance_data[current_dance]

            q_des = sample_dance(kf, lT, seq_t)
            q = data.qpos[7:7+NU]; dq = data.qvel[6:6+NU]
            tau = np.clip(KP*(q_des-q)+KD*(0.-dq), -TAU_MAX, TAU_MAX)
            data.ctrl[:] = tau
            harness.apply(model, data)
            mujoco.mj_step(model, data)
            harness.anchor(data)
            mujoco.mj_forward(model, data)

            # Changer la couleur de la headlight au rythme du beat
            c = beat_color(sim_t, bpm=120)
            model.vis.headlight.diffuse[:] = c
            model.vis.headlight.ambient[:] = c * 0.5

            viewer.sync()
            el = time.time() - t0
            if el < dt:
                time.sleep(dt - el)

    print("\n🏁 Show terminé !")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--headless", action="store_true")
    p.add_argument("--export", action="store_true", help="Exporte les trajectoires en JSON")
    p.add_argument("--once", action="store_true", help="Joue la séquence une seule fois")
    args = p.parse_args()

    if args.export:
        export_trajectories()
    else:
        run_show(headless=args.headless, loop_forever=not args.once)
