"""
=============================================================================
TALOS #225 - FORTNITE DANCE MODE 🤖💃
Unitree G1 apprend les emotes Fortnite de base en simulation MuJoCo.

Danses :
    [1] Default Dance   [2] Floss   [3] Take the L   [4] Orange Justice
    [5] Best Mates (high-five)  [0] Stop (stand)

Usage :
    python fortnite_dance.py                  # viewer interactif
    python fortnite_dance.py --dance floss    # lance directement le Floss
    python fortnite_dance.py --headless       # test 30s sans affichage
=============================================================================
"""

import os
import sys
import math
import time
import argparse
import numpy as np
import mujoco
import mujoco.viewer

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
G1_XML = os.path.join(WORKSPACE, "unitree_rl_gym", "resources", "robots", "g1_description", "g1_29dof.xml")

NU = 29  # 29 DoF (hors free joint)

# ─── Gains PD ───
KP = np.zeros(NU); KD = np.zeros(NU)
KP[0:12] = [80., 60., 60., 110., 40., 40.] * 2
KD[0:12] = [2., 2., 2., 4., 2., 2.] * 2
KP[12:15] = [60., 40., 40.]; KD[12:15] = [2., 2., 2.]
KP[15:29] = 40.; KD[15:29] = 1.5
TAU_MAX = np.full(NU, 80.); TAU_MAX[15:29] = 25.

# ─── Joint indices ───
L_HIP_P, L_HIP_R, L_HIP_Y, L_KNEE, L_ANK_P, L_ANK_R = 0, 1, 2, 3, 4, 5
R_HIP_P, R_HIP_R, R_HIP_Y, R_KNEE, R_ANK_P, R_ANK_R = 6, 7, 8, 9, 10, 11
W_YAW, W_ROLL, W_PITCH = 12, 13, 14
L_SH_P, L_SH_R, L_SH_Y, L_ELB, L_WR_R, L_WR_P, L_WR_Y = 15, 16, 17, 18, 19, 20, 21
R_SH_P, R_SH_R, R_SH_Y, R_ELB, R_WR_R, R_WR_P, R_WR_Y = 22, 23, 24, 25, 26, 27, 28


def base_posture():
    q = np.zeros(NU)
    q[L_HIP_P] = -0.15; q[L_KNEE] = 0.32; q[L_ANK_P] = -0.17
    q[R_HIP_P] = -0.15; q[R_KNEE] = 0.32; q[R_ANK_P] = -0.17
    q[L_SH_P] = 0.1; q[L_ELB] = 0.6
    q[R_SH_P] = 0.1; q[R_ELB] = 0.6
    return q


# ═══════════════════════════════════════════
#  KEYFRAMES — chaque danse = liste de (t, q[29]), boucle de durée T
# ═══════════════════════════════════════════

def dance_default():
    """🕺 Default Dance — ondulation bras alternés + hanches, boucle 2s."""
    b = base_posture(); kf = []; T = 2.0; n = 8
    for i in range(n):
        t = T * i / n; q = b.copy()
        ph = 2 * math.pi * i / n; s, c = math.sin(ph), math.cos(ph)
        bounce = 0.06 * abs(s)
        q[L_KNEE] = 0.32 + bounce; q[R_KNEE] = 0.32 + bounce
        q[L_HIP_P] = -0.15 - bounce * 0.5; q[R_HIP_P] = -0.15 - bounce * 0.5
        q[L_HIP_R] = 0.12 * s; q[R_HIP_R] = -0.12 * s; q[W_ROLL] = 0.15 * s
        q[L_SH_P] = 0.1 - 0.7 * s; q[R_SH_P] = 0.1 + 0.7 * s
        q[L_ELB] = 1.0 + 0.3 * abs(c); q[R_ELB] = 1.0 + 0.3 * abs(c)
        q[W_YAW] = 0.1 * c
        kf.append((t, q))
    return kf, T


def dance_floss():
    """🌀 Floss — bras pliés qui traversent le corps, hanches latérales, boucle 1s."""
    b = base_posture(); kf = []; T = 1.0; n = 6
    for i in range(n):
        t = T * i / n; q = b.copy()
        ph = 2 * math.pi * i / n; s, c = math.sin(ph), math.cos(ph)
        q[L_KNEE] = 0.32 + 0.08 * abs(s); q[R_KNEE] = 0.32 + 0.08 * abs(s)
        q[L_HIP_R] = 0.25 * s; q[R_HIP_R] = -0.25 * s; q[W_ROLL] = 0.2 * s
        q[L_SH_P] = -0.3 + 0.5 * s; q[R_SH_P] = -0.3 - 0.5 * s
        q[L_SH_R] = 0.4 * c; q[R_SH_R] = -0.4 * c
        q[L_ELB] = 1.4; q[R_ELB] = 1.4
        q[L_WR_R] = 0.3 * s; q[R_WR_R] = -0.3 * s
        kf.append((t, q))
    return kf, T


def dance_take_the_l():
    """💀 Take the L — L au-dessus de la tête, buste penché, boucle 2.4s."""
    b = base_posture(); kf = []; T = 2.4; n = 8
    for i in range(n):
        t = T * i / n; q = b.copy()
        ph = 2 * math.pi * i / n; s, c = math.sin(ph), math.cos(ph)
        q[L_KNEE] = 0.32 + 0.05 * abs(s); q[R_KNEE] = 0.32 + 0.05 * abs(s)
        # Bras droit levé (vertical du L)
        q[R_SH_P] = -2.6; q[R_SH_R] = -0.3 * s; q[R_ELB] = 0.1
        # Bras gauche tendu sur le côté (horizontal du L)
        q[L_SH_P] = -1.4; q[L_SH_R] = 0.8; q[L_ELB] = 0.1
        # Buste penché
        q[W_ROLL] = 0.2 * s; q[W_PITCH] = 0.1 * c; q[W_YAW] = 0.15 * s
        q[L_HIP_R] = -0.1 * s; q[R_HIP_R] = 0.1 * s
        kf.append((t, q))
    return kf, T


def dance_orange_justice():
    """🍊 Orange Justice — bras croisés secoués, boucle 1.2s."""
    b = base_posture(); kf = []; T = 1.2; n = 8
    for i in range(n):
        t = T * i / n; q = b.copy()
        ph = 2 * math.pi * i / n; s, c = math.sin(ph), math.cos(ph)
        q[L_KNEE] = 0.32 + 0.07 * abs(s); q[R_KNEE] = 0.32 + 0.07 * abs(s)
        q[L_HIP_P] = -0.15 - 0.03 * abs(s); q[R_HIP_P] = -0.15 - 0.03 * abs(s)
        q[L_SH_P] = -0.8; q[R_SH_P] = -0.8
        q[L_SH_R] = 0.9; q[R_SH_R] = -0.9
        q[L_ELB] = 1.2; q[R_ELB] = 1.2
        q[W_ROLL] = 0.25 * s; q[W_YAW] = 0.15 * c
        q[L_HIP_R] = 0.15 * s; q[R_HIP_R] = -0.15 * s
        q[L_WR_P] = 0.4 * s; q[R_WR_P] = -0.4 * s
        kf.append((t, q))
    return kf, T


def dance_best_mates():
    """🙌 Best Mates — high-five alterné avec soi-même, boucle 1.6s."""
    b = base_posture(); kf = []; T = 1.6; n = 8
    for i in range(n):
        t = T * i / n; q = b.copy()
        ph = 2 * math.pi * i / n; s, c = math.sin(ph), math.cos(ph)
        q[L_KNEE] = 0.32 + 0.04 * abs(s); q[R_KNEE] = 0.32 + 0.04 * abs(s)
        # Bras droit monte faire un high-five
        q[R_SH_P] = -1.8 - 0.6 * max(0, s)
        q[R_SH_R] = -0.3
        q[R_ELB] = 0.3 + 0.4 * max(0, s)
        # Bras gauche descend puis remonte (alternance)
        q[L_SH_P] = -1.8 - 0.6 * max(0, -s)
        q[L_SH_R] = 0.3
        q[L_ELB] = 0.3 + 0.4 * max(0, -s)
        q[W_ROLL] = 0.1 * c
        kf.append((t, q))
    return kf, T


def dance_idle():
    """🧍 Stand — posture neutre, boucle 2s (respiration)."""
    b = base_posture(); kf = []; T = 2.0
    for i in range(4):
        t = T * i / 4; q = b.copy()
        ph = 2 * math.pi * i / 4; s = math.sin(ph)
        q[L_KNEE] = 0.32 + 0.02 * abs(s); q[R_KNEE] = 0.32 + 0.02 * abs(s)
        q[W_PITCH] = 0.03 * s
        kf.append((t, q))
    return kf, T


DANCES = {
    "default":       ("🕺 Default Dance",  dance_default),
    "floss":         ("🌀 Floss",          dance_floss),
    "takethel":      ("💀 Take the L",     dance_take_the_l),
    "orangejustice": ("🍊 Orange Justice", dance_orange_justice),
    "bestmates":     ("🙌 Best Mates",     dance_best_mates),
    "idle":          ("🧍 Stand",          dance_idle),
}
DANCE_ORDER = ["default", "floss", "takethel", "orangejustice", "bestmates", "idle"]


# ═══════════════════════════════════════════
#  INTERPOLATION
# ═══════════════════════════════════════════

def cosine_ease(x):
    x = max(0.0, min(1.0, x))
    return 0.5 - 0.5 * math.cos(math.pi * x)


def sample_dance(keyframes, loop_T, t):
    t = t % loop_T
    n = len(keyframes)
    for i in range(n):
        t0, q0 = keyframes[i]
        t1, q1 = keyframes[(i + 1) % n]
        if t1 <= t0:
            t1 += loop_T
        if t0 <= t < t1:
            span = t1 - t0
            if span <= 1e-9:
                return q0.copy()
            return q0 + cosine_ease((t - t0) / span) * (q1 - q0)
    return keyframes[0][1].copy()


# ═══════════════════════════════════════════
#  HARNESS ÉLASTIQUE (maintient le robot debout)
# ═══════════════════════════════════════════

class ElasticHarness:
    """Harnais PD sur le pelvis + ancrage doux au sol.
    Maintient le robot en position verticale pour la danse sur place."""
    def __init__(self, target_pos=None):
        self.target_pos = np.array([0., 0., 0.793]) if target_pos is None else target_pos
        self.kp_pos = np.array([500., 500., 800.])
        self.kd_pos = np.array([50., 50., 60.])
        self.kp_rot = 150.
        self.kd_rot = 15.

    def apply(self, model, data):
        pid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        pos = data.xpos[pid].copy()
        vel = data.cvel[pid].copy()
        quat = data.xquat[pid].copy()

        # PD position (seulement Z fort, X/Y doux pour laisser le robot bouger)
        pos_err = self.target_pos - pos
        force = self.kp_pos * pos_err - self.kd_pos * vel[:3]
        force[0] = np.clip(force[0], -100, 100)
        force[1] = np.clip(force[1], -100, 100)
        force[2] = np.clip(force[2], -50, 400)

        # PD orientation : ramener le quaternion vers l'identité
        qw, qx, qy, qz = quat
        # erreur = partie imaginaire du quaternion (petits angles)
        rot_err = np.array([qx, qy, qz])
        if qw < 0:
            rot_err = -rot_err
        torque = self.kp_rot * rot_err - self.kd_rot * vel[3:6]
        torque = np.clip(torque, [-30, -30, -30], [30, 30, 30])

        data.xfrc_applied[pid] = [force[0], force[1], force[2], torque[0], torque[1], torque[2]]

    def anchor(self, data):
        """Ancrage dur : force la position Z et le quaternion du pelvis (anti-chute)."""
        data.qpos[2] = self.target_pos[2]   # Z
        data.qpos[0] = self.target_pos[0]   # X
        data.qpos[1] = self.target_pos[1]   # Y
        data.qpos[3:7] = [1., 0., 0., 0.]   # quaternion identité
        # Zero les vitesses de la base
        data.qvel[0:6] = 0.0


# ═══════════════════════════════════════════
#  SCÈNE
# ═══════════════════════════════════════════

def build_scene():
    """Charge le G1 directement (pas d'include) et customise la scène via l'API Python."""
    model = mujoco.MjModel.from_xml_path(G1_XML)
    data = mujoco.MjData(model)

    # Agrandir le sol existant
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id >= 0:
        model.geom_type[floor_id] = mujoco.mjtGeom.mjGEOM_PLANE
        model.geom_size[floor_id] = [15, 15, 0.1]
        model.geom_friction[floor_id] = [1.0, 0.5, 0.01]
        model.geom_rgba[floor_id] = [0.12, 0.08, 0.28, 1.0]

    # Changer le timestep
    model.opt.timestep = 0.002

    # Ajouter une caméra custom (side view)
    # Les caméras existantes dans le G1 XML sont fixes, on en ajoute une
    cam_id = model.ncam
    # En MuJoCo Python, on ne peut pas facilement ajouter une caméra à chaud,
    # on utilise la caméra libre du viewer à la place.

    return model, data


def main():
    # Fix encodage console Windows
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except Exception:
            pass

    p = argparse.ArgumentParser()
    p.add_argument("--dance", default="default", choices=list(DANCES.keys()))
    p.add_argument("--headless", action="store_true")
    p.add_argument("--seconds", type=float, default=30.0)
    args = p.parse_args()

    model, data = build_scene()
    harness = ElasticHarness()

    # Préparer les danses
    dance_data = {name: fn() for name, (_, fn) in DANCES.items()}
    state = {"current": args.dance, "t0": 0.0, "paused": False}

    # Init position
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0., 0., 0.793]
    data.qpos[3:7] = [1., 0., 0., 0.]
    qi = base_posture()
    for i in range(NU):
        data.qpos[7 + i] = qi[i]
    mujoco.mj_forward(model, data)

    dt = model.opt.timestep
    sim_t = 0.0

    def set_dance(name):
        state["current"] = name
        state["t0"] = sim_t
        print(f"\n  ▶ {DANCES[name][0]}")

    def key_cb(keycode):
        mapping = {
            ord('1'): "default", ord('2'): "floss", ord('3'): "takethel",
            ord('4'): "orangejustice", ord('5'): "bestmates", ord('0'): "idle",
        }
        if keycode in mapping:
            set_dance(mapping[keycode])
        elif keycode == 32:  # espace
            state["paused"] = not state["paused"]
            print("  ⏸ Pause" if state["paused"] else "  ▶ Play")
        elif keycode in (ord('r'), ord('R')):
            mujoco.mj_resetData(model, data)
            data.qpos[0:3] = [0., 0., 0.793]
            data.qpos[3:7] = [1., 0., 0., 0.]
            for i in range(NU):
                data.qpos[7 + i] = qi[i]
            sim_t_ref[0] = 0.0
            state["t0"] = 0.0
            print("  🔄 Reset")

    sim_t_ref = [0.0]

    print("=" * 64)
    print("  🤖 TALOS #225 — FORTNITE DANCE MODE")
    print("=" * 64)
    print("  Touches : 1=Default  2=Floss  3=TakeTheL  4=OrangeJustice  5=BestMates  0=Stand")
    print("            ESPACE=pause  R=reset")
    print("=" * 64)
    set_dance(args.dance)

    if args.headless:
        n = int(args.seconds / dt)
        falls = 0; min_z = 99; max_z = 0
        pid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        for step in range(n):
            if not state["paused"]:
                sim_t += dt
            kf, lT = dance_data[state["current"]]
            q_des = sample_dance(kf, lT, sim_t - state["t0"])
            q = data.qpos[7:7 + NU]; dq = data.qvel[6:6 + NU]
            tau = np.clip(KP * (q_des - q) + KD * (0. - dq), -TAU_MAX, TAU_MAX)
            data.ctrl[:] = tau
            harness.apply(model, data)
            mujoco.mj_step(model, data)
            harness.anchor(data)  # anti-chute : maintient le pelvis en place
            mujoco.mj_forward(model, data)
            z = data.xpos[pid][2]
            min_z = min(min_z, z); max_z = max(max_z, z)
            if z < 0.3:
                falls += 1
        print(f"\n✅ Headless {args.seconds}s — danse '{state['current']}'")
        print(f"   Pelvis Z: min={min_z:.3f} max={max_z:.3f} final={z:.3f}  chutes={falls}")
        return

    with mujoco.viewer.launch_passive(model, data, key_callback=key_cb) as viewer:
        while viewer.is_running():
            t_start = time.time()
            if not state["paused"]:
                sim_t += dt
            kf, lT = dance_data[state["current"]]
            q_des = sample_dance(kf, lT, sim_t - state["t0"])
            q = data.qpos[7:7 + NU]; dq = data.qvel[6:6 + NU]
            tau = np.clip(KP * (q_des - q) + KD * (0. - dq), -TAU_MAX, TAU_MAX)
            data.ctrl[:] = tau
            harness.apply(model, data)
            mujoco.mj_step(model, data)
            harness.anchor(data)  # anti-chute
            mujoco.mj_forward(model, data)
            viewer.sync()
            elapsed = time.time() - t_start
            if elapsed < dt:
                time.sleep(dt - elapsed)


if __name__ == "__main__":
    main()
