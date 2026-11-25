"""Protocole PENTES identique a rl/evaluate.py, mais sous MuJoCo CPU (Windows) :
  - G1 12 ddl Unitree pre-entraine (sol plat, unitree_rl_gym) : la reference 'industrielle' de depart ;
  - politiques Go1 entrainees sur GPU (MJX) rejouees en MuJoCo C : verification du transfert simulateur -> simulateur.
Plan incline 0..25 deg, consigne vx = 0,5 m/s vers le haut, 15 s, charge 3 kg, N essais (perturbation : cap initial
+-0,1 rad, poussee laterale aleatoire a t = 5 s).
  python rl/eval_cpu.py --trials 8      -> results/robots/slopes_cpu.json"""
import argparse, json, os, sys
import numpy as np
import mujoco
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from talos2.mujoco_g1.g1_policy import G1Walker, G1_XML, SIM_DT
from talos2.mujoco_g1 import quad, ext_policies as EXT

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "robots")
DEGS = (0, 5, 10, 15, 20, 25)


def plane(spec, deg):
    th = np.radians(deg)
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], quat=[np.cos(-th / 2), 0, np.sin(-th / 2), 0],
                            contype=1, conaffinity=1, friction=[0.8, 0.005, 0.0001])


def yaw_quat(y):
    return [np.cos(y / 2), 0, 0, np.sin(y / 2)]


def trial_g1(deg, rng, secs=15.0):
    s = mujoco.MjSpec.from_file(G1_XML); plane(s, deg)
    s.body("pelvis").mass += 3.0
    m = s.compile(); d = mujoco.MjData(m); W = G1Walker(); W.reset(d)
    d.qpos[2] = 0.8; d.qpos[3:7] = yaw_quat(rng.uniform(-0.1, 0.1))
    for _ in range(500):
        W.step(m, d, [0, 0, 0])
    x0, th = d.qpos[0], np.radians(deg)
    n = int(secs / SIM_DT)
    for k in range(n):
        if k == int(5 / SIM_DT):
            d.qvel[1] += rng.uniform(-0.4, 0.4)
        W.step(m, d, [0.5, 0, 0])
        if d.qpos[2] - np.tan(th) * d.qpos[0] < 0.45:
            return {"speed": 0.0, "fell": True}
    return {"speed": float((d.qpos[0] - x0) / np.cos(th) / secs), "fell": False}


def trial_go1(deg, rng, policy, secs=15.0):
    s = quad.robot_spec(3.0); plane(s, deg)
    m = s.compile(); d = mujoco.MjData(m); W = quad.Go1Walker(m, policy); W.reset(d)
    d.qpos[2] = W.home_z + 0.05; d.qpos[3:7] = yaw_quat(rng.uniform(-0.1, 0.1))
    mujoco.mj_forward(m, d)
    for _ in range(25):
        W.control_step(m, d, [0, 0, 0])
    x0, th = d.qpos[0], np.radians(deg)
    n = int(secs / quad.CTRL_DT)
    for k in range(n):
        if k == int(5 / quad.CTRL_DT):
            d.qvel[1] += rng.uniform(-0.4, 0.4)
        W.control_step(m, d, [0.5, 0, 0])
        if W.upright(d) < 0.3 or d.qpos[2] - np.tan(th) * d.qpos[0] < 0.12:
            return {"speed": 0.0, "fell": True}
    return {"speed": float((d.qpos[0] - x0) / np.cos(th) / secs), "fell": False}


def trial_ext(deg, rng, key, secs=15.0):
    """Politique pre-entrainee par un tiers (talos2/mujoco_g1/ext_policies.py), meme protocole."""
    fs, W = EXT.REGISTRY[key]
    s = fs(3.0); plane(s, deg)
    m = s.compile(); d = mujoco.MjData(m); w = W(m); w.reset(d)
    d.qpos[2] = w.home_z + 0.02; d.qpos[3:7] = yaw_quat(rng.uniform(-0.1, 0.1))
    mujoco.mj_forward(m, d)
    for _ in range(50):
        w.control_step(m, d, [0, 0, 0])
    x0, th = d.qpos[0], np.radians(deg)
    n = int(secs / EXT.CTRL_DT)
    for k in range(n):
        if k == int(5 / EXT.CTRL_DT):
            d.qvel[1] += rng.uniform(-0.4, 0.4)
        w.control_step(m, d, [0.5, 0, 0])
        if w.upright(d) < 0.5 or d.qpos[2] - np.tan(th) * d.qpos[0] < 0.45:
            return {"speed": 0.0, "fell": True}
    return {"speed": float((d.qpos[0] - x0) / np.cos(th) / secs), "fell": False}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--trials", type=int, default=8); a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rows = []
    robots = [("G1 Unitree pré-entraîné (plat) [CPU]", lambda deg, rng: trial_g1(deg, rng))]
    for run, lab in (("go1_terrain_ua", "Go1 — terrain UA [CPU, transfert]"), ("go1_sol_plat", "Go1 — sol plat [CPU, transfert]")):
        p = os.path.join(quad.RESULTS_RL, run, "policy.npz")
        if os.path.exists(p):
            robots.append((lab, lambda deg, rng, p=p: trial_go1(deg, rng, p)))
    for key in EXT.available():
        robots.append((EXT.REGISTRY[key][1].label + " [CPU]", lambda deg, rng, key=key: trial_ext(deg, rng, key)))
    for lab, fn in robots:
        for deg in DEGS:
            rng = np.random.default_rng(1000 + deg)
            R = [fn(deg, rng) for _ in range(a.trials)]
            for r in R:
                rows.append({"robot": lab, "slope_deg": deg, "speed": round(r["speed"], 3), "fell": r["fell"]})
            print("%-40s %2d deg  v=%.2f  chutes %d/%d" % (lab, deg, np.mean([r["speed"] for r in R]), sum(r["fell"] for r in R), a.trials), flush=True)
    json.dump({"rows": rows, "protocol": __doc__}, open(os.path.join(OUT, "slopes_cpu.json"), "w"))


if __name__ == "__main__":
    main()
