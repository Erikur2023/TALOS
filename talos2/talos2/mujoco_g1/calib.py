"""Domaine de consigne de la politique G1 (mesure, reproductible) : suivi vx / wz sur sol plat et vitesse en pente.
  python -m talos2.mujoco_g1.calib --out results/mujoco_calib.json"""
import argparse, json
import numpy as np
import mujoco
from .g1_policy import G1Walker, G1_XML, SIM_DT


def _yaw(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def trial(vx, wz, incline_deg=0.0, secs=8.0):
    s = mujoco.MjSpec.from_file(G1_XML)
    th = np.radians(incline_deg)
    s.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], quat=[np.cos(-th / 2), 0, np.sin(-th / 2), 0])
    m = s.compile(); d = mujoco.MjData(m); W = G1Walker(); W.reset(d); d.qpos[2] = 0.8
    for _ in range(500):
        W.step(m, d, [0, 0, 0])
    p0, yprev, yacc, n = d.qpos[:2].copy(), _yaw(d.qpos[3:7]), 0.0, int(secs / SIM_DT)
    fell = False
    for k in range(n):
        W.step(m, d, [vx, 0, wz])
        if k % 50 == 0:
            y = _yaw(d.qpos[3:7]); yacc += (y - yprev + np.pi) % (2 * np.pi) - np.pi; yprev = y
        if d.qpos[2] - np.tan(th) * d.qpos[0] < 0.45:
            fell = True; break
    T = (k + 1) * SIM_DT
    return {"vx_cmd": vx, "wz_cmd": wz, "incline_mm": round(float(np.tan(th)), 3),
            "speed": round(float(np.hypot(*(d.qpos[:2] - p0)) / T), 3), "yawrate": round(yacc / T, 3), "fell": fell}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/mujoco_calib.json"); a = ap.parse_args()
    rows = [trial(v, 0) for v in (0.1, 0.3, 0.5, 0.7, 0.9, 1.1, -0.25)]
    rows += [trial(0, w) for w in (0.2, 0.4, 0.8)] + [trial(0.5, w) for w in (0.4, 0.8)]
    rows += [trial(0.6, 0, deg) for deg in (3, 5, 8, 11, 14)]
    for r in rows:
        print(r)
    json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
