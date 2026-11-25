"""Benchmark G1 sous MuJoCo : memes graines que le sim point, avec / sans recalage.
  python -m talos2.mujoco_g1.benchmark --seeds 100 116 --relief 0.4 --out results/mujoco"""
import argparse, json, os, time
from multiprocessing import Pool
import numpy as np
from ..benchmark import wilson


DELIVERY = True     # phase de depot incluse (mission complete)


def _job(a):
    seed, matching, relief, mode, robot = a
    from . import run as R
    kw = {}
    if mode == "lidar3d":
        from .lidar3d import attach_lidar3d
        kw = attach_lidar3d()
    if mode == "osm":
        kw = {"map_kind": "osm"}
    try:
        return R.run(seed, matching=matching, relief_scale=relief, delivery=DELIVERY, robot=robot, **kw)["metrics"]
    except Exception as e:                                  # une mission qui plante est comptee comme un echec
        return {"seed": seed, "matching": matching, "success": False, "status": "error: %r" % e}


def summarize(rows):
    n = len(rows)
    if n == 0:
        return {}
    ok = sum(r["success"] for r in rows)
    lo, hi = wilson(ok, n)
    g = lambda k: np.array([r.get(k, np.nan) for r in rows], float)
    return {"n": n, "success": ok, "success_ci95": [round(lo, 3), round(hi, 3)],
            "falls": int(np.nansum(g("falls"))), "missions_with_contact": int((g("contacts") > 0).sum()),
            "err_mean_median_m": round(float(np.nanmedian(g("err_mean_m"))), 2),
            "err_p95_median_m": round(float(np.nanmedian(g("err_p95_m"))), 2),
            "final_goal_error_median_m": round(float(np.nanmedian(g("final_goal_error_m"))), 2),
            "time_median_s": round(float(np.nanmedian(g("time_s"))), 0),
            "mech_work_median_wh": round(float(np.nanmedian(g("mech_positive_work_wh"))), 1),
            **({k: round(float(np.nanmedian(g(k))), 3) for k in ("lidar3d_obst_precision", "lidar3d_obst_recall",
                                                                 "lidar3d_slope_err_median") if k in rows[0]}),
            "statuses": {s: int(sum(r["status"] == s for r in rows)) for s in sorted({r["status"] for r in rows})}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs=2, type=int, default=[100, 116])
    ap.add_argument("--relief", type=float, default=0.4)
    ap.add_argument("--mode", default="lidar2d", choices=["lidar2d", "lidar3d", "osm"])
    ap.add_argument("--no-dr", action="store_true", help="ne pas lancer la reference odometrie seule")
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--robot", default="g1", choices=["g1", "go1", "unitree_mjlab", "gr00t_wbc"])
    ap.add_argument("--policy", default=None, help="politique Go1 (results/rl/<run>/policy.npz)")
    ap.add_argument("--out", default="results/mujoco")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    seeds = list(range(*a.seeds))
    if a.policy:
        os.environ["TALOS_GO1_POLICY"] = os.path.abspath(a.policy)
    jobs = [(s, m, a.relief, a.mode, a.robot) for s in seeds for m in ((True,) if a.no_dr else (True, False))]
    t0 = time.time()
    with Pool(a.procs) as pool:
        rows = pool.map(_job, jobs, chunksize=1)
    out = {"seeds": seeds, "relief_scale": a.relief, "mode": a.mode, "robot": a.robot, "policy": a.policy,
           "with_map_matching": summarize([r for r in rows if r["matching"]]),
           "dead_reckoning_only": summarize([r for r in rows if not r["matching"]]),
           "runs": rows, "wall_s": round(time.time() - t0)}
    json.dump(out, open(os.path.join(a.out, "benchmark.json"), "w"), indent=1, default=float)
    print(json.dumps({k: out[k] for k in ("with_map_matching", "dead_reckoning_only", "wall_s")}, indent=1))


if __name__ == "__main__":
    main()
