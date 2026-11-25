"""Benchmark mesure : N missions aleatoires, avec et sans recalage sur la carte.
Usage : python -m talos2.benchmark --seeds 100 116 --out results
Les seeds d'evaluation sont DISTINCTES de celles utilisees pour regler le filtre (0-9)."""
import argparse, json, os, time
import numpy as np
from multiprocessing import Pool
from .sim import run_mission
from .params import Params


MAP, TRACKS = "trees", False


def _job(a):
    seed, matching, map_kind, tracks, delivery = a
    t = time.time()
    from .world import World
    p = Params()
    w = World(seed, tracks=tracks)
    if map_kind == "osm":
        from .osm_map import OsmMap, osm_params
        p = osm_params(p)
        prior = OsmMap(w, p, seed)
    else:
        from .prior_map import PriorMap
        prior = PriorMap(w, p, seed)
    r = run_mission(seed, matching=matching, p=p, record=False, world=w, prior=prior, delivery=delivery)
    r["metrics"]["map"] = map_kind
    r["metrics"]["wall_s"] = round(time.time() - t, 1)
    return r


def wilson(k, n, z=1.96):
    if n == 0:
        return (0, 0)
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h), min(1, c + h))


def summarize(rows):
    n = len(rows)
    if n == 0:
        return {}
    ok = sum(r["success"] for r in rows)
    lo, hi = wilson(ok, n)
    g = lambda k: np.array([r.get(k, np.nan) for r in rows], float)
    out = {
        "n": n, "success": ok, "success_rate": round(ok / n, 3), "success_ci95": [round(lo, 3), round(hi, 3)],
        "err_mean_median_m": round(float(np.nanmedian(g("err_mean_m"))), 2),
        "err_p95_median_m": round(float(np.nanmedian(g("err_p95_m"))), 2),
        "final_goal_error_median_m": round(float(np.nanmedian(g("final_goal_error_m"))), 2),
        "contacts_total": int(np.nansum(g("contacts"))),
        "missions_with_contact": int((np.nan_to_num(g("contacts")) > 0).sum()),
        "time_median_s": round(float(np.nanmedian(g("time_s"))), 0),
        "energy_median_wh": round(float(np.nanmedian(g("energy_wh"))), 1),
        "replans_median": float(np.median(g("replans"))),
        "thermal_dT_peak_median_k": round(float(np.nanmedian(g("thermal_dT_peak_k"))), 1) if "thermal_dT_peak_k" in rows[0] else None,
    }
    if any("delivered" in r for r in rows):
        ok_nav = [r for r in rows if r["success"] and "delivered" in r]
        out["delivered"] = int(sum(r["delivered"] and r["delivery_state"] == "DONE" for r in ok_nav))
        out["edge_incidents"] = int(sum(r["edge_incident"] for r in ok_nav))
        out["min_edge_dist_min_m"] = min((r["min_edge_dist_m"] for r in ok_nav), default=None)
        dd = [r["drop_edge_dist_m"] for r in ok_nav if r["drop_edge_dist_m"] is not None]
        out["drop_edge_dist_median_m"] = round(float(np.median(dd)), 2) if dd else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs=2, type=int, default=[100, 116])
    ap.add_argument("--out", default="results")
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--list", nargs="+", type=int, default=None, help="graines explicites (remplace --seeds)")
    ap.add_argument("--no-dr", action="store_true")
    ap.add_argument("--map", default="trees", choices=["trees", "osm"])
    ap.add_argument("--tracks", action="store_true", help="mondes avec chemins forestiers")
    ap.add_argument("--delivery", action="store_true", help="phase de depot a la tranchee incluse")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    seeds = a.list or list(range(*a.seeds))
    jobs = [(s, m, a.map, a.tracks, a.delivery) for s in seeds for m in ((True,) if a.no_dr else (True, False))]
    t0 = time.time()
    with Pool(a.procs) as pool:
        res = pool.map(_job, jobs, chunksize=1)
    rows = []
    for (s, m, *_), r in zip(jobs, res):
        r["metrics"].setdefault("seed", s); r["metrics"].setdefault("matching", m)
        rows.append(r["metrics"])
    with_m = [r for r in rows if r.get("matching") is True and "success" in r]
    no_m = [r for r in rows if r.get("matching") is False and "success" in r]
    out = {"params": Params().__dict__, "seeds": seeds, "map": a.map, "tracks": a.tracks,
           "with_map_matching": summarize(with_m), "dead_reckoning_only": summarize(no_m),
           "runs": rows, "wall_s": round(time.time() - t0)}
    json.dump(out, open(os.path.join(a.out, "benchmark.json"), "w"), indent=1, default=float)
    with open(os.path.join(a.out, "benchmark.csv"), "w") as f:
        keys = max((list(r.keys()) for r in rows), key=len)
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
    # courbes erreur / distance
    series = {True: [], False: []}
    for r in res:
        if "err_series" in r:
            series[r["metrics"]["matching"]].append(r["err_series"])
    np.save(os.path.join(a.out, "err_series.npy"), np.array([series], dtype=object), allow_pickle=True)
    print(json.dumps({k: out[k] for k in ("with_map_matching", "dead_reckoning_only", "wall_s")}, indent=1))


if __name__ == "__main__":
    main()
