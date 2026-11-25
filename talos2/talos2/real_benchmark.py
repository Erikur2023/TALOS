"""Benchmark terrain REEL : 2 zones du front x 12 departs x 5 niveaux d'information de carte.
Usage : python -m talos2.real_benchmark --out results/real --procs 14
Les 12 departs sont sur deux anneaux (140 m / 210 m) autour de la tranchee fictive ; meme verite terrain pour tous
les niveaux d'une zone (graine 0), seules la carte embarquee et l'activation du recalage changent."""
import argparse, json, os, time
import numpy as np
from multiprocessing import Pool
from .benchmark import summarize

LEVELS = ["trees", "sat", "osm", "dem", "none"]
SITES = ["serebryansky", "robotyne"]


def _job(a):
    site, start, level = a
    from . import real_site as rs
    from .sim import run_mission
    t = time.time()
    w, m, p, mt = rs.make(site, start, level)
    r = run_mission(start, matching=mt, p=p, record=False, world=w, prior=m, delivery=True)["metrics"]
    r.update({"site": site, "start": start, "level": level, "wall_s": round(time.time() - t, 1),
              "start_xy": [round(v, 1) for v in w.start], "straight_m": round(float(np.hypot(*(w.goal - w.start))), 1)})
    r.setdefault("success", False)
    return r


def figure(rows, out):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from . import real_site as rs
    fig, ax = plt.subplots(2, 3, figsize=(17, 10.5))
    for k, site in enumerate(SITES):
        R = [r for r in rows if r["site"] == site]
        sr = [np.mean([r["success"] for r in R if r["level"] == l]) * 100 for l in LEVELS]
        er = [np.nanmedian([r.get("err_mean_m", np.nan) for r in R if r["level"] == l]) for l in LEVELS]
        a = ax[k, 0]
        a.bar(LEVELS, sr, color=["#888", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"])
        a.set_ylim(0, 105); a.set_ylabel("missions réussies (%)"); a.set_title(rs.rf.SITES[site]["name"], fontsize=10)
        for i, v in enumerate(sr):
            a.text(i, v + 2, "%.0f%%" % v, ha="center")
        a = ax[k, 1]
        a.bar(LEVELS, er, color="#457b9d"); a.set_ylabel("erreur de position, médiane des moyennes (m)")
        a.set_yscale("log")
        for i, v in enumerate(er):
            a.text(i, v * 1.1, "%.1f" % v, ha="center")
        a = ax[k, 2]
        w = rs.RealWorld(site, rs.starts_for(site)[0], rs.GOALS[site])
        a.imshow(w.img, origin="lower", extent=[0, w.size, 0, w.size])
        for r in R:
            if r["level"] != "sat":
                continue
            a.plot(*r["start_xy"], "o", ms=9, mfc="#06d6a0" if r["success"] else "#ef476f", mec="w")
            a.text(r["start_xy"][0] + 6, r["start_xy"][1] + 6, str(r["start"]), color="w", fontsize=8)
        a.plot(*w.goal, "*", ms=18, color="#ffb703", mec="k")
        a.set_title("départs (vert = réussi avec carte satellite), étoile = tranchée fictive", fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(out, "comparison.png"), dpi=80)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/real")
    ap.add_argument("--procs", type=int, default=14)
    ap.add_argument("--starts", type=int, default=12)
    ap.add_argument("--fig-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.fig_only:
        figure(json.load(open(os.path.join(a.out, "benchmark.json")))["runs"], a.out)
        return
    jobs = [(s, i, l) for s in SITES for i in range(a.starts) for l in LEVELS]
    t0 = time.time()
    with Pool(a.procs) as pool:
        rows = pool.map(_job, jobs, chunksize=1)
    summ = {s: {l: summarize([r for r in rows if r["site"] == s and r["level"] == l]) for l in LEVELS} for s in SITES}
    json.dump({"summary": summ, "runs": rows, "wall_s": round(time.time() - t0)}, open(os.path.join(a.out, "benchmark.json"), "w"),
              indent=1, default=float)
    figure(rows, a.out)
    for s in SITES:
        for l in LEVELS:
            x = summ[s][l]
            print("%-13s %-6s %2d/%2d  err %5.2f m  p95 %5.2f m" % (s, l, x["success"], x["n"], x["err_mean_median_m"], x["err_p95_median_m"]))


if __name__ == "__main__":
    main()
