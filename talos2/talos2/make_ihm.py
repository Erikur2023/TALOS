"""Genere l'IHM autonome (un seul fichier HTML) a partir de missions simulees.
Usage : python -m talos2.make_ihm --seeds 101 105 --out ihm/talos_ihm.html"""
import argparse, json, os
from .sim import run_mission

HERE = os.path.dirname(os.path.abspath(__file__))


def mujoco_panel(res):
    """Images MuJoCo (ihm/mujoco/*.jpg) + chiffres des benchmarks MuJoCo (results/final_mujoco*/benchmark.json)."""
    import base64, glob
    imgs = []
    for f in sorted(glob.glob(os.path.join(HERE, "..", "ihm", "mujoco", "*.jpg"))):
        imgs.append("data:image/jpeg;base64," + base64.b64encode(open(f, "rb").read()).decode())
    rows = []
    for name, lab in (("final_mujoco", "LiDAR 2D"), ("final_mujoco_lidar3d", "LiDAR 3D + segmentation")):
        f = os.path.join(res, name, "benchmark.json")
        if not os.path.exists(f):
            continue
        d = json.load(open(f))
        a, b = d["with_map_matching"], d.get("dead_reckoning_only") or {}
        txt = "%s, relief x%.1f, %d missions : %d réussies (IC95 %d-%d %%), %d chutes, erreur médiane %.2f m" % (
            lab, d["relief_scale"], a["n"], a["success"], 100 * a["success_ci95"][0], 100 * a["success_ci95"][1],
            a["falls"], a["err_mean_median_m"])
        if b:
            txt += " — odométrie seule : %d/%d" % (b["success"], b["n"])
        rows.append(txt)
    return {"images": imgs, "rows": rows} if (imgs or rows) else None


LEVEL_LAB = {"trees": "relevé arbre par arbre", "sat": "carte satellite", "osm": "carte OSM", "dem": "relief seul",
             "none": "odométrie seule"}


def real_run(site, start, level):
    """Mission sur terrain reel (talos2/real_site.py), image satellite en fond de carte."""
    import base64, io
    from PIL import Image
    from . import real_site as rs, real_fetch as rf
    w, m, p, mt = rs.make(site, start, level)
    r = run_mission(start, matching=mt, p=p, record=True, world=w, prior=m, delivery=True)
    if "telemetry" not in r:
        return None
    buf = io.BytesIO()
    Image.fromarray(w.img[::-1]).resize((700, 700)).save(buf, "JPEG", quality=80)
    W = r["telemetry"]["world"]
    W["image"] = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    W["trees"] = W["trees"][::3]                                     # allege le fichier (vue verite)
    lab = "Terrain réel — %s — départ %d — %s" % (rf.SITES[site]["name"], start, LEVEL_LAB[level])
    return {"label": lab, "metrics": r["metrics"], "telemetry": r["telemetry"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[101])
    ap.add_argument("--results", default=os.path.join(HERE, "..", "results"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "ihm", "talos_ihm.html"))
    ap.add_argument("--real", nargs="*", default=["serebryansky:1:sat", "serebryansky:1:none", "robotyne:4:sat"],
                    help="missions terrain reel site:depart:niveau")
    a = ap.parse_args()
    runs = {}
    for spec in a.real:
        site, st, lvl = spec.split(":")
        rr = real_run(site, int(st), lvl)
        if rr:
            runs["real_%s_%s_%s" % (site, st, lvl)] = rr
            print(spec, rr["metrics"]["status"], "succès" if rr["metrics"]["success"] else "échec",
                  "err moy %.1f m" % rr["metrics"]["err_mean_m"])
    for s in a.seeds:
        for matching in (True, False):
            r = run_mission(s, matching=matching, record=True, delivery=True)
            if "telemetry" not in r:
                continue
            key = "s%d_%s" % (s, "map" if matching else "dr")
            lab = "Scénario %d — %s" % (s, "navigation avec recalage sur la carte" if matching else "odométrie seule (référence)")
            runs[key] = {"label": lab, "metrics": r["metrics"], "telemetry": r["telemetry"]}
            print(key, r["metrics"]["status"], "succès" if r["metrics"]["success"] else "échec",
                  "err moy %.1f m" % r["metrics"]["err_mean_m"])
    tpl = open(os.path.join(HERE, "..", "ihm", "template.html"), encoding="utf-8").read()
    html = tpl.replace("/*DATA*/null", json.dumps({"runs": runs, "mujoco": mujoco_panel(a.results)},
                                                   default=float, separators=(",", ":")))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    open(a.out, "w", encoding="utf-8").write(html)
    print("IHM ecrite :", os.path.abspath(a.out), round(len(html) / 1e6, 2), "Mo")


if __name__ == "__main__":
    main()
