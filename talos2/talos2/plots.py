"""Figures du dossier a partir de results/benchmark.json."""
import json, sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main(res="results"):
    d = json.load(open(os.path.join(res, "benchmark.json")))
    runs = d["runs"]
    A = [r for r in runs if r.get("matching") is True and "success" in r]
    B = [r for r in runs if r.get("matching") is False and "success" in r]
    A.sort(key=lambda r: r["seed"]); B.sort(key=lambda r: r["seed"])
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    seeds = [r["seed"] for r in A]
    x = np.arange(len(seeds)); w = 0.38
    ax[0].bar(x - w/2, [r["err_mean_m"] for r in A], w, label="avec recalage carte", color="#2a9fd6")
    ax[0].bar(x + w/2, [r["err_mean_m"] for r in B], w, label="odométrie seule", color="#f2994a")
    ax[0].set_xticks(x); ax[0].set_xticklabels(seeds, rotation=70, fontsize=8)
    ax[0].set_ylabel("erreur de position moyenne (m)"); ax[0].set_xlabel("scénario (graine)"); ax[0].legend(frameon=False)
    ax[0].set_title("Erreur de localisation par mission")
    for lab, S, c in (("avec recalage", A, "#2a9fd6"), ("odométrie seule", B, "#f2994a")):
        ax[1].scatter([r["time_s"] for r in S], [r["final_goal_error_m"] for r in S], c=c, label=lab, s=34)
    ax[1].axhline(5.0, color="grey", ls="--", lw=1); ax[1].text(ax[1].get_xlim()[0], 5.3, "tolérance 5 m", fontsize=8, color="grey")
    ax[1].set_xlabel("durée de mission (s)"); ax[1].set_ylabel("distance finale à l'objectif (m)"); ax[1].legend(frameon=False)
    ax[1].set_title("Arrivée à l'objectif")
    sa, sb = d["with_map_matching"], d["dead_reckoning_only"]
    names = ["Réussite", "Missions sans contact"]
    va = [sa["success_rate"], 1 - sa["missions_with_contact"] / sa["n"]]
    vb = [sb["success_rate"], 1 - sb["missions_with_contact"] / sb["n"]]
    xx = np.arange(2)
    ax[2].bar(xx - w/2, va, w, color="#2a9fd6", label="avec recalage"); ax[2].bar(xx + w/2, vb, w, color="#f2994a", label="odométrie seule")
    ax[2].set_xticks(xx); ax[2].set_xticklabels(names); ax[2].set_ylim(0, 1.05); ax[2].legend(frameon=False)
    lo, hi = sa["success_ci95"]; ax[2].errorbar([-w/2], [sa["success_rate"]], yerr=[[sa["success_rate"]-lo], [hi-sa["success_rate"]]], c="k", capsize=3)
    lo, hi = sb["success_ci95"]; ax[2].errorbar([w/2], [sb["success_rate"]], yerr=[[sb["success_rate"]-lo], [hi-sb["success_rate"]]], c="k", capsize=3)
    ax[2].set_title("Taux (IC 95 %%, n=%d)" % sa["n"])
    fig.tight_layout(); out = os.path.join(res, "benchmark.png"); fig.savefig(out, dpi=150); print(out)

if __name__ == "__main__":
    main(*sys.argv[1:])
