"""Application C2 : donnees region (si construites), itineraire, garde de cohérence, mission courte."""
import os
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
HAVE = os.path.exists(os.path.join(HERE, "..", "data", "region", "build", "front.json"))


@pytest.mark.skipif(not HAVE, reason="donnees region non construites (python -m talos_app.datasets)")
def test_region_route_avoids_front():
    from talos_app import region as RG
    from talos_app.route import plan, FRONT_MARGIN
    R = RG.get()
    assert len(R.allied) > 5 and all(R.status_at(a["lon"], a["lat"]) == "controlled" for a in R.allied)
    pos = [a for a in R.allied if a["kind"] == "position"][0]
    x, y = RG.to_xy(pos["lon"], pos["lat"])
    for ang in np.linspace(0, 2 * np.pi, 16, endpoint=False):
        ax, ay = x + 1500 * np.cos(ang), y + 1500 * np.sin(ang)
        if R.front_dist(ax, ay) > R.front_dist(x, y):
            break
    r = plan(tuple(map(float, RG.to_ll(ax, ay))), (pos["lon"], pos["lat"]))
    assert r["ok"] and r["front_min_m"] >= FRONT_MARGIN - 50 and r["length_m"] >= r["straight_m"]
    occ = [a for a in R.places if R.status_at(a["lon"], a["lat"]) == "occupied"]
    if occ:
        assert not plan((pos["lon"], pos["lat"]), (occ[0]["lon"], occ[0]["lat"]))["ok"]      # B en zone occupee : refus


def test_odom_gate_rejects_runaway_estimate():
    """Garde de coherence : une estimee qui part loin de l'odometrie est ramenee."""
    from dataclasses import replace
    from talos2.params import Params
    from talos2.world import World
    from talos2.prior_map import PriorMap
    from talos2.navigator import Navigator
    p = replace(Params(), pf_odom_gate=True)
    w = World(101); m = PriorMap(w, p, 101)
    nav = Navigator(m, p, w.start, 0.0, w.goal, 1, True)
    nav.est = nav.pf.estimate(); nav._odom_gate(0.0)
    nav.pf.s[:, 0] += 30.0                                   # saut artificiel de 30 m
    nav.est = nav.pf.estimate(); nav._odom_gate(1.0)
    assert abs(nav.est[0] - w.start[0]) < 5 and nav.gate_rejects == 1
