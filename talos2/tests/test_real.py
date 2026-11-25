"""Terrain reel (donnees en cache dans data/real/, telechargees par `python -m talos2.real_fetch`) et carte 3D."""
import os
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
HAVE = all(os.path.exists(os.path.join(HERE, "..", "data", "real", s, "osm.json")) for s in ("serebryansky", "robotyne"))


@pytest.mark.skipif(not HAVE, reason="donnees reelles non telechargees")
def test_real_world_and_maps_are_consistent():
    from talos2 import real_site as rs
    from talos2.params import Params
    w = rs.RealWorld("serebryansky", rs.starts_for("serebryansky")[0], rs.GOALS["serebryansky"])
    assert 0.3 < w.veg_img.mean() < 0.95 and len(w.trees) > 1000 and w.height.shape == (w.n, w.n)
    assert np.hypot(*(w.trees[:, :2] - w.goal).T).min() > 5.0
    ang, rr = w.lidar_scan(*w.start, 0.0, 180, 30.0, 0.0, 0.0, np.random.default_rng(0))
    assert (rr < 29.5).sum() > 10                                    # on voit des troncs depuis le depart
    for lvl in ("sat", "osm", "dem"):
        m = rs.RealMap(w, lvl, Params())
        assert m.lam.shape == m.dist.shape and np.isfinite(m.lam).all()
    sat, dem = rs.RealMap(w, "sat", Params()), rs.RealMap(w, "dem", Params())
    assert sat.lam.std() > 0.01 and dem.lam.std() < 1e-6          # le niveau 'relief seul' n'apporte rien au LiDAR


def test_sim_records_3d_map():
    from talos2.sim import run_mission
    from talos2.params import Params
    from dataclasses import replace
    r = run_mission(105, p=replace(Params(), max_time=60.0), record=True)
    v = np.array(r["telemetry"]["vox"])
    assert len(v) > 500 and set(np.unique(v[:, 3])) == {0.0, 1.0}
    assert (np.diff(v[:, 4]) >= 0).all()                              # tries par instant de premiere observation
