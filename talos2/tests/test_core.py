import numpy as np
import pytest
from talos2.params import Params
from talos2.world import World
from talos2.prior_map import PriorMap
from talos2.planner import Planner
from talos2.energy import calibration_autonomy_hours, power_w


def test_world_is_deterministic():
    a, b = World(5), World(5)
    assert np.array_equal(a.trees, b.trees) and np.array_equal(a.height, b.height)
    assert not np.array_equal(World(6).trees, a.trees)


def test_lidar_hits_a_known_circle():
    w = World(0)
    w.obst = np.array([[50.0, 10.0, 0.5]])            # un seul cylindre a 10 m a l'est
    ang, r = w.lidar_scan(40.0, 10.0, 0.0, 72, 25.0, 0.0, 0.0, np.random.default_rng(0))
    k = int(np.argmin(np.abs(ang)))
    assert abs(r[k] - 9.5) < 1e-6                     # surface du cylindre
    assert (r < 25).sum() <= 3                        # les autres rayons ne touchent rien


def test_prior_map_is_incomplete_and_raycast_sees_a_tree():
    p = Params()
    w = World(1); pr = PriorMap(w, p, 1)
    assert len(pr.trees) < len(w.trees)               # carte incomplete
    pr2 = PriorMap(w, Params(prior_keep_trees=1.0, prior_pos_sigma=0.0, prior_ghost_trees=0.0), 1)
    x, y, r = w.trees[10]
    z = pr2.raycast(np.array([x - 8.0]), np.array([y]), np.array([0.0]), 25.0)[0]
    assert z < 25.0                                   # un tronc cartographie est vu sur l'axe


def test_planner_avoids_known_trees():
    p = Params(); w = World(0); pr = PriorMap(w, p, 0); pl = Planner(pr, p)
    path = pl.plan(w.start, w.goal)
    assert path is not None and np.hypot(*(path[-1] - w.goal)) < 2.0
    for tx, ty, tr in pr.trees:
        assert np.min(np.hypot(path[:, 0] - tx, path[:, 1] - ty)) > tr + 0.3


def test_energy_is_calibrated_on_manufacturer_figure():
    p = Params(payload_kg=0.0)
    assert 1.9 < calibration_autonomy_hours(p, 1.0) < 2.1     # ~2 h annonce constructeur
    assert power_w(p, 1.0, 0.2, True) > power_w(p, 1.0, 0.0, False)   # montee + boue coutent plus


@pytest.mark.slow
def test_map_matching_beats_dead_reckoning():
    from talos2.sim import run_mission
    a = run_mission(1, matching=True, record=False)["metrics"]
    b = run_mission(1, matching=False, record=False)["metrics"]
    assert a["err_mean_m"] < b["err_mean_m"]
