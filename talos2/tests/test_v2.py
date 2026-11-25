"""Tests des ajouts de la nuit : Navigator, planificateur, carte OSM, depot, thermique, pont MuJoCo, LiDAR 3D."""
import numpy as np
import pytest
from talos2.params import Params
from talos2.world import World
from talos2.prior_map import PriorMap
from talos2.planner import Planner
from talos2.navigator import Navigator


def test_navigator_is_simulator_independent():
    p = Params(); w = World(3); pr = PriorMap(w, p, 3)
    nav = Navigator(pr, p, w.start, 0.0, w.goal, seed=3)
    v, wz = nav.command(0.0, 0.1)
    assert np.isfinite(v) and np.isfinite(wz) and abs(wz) <= p.w_cmd_max + 1e-9
    nav.on_proprio(0.5, 0.01, 0.1, v_lat=0.05)
    ang = np.linspace(-np.pi, np.pi, 72, endpoint=False)
    nav.on_scan(ang, np.full(72, p.lidar_range), 0.0, 0.0, 0.0)
    assert nav.remaining_m() > 100 and not nav.arrived


def test_tracks_do_not_change_default_world():
    a, b = World(7), World(7, tracks=False)
    assert np.array_equal(a.trees, b.trees)
    c = World(7, tracks=True)
    assert len(c.tracks) >= 2 and len(c.trees) < len(a.trees)


def test_planner_can_leave_a_blocked_start():
    p = Params(); w = World(0); pr = PriorMap(w, p, 0); pl = Planner(pr, p)
    x, y = w.start + 20
    for a in np.linspace(0, 2 * np.pi, 8, endpoint=False):     # entoure d'obstacles decouverts
        pl.add_obstacle(x + 1.2 * np.cos(a), y + 1.2 * np.sin(a), 1.0)
    path = pl.plan((x, y), w.goal)
    assert path is not None and np.hypot(*(path[-1] - w.goal)) < 2


def test_osm_map_has_no_trees_and_a_density_field():
    from talos2.osm_map import OsmMap, osm_params, LAMBDA_FOREST
    w = World(4, tracks=True); p = osm_params(Params()); m = OsmMap(w, p, 4)
    assert len(m.trees) == 0
    q = m.quality(w)
    assert q["forest_agreement"] > 0.85 and q["trees_in_mapped_forest"] > 0.8
    assert 0 < m.lam.min() < m.lam.max() <= LAMBDA_FOREST + 1e-6
    assert float(m.known_dist_at(np.array([10.0]), np.array([10.0]))[0]) >= 6.0


def test_delivery_rules():
    from talos2.delivery import Delivery, closest_on_polyline, MAX_PAYLOAD_KG
    tr = np.array([[0.0, 0.0], [10.0, 0.0]])
    with pytest.raises(ValueError):
        Delivery(tr, MAX_PAYLOAD_KG + 0.5, Params())
    c, d, _ = closest_on_polyline(np.array([5.0, 3.0]), tr)
    assert np.allclose(c, [5, 0]) and abs(d - 3) < 1e-9


def test_thermal_steady_state():
    from talos2.thermal import ThermalModel, LOSS_FRACTION, HA_W_PER_K
    th = ThermalModel()
    for _ in range(20000):
        th.step(200.0, 1.0)
    assert abs(th.dT - LOSS_FRACTION * 200 / HA_W_PER_K) < 0.05
    assert ThermalModel.detection_range_m(20) > ThermalModel.detection_range_m(5)


mujoco = pytest.importorskip("mujoco")
torch = pytest.importorskip("torch")


def _g1_available():
    from talos2.mujoco_g1.g1_policy import G1_XML, POLICY
    import os
    return os.path.exists(G1_XML) and os.path.exists(POLICY)


@pytest.mark.skipif(not _g1_available(), reason="unitree_rl_gym absent")
def test_mujoco_g1_stands_and_walks():
    from talos2.mujoco_g1.run import G1Mission, mujoco_params
    M = G1Mission(105, mujoco_params(), relief_scale=0.4)
    x0, y0, _ = M.pose
    fell = False
    for k in range(30):                                   # 3 s a 0,5 m/s
        _, _, fell = M.advance(0.5, 0.0, k * 0.1)
        if fell:
            break
    assert not fell and np.hypot(M.pose[0] - x0, M.pose[1] - y0) > 0.6


@pytest.mark.skipif(not _g1_available(), reason="unitree_rl_gym absent")
def test_lidar3d_segments_ground_and_trees():
    from talos2.mujoco_g1.run import G1Mission, mujoco_params
    from talos2.mujoco_g1.lidar3d import Lidar3D
    M = G1Mission(105, mujoco_params(), relief_scale=0.4)
    L = Lidar3D()
    for _ in range(3):
        (ang, rr), (al, ac) = L.process(M, M.rng)
    s = L.summary()
    assert s["obst_precision"] > 0.8 and s["obst_recall"] > 0.8
    assert s["slope_err_median"] < 0.05 and len(ang) == len(rr) == 72
