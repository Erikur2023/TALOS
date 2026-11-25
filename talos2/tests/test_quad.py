"""Quadrupede Go1 (modele Menagerie exporte dans rl/models/) et executeur numpy des politiques entrainees."""
import os
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "..", "rl", "models", "go1", "scene.xml")
POL = [os.path.join(HERE, "..", "results", "rl", r, "policy.npz") for r in ("go1_terrain_ua", "go1_sol_plat", "go1_ua_direct")]


def test_np_policy_forward(tmp_path):
    from talos2.mujoco_g1.quad import NpPolicy
    rng = np.random.default_rng(0)
    p = tmp_path / "p.npz"
    np.savez(p, obs_mean=np.zeros(48), obs_std=np.ones(48), W0=rng.normal(size=(48, 16)), b0=np.zeros(16),
             W1=rng.normal(size=(16, 24)), b1=np.zeros(24))
    a = NpPolicy(str(p))(rng.normal(size=48))
    assert a.shape == (12,) and np.all(np.abs(a) <= 1)


@pytest.mark.skipif(not os.path.exists(MODEL), reason="modeles Go1 non exportes (rl/export_models.py)")
def test_go1_spec_and_standing():
    import mujoco
    from talos2.mujoco_g1 import quad
    s = quad.robot_spec(3.0)
    s.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05])
    m = s.compile()
    assert m.nu == 12 and abs(m.opt.timestep - 0.004) < 1e-9
    pol = next((p for p in POL if os.path.exists(p)), None)
    if pol is None:
        pytest.skip("aucune politique Go1 entrainee")
    d = mujoco.MjData(m); W = quad.Go1Walker(m, pol); W.reset(d)
    d.qpos[2] = W.home_z + 0.02; mujoco.mj_forward(m, d)
    for _ in range(100):                                       # 2 s consigne nulle : reste debout
        W.control_step(m, d, [0, 0, 0])
    assert W.upright(d) > 0.8 and d.qpos[2] > 0.15
