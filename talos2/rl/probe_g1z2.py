import sys, jax, jax.numpy as jp, mujoco
from mujoco import mjx
from mujoco_playground import registry
m = registry.load("G1JoystickFlatTerrain").mj_model
def test(tag, it, ls, solver=None):
    mm = mujoco.MjModel.from_xml_string("<mujoco/>") if False else m
    mm.opt.iterations, mm.opt.ls_iterations = it, ls
    if solver is not None: mm.opt.solver = solver
    mx = mjx.put_model(mm); d = mjx.make_data(mx)
    d = d.replace(qpos=jp.array(m.keyframe("knees_bent").qpos), ctrl=jp.array(m.keyframe("knees_bent").ctrl))
    step = jax.jit(mjx.step)
    for i in range(500): d = step(mx, d)
    print(tag, "z", float(d.qpos[2]), "ncon", int(d.ncon), flush=True)
test("it3", 3, 5); test("it10", 10, 20); test("it30", 30, 30); test("it3_cg", 3, 5, mujoco.mjtSolver.mjSOL_CG)
