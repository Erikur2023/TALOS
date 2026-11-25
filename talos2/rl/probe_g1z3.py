import jax, jax.numpy as jp
from mujoco import mjx
from mujoco_playground import registry
env = registry.load("G1JoystickFlatTerrain")
s = env.reset(jax.random.PRNGKey(0)); d = s.data; mx = env.mjx_model
print("ctrl0", d.ctrl[:6], "qvel max", float(jp.abs(d.qvel).max()))
step = jax.jit(mjx.step)
for i in range(500):
    d = step(mx, d)
    if i in (5,50,200,499): print(i, "z", float(d.qpos[2]), "ncon", int(d.ncon), flush=True)
mx2 = mjx.put_model(env.mj_model)
d = s.data
for i in range(500): d = step(mx2, d)
print("put_model frais z", float(d.qpos[2]))
print("diff opt", env.mj_model.opt.iterations, mx.opt.iterations, mx.opt.ls_iterations, mx.opt.timestep, mx.opt.disableflags)
