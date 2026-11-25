import jax, jax.numpy as jp, mujoco
from mujoco_playground import registry
env = registry.load("G1JoystickFlatTerrain")
m = env.mj_model
print("ngeom", m.ngeom, "npair", m.npair, "opt", m.opt.timestep, m.opt.iterations, m.opt.ls_iterations, "cone", m.opt.cone)
s = jax.jit(env.reset)(jax.random.PRNGKey(0)); st = jax.jit(env.step)
print("z0", float(s.data.qpos[2]))
for i in range(100): s = st(s, jp.zeros(env.action_size))
print("z100 zero-action", float(s.data.qpos[2]), "ncon", int(s.data.ncon) if hasattr(s.data,'ncon') else '?')
d = mujoco.MjData(m); d.qpos[:] = m.keyframe("knees_bent").qpos if m.nkey else d.qpos
mujoco.mj_forward(m, d); 
for i in range(500): mujoco.mj_step(m, d)
print("CPU z", d.qpos[2], "ncon", d.ncon)
