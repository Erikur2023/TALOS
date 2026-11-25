import sys, jax, jax.numpy as jp
if sys.argv[1] == "hi": jax.config.update("jax_default_matmul_precision", "highest")
from mujoco_playground import registry
env = registry.load("G1JoystickFlatTerrain")
st = jax.jit(env.step); rs = jax.jit(env.reset)
s = rs(jax.random.PRNGKey(0)); k = jax.random.PRNGKey(1)
for i in range(300):
    k, kk = jax.random.split(k)
    s = st(s, jax.random.uniform(kk, (env.action_size,), minval=-1, maxval=1))
    if jp.isnan(s.data.qpos).any() or jp.isnan(s.reward):
        print(sys.argv[1], "NaN au pas", i); break
else: print(sys.argv[1], "OK 300 pas, z", float(s.data.qpos[2]))
