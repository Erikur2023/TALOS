import jax, jax.numpy as jp
from mujoco_playground import registry
env = registry.load("G1JoystickFlatTerrain")
N=256
rs = jax.jit(jax.vmap(env.reset)); st = jax.jit(jax.vmap(env.step))
s = rs(jax.random.split(jax.random.PRNGKey(0), N)); k = jax.random.PRNGKey(1)
for i in range(400):
    k, kk = jax.random.split(k)
    s = st(s, jax.random.uniform(kk, (N, env.action_size), minval=-1, maxval=1))
    bad = {n: int(jp.isnan(v).sum()) for n, v in s.metrics.items() if jp.isnan(v).any()}
    if jp.isnan(s.reward).any() or bad or jp.isnan(s.obs['state'] if isinstance(s.obs,dict) else s.obs).any():
        print("pas", i, "reward nan", int(jp.isnan(s.reward).sum()), bad, "qpos nan", int(jp.isnan(s.data.qpos).any(1).sum())); break
else: print("aucun NaN sur", N, "x 400 pas")
