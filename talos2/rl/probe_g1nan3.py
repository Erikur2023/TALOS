import sys, functools, jax, jax.numpy as jp
from mujoco_playground import registry, wrapper
name = "G1JoystickFlatTerrain"; env = registry.load(name); N = 128
rf = registry.get_domain_randomizer(name)
r = functools.partial(rf, rng=jax.random.split(jax.random.PRNGKey(3), N)) if sys.argv[1] == "dr" else None
w = wrapper.wrap_for_brax_training(env, episode_length=1000, action_repeat=1, randomization_fn=r)
s = jax.jit(w.reset)(jax.random.split(jax.random.PRNGKey(0), N)); st = jax.jit(w.step); k = jax.random.PRNGKey(1)
o = s.obs['state'] if isinstance(s.obs, dict) else s.obs
print(sys.argv[1], "obs nan au reset", int(jp.isnan(o).any(1).sum()))
for i in range(300):
    k, kk = jax.random.split(k)
    s = st(s, jax.random.uniform(kk, (N, env.action_size), minval=-1, maxval=1))
    if jp.isnan(s.reward).any():
        print(sys.argv[1], "reward NaN pas", i, int(jp.isnan(s.reward).sum()), "envs"); break
else: print(sys.argv[1], "aucun NaN")
