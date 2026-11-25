"""Vitesse reellement atteinte par une politique (env MJX officiel, 64 robots, cap aleatoire), par consigne."""
import os, sys
import numpy as np, jax, jax.numpy as jp
from mujoco_playground import registry
import terrain as T
from evaluate import policy_for, local_v

run, env_name = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "Go1JoystickRoughTerrain"
env = registry.load(env_name)
T.add_payload(env, "trunk" if "Go1" in env_name else "torso_link", 3.0)
if "Rough" in env_name:
    T.apply(env, T.slope_plane(0))
pol = policy_for(run, env_name, env)
n = 64
for vx in (0.0, 0.5, 1.0):
    st = jax.jit(jax.vmap(env.reset))(jax.random.split(jax.random.PRNGKey(1), n))
    cmd = jp.tile(jp.array([vx, 0.0, 0.0]), (n, 1))
    step = jax.jit(jax.vmap(env.step)); act = jax.jit(jax.vmap(lambda o, k: pol(o, k)[0]))
    vs = []
    for k in range(250):
        st = st.replace(info={**st.info, "command": cmd, "steps_until_next_cmd": jp.full((n,), 10000, jp.int32)})
        st = step(st, act(st.obs, jax.random.split(jax.random.PRNGKey(k), n)))
        if k > 100:
            vs.append(local_v(st.data)[:, 0])
    print(run, "consigne vx=%.1f -> vitesse avant moyenne %.3f m/s, chutes %d/%d" % (vx, float(np.mean(vs)), int(np.asarray(st.done).sum()), n), flush=True)
