"""Compare la politique Go1 : (1) env MJX officiel, (2) MuJoCo CPU avec le modele de l'env + notre executeur numpy."""
import os, sys, pickle
import numpy as np, jax, jax.numpy as jp, mujoco
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mujoco_playground import registry
import terrain as T
from talos2.mujoco_g1.quad import NpPolicy
from evaluate import policy_for

run = sys.argv[1] if len(sys.argv) > 1 else "go1_terrain_ua"
env = registry.load("Go1JoystickRoughTerrain")
T.add_payload(env, "trunk", 3.0); T.apply(env, T.slope_plane(0))
pol = policy_for(run, "Go1JoystickRoughTerrain", env)
npp = NpPolicy(os.path.join(os.path.dirname(__file__), "..", "results", "rl", run, "policy.npz"))
st = jax.jit(env.reset)(jax.random.PRNGKey(0))
cmd = jp.array([0.5, 0, 0])
x0 = float(st.data.qpos[0]); step = jax.jit(env.step)
for k in range(250):
    st = st.replace(info={**st.info, "command": cmd, "steps_until_next_cmd": 10000})
    a_j = pol(st.obs, jax.random.PRNGKey(k))[0]
    if k in (0, 100):
        a_n = npp(np.asarray(st.obs["state"]))
        print("k", k, "action jax", np.round(np.asarray(a_j)[:4], 3), "numpy", np.round(a_n[:4], 3))
        print("obs", np.round(np.asarray(st.obs["state"]), 2))
    st = step(st, a_j)
print("MJX : distance en 5 s", float(st.data.qpos[0]) - x0, "done", float(st.done))
# CPU avec le meme modele
m = env.mj_model; d = mujoco.MjData(m)
d.qpos[:] = m.keyframe("home").qpos; d.ctrl[:] = d.qpos[7:]
mujoco.mj_forward(m, d)
default = m.keyframe("home").qpos[7:].copy(); last = np.zeros(12)
imu = m.site("imu").id
def sens(n): i = m.sensor(n).id; a = m.sensor_adr[i]; return d.sensordata[a:a + m.sensor_dim[i]]
x0 = d.qpos[0]
for k in range(250):
    grav = d.site_xmat[imu].reshape(3, 3).T @ np.array([0, 0, -1.0])
    obs = np.concatenate([sens("local_linvel"), sens("gyro"), grav, d.qpos[7:] - default, d.qvel[6:], last, [0.5, 0, 0]])
    a = npp(obs); last = a
    d.ctrl[:] = default + 0.5 * a
    for _ in range(5):
        mujoco.mj_step(m, d)
print("CPU (modele env) : distance en 5 s", d.qpos[0] - x0)
