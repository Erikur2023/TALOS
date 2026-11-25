"""Le Go1 peut-il physiquement avancer dans l'env Playground ? Trot en boucle ouverte (MJX et MuJoCo CPU) + contacts."""
import numpy as np, jax, jax.numpy as jp, mujoco
from mujoco import mjx
from mujoco_playground import registry
env = registry.load("Go1JoystickFlatTerrain")
m = env.mj_model
print("mujoco", mujoco.__version__, "nconmax?", getattr(m.opt, "cone", None), "condim floor", m.geom_condim[m.geom("floor").id],
      "condim FR", m.geom_condim[m.geom("FR").id], "friction FR", m.geom_friction[m.geom("FR").id], "floor", m.geom_friction[m.geom("floor").id],
      "priority", m.geom_priority[m.geom("floor").id], m.geom_priority[m.geom("FR").id])
default = m.keyframe("home").qpos[7:]
def gait(t):
    ph = 2 * np.pi * 2.0 * t
    a = np.zeros(12)
    for leg, off in enumerate((0, np.pi, np.pi, 0)):          # FR FL RR RL : trot
        a[leg * 3 + 1] = 0.35 * np.sin(ph + off)
        a[leg * 3 + 2] = 0.45 * max(0, np.sin(ph + off + np.pi / 2))
    return a
# CPU
d = mujoco.MjData(m); d.qpos[:] = m.keyframe("home").qpos; mujoco.mj_forward(m, d)
for k in range(250):
    d.ctrl[:] = default + gait(k * 0.02)
    for _ in range(5):
        mujoco.mj_step(m, d)
print("CPU trot boucle ouverte : dx=%.2f dy=%.2f z=%.2f ncon=%d" % (d.qpos[0], d.qpos[1], d.qpos[2], d.ncon))
# MJX via env.step avec actions = gait/0.5
st = jax.jit(env.reset)(jax.random.PRNGKey(0))
q = st.data.qpos.at[3:7].set(jp.array([1.0, 0, 0, 0])).at[0:2].set(0.0)
st = st.replace(data=mjx.forward(env.mjx_model, st.data.replace(qpos=q, qvel=jp.zeros_like(st.data.qvel))))
step = jax.jit(env.step)
for k in range(250):
    st = step(st, jp.array(gait(k * 0.02) / 0.5))
print("MJX trot boucle ouverte : dx=%.2f dy=%.2f z=%.2f done=%s" % (float(st.data.qpos[0]), float(st.data.qpos[1]), float(st.data.qpos[2]), bool(st.done)))
print("rewards", {k: round(float(v), 3) for k, v in st.metrics.items() if k.startswith("reward/")})
