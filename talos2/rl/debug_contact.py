import numpy as np, mujoco
from mujoco_playground import registry
env = registry.load("Go1JoystickFlatTerrain")
m = env.mj_model
d = mujoco.MjData(m); d.qpos[:] = m.keyframe("home").qpos; d.ctrl[:] = d.qpos[7:]
mujoco.mj_forward(m, d)
print("z0", d.qpos[2], "ngeom", m.ngeom, "opt", m.opt.iterations, m.opt.ls_iterations, m.opt.integrator, "disableflags", m.opt.disableflags)
for k in range(500):
    mujoco.mj_step(m, d)
    if k % 100 == 0:
        print(k, "z=%.3f ncon=%d" % (d.qpos[2], d.ncon), [(m.geom(c.geom1).name, m.geom(c.geom2).name, round(c.dist, 3)) for c in d.contact[:d.ncon]][:4])
for g in ("floor", "FR", "FL", "RR", "RL"):
    i = m.geom(g).id
    print(g, "type", m.geom_type[i], "contype", m.geom_contype[i], "conaff", m.geom_conaffinity[i], "size", m.geom_size[i], "body", m.body(m.geom_bodyid[i]).name)
print("site FR z", d.site_xpos[m.site("FR").id])
