import inspect, sys
from mujoco_playground import registry
for name in sys.argv[1:]:
    env = registry.load(name)
    cls = type(env)
    print("=====", name)
    print(inspect.getsource(cls.reset)[:1800])
    print(inspect.getsource(cls.step)[:2600])
    m = env.mj_model
    import mujoco
    for i in range(m.ngeom):
        if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_HFIELD:
            print("hfield geom pos", m.geom_pos[i], "name", m.geom(i).name)
    print("config", env._config)
    print("xml", getattr(env, "_xml_path", None))
    print(inspect.getsource(cls.__init__)[:1500])
