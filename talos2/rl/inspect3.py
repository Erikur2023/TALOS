import inspect, sys
from mujoco_playground import registry
from mujoco_playground._src.locomotion.go1 import base as go1base, joystick as go1j
from mujoco_playground._src.locomotion.go1 import go1_constants as C
print(inspect.getsource(go1j.Joystick._get_obs))
print(inspect.getsource(go1j.Joystick._post_init))
for n in ("get_gyro", "get_gravity", "get_local_linvel", "get_upvector"):
    if hasattr(go1base.Go1Env, n):
        print(inspect.getsource(getattr(go1base.Go1Env, n)))
print(inspect.getsource(go1base.Go1Env.__init__))
print({k: getattr(C, k) for k in dir(C) if k.isupper()})
print(inspect.getsource(go1j.Joystick._get_termination))
env = registry.load("Go1JoystickRoughTerrain")
m = env.mj_model
print("keyframes", [m.key(i).name for i in range(m.nkey)], "home qpos", m.key_qpos[0])
print("gainprm", m.actuator_gainprm[:2, :3], "biasprm", m.actuator_biasprm[:2, :3], "opt dt", m.opt.timestep)
print("sensors", [m.sensor(i).name for i in range(m.nsensor)])
