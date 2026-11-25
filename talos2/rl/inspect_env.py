import inspect, sys
from mujoco_playground import registry
from mujoco_playground.config import locomotion_params
for name in sys.argv[1:]:
    env = registry.load(name)
    m = env.mj_model
    print("=====", name, type(env).__mro__[:3])
    print("attrs:", [a for a in vars(env) if "mj" in a.lower() or "model" in a.lower()])
    print("hfield:", m.nhfield, [(m.hfield_nrow[i], m.hfield_ncol[i], m.hfield_size[i]) for i in range(m.nhfield)])
    print("bodies:", [m.body(i).name for i in range(min(m.nbody, 6))], "mass", m.body_mass[:4])
    print("obs size", env.observation_size, "act", env.action_size, "ctrl_dt", env.dt)
    print("ppo:", dict(locomotion_params.brax_ppo_config(name)))
    print("dr:", registry.get_domain_randomizer(name))
    src = inspect.getsource(type(env)._get_obs)
    print(src[:3000])
