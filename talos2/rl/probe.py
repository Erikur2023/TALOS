import jax
print("jax", jax.__version__, jax.devices())
from mujoco_playground import registry
envs = registry.locomotion.ALL_ENVS
print([e for e in envs if "G1" in e or "Go" in e or "Rough" in e])
