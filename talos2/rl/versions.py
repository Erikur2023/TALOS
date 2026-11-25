import importlib.metadata as md
for p in ("brax", "jax", "jaxlib", "mujoco", "mujoco-mjx", "playground", "flax", "optax", "ml_collections"):
    try:
        print(p, md.version(p))
    except Exception:
        print(p, "-")
try:
    print("playground requires:", [r for r in md.requires("playground") if any(k in r for k in ("brax", "jax", "mujoco"))])
except Exception as e:
    print(e)
