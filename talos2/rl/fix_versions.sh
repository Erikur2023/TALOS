#!/usr/bin/env bash
# Pile coherente avec MuJoCo Playground 0.0.5 (ete 2025) : MuJoCo / MJX 3.3, Brax 0.12, JAX 0.6 (CUDA 12, Blackwell).
# (L'installation par defaut tirait MuJoCo 4.0 + Brax 0.14 : la politique n'apprenait qu'a rester debout.)
export PATH="$HOME/.local/bin:$PATH"
source ~/talos_rl/bin/activate
uv pip install -q "jax[cuda12]==0.6.2" "brax==0.12.4" "mujoco==3.3.3" "mujoco-mjx==3.3.3" "flax==0.10.6" 2>&1 | tail -3
python -c "import jax, brax, mujoco; print('jax', jax.__version__, jax.devices(), 'brax', brax.__version__, 'mujoco', mujoco.__version__)"
