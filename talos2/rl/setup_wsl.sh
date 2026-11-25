#!/usr/bin/env bash
# Environnement d'entrainement (WSL Ubuntu + GPU NVIDIA) : JAX CUDA + MuJoCo MJX + MuJoCo Playground (Apache-2.0).
set -e
cd ~
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ -d ~/talos_rl ] || uv venv -p 3.12 ~/talos_rl
source ~/talos_rl/bin/activate
uv pip install -q "jax[cuda12]" playground mediapy
python -c "import jax; print('jax', jax.__version__, jax.devices())"
python -c "import mujoco_playground as mp; print('playground ok', len(mp.locomotion.ALL_ENVS), 'envs locomotion'); print([e for e in mp.locomotion.ALL_ENVS if 'Go' in e or 'G1' in e])"
