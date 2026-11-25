#!/usr/bin/env bash
# Lance une commande python dans l'environnement d'entrainement WSL : bash rl/run.sh script.py args...
export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export MUJOCO_GL=egl
source ~/talos_rl/bin/activate
cd /mnt/c/Users/fredb/Documents/Hackathon/talos2/rl
python "$@" 2>&1 | grep -v -E "Warning|warn\("
