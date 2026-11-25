#!/usr/bin/env bash
R=/mnt/c/Users/fredb/Documents/Hackathon/talos2/rl/run.sh
bash $R train.py --env G1JoystickFlatTerrain --run diag_g1_defaut  --terrain default --payload 0 --no-shaping --steps 6e6 --evals 3
bash $R train.py --env G1JoystickFlatTerrain --run diag_g1_shaping --terrain default --payload 0 --steps 6e6 --evals 3
bash $R train.py --env G1JoystickFlatTerrain --run diag_g1_charge  --terrain default --payload 3 --no-shaping --steps 6e6 --evals 3
echo DIAG_FINI
