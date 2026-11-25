#!/usr/bin/env bash
R=/mnt/c/Users/fredb/Documents/Hackathon/talos2/rl/run.sh
bash $R train.py --env G1JoystickFlatTerrain   --run g1_sol_plat    --terrain default --payload 3 --steps 100e6 --evals 8
bash $R train.py --env G1JoystickRoughTerrain  --run g1_ua_doux     --terrain ua_doux --payload 3 --steps 40e6 --evals 5 --restore g1_sol_plat
bash $R train.py --env G1JoystickRoughTerrain  --run g1_terrain_ua  --terrain ua      --payload 3 --steps 60e6 --evals 6 --restore g1_ua_doux
bash $R evaluate.py; echo TOUT_TERMINE
