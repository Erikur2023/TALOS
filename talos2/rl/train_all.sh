#!/usr/bin/env bash
# Entrainements du comparatif : recompense mise en forme (train.py:SHAPING) + CURRICULUM plat -> UA doux -> UA complet.
# Contre-exemples conserves : go1_ua_direct (ni mise en forme ni curriculum), go1_plat_sans_mise_en_forme (150 M pas) :
# tous deux convergent vers 'rester debout'.
R=/mnt/c/Users/fredb/Documents/Hackathon/talos2/rl/run.sh
bash $R train.py --env Go1JoystickFlatTerrain --run go1_sol_plat   --terrain default --payload 3 --steps 40e6 --num-envs 8192 --evals 5
bash $R train.py --env Go1JoystickRoughTerrain --run go1_ua_doux    --terrain ua_doux --payload 3 --steps 40e6 --num-envs 8192 --evals 5 --restore go1_sol_plat
bash $R train.py --env Go1JoystickRoughTerrain --run go1_terrain_ua --terrain ua      --payload 3 --steps 60e6 --num-envs 8192 --evals 6 --restore go1_ua_doux
bash $R train.py --env G1JoystickFlatTerrain   --run g1_sol_plat    --terrain default --payload 3 --steps 100e6 --evals 8
bash $R train.py --env G1JoystickRoughTerrain  --run g1_ua_doux     --terrain ua_doux --payload 3 --steps 40e6 --evals 5 --restore g1_sol_plat
bash $R train.py --env G1JoystickRoughTerrain  --run g1_terrain_ua  --terrain ua      --payload 3 --steps 60e6 --evals 6 --restore g1_ua_doux
bash $R evaluate.py
echo TOUT_TERMINE
