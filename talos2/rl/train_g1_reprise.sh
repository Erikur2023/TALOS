#!/usr/bin/env bash
# Reprise apres la mise en veille du PC : 8,8 M pas deja faits (checkpoint), on termine les ~52 M restants.
R=/mnt/c/Users/fredb/Documents/Hackathon/talos2/rl/run.sh
bash $R train.py --env G1JoystickRoughTerrain --run g1_terrain_ua --terrain ua --payload 3 --steps 52e6 --evals 13 --restore /mnt/c/Users/fredb/Documents/Hackathon/talos2/results/rl/g1_terrain_ua/reprise_8.8M.pkl
bash $R evaluate.py
echo TOUT_TERMINE
