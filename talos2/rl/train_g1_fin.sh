#!/usr/bin/env bash
R=/mnt/c/Users/fredb/Documents/Hackathon/talos2/rl/run.sh
bash $R train.py --env G1JoystickRoughTerrain --run g1_terrain_ua --terrain ua --payload 3 --steps 60e6 --evals 15 --restore g1_ua_doux
bash $R evaluate.py
echo TOUT_TERMINE
