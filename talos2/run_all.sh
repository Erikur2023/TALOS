#!/usr/bin/env bash
# Relance TOUS les chiffres publies dans README.md (environ 1 h 30 sur 16 coeurs).
# Graines : 100-115 = jeu d'evaluation historique ; 400-431 = jeu jamais regarde pendant le developpement.
set -e
cd "$(dirname "$0")"
P=${PROCS:-14}
python -m talos2.mujoco_g1.calib --out results/mujoco_calib.json
python -m talos2.benchmark --seeds 100 116 --delivery --procs $P --out results/final_point
python -m talos2.benchmark --seeds 400 432 --delivery --procs $P --out results/final_point_holdout
python -m talos2.benchmark --seeds 100 116 --tracks --procs $P --out results/final_tracks_treemap
python -m talos2.benchmark --seeds 100 116 --tracks --map osm --procs $P --out results/final_tracks_osm
python -m talos2.mujoco_g1.benchmark --seeds 100 116 --relief 0.4 --procs $P --out results/final_mujoco
python -m talos2.mujoco_g1.benchmark --seeds 100 116 --relief 0.4 --mode lidar3d --no-dr --procs $P --out results/final_mujoco_lidar3d
python -m talos2.plots results/final_point
# terrain reel (Ukraine) : telechargement (cache data/real/), puis 2 zones x 12 departs x 5 niveaux de carte
python -m talos2.real_fetch --site serebryansky
python -m talos2.real_fetch --site robotyne
python -m talos2.real_benchmark --procs $P --out results/real
python -m talos2.make_ihm --seeds 105 111
python -m talos_app.datasets   # application C2 : donnees region
