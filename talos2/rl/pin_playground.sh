#!/usr/bin/env bash
# La version 2026 de Playground a retire les taches 'locomotion' (joystick terrain accidente) : on epingle 0.0.5.
export PATH="$HOME/.local/bin:$PATH"
source ~/talos_rl/bin/activate
uv pip install -q "playground==0.0.5" 2>&1 | tail -3
cd /mnt/c/Users/fredb/Documents/Hackathon/talos2/rl && python probe.py 2>&1 | grep -v -i warn | tail -4
