"""Capture des frames des danses Fortnite en PNG pour prévisualisation."""
import os, sys, math, time
import numpy as np
import mujoco
import mujoco.viewer

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fortnite_dance import (G1_XML, NU, KP, KD, TAU_MAX, build_scene, base_posture,
                            DANCES, sample_dance, ElasticHarness)

def capture(dance_name, n_frames=8, out_prefix=None):
    model, data = build_scene()
    model.vis.global_.offwidth = 1920
    model.vis.global_.offheight = 1080
    harness = ElasticHarness()
    fn = DANCES[dance_name][1]
    kf, loop_T = fn()

    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0., 0., 0.793]
    data.qpos[3:7] = [1., 0., 0., 0.]
    qi = base_posture()
    for i in range(NU):
        data.qpos[7 + i] = qi[i]
    mujoco.mj_forward(model, data)

    dt = model.opt.timestep
    renderer = mujoco.Renderer(model, height=720, width=1280)

    # Caméra free via le viewer interne : on utilise une caméra fixe du modèle
    # ou on passe par renderer avec camera=-1 (free camera) qu'on configure via data.cam
    # Méthode simple : utiliser scene camera par défaut du renderer
    renderer.update_scene(data)  # caméra par défaut
    # On configure la caméra via les propriétés du modèle
    # Alternative : utiliser une caméra nommée du modèle si existe
    pass

    prefix = out_prefix or f"dance_{dance_name}"
    # Avancer la danse et capturer à intervalles réguliers
    total_time = loop_T
    steps_per_frame = int((total_time / n_frames) / dt)
    for f in range(n_frames):
        target_t = loop_T * f / n_frames
        # Simuler jusqu'à target_t
        while data.time < target_t:
            q_des = sample_dance(kf, loop_T, data.time)
            q = data.qpos[7:7+NU]; dq = data.qvel[6:6+NU]
            tau = np.clip(KP*(q_des-q)+KD*(0.-dq), -TAU_MAX, TAU_MAX)
            data.ctrl[:] = tau
            harness.apply(model, data)
            mujoco.mj_step(model, data)
            harness.anchor(data)
            mujoco.mj_forward(model, data)
        renderer.update_scene(data)
        pixels = renderer.render()
        from PIL import Image
        img = Image.fromarray(pixels)
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{prefix}_frame{f}.png")
        img.save(path)
        print(f"  📸 {path}")
    renderer.close()

if __name__ == "__main__":
    dances = sys.argv[1:] if len(sys.argv) > 1 else ["default", "floss", "takethel", "orangejustice", "bestmates"]
    for d in dances:
        print(f"\n🎬 Capture '{d}'...")
        capture(d, n_frames=6)
