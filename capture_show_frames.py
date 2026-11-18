"""Capture de belles frames du Fortnite Dance Show (scène boîte de nuit)."""
import os, sys, math
import numpy as np
import mujoco
from PIL import Image

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WORKSPACE)
from fortnite_dance import (NU, KP, KD, TAU_MAX, base_posture, DANCES, sample_dance, ElasticHarness)
from fortnite_dance_show import make_nightclub_scene, SHOW_SEQUENCE, beat_color

def capture_show(n_per_dance=4):
    model, data = make_nightclub_scene()
    harness = ElasticHarness()
    dance_data = {name: fn() for name, (_, fn) in DANCES.items()}

    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0., 0., 0.793]
    data.qpos[3:7] = [1., 0., 0., 0.]
    qi = base_posture()
    for i in range(NU):
        data.qpos[7+i] = qi[i]
    mujoco.mj_forward(model, data)

    dt = model.opt.timestep
    renderer = mujoco.Renderer(model, height=720, width=1280)

    sim_t = 0.0
    seq_t = 0.0
    seq_idx = 0
    current_dance, current_loops = SHOW_SEQUENCE[0]
    captured = 0

    total_steps = int(40.0 / dt)  # 40s de simu max

    for step in range(total_steps):
        sim_t += dt
        seq_t += dt
        kf, lT = dance_data[current_dance]
        if seq_t >= lT * current_loops:
            seq_idx += 1
            if seq_idx >= len(SHOW_SEQUENCE):
                break
            current_dance, current_loops = SHOW_SEQUENCE[seq_idx]
            seq_t = 0.0
            kf, lT = dance_data[current_dance]

        q_des = sample_dance(kf, lT, seq_t)
        q = data.qpos[7:7+NU]; dq = data.qvel[6:6+NU]
        tau = np.clip(KP*(q_des-q)+KD*(0.-dq), -TAU_MAX, TAU_MAX)
        data.ctrl[:] = tau
        harness.apply(model, data)
        mujoco.mj_step(model, data)
        harness.anchor(data)
        mujoco.mj_forward(model, data)

        # Capturer à intervalles réguliers dans chaque danse
        progress = (seq_t % lT) / lT
        if step % int((lT * current_loops / n_per_dance) / dt) < 1 and progress < 0.9:
            c = beat_color(sim_t, 120)
            model.vis.headlight.diffuse[:] = c
            model.vis.headlight.ambient[:] = c * 0.6
            renderer.update_scene(data)
            pixels = renderer.render()
            img = Image.fromarray(pixels)
            path = os.path.join(WORKSPACE, f"show_{current_dance}_{captured:02d}.png")
            img.save(path)
            print(f"  📸 {os.path.basename(path)}  @ {sim_t:.1f}s  beat_color={c.round(2)}")
            captured += 1

    renderer.close()
    print(f"\n✅ {captured} frames capturées")

if __name__ == "__main__":
    capture_show(n_per_dance=4)
