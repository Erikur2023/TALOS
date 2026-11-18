import os
import mujoco
import numpy as np
import torch

from run_sim import (
    KPS, KDS, DEFAULT_ANGLES, DOF_POS_SCALE, DOF_VEL_SCALE, ANG_VEL_SCALE,
    ACTION_SCALE, CMD_SCALE, get_gravity_orientation, SIMULATION_DT, CONTROL_DECIMATION
)

# Test XML with 6 steps of 4cm rise and chamfered lip
test_stairs_xml = """<mujoco model="stair_test_chamfer">
  <include file="g1_12dof.xml"/>
  <visual>
    <headlight diffuse="0.8 0.8 0.8" ambient="0.2 0.2 0.2"/>
  </visual>
  <worldbody>
    <light pos="2 0 4" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="5 5 0.05"/>
    
    <!-- Marches réalistes avec bord adouci (4cm de dénivelé par marche) -->
    <geom name="step1" type="box" size="0.16 0.8 0.020" pos="2.00 0.0 0.020" rgba="0.8 0.7 0.6 1"/>
    <geom name="lip1" type="capsule" size="0.020 0.8" pos="1.84 0.0 0.020" euler="1.5708 0 0" rgba="0.8 0.7 0.6 1"/>

    <geom name="step2" type="box" size="0.16 0.8 0.040" pos="2.32 0.0 0.040" rgba="0.7 0.6 0.5 1"/>
    <geom name="lip2" type="capsule" size="0.020 0.8" pos="2.16 0.0 0.040" euler="1.5708 0 0" rgba="0.7 0.6 0.5 1"/>

    <geom name="step3" type="box" size="0.16 0.8 0.060" pos="2.64 0.0 0.060" rgba="0.8 0.7 0.6 1"/>
    <geom name="lip3" type="capsule" size="0.020 0.8" pos="2.48 0.0 0.060" euler="1.5708 0 0" rgba="0.8 0.7 0.6 1"/>

    <geom name="step4" type="box" size="0.16 0.8 0.080" pos="2.96 0.0 0.080" rgba="0.7 0.6 0.5 1"/>
    <geom name="lip4" type="capsule" size="0.020 0.8" pos="2.80 0.0 0.080" euler="1.5708 0 0" rgba="0.7 0.6 0.5 1"/>

    <geom name="step5" type="box" size="0.16 0.8 0.100" pos="3.28 0.0 0.100" rgba="0.8 0.7 0.6 1"/>
    <geom name="lip5" type="capsule" size="0.020 0.8" pos="3.12 0.0 0.100" euler="1.5708 0 0" rgba="0.8 0.7 0.6 1"/>

    <!-- Palier supérieur de la maison (Mezzanine / 1er étage) -->
    <geom name="landing" type="box" size="0.70 0.8 0.120" pos="4.14 0.0 0.120" rgba="0.6 0.5 0.4 1"/>
  </worldbody>
</mujoco>"""

test_xml_path = "unitree_rl_gym/resources/robots/g1_description/test_stairs_tmp.xml"
with open(test_xml_path, "w", encoding="utf-8") as f:
    f.write(test_stairs_xml)

m = mujoco.MjModel.from_xml_path(test_xml_path)
d = mujoco.MjData(m)
policy = torch.jit.load("unitree_rl_gym/deploy/pre_train/g1/motion.pt").eval()

action = np.zeros(12, dtype=np.float32)
target_dof_pos = DEFAULT_ANGLES.copy()
obs = np.zeros(47, dtype=np.float32)
cmd = np.array([0.45, 0.0, 0.0], dtype=np.float32)

for s in range(int(11.0 / SIMULATION_DT)):
    tau = (target_dof_pos - d.qpos[7:]) * KPS - d.qvel[6:] * KDS
    d.ctrl[:] = tau
    mujoco.mj_step(m, d)
    
    if s % CONTROL_DECIMATION == 0:
        sim_time = s * SIMULATION_DT
        qj = (d.qpos[7:] - DEFAULT_ANGLES) * DOF_POS_SCALE
        dqj = d.qvel[6:] * DOF_VEL_SCALE
        quat = d.qpos[3:7]
        omega = d.qvel[3:6] * ANG_VEL_SCALE
        grav = get_gravity_orientation(quat)
        
        period = 0.8
        phase_clock = (sim_time % period) / period
        obs[:3] = omega
        obs[3:6] = grav
        obs[6:9] = cmd * CMD_SCALE
        obs[9:21] = qj
        obs[21:33] = dqj
        obs[33:45] = action
        obs[45:47] = np.array([np.sin(2*np.pi*phase_clock), np.cos(2*np.pi*phase_clock)], dtype=np.float32)
        
        with torch.no_grad():
            action = policy(torch.from_numpy(obs).unsqueeze(0)).numpy().squeeze()
            
        target_dof_pos = action * ACTION_SCALE + DEFAULT_ANGLES
        
        # Réflexe anti-chute adaptatif pour escaliers
        pitch_err = grav[0]
        roll_err = grav[1]
        if abs(pitch_err) > 0.03:
            target_dof_pos[0] -= pitch_err * 0.35
            target_dof_pos[4] -= pitch_err * 0.25
            target_dof_pos[6] -= pitch_err * 0.35
            target_dof_pos[10] -= pitch_err * 0.25
        if abs(roll_err) > 0.03:
            target_dof_pos[1] += roll_err * 0.28
            target_dof_pos[5] += roll_err * 0.16
            target_dof_pos[7] += roll_err * 0.28
            target_dof_pos[11] += roll_err * 0.16
            
    if s % 250 == 0:
        print(f"t={s*SIMULATION_DT:.1f}s | POS: X={d.qpos[0]:.2f}m, Z={d.qpos[2]:.2f}m | Pitch={np.degrees(-grav[0]):.1f}°")
        if d.qpos[2] < 0.35:
            print("Chute détectée !")
            break

print(f"Final POS: X={d.qpos[0]:.2f}m, Z={d.qpos[2]:.2f}m")
