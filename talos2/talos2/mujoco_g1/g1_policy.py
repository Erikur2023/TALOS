"""Politique de marche G1 12 ddl pre-entrainee (unitree_rl_gym, BSD-3, (c) Unitree Robotics).
Boucle reprise de unitree_rl_gym/deploy/deploy_mujoco/deploy_mujoco.py et configs/g1.yaml (credit : Unitree),
reecrite en classe pour etre appelee par notre boucle de mission. Rien n'est modifie dans la politique."""
import os
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RL_GYM = os.path.join(ROOT, "unitree_rl_gym")
G1_XML = os.path.join(RL_GYM, "resources", "robots", "g1_description", "g1_12dof.xml")
POLICY = os.path.join(RL_GYM, "deploy", "pre_train", "g1", "motion.pt")

# configs/g1.yaml (unitree_rl_gym)
SIM_DT = 0.002
DECIM = 10
KPS = np.array([100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40], np.float32)
KDS = np.array([2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2], np.float32)
DEFAULT = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0], np.float32)
ANG_VEL_SCALE, DOF_POS_SCALE, DOF_VEL_SCALE, ACTION_SCALE = 0.25, 1.0, 0.05, 0.25
CMD_SCALE = np.array([2.0, 2.0, 0.25], np.float32)
NA, NOBS = 12, 47


def gravity_orientation(q):
    qw, qx, qy, qz = q
    return np.array([2 * (-qz * qx + qw * qy), -2 * (qz * qy + qw * qx), 1 - 2 * (qw * qw + qz * qz)])


class G1Walker:
    """Applique la politique a 50 Hz et le PD a 500 Hz sur un MjData dont qpos[7:19] = 12 articulations des jambes."""
    def __init__(self):
        torch.set_num_threads(1)
        self.policy = torch.jit.load(POLICY)
        self.action = np.zeros(NA, np.float32)
        self.target = DEFAULT.copy()
        self.obs = np.zeros(NOBS, np.float32)
        self.counter = 0

    def reset(self, d):
        d.qpos[7:19] = DEFAULT
        self.action[:] = 0
        self.target = DEFAULT.copy()
        self.counter = 0

    def step(self, m, d, cmd):
        """Un pas physique (SIM_DT). cmd = [vx, vy, wz] repere corps."""
        import mujoco
        d.ctrl[:NA] = (self.target - d.qpos[7:19]) * KPS - d.qvel[6:18] * KDS
        mujoco.mj_step(m, d)
        self.counter += 1
        if self.counter % DECIM == 0:
            o = self.obs
            o[:3] = d.qvel[3:6] * ANG_VEL_SCALE
            o[3:6] = gravity_orientation(d.qpos[3:7])
            o[6:9] = np.asarray(cmd, np.float32) * CMD_SCALE
            o[9:21] = (d.qpos[7:19] - DEFAULT) * DOF_POS_SCALE
            o[21:33] = d.qvel[6:18] * DOF_VEL_SCALE
            o[33:45] = self.action
            ph = (self.counter * SIM_DT) % 0.8 / 0.8
            o[45:47] = np.sin(2 * np.pi * ph), np.cos(2 * np.pi * ph)
            with torch.no_grad():
                self.action = self.policy(torch.from_numpy(o).unsqueeze(0)).numpy().squeeze()
            self.target = self.action * ACTION_SCALE + DEFAULT
