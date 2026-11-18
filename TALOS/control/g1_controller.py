"""
TALOS G1 Contrôleur Articulaire & Générateur de Postures Tactiques
Basé sur les spécifications officielles Unitree (unitree_sdk2 / unitree_rl_gym)
Équipe TALOS (#225 - ENSTA Brest / tutelle DGA)
"""

import numpy as np
import math

class TalosElasticHarness:
    """
    Harnais de sustentation officiel Unitree (ElasticBand) avec régulation d'attitude critique.
    """
    def __init__(self, target_z=0.795, stiffness_z=260.0, damping_z=55.0, kp_tilt=45.0, kd_tilt=8.0):
        self.target_z = target_z
        self.stiffness_z = stiffness_z
        self.damping_z = damping_z
        self.kp_tilt = kp_tilt
        self.kd_tilt = kd_tilt
        self.enabled = True

    def compute_forces_and_torques(self, current_pos, current_vel, rpy_deg, gyro_rad_s):
        if not self.enabled:
            return np.zeros(3), np.zeros(3)
        
        fz = self.stiffness_z * (self.target_z - current_pos[2]) - self.damping_z * current_vel[2]
        fy = -100.0 * current_pos[1] - 25.0 * current_vel[1]
        fz = np.clip(fz, -20.0, 380.0)
        force = np.array([0.0, fy, fz])

        roll_rad = math.radians(rpy_deg[0])
        pitch_rad = math.radians(rpy_deg[1])
        
        tau_x = -self.kp_tilt * roll_rad - self.kd_tilt * gyro_rad_s[0]
        tau_y = -self.kp_tilt * pitch_rad - self.kd_tilt * gyro_rad_s[1]
        tau_x = np.clip(tau_x, -35.0, 35.0)
        tau_y = np.clip(tau_y, -35.0, 35.0)
        
        torque = np.array([tau_x, tau_y, 0.0])
        return force, torque


class TalosG1Controller:
    MODE_STAND = "STAND"
    MODE_TACTICAL_READY = "TACTICAL_READY"
    MODE_POINT_TARGET = "POINT_TARGET"
    MODE_EMERGENCY_DAMP = "EMERGENCY_DAMP"

    def __init__(self, nu=29):
        self.nu = nu
        self.mode = self.MODE_STAND
        
        self.q_des = np.zeros(nu, dtype=float)
        self.dq_des = np.zeros(nu, dtype=float)
        self._set_stand_posture()

        self.kp_nominal = np.zeros(nu, dtype=float)
        self.kd_nominal = np.zeros(nu, dtype=float)
        self._init_gains()

        self.tau_cmd = np.zeros(nu, dtype=float)
        self.gait_phase = 0.0
        self.gait_freq = 0.9  # Hz
        
        self.harness = TalosElasticHarness()

    def _init_gains(self):
        # Gains officiels Unitree G1 (g1_low_level_example.py)
        # Legs (0..11) : [60, 60, 60, 100, 40, 40] * 2
        self.kp_nominal[0:12] = [80.0, 60.0, 60.0, 110.0, 40.0, 40.0] * 2
        self.kd_nominal[0:12] = [2.0, 2.0, 2.0, 4.0, 2.0, 2.0] * 2

        # Buste / Taille (12..14)
        self.kp_nominal[12:15] = [60.0, 40.0, 40.0]
        self.kd_nominal[12:15] = [2.0, 2.0, 2.0]

        # Bras (15..28) : 40 Nm/rad, 1.5 Nms/rad
        self.kp_nominal[15:29] = 40.0
        self.kd_nominal[15:29] = 1.5

    def _set_stand_posture(self):
        """Posture naturelle militaire (coudes fléchis à 75°, bras près du corps)."""
        self.q_des.fill(0.0)
        # Jambes : flexion légère de garde
        self.q_des[0] = -0.15   # L hip pitch
        self.q_des[3] = 0.30    # L knee
        self.q_des[4] = -0.15   # L ankle pitch
        
        self.q_des[6] = -0.15   # R hip pitch
        self.q_des[9] = 0.30    # R knee
        self.q_des[10] = -0.15  # R ankle pitch

        # Bras tactiques naturels (plus de pose zombie !)
        # Coudes fléchis à ~75° (1.3 rad), avant-bras orientés vers l'avant
        self.q_des[15] = 0.20   # L shoulder pitch
        self.q_des[16] = 0.20   # L shoulder roll
        self.q_des[18] = 1.25   # L elbow flex (naturel)
        
        self.q_des[22] = 0.20   # R shoulder pitch
        self.q_des[23] = -0.20  # R shoulder roll
        self.q_des[25] = 1.25   # R elbow flex (naturel)

    def set_mode(self, new_mode):
        self.mode = new_mode
        if new_mode == self.MODE_STAND:
            self._set_stand_posture()
        elif new_mode == self.MODE_TACTICAL_READY:
            self._set_stand_posture()
            # Garde haute d'intervention (coudes à 90°, mains prêtes)
            self.q_des[15] = -0.30  # L shoulder forward
            self.q_des[16] = 0.25   # L shoulder roll
            self.q_des[18] = 1.50   # L elbow 90°
            self.q_des[22] = -0.30  # R shoulder forward
            self.q_des[23] = -0.25  # R shoulder roll
            self.q_des[25] = 1.50   # R elbow 90°
        elif new_mode == self.MODE_POINT_TARGET:
            self._set_stand_posture()
            # Désignation tactique directe : bras droit tendu horizontalement vers la cible
            self.q_des[22] = -1.35  # R shoulder pitch horizontal droit devant
            self.q_des[23] = 0.00   # R shoulder roll neutre
            self.q_des[24] = 0.00   # R shoulder yaw neutre
            self.q_des[25] = 0.05   # R elbow tendu (visée)
            self.q_des[27] = -0.10  # R wrist pitch orienté vers la cible
        elif new_mode == self.MODE_EMERGENCY_DAMP:
            self.q_des.fill(0.0)

    def update_gait(self, vx, dt, high_clearance=False):
        if abs(vx) < 0.02 or self.mode == self.MODE_EMERGENCY_DAMP:
            return

        self.gait_phase += 2.0 * math.pi * self.gait_freq * dt
        s = math.sin(self.gait_phase)

        if high_clearance:
            swing_amp = 0.20 * (vx / 0.20)
            knee_amp = 0.55 * (vx / 0.20)
            ankle_amp = 0.18
        else:
            swing_amp = 0.25 * (vx / 0.25)
            knee_amp = 0.35 * (vx / 0.25)
            ankle_amp = 0.10

        # Jambes
        self.q_des[0] = -0.15 + swing_amp * s
        self.q_des[3] = 0.30 + knee_amp * max(0.0, s)
        self.q_des[4] = -0.15 - ankle_amp * s

        self.q_des[6] = -0.15 - swing_amp * s
        self.q_des[9] = 0.30 + knee_amp * max(0.0, -s)
        self.q_des[10] = -0.15 + ankle_amp * s

        # Balancement fluide des bras tout en maintenant les coudes fléchis
        if self.mode == self.MODE_STAND:
            self.q_des[15] = 0.20 - 0.15 * s
            self.q_des[22] = 0.20 + 0.15 * s

    def compute_torques(self, q_actual, dq_actual, watchdog_tripped=False):
        if watchdog_tripped or self.mode == self.MODE_EMERGENCY_DAMP:
            self.mode = self.MODE_EMERGENCY_DAMP
            kp = np.zeros(self.nu)
            kd = np.ones(self.nu) * 10.0
            self.tau_cmd = kp * (self.q_des - q_actual) + kd * (0.0 - dq_actual)
            return np.clip(self.tau_cmd, -40.0, 40.0)

        self.tau_cmd = self.kp_nominal * (self.q_des - q_actual) + self.kd_nominal * (self.dq_des - dq_actual)
        
        max_torques = np.full(self.nu, 70.0)
        max_torques[15:29] = 30.0
        return np.clip(self.tau_cmd, -max_torques, max_torques)
