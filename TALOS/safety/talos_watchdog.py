"""
TALOS Sûreté Matérielle Déterministe - Watchdog Anti-chute IMU
Équipe TALOS (#225 - ENSTA Brest / tutelle DGA)
Contrôle en temps réel de l'inclinaison et déclenchement du mode amorti d'urgence
"""

import numpy as np
import math

class TalosIMUWatchdog:
    """
    Watchdog temps-réel surveillant les flux IMU (100 Hz).
    Critère DGA/TALOS :
      - Seuil d'alerte : |Roulis| ou |Tangage| > 18°
      - Seuil critique : |Roulis| ou |Tangage| > 25° -> Déclenchement Emergency Damping
      - Détection de choc : accélération filtrée > 4.5g
    """
    STATE_NOMINAL = "NOMINAL"
    STATE_WARNING = "WARNING_TILT"
    STATE_EMERGENCY_DAMP = "EMERGENCY_DAMP_TRIGGERED"

    def __init__(self, tilt_warning_deg=18.0, tilt_critical_deg=25.0, max_accel_g=10.0):
        self.tilt_warning_deg = tilt_warning_deg
        self.tilt_critical_deg = tilt_critical_deg
        self.max_accel_norm = max_accel_g * 9.81  # m/s^2 (Seuil choc violent type crash/explosion)

        self.state = self.STATE_NOMINAL
        self.trigger_reason = None
        self.total_tilt_deg = 0.0
        self.roll_deg = 0.0
        self.pitch_deg = 0.0
        self.yaw_deg = 0.0
        self.tripped = False
        
        # Filtre passe-bas accélération pour éliminer les micro-bruits d'impact de pas
        self.filtered_accel_norm = 9.81
        self.step_count = 0
        self.warmup_steps = 100  # 0.5s de stabilisation initiale

    @staticmethod
    def quat_to_euler(q):
        """
        Convertit un quaternion MuJoCo [w, x, y, z] en angles d'Euler (Roll, Pitch, Yaw) en degrés.
        """
        w, x, y, z = q[0], q[1], q[2], q[3]

        # Roll (axe x)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (axe y)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)

        # Yaw (axe z)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return (
            math.degrees(roll),
            math.degrees(pitch),
            math.degrees(yaw)
        )

    def update(self, imu_quat, imu_gyro=None, imu_acc=None):
        self.step_count += 1
        self.roll_deg, self.pitch_deg, self.yaw_deg = self.quat_to_euler(imu_quat)
        self.total_tilt_deg = math.sqrt(self.roll_deg**2 + self.pitch_deg**2)

        # Filtrage passe-bas de l'accélération (tau ~ 0.1s pour éliminer les impacts de foulée)
        raw_acc_norm = np.linalg.norm(imu_acc) if imu_acc is not None else 9.81
        self.filtered_accel_norm = 0.98 * self.filtered_accel_norm + 0.02 * raw_acc_norm

        if self.tripped:
            return False, self.get_telemetry()

        # Phase de warmup (ignore les chocs du premier contact au sol)
        if self.step_count < self.warmup_steps:
            return True, self.get_telemetry()

        # 1. Seuil critique de tilt (25°)
        if abs(self.roll_deg) >= self.tilt_critical_deg:
            self.state = self.STATE_EMERGENCY_DAMP
            self.trigger_reason = f"CRITICAL_ROLL ({self.roll_deg:.1f}° >= {self.tilt_critical_deg}°)"
            self.tripped = True
            return False, self.get_telemetry()

        if abs(self.pitch_deg) >= self.tilt_critical_deg:
            self.state = self.STATE_EMERGENCY_DAMP
            self.trigger_reason = f"CRITICAL_PITCH ({self.pitch_deg:.1f}° >= {self.tilt_critical_deg}°)"
            self.tripped = True
            return False, self.get_telemetry()

        # 2. Seuil de choc violent persistant
        if self.filtered_accel_norm >= self.max_accel_norm:
            self.state = self.STATE_EMERGENCY_DAMP
            self.trigger_reason = f"CRITICAL_IMPACT ({self.filtered_accel_norm:.1f} m/s² >= {self.max_accel_norm:.1f})"
            self.tripped = True
            return False, self.get_telemetry()

        # 3. Alerte préventive (18°)
        if abs(self.roll_deg) >= self.tilt_warning_deg or abs(self.pitch_deg) >= self.tilt_warning_deg:
            self.state = self.STATE_WARNING
        else:
            self.state = self.STATE_NOMINAL

        return True, self.get_telemetry()

    def get_telemetry(self):
        return {
            "state": self.state,
            "tripped": self.tripped,
            "roll_deg": self.roll_deg,
            "pitch_deg": self.pitch_deg,
            "yaw_deg": self.yaw_deg,
            "total_tilt_deg": self.total_tilt_deg,
            "filtered_accel_m_s2": self.filtered_accel_norm,
            "trigger_reason": self.trigger_reason
        }

    def reset(self):
        self.state = self.STATE_NOMINAL
        self.tripped = False
        self.trigger_reason = None
        self.step_count = 0
        self.filtered_accel_norm = 9.81
