"""
TALOS Odométrie Laser-Inertielle (Point-LIO + IMU 100 Hz)
Équipe TALOS (#225 - ENSTA Brest / tutelle DGA)
Estimation d'état 6-DOF continue en environnement GPS-denied
"""

import numpy as np
import time

class TalosOdometryEstimator:
    """
    Simulateur de fusion Point-LIO / IMU haute fréquence.
    Compense les oscillations et secousses de la démarche bipède du G1.
    Fournit la pose 3D (x, y, z), l'orientation (R, P, Y) et les vitesses linéaires/angulaires.
    """
    def __init__(self, init_pos=None, init_yaw=0.0):
        self.pos = np.array(init_pos if init_pos is not None else [0.0, 0.0, 0.79], dtype=float)
        self.vel = np.zeros(3, dtype=float)
        self.rpy = np.array([0.0, 0.0, init_yaw], dtype=float)
        self.last_time = time.perf_counter()
        
        # Statistiques et métriques
        self.total_distance = 0.0
        self.trajectory = []
        self.drift_correction_count = 0

    def update_from_sim(self, frame_pos, frame_vel, rpy, dt=0.005):
        """
        Met à jour l'estimation à partir du modèle physique MuJoCo
        avec modélisation du filtrage Point-LIO.
        """
        # Mise à jour avec filtrage passe-bas pour lisser les micro-impacts des talons
        alpha_pos = 0.95
        prev_pos = self.pos.copy()
        
        self.pos = alpha_pos * self.pos + (1 - alpha_pos) * np.array(frame_pos)
        self.vel = np.array(frame_vel)
        self.rpy = np.array(rpy)

        # Calcul distance parcourue (plan horizontal x-y)
        delta_dist = np.linalg.norm(self.pos[:2] - prev_pos[:2])
        self.total_distance += delta_dist

        # Enregistrement pour trajectoire
        if len(self.trajectory) == 0 or np.linalg.norm(self.pos - self.trajectory[-1]) > 0.05:
            self.trajectory.append(self.pos.copy())

        return self.get_state()

    def get_state(self):
        return {
            "x": float(self.pos[0]),
            "y": float(self.pos[1]),
            "z": float(self.pos[2]),
            "roll": float(self.rpy[0]),
            "pitch": float(self.rpy[1]),
            "yaw": float(self.rpy[2]),
            "vx": float(self.vel[0]),
            "vy": float(self.vel[1]),
            "vz": float(self.vel[2]),
            "total_distance_m": float(self.total_distance),
            "trajectory_points": len(self.trajectory)
        }
