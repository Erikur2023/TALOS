"""IMU (gyro) + odometrie jambes simules. Pas de GNSS : brouille."""
import numpy as np


class ImuOdometry:
    def __init__(self, p, seed=0):
        self.p = p
        self.rng = np.random.default_rng(20_000 + seed)
        self.bias = self.rng.normal(0, p.gyro_bias_sd)
        self.scale = 1.0 + self.rng.normal(0, p.odom_scale_err)

    def measure(self, v_cmd, w_true, dt):
        """Odometrie jambes = ce que les jambes 'croient' avoir parcouru (ne voit PAS le glissement)."""
        p = self.p
        self.bias += self.rng.normal(0, p.gyro_bias_walk * np.sqrt(dt))
        gyro = w_true + self.bias + self.rng.normal(0, p.gyro_noise)
        v_meas = v_cmd * self.scale + self.rng.normal(0, p.odom_noise)
        return v_meas, gyro


class DeadReckoning:
    """Estimation par integration seule (reference 'sans recalage')."""
    def __init__(self, x0, y0, yaw0):
        self.x, self.y, self.yaw = x0, y0, yaw0

    def step(self, v, gyro, dt, v_lat=0.0):
        self.yaw += gyro * dt
        self.x += v * np.cos(self.yaw) * dt
        self.y += v * np.sin(self.yaw) * dt
        if v_lat:
            self.x -= v_lat * np.sin(self.yaw) * dt
            self.y += v_lat * np.cos(self.yaw) * dt
