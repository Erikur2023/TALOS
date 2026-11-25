"""Localisation sans GPS : filtre a particules qui recale le LiDAR sur la carte a priori.

Etat par particule : x, y, cap, echelle d'odometrie (glissement), biais du gyroscope.
Le filtre APPREND donc le glissement et la derive du gyro quand la carte lui donne de l'information,
et son incertitude grossit honnetement quand il n'y a rien a voir (clairiere).
"""
import numpy as np


class ParticleFilter:
    def __init__(self, prior, p, x0, y0, yaw0, seed=0, init_xy_sd=1.0, init_yaw_sd=0.05):
        self.prior, self.p = prior, p
        self.rng = np.random.default_rng(30_000 + seed)
        n = p.n_particles
        self.s = np.empty((n, 6))      # x, y, cap, echelle sec, biais gyro, echelle boue
        self.s[:, 0] = x0 + self.rng.normal(0, init_xy_sd, n)
        self.s[:, 1] = y0 + self.rng.normal(0, init_xy_sd, n)
        self.s[:, 2] = yaw0 + self.rng.normal(0, init_yaw_sd, n)
        self.s[:, 3] = 1.0 + self.rng.normal(0, 0.03, n)
        self.s[:, 5] = np.clip(0.88 + self.rng.normal(0, 0.10, n), 0.5, 1.2)   # glissement en boue : inconnu a priori
        self.s[:, 4] = self.rng.normal(0, 0.0004, n)
        self.w = np.full(n, 1.0 / n)
        self.last_n_inliers = 0
        self.travel_since_good = 0.0

    def predict(self, v_meas, gyro, dt, v_lat=0.0):
        s = self.s
        n = len(s)
        s[:, 3] = np.clip(s[:, 3] + self.rng.normal(0, 0.004 * np.sqrt(dt), n), 0.85, 1.15)
        s[:, 5] = np.clip(s[:, 5] + self.rng.normal(0, 0.02 * np.sqrt(dt), n), 0.5, 1.2)
        s[:, 4] += self.rng.normal(0, 5.0e-5 * np.sqrt(dt), n)
        in_mud = self.prior.mud_at(s[:, 0], s[:, 1])
        fac = np.where(in_mud, s[:, 5], s[:, 3])        # echelle sec / echelle boue : apprises
        v = v_meas * fac + self.rng.normal(0, 0.07 * (abs(v_meas) + 0.1), n)
        self.travel_since_good += abs(v_meas) * dt
        s[:, 2] += (gyro - s[:, 4]) * dt + self.rng.normal(0, 0.003 * np.sqrt(dt), n)
        s[:, 0] += v * np.cos(s[:, 2]) * dt
        s[:, 1] += v * np.sin(s[:, 2]) * dt
        if v_lat:                                       # odometrie laterale (G1 : derive en marchant / en pivotant)
            s[:, 0] -= v_lat * np.sin(s[:, 2]) * dt
            s[:, 1] += v_lat * np.cos(s[:, 2]) * dt

    def update(self, ang, ranges):
        """Modele de faisceau : portee attendue (lancer de rayons sur la carte) contre portee mesuree.
        Un arbre cartographie qui n'est pas vu penalise la particule ; un obstacle non cartographie
        (mesure plus courte que l'attendu) est tolere : la carte est incomplete, pas fausse."""
        p = self.p
        if p.pf_mode == "endpoint":
            return self._update_endpoint(ang, ranges)
        if p.pf_mode == "density":
            return self._update_density(ang, ranges)
        zmax = p.lidar_range
        K = p.pf_rays_used
        sel = np.linspace(0, len(ang) - 1, K).astype(int)
        a, z = ang[sel], ranges[sel]
        theta = self.s[:, 2:3] + a[None, :]
        zexp = self.prior.raycast(self.s[:, 0:1], self.s[:, 1:2], theta, zmax)
        is_max = z >= zmax - 0.5
        exp_max = zexp >= zmax - 0.5
        gauss = np.exp(-0.5 * ((z[None, :] - zexp) / p.pf_sigma_hit) ** 2)
        short = (z[None, :] < zexp - 0.5).astype(float)
        lik_hit = 0.55 * gauss + 0.25 * short + p.pf_w_rand * 0.2
        lik_max = np.where(exp_max, 0.7, p.pf_unseen_lik)            # mesure 'rien' : coherent si la carte est vide aussi
        lik = np.where(is_max[None, :], lik_max, lik_hit)
        ll = np.log(lik).sum(axis=1) * p.pf_temper
        ll -= ll.max()
        self.w *= np.exp(ll)
        self.w += 1e-300
        self.w /= self.w.sum()
        self.last_n_inliers = int(((gauss > 0.5) & ~is_max[None, :] & ~exp_max).any(axis=0).sum())
        if self.last_n_inliers >= 3:
            self.travel_since_good = 0.0
        if 1.0 / np.sum(self.w ** 2) < 0.5 * len(self.w):
            self._resample()

    def update_heading(self, yaw_meas, sigma):
        """Cap mesure par un compas magnetique (bruit + perturbation locale, voir talos_app/mission.py)."""
        d = (self.s[:, 2] - yaw_meas + np.pi) % (2 * np.pi) - np.pi
        self.w *= np.exp(-0.5 * (d / sigma) ** 2)
        self.w += 1e-300
        self.w /= self.w.sum()
        if 1.0 / np.sum(self.w ** 2) < 0.5 * len(self.w):
            self._resample()

    def update_relief(self, along_m, across_m):
        """Pente mesuree (centrale inertielle : tangage/roulis) comparee a la pente du MNT embarque."""
        p = self.p
        gx, gy = self.prior.grad_at(self.s[:, 0], self.s[:, 1])
        c, s = np.cos(self.s[:, 2]), np.sin(self.s[:, 2])
        al, ac = gx * c + gy * s, -gx * s + gy * c
        ll = -0.5 * ((along_m - al) ** 2 + (across_m - ac) ** 2) / p.pf_relief_sigma ** 2 * p.pf_relief_temper
        ll -= ll.max()
        self.w *= np.exp(ll)
        self.w += 1e-300
        self.w /= self.w.sum()
        if 1.0 / np.sum(self.w ** 2) < 0.5 * len(self.w):
            self._resample()

    def _update_endpoint(self, ang, ranges):
        """Variante : champ de vraisemblance sur les points d'impact uniquement (pas d'info 'espace libre')."""
        p = self.p
        idx = np.flatnonzero(ranges < p.lidar_range - 0.5)
        self.last_n_inliers = 0
        if len(idx) < 3:
            return
        if len(idx) > p.pf_rays_used:
            idx = self.rng.choice(idx, p.pf_rays_used, replace=False)
        a, r = ang[idx], ranges[idx]
        ex = self.s[:, 0:1] + r[None, :] * np.cos(self.s[:, 2:3] + a[None, :])
        ey = self.s[:, 1:2] + r[None, :] * np.sin(self.s[:, 2:3] + a[None, :])
        d = self.prior.dist_at(ex, ey)
        lik = p.pf_w_hit * np.exp(-0.5 * (d / p.pf_sigma_hit) ** 2) + p.pf_w_rand
        ll = np.log(lik).sum(axis=1) * p.pf_temper
        ll -= ll.max()
        self.w *= np.exp(ll)
        self.w += 1e-300
        self.w /= self.w.sum()
        self.last_n_inliers = int((d[self.w.argmax()] < 0.8).sum())
        if self.last_n_inliers >= 3:
            self.travel_since_good = 0.0
        if 1.0 / np.sum(self.w ** 2) < 0.5 * len(self.w):
            self._resample()

    def _update_density(self, ang, ranges, step=2.0):
        """Carte sans objets (OSM) : modele de densite. Chaque rayon est un processus de Poisson de taux lambda(x)
        (foret / clairiere / chemin selon la carte). log L = -integrale(lambda) [+ log lambda(impact) si impact].
        Contrairement au modele 'point d'impact', un long rayon libre en foret cartographiee est penalise."""
        p = self.p
        K = min(p.pf_rays_used, 24)
        sel = np.linspace(0, len(ang) - 1, K).astype(int)
        a, r = ang[sel], ranges[sel]
        hit = r < p.lidar_range - 0.5
        S = int(np.ceil(p.lidar_range / step))
        sd = (np.arange(S) + 0.5) * step                                 # abscisses le long du rayon
        th = self.s[:, 2:3] + a[None, :]                                 # N x K
        c, s_ = np.cos(th)[:, :, None], np.sin(th)[:, :, None]
        X = self.s[:, 0, None, None] + c * sd[None, None, :]
        Y = self.s[:, 1, None, None] + s_ * sd[None, None, :]
        lam = self.prior.lam_at(X, Y)                                    # N x K x S
        inside = (sd[None, :] < r[:, None])                              # K x S
        integ = (lam * inside[None]).sum(axis=2) * step
        k = 1.0
        if p.pf_lam_calib:
            # auto-etalonnage de la densite : la carte donne OU est la vegetation, pas combien de troncs par m.
            # Nombre d'impacts observes / attendus (sous l'estimee courante), moyenne glissante -> facteur d'echelle.
            exp_hits = float(self.w @ integ.sum(axis=1))
            self.lam_obs = 0.97 * getattr(self, "lam_obs", 0.0) + float(hit.sum())
            self.lam_exp = 0.97 * getattr(self, "lam_exp", 0.0) + exp_hits
            if self.lam_exp > 20:
                k = float(np.clip(self.lam_obs / self.lam_exp, 0.25, 4.0))
            self.lam_k = k
        ll = -integ * k
        if hit.any():
            lam_end = self.prior.lam_at(self.s[:, 0:1] + r[None, :] * np.cos(th), self.s[:, 1:2] + r[None, :] * np.sin(th))
            ll = ll + np.where(hit[None, :], np.log(lam_end * k + 1e-4), np.log(0.02))
        ll = np.maximum(ll, -6.0).sum(axis=1) * p.pf_temper            # plancher par rayon : objets non cartographies
        ll -= ll.max()
        self.w *= np.exp(ll)
        self.w += 1e-300
        self.w /= self.w.sum()
        self.last_n_inliers = int(hit.sum())
        if 1.0 / np.sum(self.w ** 2) < 0.5 * len(self.w):
            self._resample()

    def _resample(self):
        n = len(self.w)
        pos = (self.rng.random() + np.arange(n)) / n
        i = np.minimum(np.searchsorted(np.cumsum(self.w), pos), n - 1)
        self.s = self.s[i].copy()
        self.s[:, 0:2] += self.rng.normal(0, 0.10, (n, 2))
        self.s[:, 2] += self.rng.normal(0, 0.01, n)
        self.s[:, 3] += self.rng.normal(0, 0.002, n)
        self.s[:, 5] += self.rng.normal(0, 0.01, n)
        self.s[:, 4] += self.rng.normal(0, 1e-5, n)
        # particules d'injection : filet de securite si la derive depasse le nuage (MCL augmente)
        k = max(2, int(0.04 * n))
        ex, ey, eyaw, sg, sc, bi = self._mean()
        sd = min(1.0 + 0.12 * self.travel_since_good, self.p.pf_inject_sd_max)
        idx = self.rng.choice(n, k, replace=False)
        self.s[idx, 0] = ex + self.rng.normal(0, sd, k)
        self.s[idx, 1] = ey + self.rng.normal(0, sd, k)
        self.s[idx, 2] = eyaw + self.rng.normal(0, 0.05 + 0.01 * sd, k)
        self.s[idx, 3] = np.clip(sc + self.rng.normal(0, 0.02, k), 0.85, 1.15)
        self.s[idx, 5] = np.clip(self.s[:, 5].mean() + self.rng.normal(0, 0.08, k), 0.5, 1.2)
        self.s[idx, 4] = bi + self.rng.normal(0, 1e-4, k)
        self.w = np.full(n, 1.0 / n)

    def _mean(self):
        w, s = self.w, self.s
        x = float(np.sum(w * s[:, 0])); y = float(np.sum(w * s[:, 1]))
        yaw = float(np.arctan2(np.sum(w * np.sin(s[:, 2])), np.sum(w * np.cos(s[:, 2]))))
        var = np.sum(w * ((s[:, 0] - x) ** 2 + (s[:, 1] - y) ** 2))
        return x, y, yaw, float(np.sqrt(var / 2)), float(np.sum(w * s[:, 3])), float(np.sum(w * s[:, 4]))

    def estimate(self):
        return self._mean()
