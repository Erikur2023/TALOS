"""LiDAR 3D simule (type Livox MID-360 du G1) + segmentation sol / obstacles + pente locale.

Modele (ASSUMPTION, a recaler sur la fiche technique) : 360 deg en azimut (72 colonnes), 8 nappes d'elevation
de -52 a +7 deg (champ vertical 59 deg du MID-360, capteur retourne), portee 25 m (on limite comme le 2D), bruit 2 cm,
3 % de pertes. Le capteur voit le SOL (hfield) et les obstacles : il n'a pas l'etiquette, il doit segmenter.
Le corps du robot est exclu (groupes de geoms) : sur le vrai robot il faut un masque d'auto-occultation.

Sorties pour le Navigator (meme interface que le LiDAR 2D) :
  - pseudo-scan 2D : pour chaque azimut, portee horizontale du plus proche point 'obstacle' (0,25 a 2,0 m au-dessus du sol) ;
  - pente le long / en travers du cap, estimee par un plan ajuste sur les points 'sol' a moins de 6 m
    (remplace 'pente vraie + bruit' de l'etape 2).
"""
import numpy as np
import mujoco
from . import scene as S

# MID-360 : champ vertical 59 deg (-7..+52). Sur le G1 il est monte retourne dans la tete (ASSUMPTION a verifier
# sur le robot) -> champ -52..+7 deg : il voit le sol proche, indispensable pour la pente et la segmentation.
ELEV_DEG = np.array([-52.0, -35.0, -22.0, -13.0, -7.0, -3.0, 1.0, 7.0])
N_AZ = 72


class Lidar3D:
    def __init__(self, max_range=25.0, sigma=0.02, dropout=0.03, ground_band=0.25, obst_top=2.0, max_ground_slope=0.3):
        az = np.linspace(-np.pi, np.pi, N_AZ, endpoint=False)
        self.az, self.el = np.meshgrid(az, np.radians(ELEV_DEG))          # (n_el, n_az)
        self.az, self.el = self.az.ravel(), self.el.ravel()
        self.n = self.az.size
        self.max_range, self.sigma, self.dropout = max_range, sigma, dropout
        self.ground_band, self.obst_top, self.max_ground_slope = ground_band, obst_top, max_ground_slope
        self.gg = np.zeros(6, np.uint8); self.gg[S.G_GROUND] = 1; self.gg[S.G_OBST] = 1
        self.gid = np.zeros(self.n, np.int32); self.dist = np.zeros(self.n)
        self.stats = {"scans": 0, "tp": 0, "fp": 0, "fn": 0, "slope_err": []}
        self.last = None

    def cloud(self, m, d, origin, yaw, rng):
        """Nuage en repere monde (capteur stabilise en roulis/tangage par la centrale : ASSUMPTION)."""
        a = yaw + self.az
        vec = np.c_[np.cos(self.el) * np.cos(a), np.cos(self.el) * np.sin(a), np.sin(self.el)]
        mujoco.mj_multiRay(m, d, np.asarray(origin, float), vec.ravel(), self.gg, 1, -1, self.gid, self.dist, None,
                           self.n, self.max_range)
        r = self.dist.copy()
        ok = (r > 0) & (rng.random(self.n) > self.dropout)
        r = r + rng.normal(0, self.sigma, self.n)
        pts = np.asarray(origin) + vec * r[:, None]
        return pts[ok], self.gid[ok].copy(), self.az[ok]

    def process(self, M, rng):
        """Retourne (angles, portees) pseudo-2D, (pente_long, pente_travers) et met a jour les statistiques."""
        x, y, yaw = M.pose
        origin = np.array([x, y, float(M.d.qpos[2]) + 0.45])
        pts, gid, az = self.cloud(M.m, M.d, origin, yaw, rng)
        rel = pts - origin
        hd = np.hypot(rel[:, 0], rel[:, 1])
        # --- sol : plan robuste sur les points proches et bas (2 passes : moindres carres puis rejet) ---
        near = hd < 6.0
        cand = near & (rel[:, 2] < -0.6)
        plane = None
        if cand.sum() >= 6:
            A = np.c_[rel[cand, 0], rel[cand, 1], np.ones(cand.sum())]
            z = rel[cand, 2]
            for _ in range(2):
                coef, *_ = np.linalg.lstsq(A, z, rcond=None)
                res = np.abs(A @ coef - z)
                keep = res < max(0.1, 2.5 * np.median(res))
                A, z = A[keep], z[keep]
                if len(z) < 6:
                    break
            plane = coef
        # hauteur au-dessus du sol : plan local pres du robot, au-dela on prend le point le plus bas de la colonne
        if plane is not None:
            zg = rel[:, 0] * plane[0] + rel[:, 1] * plane[1] + plane[2]
        else:
            zg = np.full(len(rel), -1.25)
        # loin du plan : filtre de sol radial par colonne d'azimut (du plus proche au plus lointain, un point est
        # 'sol' si la montee depuis le dernier point sol reste sous une pente plausible)
        col = np.round(az / (2 * np.pi / N_AZ)).astype(int) % N_AZ
        h_ray = np.zeros(len(rel))
        for c in np.unique(col):
            idx = np.flatnonzero(col == c)
            idx = idx[np.argsort(hd[idx])]
            g_r, g_z = 0.0, (plane[2] if plane is not None else -1.25)
            for i in idx:
                if hd[i] < 6.0 and plane is not None:
                    hh = rel[i, 2] - zg[i]
                    if hh < self.ground_band:
                        g_r, g_z = hd[i], zg[i] + hh
                    h_ray[i] = hh
                    continue
                rise = rel[i, 2] - g_z
                if rise <= max(0.15, self.max_ground_slope * (hd[i] - g_r)):
                    g_r, g_z = hd[i], rel[i, 2]
                    h_ray[i] = 0.0
                else:
                    h_ray[i] = rise - self.max_ground_slope * (hd[i] - g_r)
        zg = rel[:, 2] - h_ray
        h = rel[:, 2] - zg
        is_obst = (h > self.ground_band) & (h < self.obst_top)
        # --- statistiques de segmentation (verite = geom touche, connue seulement en simulation) ---
        truth = np.isin(gid, M.info["obst_geom"])
        band = (h < self.obst_top)                           # on ne juge que la tranche utile
        st = self.stats
        st["scans"] += 1
        st["tp"] += int((is_obst & truth & band).sum()); st["fp"] += int((is_obst & ~truth).sum())
        st["fn"] += int((~is_obst & truth & band & (h > 0.05)).sum())
        # --- pseudo-scan 2D ---
        rr = np.full(N_AZ, self.max_range)
        col = np.round((az) / (2 * np.pi / N_AZ)).astype(int) % N_AZ
        np.minimum.at(rr, col[is_obst], hd[is_obst])
        ang = np.linspace(-np.pi, np.pi, N_AZ, endpoint=False)
        rr = np.r_[rr[N_AZ // 2:], rr[:N_AZ // 2]]           # colonnes indexees depuis -pi comme le 2D
        # --- pente ---
        slope = (None, None)
        if plane is not None:
            gx, gy = plane[0], plane[1]
            al, ac = gx * np.cos(yaw) + gy * np.sin(yaw), -gx * np.sin(yaw) + gy * np.cos(yaw)
            slope = (float(al), float(ac))
            tx, ty = M.w.grad_at(x, y)
            st["slope_err"].append(float(np.hypot(al - (tx * np.cos(yaw) + ty * np.sin(yaw)),
                                                  ac - (-tx * np.sin(yaw) + ty * np.cos(yaw)))))
        self.last = {"n_pts": int(len(pts)), "n_obst": int(is_obst.sum()), "plane": plane}
        return (ang, rr), slope

    def summary(self):
        st = self.stats
        tp, fp, fn = st["tp"], st["fp"], st["fn"]
        se = np.array(st["slope_err"]) if st["slope_err"] else np.zeros(1)
        return {"obst_precision": round(tp / max(1, tp + fp), 3), "obst_recall": round(tp / max(1, tp + fn), 3),
                "slope_err_median": round(float(np.median(se)), 3), "slope_err_p95": round(float(np.percentile(se, 95)), 3)}


def attach_lidar3d():
    """kwargs pour mujoco_g1.run.run : remplace le LiDAR 2D et la pente 'verite + bruit' par le LiDAR 3D."""
    state = {}

    def scan_fn(M):
        if "l" not in state:
            state["l"] = Lidar3D(max_range=M.p.lidar_range)
            M.lidar3d = state["l"]
        scan, state["slope"] = state["l"].process(M, M.rng)
        return scan

    def slope_fn(M):
        s = state.get("slope", (None, None))
        return s if s[0] is not None else M.slope_meas()

    return {"scan_fn": scan_fn, "slope_fn": slope_fn}
