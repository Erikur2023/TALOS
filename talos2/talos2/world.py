"""Monde simule : relief, boue, foret, anomalies, tranchee. Contient la VERITE TERRAIN."""
import numpy as np
from scipy.ndimage import gaussian_filter


def _noise(rng, shape, sigma):
    n = gaussian_filter(rng.standard_normal(shape), sigma, mode="wrap")
    return (n - n.mean()) / (n.std() + 1e-9)


def _dist_to_polyline(xy, pts):
    d = np.full(len(xy), np.inf)
    for a, b in zip(pts[:-1], pts[1:]):
        ab = b - a
        t = np.clip(((xy - a) @ ab) / (ab @ ab + 1e-9), 0, 1)
        d = np.minimum(d, np.hypot(*(xy - (a + t[:, None] * ab)).T))
    return d


class World:
    def __init__(self, seed=0, size=300, res=1.0, tracks=False):
        rng = np.random.default_rng(seed)
        self.seed, self.size, self.res = seed, float(size), float(res)
        n = int(size / res)
        self.n = n

        h = _noise(rng, (n, n), 18) * 4.0 + _noise(rng, (n, n), 6) * 1.0
        self.height = h - h.min()
        gy, gx = np.gradient(self.height, res)
        self.slope = np.hypot(gx, gy)
        self.gx, self.gy = gx, gy

        mud_field = _noise(rng, (n, n), 10)
        low = self.height < np.median(self.height)
        self.mud = (mud_field > 0.55) & low

        dens = np.clip((_noise(rng, (n, n), 25) + 0.1) * 0.55, 0, 1)

        # depart et but (opposes, avec alea)
        self.start = np.array([rng.uniform(15, 30), rng.uniform(15, 30)])
        self.goal = np.array([size - rng.uniform(25, 40), size - rng.uniform(25, 40)])

        # arbres
        cand = rng.uniform(3, size - 3, size=(16000, 2))
        p = dens[(cand[:, 1] / res).astype(int), (cand[:, 0] / res).astype(int)]
        keep = rng.random(len(cand)) < p * 0.9
        trees = cand[keep]
        for pt in (self.start, self.goal):
            trees = trees[np.hypot(*(trees - pt).T) > 5.0]
        radii = rng.uniform(0.15, 0.5, size=len(trees))
        self.trees = np.column_stack([trees, radii])           # x, y, r

        # anomalies (absentes de la carte) dans un couloir autour de la ligne depart-but
        d = self.goal - self.start
        L = np.linalg.norm(d)
        u, nrm = d / L, np.array([-d[1], d[0]]) / L
        an = []
        for _ in range(6):
            s = rng.uniform(0.15, 0.85) * L
            o = rng.uniform(-25, 25)
            c = self.start + u * s + nrm * o
            an.append([c[0], c[1], rng.uniform(1.0, 2.2)])
        self.anomalies = np.array(an)

        # tranchee (zigzag termine au but) : ou se trouve le groupe a ravitailler
        t0 = self.goal + np.array([6.0, 4.0])
        self.trench = np.array([t0, t0 + [-10, 0], t0 + [-10, -9], t0 + [-20, -9]])

        # chemins forestiers (optionnel, alea separe : les mondes sans chemins restent identiques)
        self.tracks = []
        if tracks:
            self._make_tracks(np.random.default_rng(70_000 + seed))
        # zone ou il y a des arbres (verite) : densite locale lissee -> 'foret' / 'clairiere'
        cnt = np.zeros((n, n))
        np.add.at(cnt, (np.clip((self.trees[:, 1] / res).astype(int), 0, n - 1),
                        np.clip((self.trees[:, 0] / res).astype(int), 0, n - 1)), 1.0)
        self.tree_density = gaussian_filter(cnt, 4.0 / res) * (1.0 / res ** 2)      # arbres / m2
        # tous les obstacles que le LiDAR peut toucher
        self.obst = np.vstack([self.trees, self.anomalies])
        self.obst_kind = np.r_[np.zeros(len(self.trees), int), np.ones(len(self.anomalies), int)]

    def _make_tracks(self, rng, width=3.0):
        """2 a 3 chemins forestiers (courbes lisses traversant la carte) ; on y abat les arbres."""
        L = self.size
        for k in range(int(rng.integers(2, 4))):
            a, b = rng.uniform(0.1, 0.9, 2) * L
            if k % 2 == 0:
                p0, p1 = np.array([0.0, a]), np.array([L, b])
            else:
                p0, p1 = np.array([a, 0.0]), np.array([b, L])
            s = np.linspace(0, 1, 60)[:, None]
            line = p0 + s * (p1 - p0)
            nrm = np.array([-(p1 - p0)[1], (p1 - p0)[0]]) / np.linalg.norm(p1 - p0)
            wig = gaussian_filter(rng.standard_normal(60), 6, mode="nearest")
            wig = wig / (np.abs(wig).max() + 1e-9) * rng.uniform(10, 30)
            pts = line + wig[:, None] * nrm
            self.tracks.append(pts)
            d = _dist_to_polyline(self.trees[:, :2], pts)
            self.trees = self.trees[d > width / 2 + self.trees[:, 2]]

    # ----- requetes -----
    def _ij(self, x, y):
        i = np.clip((np.asarray(y) / self.res).astype(int), 0, self.n - 1)
        j = np.clip((np.asarray(x) / self.res).astype(int), 0, self.n - 1)
        return i, j

    def height_at(self, x, y):
        i, j = self._ij(x, y)
        return self.height[i, j]

    def slope_at(self, x, y):
        i, j = self._ij(x, y)
        return self.slope[i, j]

    def grad_at(self, x, y):
        i, j = self._ij(x, y)
        return self.gx[i, j], self.gy[i, j]

    def mud_at(self, x, y):
        i, j = self._ij(x, y)
        return self.mud[i, j]

    def lidar_scan(self, x, y, yaw, n_rays, max_range, sigma, dropout, rng):
        """Coupe horizontale 360 deg. Retourne (angles_corps, portees). portee=max_range si rien."""
        ang = np.linspace(-np.pi, np.pi, n_rays, endpoint=False)
        dx, dy = np.cos(yaw + ang), np.sin(yaw + ang)
        ranges = np.full(n_rays, max_range)
        rel = self.obst[:, :2] - [x, y]
        dist = np.hypot(rel[:, 0], rel[:, 1])
        m = dist < max_range + self.obst[:, 2]
        if m.any():
            c, r = rel[m], self.obst[m, 2]
            t = dx[:, None] * c[None, :, 0] + dy[:, None] * c[None, :, 1]
            perp2 = (c[:, 0] ** 2 + c[:, 1] ** 2)[None, :] - t ** 2
            hit = (perp2 < r[None, :] ** 2) & (t > 0)
            th = np.where(hit, t - np.sqrt(np.maximum(r[None, :] ** 2 - perp2, 0)), np.inf)
            ranges = np.minimum(ranges, th.min(axis=1))
        ranges = np.minimum(ranges, max_range)
        noisy = ranges + rng.normal(0, sigma, n_rays) * (ranges < max_range)
        lost = rng.random(n_rays) < dropout
        noisy[lost] = max_range
        return ang, noisy

    def contact(self, x, y, robot_radius):
        """Obstacle en contact avec le robot ? Retourne (index ou -1, vecteur de repoussement)."""
        rel = np.array([x, y]) - self.obst[:, :2]
        d = np.hypot(rel[:, 0], rel[:, 1])
        pen = self.obst[:, 2] + robot_radius - d
        k = int(np.argmax(pen))
        if pen[k] > 0:
            return k, rel[k] / (d[k] + 1e-9) * pen[k]
        return -1, np.zeros(2)
