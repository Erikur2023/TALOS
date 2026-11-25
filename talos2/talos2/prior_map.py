"""Carte a priori (la 'memoire' du robot) : version INCOMPLETE et IMPARFAITE du monde reel.

Elle ne contient ni les anomalies recentes, ni ~30 % des arbres, a des positions decalees,
plus quelques faux objets. C'est ce qui rend le recalage non trivial.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt


class PriorMap:
    kind = "trees"
    def __init__(self, world, p, seed=0):
        rng = np.random.default_rng(10_000 + seed)
        self.size, self.res = world.size, p.prior_res
        t = world.trees
        keep = rng.random(len(t)) < p.prior_keep_trees
        kept = t[keep].copy()
        kept[:, :2] += rng.normal(0, p.prior_pos_sigma, size=(len(kept), 2))
        n_ghost = int(p.prior_ghost_trees * len(t))
        ghost = np.column_stack([rng.uniform(3, world.size - 3, (n_ghost, 2)),
                                 rng.uniform(0.15, 0.5, n_ghost)])
        self.trees = np.vstack([kept, ghost])

        self.n = int(world.size / self.res)
        occ = np.zeros((self.n, self.n), bool)
        for x, y, r in self.trees:                      # troncs rasterises en disques
            rr = int(np.ceil(r / self.res)) + 1
            ci, cj = int(y / self.res), int(x / self.res)
            i0, i1 = max(0, ci - rr), min(self.n, ci + rr + 1)
            j0, j1 = max(0, cj - rr), min(self.n, cj + rr + 1)
            ii, jj = np.mgrid[i0:i1, j0:j1]
            occ[i0:i1, j0:j1] |= np.hypot((ii + 0.5) * self.res - y, (jj + 0.5) * self.res - x) <= r
        # distance (m) a la SURFACE du plus proche objet cartographie, plafonnee
        self.dist = np.minimum(distance_transform_edt(~occ) * self.res, 6.0).astype(np.float32)

        # MNT approximatif (erreur) et boue partiellement connue
        from scipy.ndimage import gaussian_filter
        e = gaussian_filter(rng.standard_normal(world.height.shape), 8, mode='wrap')
        e = 0.5 * (e - e.mean()) / (e.std() + 1e-9)       # ASSUMPTION erreur MNT 0,5 m, longueur de correlation ~10 m
        self.height = world.height + e
        self.pgy, self.pgx = np.gradient(self.height, world.res)
        self.slope = world.slope
        self.mud = world.mud & (rng.random(world.mud.shape) > 0.2)
        self.world_res = world.res

    def dist_at(self, x, y):
        i = np.clip((np.asarray(y) / self.res).astype(int), 0, self.n - 1)
        j = np.clip((np.asarray(x) / self.res).astype(int), 0, self.n - 1)
        return self.dist[i, j]

    def known_dist_at(self, x, y):
        """Distance aux objets individuels connus de la carte (ici : les arbres cartographies)."""
        return self.dist_at(x, y)

    def raycast(self, px, py, theta, zmax, max_iter=60):
        """Portee attendue (m) pour des rayons partant de (px,py) selon theta (tableaux de meme forme).
        Sphere tracing sur le champ de distance de la carte a priori."""
        shape = np.shape(theta)
        px, py, th = (np.asarray(a, float).ravel() for a in np.broadcast_arrays(px, py, theta))
        dx, dy = np.cos(th), np.sin(th)
        t = np.zeros(px.size)
        hit = np.zeros(px.size, bool)
        act = np.arange(px.size)
        for _ in range(max_iter):
            if act.size == 0:
                break
            d = self.dist_at(px[act] + dx[act] * t[act], py[act] + dy[act] * t[act])
            h = d <= 0.08
            hit[act[h]] = True
            go = ~h
            t[act[go]] += np.maximum(d[go] - 0.03, 0.12)
            act = act[go & (t[act] < zmax)]
        t = np.where(hit, t, zmax)
        return np.minimum(t, zmax).reshape(shape)

    def mud_at(self, x, y):
        i = np.clip((np.asarray(y) / self.world_res).astype(int), 0, self.mud.shape[0] - 1)
        j = np.clip((np.asarray(x) / self.world_res).astype(int), 0, self.mud.shape[1] - 1)
        return self.mud[i, j]

    def grad_at(self, x, y):
        i = np.clip((np.asarray(y) / self.world_res).astype(int), 0, self.pgx.shape[0] - 1)
        j = np.clip((np.asarray(x) / self.world_res).astype(int), 0, self.pgx.shape[1] - 1)
        return self.pgx[i, j], self.pgy[i, j]
