"""Planification A* sur la carte a priori + obstacles decouverts. Retourne un chemin en metres."""
import heapq
import numpy as np
from scipy.ndimage import binary_dilation


class Planner:
    def __init__(self, prior, p, res=1.0):
        self.prior, self.p, self.res = prior, p, res
        self.n = int(prior.size / res)
        occ = np.zeros((self.n, self.n), bool)
        for x, y, r in prior.trees:
            i, j = int(y / res), int(x / res)
            if 0 <= i < self.n and 0 <= j < self.n:
                occ[i, j] = True
        k = int(np.ceil(p.plan_inflation / res)) + 1
        self.static_block = binary_dilation(occ, iterations=k)
        self.static_tight = binary_dilation(occ, iterations=max(1, k - 1))
        b = int(np.ceil(getattr(p, "plan_border", 0.0) / res))
        if b > 0:
            for m_ in (self.static_block, self.static_tight):
                m_[:b, :] = m_[-b:, :] = True
                m_[:, :b] = m_[:, -b:] = True
        self.dyn_block = np.zeros_like(self.static_block)
        sl = prior.slope[::int(res / prior.world_res)][:self.n, :self.n] if res != prior.world_res else prior.slope
        md = prior.mud
        self.cost = 1.0 + p.plan_slope_cost * np.clip(sl, 0, 0.6) + (p.plan_mud_cost - 1.0) * md
        if getattr(prior, "extra_cost", None) is not None:   # carte OSM : foret / chemins
            self.cost = np.maximum(0.5, self.cost + prior.extra_cost[:self.n, :self.n])
        smax = getattr(p, "plan_slope_max", 0.0)           # capacite du robot (G1 + politique plate : mesure MuJoCo)
        if smax:                                           # penalite forte (pas d'interdiction : la connectivite casse)
            self.cost = self.cost + p.plan_slope_penalty * np.clip(sl - smax, 0, None)

    def add_obstacle(self, x, y, radius=1.2):
        r = int(np.ceil(radius / self.res))
        ci, cj = int(y / self.res), int(x / self.res)
        for i in range(max(0, ci - r), min(self.n, ci + r + 1)):
            for j in range(max(0, cj - r), min(self.n, cj + r + 1)):
                if np.hypot(i - ci, j - cj) * self.res <= radius:
                    self.dyn_block[i, j] = True

    def plan(self, start_xy, goal_xy):
        path = self._plan(start_xy, goal_xy, self.static_block)
        return path if path is not None else self._plan(start_xy, goal_xy, self.static_tight)

    def _plan(self, start_xy, goal_xy, static):
        res, n = self.res, self.n
        blocked = static | self.dyn_block
        s = (int(start_xy[1] / res), int(start_xy[0] / res))
        g = (int(goal_xy[1] / res), int(goal_xy[0] / res))
        s = (min(max(s[0], 0), n - 1), min(max(s[1], 0), n - 1))
        g = (min(max(g[0], 0), n - 1), min(max(g[1], 0), n - 1))
        # but toujours libre ; autour du depart, les cellules bloquees restent franchissables mais tres cheres
        # (avant : liberees d'office sur 5 x 5 m -> on replanifiait a travers l'obstacle qui nous bloque)
        blocked = blocked.copy()
        blocked[max(0, g[0] - 2):g[0] + 3, max(0, g[1] - 2):g[1] + 3] = False
        cost = self.cost
        win = (slice(max(0, s[0] - 2), s[0] + 3), slice(max(0, s[1] - 2), s[1] + 3))
        if blocked[win].any():
            cost = self.cost.copy()
            cost[win] += self.p.plan_start_escape_cost * blocked[win]
            blocked[win] = False
        moves = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                 (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142)]

        def h(i, j):
            di, dj = abs(i - g[0]), abs(j - g[1])
            return (max(di, dj) + 0.4142 * min(di, dj)) * res

        openq = [(h(*s), 0.0, s)]
        best, came = {s: 0.0}, {}
        while openq:
            f, gc, cur = heapq.heappop(openq)
            if cur == g:
                break
            if gc > best.get(cur, 1e18):
                continue
            for di, dj, sc in moves:
                ni, nj = cur[0] + di, cur[1] + dj
                if not (0 <= ni < n and 0 <= nj < n) or blocked[ni, nj]:
                    continue
                if di and dj and (blocked[cur[0] + di, cur[1]] or blocked[cur[0], cur[1] + dj]):
                    continue
                ng = gc + sc * res * 0.5 * (cost[cur] + cost[ni, nj])
                if ng < best.get((ni, nj), 1e18):
                    best[(ni, nj)] = ng
                    came[(ni, nj)] = cur
                    heapq.heappush(openq, (ng + h(ni, nj), ng, (ni, nj)))
        if g not in came and g != s:
            return None
        path, cur = [g], g
        while cur != s:
            cur = came[cur]
            path.append(cur)
        path.reverse()
        pts = np.array([[(j + 0.5) * res, (i + 0.5) * res] for i, j in path])
        return self._smooth(pts)

    @staticmethod
    def _smooth(pts, k=3):
        if len(pts) < 2 * k:
            return pts
        out = pts.copy()
        for _ in range(2):
            out[1:-1] = np.array([pts[max(0, i - k):i + k + 1].mean(axis=0) for i in range(1, len(pts) - 1)])
            pts = out.copy()
        return out
