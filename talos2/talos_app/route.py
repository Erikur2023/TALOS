"""Planification STRATEGIQUE A -> B sur la region (grille 20 m) : couverture du sol, routes / pistes OSM, relief,
distance a la ligne de contact. Le robot suit ensuite cet itineraire troncon par troncon avec son propre
planificateur local (talos2.planner) et sa carte embarquee.

Profils :
  rapide    : pistes et routes privilegiees, foret evitee (plus lente) ;
  couvert   : progression sous couvert vegetal (foret, haies) privilegiee, champs ouverts penalises (drones) ;
  equilibre : entre les deux.
Contraintes dures : ni zone occupee ni zone grise, marge FRONT_MARGIN a la ligne, pas d'eau.
"""
import heapq, json, os
import numpy as np
from .region import get as region, to_xy, to_ll, WIDTH_M, HEIGHT_M
from .datasets import BUILD

RES = 20.0
FRONT_MARGIN = 800.0        # m : interdiction
DANGER_RANGE = 4000.0       # m : penalite decroissante
PROFILES = {
    #             foret arbustes prairie cultures bati  nu   humide route piste
    "rapide":    (2.2,  1.6,     1.1,    1.2,     1.6,  1.1, 4.0,   0.6,  0.75),
    "equilibre": (1.4,  1.3,     1.3,    1.4,     1.6,  1.3, 4.0,   0.9,  0.9),
    "couvert":   (0.9,  1.1,     2.2,    2.4,     1.5,  2.4, 4.0,   1.8,  1.3),
}
LC_IDX = {10: 0, 20: 1, 30: 2, 40: 3, 50: 4, 60: 5, 90: 6}

_segs = None


def _load_lines():
    """Segments routes / pistes (repere metrique) pour la rasterisation locale."""
    global _segs
    if _segs is None:
        out = {}
        for name in ("roads", "tracks", "tree_rows"):
            f = os.path.join(BUILD, name + ".geojson")
            segs = []
            for ft in json.load(open(f, encoding="utf-8"))["features"]:
                c = np.array(ft["geometry"]["coordinates"])
                x, y = to_xy(c[:, 0], c[:, 1])
                p = np.c_[x, y]
                segs.append(np.c_[p[:-1], p[1:]])
            out[name] = np.vstack(segs) if segs else np.zeros((0, 4))
        _segs = out
    return _segs


def rasterize_lines(segs, x0, y0, nx, ny, res):
    m = np.zeros((ny, nx), bool)
    sel = (np.maximum(segs[:, 0], segs[:, 2]) >= x0) & (np.minimum(segs[:, 0], segs[:, 2]) <= x0 + nx * res) & \
          (np.maximum(segs[:, 1], segs[:, 3]) >= y0) & (np.minimum(segs[:, 1], segs[:, 3]) <= y0 + ny * res)
    for ax, ay, bx, by in segs[sel]:
        L = max(abs(bx - ax), abs(by - ay))
        k = max(2, int(L / (res / 2)) + 1)
        t = np.linspace(0, 1, k)
        j = ((ax + t * (bx - ax) - x0) / res).astype(int)
        i = ((ay + t * (by - ay) - y0) / res).astype(int)
        ok = (i >= 0) & (i < ny) & (j >= 0) & (j < nx)
        m[i[ok], j[ok]] = True
    return m


def plan(a_ll, b_ll, profile="equilibre", pad=1500.0):
    R = region()
    for name, p in (("A", a_ll), ("B", b_ll)):
        st = R.status_at(*p)
        if st != "controlled":
            return {"ok": False, "error": "%s est en zone %s : mission refusee" % (name, "occupee" if st == "occupied" else "grise")}
    ax, ay = to_xy(*a_ll); bx, by = to_xy(*b_ll)
    x0 = max(0.0, min(ax, bx) - pad); y0 = max(0.0, min(ay, by) - pad)
    x1 = min(WIDTH_M, max(ax, bx) + pad); y1 = min(HEIGHT_M, max(ay, by) + pad)
    nx, ny = int((x1 - x0) / RES), int((y1 - y0) / RES)
    if nx * ny > 2_500_000:
        return {"ok": False, "error": "A et B trop eloignes pour un robot a pied (> ~30 km)"}
    c = (np.arange(nx) + 0.5) * RES + x0, (np.arange(ny) + 0.5) * RES + y0
    X, Y = np.meshgrid(*c)
    w = PROFILES[profile]
    lc = R.landcover_at(X, Y)
    cost = np.full(lc.shape, 1.3)
    for k, idx in LC_IDX.items():
        cost[lc == k] = w[idx]
    blocked = lc == 80                                                    # eau
    L = _load_lines()
    road = rasterize_lines(L["roads"], x0, y0, nx, ny, RES)
    track = rasterize_lines(L["tracks"], x0, y0, nx, ny, RES)
    rows = rasterize_lines(L["tree_rows"], x0, y0, nx, ny, RES)
    cost = np.where(track, np.minimum(cost, w[8]), cost)
    cost = np.where(road, w[7], cost)
    if profile == "couvert":
        cost = np.where(rows, np.minimum(cost, 1.0), cost)
    blocked &= ~(road | track)                                            # ponts
    from scipy.ndimage import map_coordinates
    z = map_coordinates(R.dem, [Y / R.dem_dy - 0.5, X / R.dem_dx - 0.5], order=1, mode="nearest")
    gy, gx = np.gradient(z, RES)
    slope = np.hypot(gx, gy)
    cost += 8 * np.clip(slope - 0.08, 0, None)
    fd = R.front_dist(X, Y)
    blocked |= fd < FRONT_MARGIN
    cost += 3.0 * np.clip(1 - (fd - FRONT_MARGIN) / DANGER_RANGE, 0, 1)
    s = (int((ay - y0) / RES), int((ax - x0) / RES))
    g = (int((by - y0) / RES), int((bx - x0) / RES))
    for q in (s, g):
        blocked[max(0, q[0] - 1):q[0] + 2, max(0, q[1] - 1):q[1] + 2] = False
    path = _astar(cost, blocked, s, g)
    if path is None:
        return {"ok": False, "error": "aucun itineraire hors zone dangereuse"}
    pts = np.array([[x0 + (j + 0.5) * RES, y0 + (i + 0.5) * RES] for i, j in path])
    pts[0], pts[-1] = (ax, ay), (bx, by)
    pts = _simplify(pts)
    lon, lat = to_ll(pts[:, 0], pts[:, 1])
    seg = np.hypot(*np.diff(pts, axis=0).T)
    # statistiques le long du trace (echantillons tous les 10 m)
    s_ = np.r_[0, np.cumsum(seg)]
    d = np.arange(0, s_[-1], 10.0)
    px, py = np.interp(d, s_, pts[:, 0]), np.interp(d, s_, pts[:, 1])
    lcp = R.landcover_at(px, py)
    fdp = R.front_dist(px, py)
    length = float(s_[-1])
    return {"ok": True, "profile": profile, "coords": np.c_[lon, lat].round(6).tolist(), "xy": pts.round(1).tolist(),
            "length_m": round(length), "straight_m": round(float(np.hypot(bx - ax, by - ay))),
            "eta_g1_min": round(length / 0.45 / 60), "cover_pct": round(100 * float(np.isin(lcp, (10, 20)).mean())),
            "open_pct": round(100 * float(np.isin(lcp, (30, 40, 60)).mean())),
            "front_min_m": round(float(fdp.min())), "climb_m": round(float(np.clip(np.diff(_z(R, px, py)), 0, None).sum()))}


def _z(R, x, y):
    from scipy.ndimage import map_coordinates
    return map_coordinates(R.dem, [np.asarray(y) / R.dem_dy - 0.5, np.asarray(x) / R.dem_dx - 0.5], order=1, mode="nearest")


def _astar(cost, blocked, s, g):
    ny, nx = cost.shape
    moves = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0), (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]
    cmin = float(cost.min())
    h = lambda i, j: cmin * np.hypot(i - g[0], j - g[1])
    best = np.full(cost.shape, np.inf); best[s] = 0
    came = -np.ones(cost.shape + (2,), np.int32)
    openq = [(h(*s), 0.0, s)]
    while openq:
        f, gc, cur = heapq.heappop(openq)
        if cur == g:
            break
        if gc > best[cur]:
            continue
        ci, cj = cur
        for di, dj, sc in moves:
            ni, nj = ci + di, cj + dj
            if 0 <= ni < ny and 0 <= nj < nx and not blocked[ni, nj]:
                ng = gc + sc * 0.5 * (cost[ci, cj] + cost[ni, nj])
                if ng < best[ni, nj]:
                    best[ni, nj] = ng; came[ni, nj] = cur
                    heapq.heappush(openq, (ng + h(ni, nj), ng, (ni, nj)))
    if not np.isfinite(best[g]):
        return None
    path, cur = [g], g
    while cur != s:
        cur = tuple(came[cur]); path.append(cur)
    return path[::-1]


def _simplify(pts, tol=6.0):
    """Douglas-Peucker."""
    if len(pts) < 3:
        return pts
    a, b = pts[0], pts[-1]
    ab = b - a
    n = np.hypot(*ab) + 1e-9
    d = np.abs(ab[0] * (pts[:, 1] - a[1]) - ab[1] * (pts[:, 0] - a[0])) / n
    k = int(np.argmax(d))
    if d[k] > tol:
        return np.vstack([_simplify(pts[:k + 1], tol)[:-1], _simplify(pts[k:], tol)])
    return np.array([a, b])
