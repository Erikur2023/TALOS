"""Region d'operation (~100 km de front) : repere metrique, couverture du sol, relief, ligne de front, positions.

Repere : metres Est / Nord depuis le coin sud-ouest de BBOX (equirectangulaire a la latitude moyenne ; erreur
< 0,3 % sur la zone, sans effet sur une mission de quelques km qui travaille dans des tuiles locales de 400 m).

POSITIONS ALLIEES : entierement FICTIVES (generees pres de la ligne, cote non occupe). Aucune donnee reelle d'unite
n'est utilisee ; les points 'unites' de DeepState sont ignores a la construction (datasets.build_front).
"""
import json, os, io
import numpy as np
from functools import lru_cache
from .datasets import BUILD, BBOX, WC

LON0, LAT0, LON1, LAT1 = BBOX
LATM = (LAT0 + LAT1) / 2
MX = 111_320.0 * np.cos(np.radians(LATM))
MY = 110_574.0
WIDTH_M, HEIGHT_M = (LON1 - LON0) * MX, (LAT1 - LAT0) * MY


def to_xy(lon, lat):
    return (np.asarray(lon) - LON0) * MX, (np.asarray(lat) - LAT0) * MY


def to_ll(x, y):
    return LON0 + np.asarray(x) / MX, LAT0 + np.asarray(y) / MY


class Region:
    def __init__(self):
        meta = json.load(open(os.path.join(BUILD, "meta.json")))
        self.lc = np.load(os.path.join(BUILD, "landcover.npy"), mmap_mode="r")        # sud -> nord, ~10 m
        self.dem = np.load(os.path.join(BUILD, "dem.npy"))                              # sud -> nord, ~30 m
        self.lc_dx, self.lc_dy = WIDTH_M / self.lc.shape[1], HEIGHT_M / self.lc.shape[0]
        self.dem_dx, self.dem_dy = WIDTH_M / self.dem.shape[1], HEIGHT_M / self.dem.shape[0]
        self.front = json.load(open(os.path.join(BUILD, "front.json")))
        from shapely.geometry import shape
        from shapely.prepared import prep
        self.occ = shape(self.front["occupied"])
        self.grey = shape(self.front["grey"]) if self.front.get("grey") else None
        self.line = shape(self.front["line"])
        self._occ_p = prep(self.occ)
        self._grey_p = prep(self.grey) if self.grey is not None else None
        self.places = json.load(open(os.path.join(BUILD, "places.json"), encoding="utf-8"))
        self.buildings = np.load(os.path.join(BUILD, "buildings.npy"))
        self._dist_grid()
        self.allied = self._allied_positions()

    # ------------------------------------------------------------------ requetes
    def landcover_at(self, x, y):
        i = np.clip((np.asarray(y) / self.lc_dy).astype(int), 0, self.lc.shape[0] - 1)
        j = np.clip((np.asarray(x) / self.lc_dx).astype(int), 0, self.lc.shape[1] - 1)
        return self.lc[i, j]

    def dem_window(self, x0, y0, size, res=1.0):
        """MNT reechantillonne (bicubique) sur une fenetre locale [x0, x0+size] x [y0, y0+size] au pas res."""
        from scipy.ndimage import map_coordinates
        n = int(size / res)
        c = (np.arange(n) + 0.5) * res
        X, Y = np.meshgrid(x0 + c, y0 + c)
        return map_coordinates(self.dem, [Y / self.dem_dy - 0.5, X / self.dem_dx - 0.5], order=3, mode="nearest")

    def lc_window(self, x0, y0, size, res=0.5):
        n = int(size / res)
        c = (np.arange(n) + 0.5) * res
        X, Y = np.meshgrid(x0 + c, y0 + c)
        return self.landcover_at(X, Y)

    def status_at(self, lon, lat):
        from shapely.geometry import Point
        p = Point(lon, lat)
        if self._occ_p.contains(p):
            return "occupied"
        if self._grey_p is not None and self._grey_p.contains(p):
            return "grey"
        return "controlled"

    def _dist_grid(self, step=250.0):
        """Distance (m) a la ligne de contact, signee (< 0 cote occupe / gris), sur une grille de 250 m."""
        from shapely.geometry import Point
        nx, ny = int(WIDTH_M / step) + 1, int(HEIGHT_M / step) + 1
        lines = [np.array(g.coords) for g in getattr(self.line, "geoms", [self.line]) if g.length > 0]
        P = np.vstack([np.c_[to_xy(*L[:, :2].T)] for L in lines])
        from scipy.spatial import cKDTree
        tree = cKDTree(P)
        gx, gy = np.meshgrid((np.arange(nx)) * step, (np.arange(ny)) * step)
        d, _ = tree.query(np.c_[gx.ravel(), gy.ravel()])
        lon, lat = to_ll(gx.ravel(), gy.ravel())
        sign = np.array([-1.0 if self.status_at(a, b) != "controlled" else 1.0 for a, b in zip(lon, lat)])
        self.dgrid = (d * sign).reshape(ny, nx).astype(np.float32)
        self.dstep = step
        self._line_tree = tree

    def front_dist(self, x, y):
        """Distance signee a la ligne (m), interpolee ; > 0 = cote controle."""
        from scipy.ndimage import map_coordinates
        xa, ya = np.asarray(x, float), np.asarray(y, float)
        out = map_coordinates(self.dgrid, [np.atleast_1d(ya).ravel() / self.dstep, np.atleast_1d(xa).ravel() / self.dstep],
                              order=1, mode="nearest")
        return out.reshape(xa.shape) if xa.ndim else float(out[0])

    # ------------------------------------------------------------------ positions fictives
    def _allied_positions(self, seed=7):
        rng = np.random.default_rng(seed)
        out = []
        lines = sorted([np.array(g.coords)[:, :2] for g in getattr(self.line, "geoms", [self.line])],
                       key=lambda a: -len(a))
        k = 0
        for L in lines:
            xy = np.c_[to_xy(*L.T)]
            seg = np.hypot(*np.diff(xy, axis=0).T)
            s = np.r_[0, np.cumsum(seg)]
            for d0 in np.arange(2000, s[-1] - 1000, 7000):
                i = np.searchsorted(s, d0)
                p, t = xy[i], xy[min(i + 1, len(xy) - 1)] - xy[max(i - 1, 0)]
                nrm = np.array([-t[1], t[0]]) / (np.hypot(*t) + 1e-9)
                for sgn in (1, -1):
                    q = p + sgn * nrm * rng.uniform(1500, 3000)
                    if 0 < q[0] < WIDTH_M and 0 < q[1] < HEIGHT_M and self.front_dist(*q) > 1200:
                        break
                else:
                    continue
                k += 1
                lon, lat = to_ll(*q)
                out.append({"id": "POS-%02d" % k, "kind": "position", "label": "Position avancee %02d (FICTIF)" % k,
                            "lon": round(float(lon), 5), "lat": round(float(lat), 5),
                            "front_m": round(float(self.front_dist(*q)))})
                # point logistique 8-14 km en arriere
                q2 = q + (q - p) / (np.hypot(*(q - p)) + 1e-9) * rng.uniform(6000, 11000)
                if 0 < q2[0] < WIDTH_M and 0 < q2[1] < HEIGHT_M and self.front_dist(*q2) > 5000 and k % 2 == 1:
                    lon, lat = to_ll(*q2)
                    out.append({"id": "LOG-%02d" % k, "kind": "depot", "label": "Point logistique %02d (FICTIF)" % k,
                                "lon": round(float(lon), 5), "lat": round(float(lat), 5),
                                "front_m": round(float(self.front_dist(*q2)))})
        return out

    # ------------------------------------------------------------------ calques image (projection Web Mercator)
    def overlay_png(self, kind, max_px=None):
        """Image du calque reechantillonnee en lignes Mercator (pour une source image MapLibre exacte)."""
        from PIL import Image
        H = max_px or (1400 if kind == "hillshade" else 2400)
        W = int(H * WIDTH_M / HEIGHT_M)
        merc = lambda lat: np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))
        my = np.linspace(merc(LAT1), merc(LAT0), H)                         # haut -> bas
        lat = np.degrees(2 * np.arctan(np.exp(my)) - np.pi / 2)
        y = (lat - LAT0) * MY
        x = (np.arange(W) + 0.5) / W * WIDTH_M
        X, Y = np.meshgrid(x, y)
        if kind == "landcover":
            pal = {10: (0, 100, 0, 170), 20: (255, 187, 34, 120), 30: (255, 255, 76, 60), 40: (240, 150, 255, 60),
                   50: (250, 0, 0, 150), 60: (180, 180, 180, 80), 80: (0, 100, 200, 170), 90: (0, 150, 160, 150)}
            c = self.landcover_at(X, Y)
            rgba = np.zeros(c.shape + (4,), np.uint8)
            for k, v in pal.items():
                rgba[c == k] = v
        elif kind == "hillshade":
            from scipy.ndimage import map_coordinates
            z = map_coordinates(self.dem, [Y / self.dem_dy - 0.5, X / self.dem_dx - 0.5], order=1, mode="nearest")
            gy, gx = np.gradient(z, HEIGHT_M / H, WIDTH_M / W)
            sh = np.clip(0.5 + (gx * 0.7 + gy * 0.7) * 6, 0, 1)
            rgba = np.zeros(z.shape + (4,), np.uint8)
            v = (sh * 255).astype(np.uint8)
            rgba[..., 0] = rgba[..., 1] = rgba[..., 2] = v
            rgba[..., 3] = (np.abs(sh - 0.5) * 2 * 200).astype(np.uint8)
        elif kind == "danger":
            d = self.front_dist(X, Y)
            a = np.clip(1 - d / 5000, 0, 1)
            rgba = np.zeros(d.shape + (4,), np.uint8)
            rgba[..., 0] = 255; rgba[..., 1] = (120 * (1 - a)).astype(np.uint8)
            rgba[..., 3] = (a * 110).astype(np.uint8)
            rgba[d < 0] = (0, 0, 0, 0)
        else:
            raise KeyError(kind)
        buf = io.BytesIO()
        Image.fromarray(rgba, "RGBA").save(buf, "PNG", optimize=True)
        return buf.getvalue()

    def summary(self):
        lc = np.asarray(self.lc[::20, ::20])
        frac = {WC.get(int(k), str(k)): round(float((lc == k).mean()), 3) for k in np.unique(lc)}
        return {"bbox": BBOX, "width_km": round(WIDTH_M / 1000, 1), "height_km": round(HEIGHT_M / 1000, 1),
                "front_date": self.front.get("date"), "front_source": self.front.get("source"),
                "front_length_km": round(float(sum(np.hypot(*np.diff(np.c_[to_xy(*np.array(g.coords)[:, :2].T)], axis=0).T).sum()
                                                   for g in getattr(self.line, "geoms", [self.line]))) / 1000, 1),
                "landcover_fraction": frac, "n_places": len(self.places), "n_buildings": int(len(self.buildings)),
                "dem_range_m": [round(float(self.dem.min())), round(float(self.dem.max()))]}


@lru_cache(1)
def get():
    return Region()
