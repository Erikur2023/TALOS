"""Terrain REEL (zone du front en Ukraine) : verite terrain simulee construite depuis l'imagerie, le MNT et OSM,
et cartes embarquees a niveaux d'information decroissants.

Ce qui est reel : relief (MNT SRTM), emprise de la vegetation (imagerie Esri ~0,5 m), chemins, batiments, zones
humides (OSM). Ce qui est SIMULE (ASSUMPTION, voir constantes) : les troncs individuels (tires dans le masque de
vegetation), le micro-relief, les cratères / debris non cartographies, l'evolution depuis la prise de vue, la tranchee
a ravitailler (position FICTIVE) et les erreurs de chaque carte.

Niveaux d'information de la carte embarquee (du plus riche au plus pauvre) :
  trees : releve arbre par arbre (PriorMap, borne haute irrealiste)
  sat   : masque de vegetation extrait de l'imagerie (contours fins), batiments, chemins ; image datee et mal geo-referencee
  osm   : polygones foret / haies / batiments / chemins d'OpenStreetMap
  dem   : relief seul (MNT 30 m reechantillonne)
  none  : MNT pour planifier, pas de recalage (odometrie seule)
"""
import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt, binary_opening, binary_closing, label, \
    uniform_filter, binary_dilation
from . import real_fetch as rf
from .world import World, _dist_to_polyline

STEM_DENSITY = 0.05      # ASSUMPTION troncs / m2 dans la vegetation (pinede, espacement ~4,5 m)
MICRO_RELIEF = 0.3       # ASSUMPTION m, relief < 30 m absent du MNT SRTM (correle ~4 m)
CRATERS_PER_HA = 4.0     # ASSUMPTION cratères / debris non cartographies par hectare
IMG_CHANGE = 0.12        # ASSUMPTION part des massifs de vegetation disparus depuis l'image (tirs, incendie)
GEOREF_SD = 2.5          # ASSUMPTION m, erreur de geo-referencement de l'imagerie (globale)
OSM_SD = 3.0             # ASSUMPTION m, erreur de position des objets OSM (globale)
LAMBDA_VEG = 0.06        # /m, taux d'arret LiDAR suppose en vegetation (meme valeur que osm_map, non regle)
LAMBDA_OPEN = 0.004
LAMBDA_WALL = 0.5        # /m, mur / ruine
VEG_RES = 0.5


def _poly_mask(polys, n, res):
    from matplotlib.path import Path
    m = np.zeros((n, n), bool)
    c = (np.arange(n) + 0.5) * res
    X, Y = np.meshgrid(c, c)
    P = np.c_[X.ravel(), Y.ravel()]
    for q in polys:
        if len(q) < 3:
            continue
        x0, y0 = q.min(0); x1, y1 = q.max(0)
        sel = (P[:, 0] >= x0) & (P[:, 0] <= x1) & (P[:, 1] >= y0) & (P[:, 1] <= y1)
        if sel.any():
            m.ravel()[np.flatnonzero(sel)[Path(q).contains_points(P[sel])]] = True
    return m


def _line_mask(lines, n, res, width):
    m = np.zeros((n, n), bool)
    c = (np.arange(n) + 0.5) * res
    X, Y = np.meshgrid(c, c)
    P = np.c_[X.ravel(), Y.ravel()]
    for q in lines:
        m.ravel()[_dist_to_polyline(P, q) < width / 2] = True
    return m


def load_osm(site):
    """-> dict de listes de polylignes locales (m) par categorie."""
    out = {"forest": [], "tree_row": [], "track": [], "road": [], "building": [], "wet": []}
    for e in rf.osm(site)["elements"]:
        tg = e.get("tags", {})
        geoms = []
        if e["type"] == "way" and "geometry" in e:
            geoms = [e["geometry"]]
        elif e["type"] == "relation":
            geoms = [m["geometry"] for m in e.get("members", []) if m.get("role") == "outer" and "geometry" in m]
        for g in geoms:
            x, y = rf.to_local(site, [p["lat"] for p in g], [p["lon"] for p in g])
            q = np.c_[x, y]
            if tg.get("landuse") == "forest" or tg.get("natural") == "wood":
                out["forest"].append(q)
            elif tg.get("natural") == "tree_row":
                out["tree_row"].append(q)
            elif "building" in tg:
                out["building"].append(q)
            elif tg.get("natural") in ("wetland", "water") or "waterway" in tg:
                out["wet"].append(q)
            elif tg.get("highway") in ("track", "path", "footway", "unclassified", "service"):
                out["track"].append(q)
            elif "highway" in tg:
                out["road"].append(q)
    return out


def veg_mask_from_image(img, res=VEG_RES, dark=60.0, rel=12.0):
    """Vegetation haute = pixels sombres (couronnes + ombres portees) ou nettement plus sombres que leur voisinage
    (~20 m : buissons isoles dans une prairie). Seuils fixes choisis a l'oeil sur les deux images (pas d'apprentissage,
    pas de verite terrain) ; puis ouverture / fermeture pour enlever le bruit."""
    L = img.astype(float).mean(-1)
    Ls = gaussian_filter(L, 0.75 / res)
    m = (Ls < dark) | (gaussian_filter(L, 10 / res) - Ls > rel)
    k = np.ones((3, 3), bool)
    return binary_closing(binary_opening(m, k), k)


class RealWorld(World):
    """Verite terrain sur une zone reelle. Interface identique a World (lidar_scan, contact, ...)."""

    def __init__(self, site, start, goal, seed=0, dem=None, osm=None, img=None, craters_per_ha=None, lc_veg=None):
        rng = np.random.default_rng(90_000 + seed)
        self.site, self.seed = site, seed
        self.size, self.res = rf.SITES[site]["size"], 1.0
        n = self.n = int(self.size / self.res)
        # --- relief : MNT reel + micro-relief ---
        dem = rf.dem(site, self.res) if dem is None else dem
        self.dem = dem - dem.min()
        micro = gaussian_filter(rng.standard_normal((n, n)), 4)
        self.height = self.dem + MICRO_RELIEF * micro / (micro.std() + 1e-9)
        self.gy, self.gx = np.gradient(self.height, self.res)
        self.slope = np.hypot(self.gx, self.gy)
        # --- OSM et image ---
        self.osm = load_osm(site) if osm is None else osm
        self.img = rf.imagery(site, VEG_RES) if img is None else img
        self.lc_veg = lc_veg                       # vegetation selon la couverture du sol 10 m (WorldCover), optionnel
        nv = int(self.size / VEG_RES)
        veg_img = veg_mask_from_image(self.img)
        self.veg_img = veg_img
        # verite = image + evolution depuis la prise de vue (massifs disparus)
        lab, k = label(veg_img)
        gone = np.r_[False, rng.random(k) < IMG_CHANGE]
        veg = veg_img & ~gone[lab]
        self.bld = _poly_mask(self.osm["building"], nv, VEG_RES)
        trk = _line_mask(self.osm["track"] + self.osm["road"], nv, VEG_RES, 3.0)
        veg &= ~trk & ~self.bld
        self.veg = veg
        # --- troncs (tires dans la vegetation vraie) ---
        cells = np.flatnonzero(veg.ravel())
        nt = rng.poisson(STEM_DENSITY * len(cells) * VEG_RES ** 2)
        c = rng.choice(cells, nt)
        xy = np.c_[(c % nv + rng.random(nt)) * VEG_RES, (c // nv + rng.random(nt)) * VEG_RES]
        self.start, self.goal = np.asarray(start, float), np.asarray(goal, float)
        for pt in (self.start, self.goal):
            xy = xy[np.hypot(*(xy - pt).T) > 5.0]
        self.trees = np.c_[xy, rng.uniform(0.12, 0.35, len(xy))]
        # --- murs (ruines) : cercles le long des contours de batiments OSM ---
        walls = []
        for q in self.osm["building"]:
            for a, b in zip(q[:-1], q[1:]):
                L = np.hypot(*(b - a))
                for s in np.linspace(0, 1, max(2, int(L / 0.8) + 1)):
                    walls.append([*(a + s * (b - a)), 0.45])
        walls = np.array(walls, float).reshape(-1, 3)
        walls = walls[(walls[:, 0] > 0) & (walls[:, 0] < self.size) & (walls[:, 1] > 0) & (walls[:, 1] < self.size)]
        # --- cratères / debris non cartographies ---
        na = rng.poisson((CRATERS_PER_HA if craters_per_ha is None else craters_per_ha) * self.size ** 2 / 1e4)
        an = np.c_[rng.uniform(5, self.size - 5, (na, 2)), rng.uniform(0.8, 2.0, na)]
        for pt in (self.start, self.goal):
            an = an[np.hypot(*(an[:, :2] - pt).T) > 6.0]
        self.anomalies = an
        # --- sol meuble : zones humides OSM + bas-fonds ---
        wet = _poly_mask([q for q in self.osm["wet"] if len(q) > 3 and np.allclose(q[0], q[-1])], n, self.res)
        low = self.height < np.percentile(self.height, 12)
        noise = gaussian_filter(rng.standard_normal((n, n)), 8)
        self.mud = wet | (low & (noise > 0.3 * noise.std()))
        # --- tranchee (FICTIVE) au but ---
        t0 = self.goal + np.array([6.0, 4.0])
        self.trench = np.array([t0, t0 + [-10, 0], t0 + [-10, -9], t0 + [-20, -9]])
        self.tracks = self.osm["track"] + self.osm["road"]
        cnt = np.zeros((n, n))
        np.add.at(cnt, (np.clip(self.trees[:, 1].astype(int), 0, n - 1), np.clip(self.trees[:, 0].astype(int), 0, n - 1)), 1.0)
        self.tree_density = gaussian_filter(cnt, 4.0)
        self.walls = walls
        self.obst = np.vstack([self.trees, walls, self.anomalies])
        self.obst_kind = np.r_[np.zeros(len(self.trees) + len(walls), int), np.ones(len(self.anomalies), int)]
        self.tree_h = rng.uniform(6, 18, len(self.obst))                   # hauteur (pour la vue 3D)
        self.tree_h[len(self.trees):len(self.trees) + len(walls)] = 2.5
        self.tree_h[len(self.trees) + len(walls):] = 0.8


class RealMap:
    """Carte embarquee de niveau 'sat', 'osm' ou 'dem' (interface commune avec OsmMap / PriorMap)."""

    def __init__(self, world, level, p, seed=0):
        assert level in ("sat", "osm", "dem", "lc")
        rng = np.random.default_rng(95_000 + seed)
        self.kind, self.level = "osm", level
        self.size, self.res, self.world_res = world.size, VEG_RES, world.res
        n = self.n = int(world.size / VEG_RES)
        sh = rng.normal(0, GEOREF_SD if level == "sat" else OSM_SD, 2)
        shift = lambda qs: [q + sh for q in qs]
        trees = np.zeros((0, 3))
        if level == "sat":
            from scipy.ndimage import shift as nd_shift
            veg = nd_shift(world.veg_img.astype(float), (sh[1] / VEG_RES, sh[0] / VEG_RES), order=0) > 0.5
            tracks = shift(world.tracks)
            blds = shift(world.osm["building"])                          # emprises visibles sur l'image
        elif level == "lc":                                              # couverture du sol 10 m (ESA WorldCover)
            veg = world.lc_veg
            tracks = shift(world.tracks)
            blds = shift(world.osm["building"])
        elif level == "osm":
            veg = _poly_mask(shift(world.osm["forest"]), n, VEG_RES) | \
                _line_mask(shift(world.osm["tree_row"]), n, VEG_RES, 8.0)
            tracks = shift(world.tracks)
            blds = shift(world.osm["building"])
        else:
            veg, tracks, blds = None, [], []
        self.tracks = tracks
        if veg is not None:
            trk = _line_mask(tracks, n, VEG_RES, 3.0) if tracks else np.zeros((n, n), bool)
            bm = _poly_mask(blds, n, VEG_RES) if blds else np.zeros((n, n), bool)
            wall = bm & ~binary_opening(bm, np.ones((3, 3), bool))
            allowed = (veg & ~trk) | wall
            self.allowed, self.veg = allowed, veg
            self.dist = np.minimum(distance_transform_edt(~allowed) * VEG_RES, 6.0).astype(np.float32)
            lam = np.where(veg & ~trk, LAMBDA_VEG, LAMBDA_OPEN)
            lam = np.where(wall, LAMBDA_WALL, lam)
            self.lam = gaussian_filter(lam, 1.5 / VEG_RES).astype(np.float32)
            up = int(round(world.res / VEG_RES))
            vw = veg[::up, ::up][:world.n, :world.n]
            tw = trk[::up, ::up][:world.n, :world.n]
            self.extra_cost = 0.8 * vw - 0.5 * tw
            # batiments = obstacles connus pour le planificateur (anneau de cercles)
            ys, xs = np.nonzero(bm[::2, ::2])
            trees = np.c_[(xs * 2 + 1) * VEG_RES, (ys * 2 + 1) * VEG_RES, np.full(len(xs), 0.6)]
        else:
            self.allowed = np.zeros((n, n), bool)
            self.dist = np.full((n, n), 6.0, np.float32)
            mean_lam = float(np.mean(np.where(world.veg_img, LAMBDA_VEG, LAMBDA_OPEN)))
            self.lam = np.full((n, n), mean_lam, np.float32)                # aucune structure : pas d'info
            self.extra_cost = np.zeros((world.n, world.n))
        self.trees = trees
        # MNT : le MNT reel (sans micro-relief, qui n'est pas cartographie)
        self.height = world.dem
        self.pgy, self.pgx = np.gradient(self.height, world.res)
        self.slope = np.hypot(self.pgx, self.pgy)
        self.mud = (world.mud & (rng.random(world.mud.shape) > 0.2)) if level != "dem" else np.zeros_like(world.mud)

    def _ij(self, x, y):
        return (np.clip((np.asarray(y) / self.res).astype(int), 0, self.n - 1),
                np.clip((np.asarray(x) / self.res).astype(int), 0, self.n - 1))

    def dist_at(self, x, y):
        return self.dist[self._ij(x, y)]

    def lam_at(self, x, y):
        return self.lam[self._ij(x, y)]

    def known_dist_at(self, x, y):
        return np.full(np.shape(x), 6.0, np.float32)

    def mud_at(self, x, y):
        i = np.clip((np.asarray(y) / self.world_res).astype(int), 0, self.mud.shape[0] - 1)
        j = np.clip((np.asarray(x) / self.world_res).astype(int), 0, self.mud.shape[1] - 1)
        return self.mud[i, j]

    def grad_at(self, x, y):
        i = np.clip((np.asarray(y) / self.world_res).astype(int), 0, self.pgx.shape[0] - 1)
        j = np.clip((np.asarray(x) / self.world_res).astype(int), 0, self.pgx.shape[1] - 1)
        return self.pgx[i, j], self.pgy[i, j]


# Objectif (tranchee fictive) et departs : anneau autour de l'objectif, deux distances
GOALS = {"serebryansky": (250.0, 250.0), "robotyne": (250.0, 250.0)}


def starts_for(site, k=12, radii=(140.0, 210.0)):
    gx, gy = GOALS[site]
    size = rf.SITES[site]["size"]
    out = []
    for i in range(k):
        a = 2 * np.pi * i / k + 0.3
        r = radii[i % len(radii)]
        out.append((float(np.clip(gx + r * np.cos(a), 15, size - 15)), float(np.clip(gy + r * np.sin(a), 15, size - 15))))
    return out


def make(site, start_idx, level, p=None, seed=0):
    """-> (world, prior, params, matching) pour une mission."""
    from .params import Params
    from .osm_map import osm_params
    p = p or Params()
    st = starts_for(site)[start_idx]
    w = RealWorld(site, st, GOALS[site], seed=seed)
    if level == "trees":
        from .prior_map import PriorMap
        return w, PriorMap(w, p, seed), p, True
    from dataclasses import replace
    # terrain reel : densite LiDAR auto-etalonnee (la carte ne donne pas le nombre de troncs) et injection bornee a
    # 3 m (sinon, avec une carte peu informative, les particules injectees derivent : voir NOTES_NUIT.md)
    p = replace(osm_params(p), pf_lam_calib=True, pf_inject_sd_max=3.0)
    m = RealMap(w, "dem" if level == "none" else level, p, seed)
    return w, m, p, level != "none"
