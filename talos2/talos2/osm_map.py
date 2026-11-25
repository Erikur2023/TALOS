"""Carte embarquee 'realiste' (style OpenStreetMap + MNT) : PAS d'arbres individuels.

Ce qu'une vraie carte fournit, et ce que l'on en simule (toutes les erreurs sont des ASSUMPTION) :
  - MNT (relief) : erreur 0,5 m, correlee sur ~10 m (comme PriorMap) ;
  - polygones 'foret' (landuse=forest / natural=wood) : generalises (pas de clairiere ni de bosquet < ~100 m2),
    lisiere deplacee de ~2-3 m (bruit correle) ;
  - chemins (highway=track) : decales globalement (sigma 2 m) + ondulation (sigma 1 m) ;
  - sol humide (natural=wetland) : zones de boue generalisees, 20 % non cartographiees.

Recalage : un impact LiDAR doit tomber la ou des arbres PEUVENT etre (foret de la carte, hors chemins).
Le champ de distance `dist` vaut 0 en foret et croit dans les clairieres / sur les chemins : l'information vient des
lisieres et des chemins (et du relief), pas des troncs. Les troncs sont tous des obstacles 'decouverts' pour la nav.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, binary_opening, binary_closing

FOREST_DENSITY = 0.02     # ASSUMPTION arbres/m2 au-dessus duquel une zone est 'foret' (1 arbre / 50 m2)
LAMBDA_FOREST = 0.06      # ASSUMPTION /m : probabilite d'arret d'un rayon par metre en foret (libre parcours ~17 m) ;
                          #   une carte OSM ne donne pas la densite : valeur 'foret typique', non reglee sur la verite
LAMBDA_OPEN = 0.004       # /m hors foret (arbres isoles, objets non cartographies)


def _corr_noise(rng, shape, sigma_cells):
    e = gaussian_filter(rng.standard_normal(shape), sigma_cells, mode="wrap")
    return (e - e.mean()) / (e.std() + 1e-9)


def _disk(r):
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


class OsmMap:
    kind = "osm"

    def __init__(self, world, p, seed=0, res=0.5, edge_err_m=2.5, track_offset_sd=2.0, track_wiggle_sd=1.0,
                 track_width=3.0):
        rng = np.random.default_rng(15_000 + seed)
        self.size, self.res, self.world_res = world.size, res, world.res
        self.trees = np.zeros((0, 3))                    # aucun arbre individuel
        n = int(world.size / res)
        self.n = n
        # --- foret (verite, sur la grille du monde) puis generalisation + bruit de lisiere ---
        dens = world.tree_density
        # bord deplace : seuil applique a une densite bruitee (bruit correle ~15 m) -> lisiere decalee de quelques m
        g = np.hypot(*np.gradient(gaussian_filter(dens, 2)))
        noisy = dens + edge_err_m * np.median(g[g > 0]) * _corr_noise(rng, dens.shape, 15 / world.res)
        forest = noisy > FOREST_DENSITY
        k = _disk(int(round(5 / world.res)))
        forest = binary_closing(binary_opening(forest, k), k)          # generalisation (~100 m2)
        self.forest_world = forest
        self.forest_true = dens > FOREST_DENSITY
        up = int(round(world.res / res))
        forest_hr = np.kron(forest, np.ones((up, up), bool))[:n, :n]
        # --- chemins ---
        self.tracks = []
        track = np.zeros((n, n), bool)
        yy, xx = (np.mgrid[0:n, 0:n] + 0.5) * res
        for pts in world.tracks:
            off = rng.normal(0, track_offset_sd, 2)
            wig = gaussian_filter(rng.standard_normal((len(pts), 2)), (3, 0)) * track_wiggle_sd * 3
            q = pts + off + wig
            self.tracks.append(q)
            for a, b in zip(q[:-1], q[1:]):
                x0, x1 = sorted((a[0], b[0])); y0, y1 = sorted((a[1], b[1]))
                pad = track_width
                j0, j1 = max(0, int((x0 - pad) / res)), min(n, int((x1 + pad) / res) + 1)
                i0, i1 = max(0, int((y0 - pad) / res)), min(n, int((y1 + pad) / res) + 1)
                X, Y = xx[i0:i1, j0:j1], yy[i0:i1, j0:j1]
                ab = b - a
                t = np.clip(((X - a[0]) * ab[0] + (Y - a[1]) * ab[1]) / (ab @ ab + 1e-9), 0, 1)
                d = np.hypot(X - (a[0] + t * ab[0]), Y - (a[1] + t * ab[1]))
                track[i0:i1, j0:j1] |= d < track_width / 2
        self.track = track
        allowed = forest_hr & ~track                       # la ou un tronc est plausible selon la carte
        self.allowed = allowed
        self.dist = np.minimum(distance_transform_edt(~allowed) * res, 6.0).astype(np.float32)
        # taux d'arret des rayons LiDAR (modele de densite) ; lissage = incertitude de lisiere
        self.lam = gaussian_filter(np.where(allowed, LAMBDA_FOREST, LAMBDA_OPEN), 1.5 / res).astype(np.float32)
        # --- MNT ---
        e = 0.5 * _corr_noise(rng, world.height.shape, 8)          # ASSUMPTION erreur MNT 0,5 m
        self.height = world.height + e
        self.pgy, self.pgx = np.gradient(self.height, world.res)
        self.slope = world.slope
        # --- sol humide ---
        wet = binary_opening(world.mud, _disk(2))
        self.mud = wet & (rng.random(world.mud.shape) > 0.2)
        # --- cout de planification : foret dense = troncs inconnus -> plus lent ; chemin = prefere ---
        trk_w = track[::up, ::up][:world.n, :world.n] if up > 1 else track
        self.extra_cost = 0.8 * forest - 0.5 * trk_w

    # interface commune avec PriorMap
    def dist_at(self, x, y):
        i = np.clip((np.asarray(y) / self.res).astype(int), 0, self.n - 1)
        j = np.clip((np.asarray(x) / self.res).astype(int), 0, self.n - 1)
        return self.dist[i, j]

    def lam_at(self, x, y):
        i = np.clip((np.asarray(y) / self.res).astype(int), 0, self.n - 1)
        j = np.clip((np.asarray(x) / self.res).astype(int), 0, self.n - 1)
        return self.lam[i, j]

    def known_dist_at(self, x, y):
        """Distance aux objets INDIVIDUELS connus : aucun ici -> tout impact est un obstacle a cartographier."""
        return np.full(np.shape(x), 6.0, np.float32)

    def mud_at(self, x, y):
        i = np.clip((np.asarray(y) / self.world_res).astype(int), 0, self.mud.shape[0] - 1)
        j = np.clip((np.asarray(x) / self.world_res).astype(int), 0, self.mud.shape[1] - 1)
        return self.mud[i, j]

    def grad_at(self, x, y):
        i = np.clip((np.asarray(y) / self.world_res).astype(int), 0, self.pgx.shape[0] - 1)
        j = np.clip((np.asarray(x) / self.world_res).astype(int), 0, self.pgx.shape[1] - 1)
        return self.pgx[i, j], self.pgy[i, j]

    def quality(self, world):
        """Accord carte / verite (pour le rapport) : part des cellules foret/non-foret correctement cartographiees."""
        return {"forest_agreement": round(float((self.forest_world == self.forest_true).mean()), 3),
                "forest_fraction_true": round(float(self.forest_true.mean()), 3),
                "trees_in_mapped_forest": round(float(self.allowed[
                    np.clip((world.trees[:, 1] / self.res).astype(int), 0, self.n - 1),
                    np.clip((world.trees[:, 0] / self.res).astype(int), 0, self.n - 1)].mean()), 3)}


def osm_params(p):
    """Le recalage sur des structures est plus flou que sur des troncs : vraisemblance plus large."""
    from dataclasses import replace
    return replace(p, pf_mode="density", map_hits_as_obstacles=True, n_particles=3000)

