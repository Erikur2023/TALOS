"""Phase finale : depot du ravitaillement au bord de la tranchee, puis retrait.

Machine a etats (pilotee avec l'ESTIMEE de position, comme la navigation) :
  APPROACH  -> rejoindre le point de depot : sur le bord de tranchee le plus proche, a distance de securite
               (standoff), face a la tranchee, a vitesse reduite ;
  DROP      -> immobile, bras qui deposent la charge (duree fixe), charge -> 0 kg ;
  RETREAT   -> demi-tour sur place puis marche AVANT vers un point deja parcouru (15 m en arriere sur le chemin
               de navigation) : pas de marche arriere a l'aveugle pres d'une tranchee en zigzag (chute MuJoCo) ;
  DONE.

Ou est la tranchee ? Les soldats transmettent sa position (ordre de mission, erreur ORDER_SD) ; a moins de
PERCEPTION_RANGE le LiDAR 3D voit le bord (depression) et on recale la cible sur ce bord percu (erreur EDGE_SD).
Dans le sim point, cette perception est modelisee (ASSUMPTION) ; sous MuJoCo la tranchee n'est qu'un decor.

Charge : le G1 porte <= 3 kg a bout de bras (SPEC). Une charge superieure est refusee avant le depart.
"""
import numpy as np

ORDER_SD = 1.0            # ASSUMPTION m, erreur sur la position de tranchee transmise
EDGE_SD = 0.15            # ASSUMPTION m, erreur du bord percu par le LiDAR 3D
PERCEPTION_RANGE = 6.0    # ASSUMPTION m
STANDOFF = 1.2            # m entre le robot et le bord au moment du depot (bras ~0,6 m + marge)
CLEAR_DIST = 2.5          # m : en retrait, on recule face au bord jusqu'a cette distance avant de pivoter
GUARD_DIST = 1.0          # m : sous ce seuil (estime) en depot / retrait, on s'eloigne activement du bord
MIN_SAFE = 0.6            # m : en dessous, risque de chute dans la tranchee (compte comme incident)
DROP_TIME = 8.0           # s
RETREAT_BACK = 3.0        # m de recul avant demi-tour
RETREAT_DIST = 12.0       # m du bord pour considerer le retrait termine
MAX_PAYLOAD_KG = 3.0      # SPEC G1 (bras)


def trench_order_for(trench_true, seed):
    """Position de tranchee transmise par les soldats (erreur globale ORDER_SD)."""
    return np.asarray(trench_true, float) + np.random.default_rng(80_000 + seed).normal(0, ORDER_SD, 2)


def repulsion(q, pts, radius=3.0):
    """Direction qui s'eloigne de TOUS les segments proches (coins de tranchee en zigzag), ponderee 1/d^2."""
    v = np.zeros(2)
    for a, b in zip(pts[:-1], pts[1:]):
        c, d, _ = closest_on_polyline(q, np.array([a, b]))
        if d < radius:
            v += (q - c) / max(d, 0.05) ** 3
    n = np.hypot(*v)
    return v / n if n > 0 else None


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def closest_on_polyline(q, pts):
    best, bd, seg = None, 1e18, 0
    for k, (a, b) in enumerate(zip(pts[:-1], pts[1:])):
        ab = b - a
        t = np.clip((q - a) @ ab / (ab @ ab + 1e-9), 0, 1)
        c = a + t * ab
        d = float(np.hypot(*(q - c)))
        if d < bd:
            best, bd, seg = c, d, k
    return best, bd, seg


class Delivery:
    def __init__(self, trench_true, payload_kg, p, seed=0, v_approach=0.3, retreat_to=None, order=None,
                 standoff=STANDOFF, guard_dist=GUARD_DIST, v_guard=0.15):
        if payload_kg > MAX_PAYLOAD_KG:
            raise ValueError("charge %.1f kg > %.1f kg (capacite des bras du G1)" % (payload_kg, MAX_PAYLOAD_KG))
        rng = np.random.default_rng(80_000 + seed)
        self.rng, self.p = rng, p
        self.trench_true = np.asarray(trench_true, float)
        self.trench_order = trench_order_for(trench_true, seed) if order is None else np.asarray(order, float)
        rng.normal(0, ORDER_SD, 2)                                               # (meme tirage que l'ordre)
        self.payload = payload_kg
        self.state, self.t_state = "APPROACH", None
        self.v_approach = v_approach
        self.edge_seen = False
        self.min_true_edge_dist = 1e9
        self.log = []
        self.drop_true_dist = None
        self.target = None
        rt = None if retreat_to is None else np.atleast_2d(np.asarray(retreat_to, float))
        self.retreat_wps = [] if rt is None else list(rt[:-1])          # points intermediaires (chemin a rebours)
        self.retreat_to = None if rt is None else rt[-1]
        self.via = None
        self.cleared = False
        self.nav_retreat = False
        self.standoff, self.guard_dist, self.v_guard = standoff, guard_dist, v_guard

    def perceive(self, true_xy, est_xy):
        """Bord percu (si proche) : on recale la tranchee de l'ordre sur le bord vu (dans le repere de l'estimee)."""
        c_true, d_true, _ = closest_on_polyline(np.asarray(true_xy), self.trench_true)
        self.min_true_edge_dist = min(self.min_true_edge_dist, d_true)
        if d_true < PERCEPTION_RANGE:
            # vecteur robot -> bord mesure en repere robot puis reporte depuis l'estimee (erreur de loc. incluse)
            rel = c_true - np.asarray(true_xy) + self.rng.normal(0, EDGE_SD, 2)
            seen = np.asarray(est_xy) + rel
            c_ord, _, _ = closest_on_polyline(seen, self.trench_order)
            self.trench_order = self.trench_order + 0.3 * (seen - c_ord)       # recalage progressif
            self.edge_seen = True
        return d_true

    def command(self, t, est):
        ex, ey, eyaw = est[:3]
        q = np.array([ex, ey])
        c, d, _ = closest_on_polyline(q, self.trench_order)
        if self.t_state is None:
            self.t_state = t
        # garde de bord (toutes phases) : le G1 derive en piétinant sur place (mesure MuJoCo : ~0,6-1 m en 8 s
        # de depot) -> si le bord estime passe sous GUARD_DIST, on recule perpendiculairement au bord
        if self.state in ("DROP", "RETREAT"):
            g = self.guard(est)
            if g is not None:
                return g
        if self.state == "APPROACH":
            if self.via is None:
                self.via = q.copy()                     # point d'arrivee : on repassera par la (chemin sur)
            u = (q - c) / (d + 1e-9)
            self.target = c + u * self.standoff
            to = self.target - q
            dist = float(np.hypot(*to))
            if dist < 0.25 and self.edge_seen:
                face = wrap(np.arctan2(c[1] - ey, c[0] - ex) - eyaw)
                if abs(face) < 0.15:
                    return self._go("DROP", t)
                return 0.0, float(np.clip(1.5 * face, -0.5, 0.5))
            err = wrap(np.arctan2(to[1], to[0]) - eyaw)
            v = min(self.v_approach, 0.6 * dist) * max(0.0, np.cos(err)) if abs(err) < 1.0 else 0.0
            if d < self.standoff - 0.2:                       # trop pres du bord (selon l'estimee) : on recule
                v = -0.15
            return float(v), float(np.clip(1.5 * err, -0.5, 0.5))
        if self.state == "DROP":
            if t - self.t_state >= DROP_TIME:
                self.payload = 0.0
                self.back_from = q.copy()
                return self._go("RETREAT", t)
            return 0.0, 0.0
        if self.state == "RETREAT" and not self.cleared:
            # 1) s'ecarter de la tranchee (tous segments proches) en reculant, sans pivoter pres du bord
            if d < CLEAR_DIST:
                u = repulsion(q, self.trench_order)
                if u is None:
                    self.cleared = True
                else:
                    face = wrap(np.arctan2(-u[1], -u[0]) - eyaw)       # dos tourne vers l'exterieur
                    if abs(face) > 0.4:
                        return 0.0, float(np.clip(1.5 * face, -0.4, 0.4))
                    return -0.2, float(np.clip(1.5 * face, -0.3, 0.3))
            self.cleared = True
        if self.state == "RETREAT":
            if self.via is not None and np.hypot(*(self.via - q)) < 1.0:
                self.via = self.retreat_wps.pop(0) if self.retreat_wps else None
            goal = self.via if self.via is not None else (
                self.retreat_to if self.retreat_to is not None else self.back_from + (self.back_from - c) * 10)
            to = goal - q
            err = wrap(np.arctan2(to[1], to[0]) - eyaw)
            if self.via is None and (np.hypot(*to) < 1.5 or d > RETREAT_DIST):
                return self._go("DONE", t)
            if abs(err) > 0.3:                                    # demi-tour sur place d'abord
                return 0.0, float(np.clip(1.5 * err, -0.5, 0.5))
            return 0.35 * float(np.cos(err)), float(np.clip(1.5 * err, -0.5, 0.5))
        return 0.0, 0.0

    def step(self, nav, t, v_nav, w_nav):
        """Consigne pendant la phase finale. Apres le recul, le retrait est confie au Navigator (son planificateur
        connait la tranchee comme zone interdite) ; la garde de bord reste prioritaire."""
        if self.state == "RETREAT" and self.cleared:
            if not self.nav_retreat:
                self.nav_retreat = True
                nav.retarget(self.retreat_to if self.retreat_to is not None else nav.retreat_point()[-1], t)
                return 0.0, 0.0
            g = self.guard(nav.est)
            if g is not None:
                return g
            if nav.arrived:
                return self._go("DONE", t)
            return v_nav, w_nav
        return self.command(t, nav.est)

    def guard(self, est):
        ex, ey, eyaw = est[:3]
        q = np.array([ex, ey])
        _, d, _ = closest_on_polyline(q, self.trench_order)
        if self.edge_seen and d < self.guard_dist:
            u = repulsion(q, self.trench_order)
            if u is None:
                return None
            away = wrap(np.arctan2(u[1], u[0]) - eyaw)
            if abs(away) > np.pi / 2:
                return -self.v_guard, float(np.clip(1.0 * wrap(away + np.pi), -0.3, 0.3))
            return self.v_guard, float(np.clip(1.0 * away, -0.3, 0.3))
        return None

    def _go(self, state, t):
        self.log.append({"t": round(t, 1), "type": "delivery", "text": {
            "DROP": "au point de depot : depot de la charge",
            "RETREAT": "charge deposee : retrait",
            "DONE": "retrait termine (point de repli atteint)"}[state]})
        self.state, self.t_state = state, t
        return 0.0, 0.0

    def on_true_state(self, true_xy):
        """Appele par le simulateur : enregistre la distance vraie au bord au moment du depot."""
        if self.state == "DROP" and self.drop_true_dist is None:
            self.drop_true_dist = float(closest_on_polyline(np.asarray(true_xy), self.trench_true)[1])

    def metrics(self):
        return {"delivery_state": self.state, "delivered": self.payload == 0.0,
                "drop_edge_dist_m": None if self.drop_true_dist is None else round(self.drop_true_dist, 2),
                "min_edge_dist_m": round(self.min_true_edge_dist, 2),
                "edge_incident": bool(self.min_true_edge_dist < MIN_SAFE)}
