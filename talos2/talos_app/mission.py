"""Execution d'une mission A -> B en direct.

L'itineraire strategique (route.py) est decoupe en troncons de <= LEG_M. Chaque troncon se joue dans une TUILE de
TILE_M x TILE_M m construite a la volee depuis les donnees reelles :
  verite : imagerie Esri (vegetation) + MNT Copernicus 30 m + batiments / pistes OSM + troncs, micro-relief et
           crateres simules (densite croissante pres de la ligne de contact) ;
  carte embarquee : niveau choisi (sat / lc / dem / none).
Le meme Navigator (talos2) pilote soit le simulateur rapide (point), soit le G1 sous MuJoCo (politique Unitree).
Au passage d'une tuile a l'autre, le nuage de particules est transfere (l'incertitude de position est conservee) ;
sous MuJoCo le robot est re-pose dans la nouvelle scene (pause de chargement visible).
"""
import io, threading, time, traceback
from dataclasses import replace
import numpy as np

from talos2 import real_fetch as rf
from talos2.real_site import RealWorld, RealMap, VEG_RES
from talos2.params import Params
from talos2.osm_map import osm_params
from talos2.navigator import Navigator
from talos2.sensors import ImuOdometry
from talos2.energy import power_w
from talos2.thermal import ThermalModel
from talos2.delivery import Delivery, trench_order_for
from talos2.sim import _vox_add
from .region import get as region, to_ll, to_xy, WIDTH_M, HEIGHT_M
from .route import _load_lines

TILE_M = 400.0
LEG_M = 240.0
DT = 0.1


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def tile_osm(x0, y0, size):
    """Pistes / routes / haies / batiments OSM (extrait Geofabrik local) dans la tuile, en coordonnees de tuile."""
    R = region()
    L = _load_lines()
    out = {"forest": [], "tree_row": [], "track": [], "road": [], "building": [], "wet": []}
    for name, key in (("roads", "road"), ("tracks", "track"), ("tree_rows", "tree_row")):
        s = L[name]
        sel = (np.maximum(s[:, 0], s[:, 2]) >= x0 - 20) & (np.minimum(s[:, 0], s[:, 2]) <= x0 + size + 20) & \
              (np.maximum(s[:, 1], s[:, 3]) >= y0 - 20) & (np.minimum(s[:, 1], s[:, 3]) <= y0 + size + 20)
        for ax, ay, bx, by in s[sel]:
            out[key].append(np.array([[ax - x0, ay - y0], [bx - x0, by - y0]]))
    b = R.buildings
    bx, by = to_xy(b[:, 0], b[:, 1])
    sel = (bx > x0 - 30) & (bx < x0 + size + 30) & (by > y0 - 30) & (by < y0 + size + 30)
    from .region import MX, MY
    for cx, cy, dlon, dlat in zip(bx[sel] - x0, by[sel] - y0, b[sel, 2], b[sel, 3]):
        hx, hy = max(2.0, dlon * MX / 2), max(2.0, dlat * MY / 2)
        if hx > 40 or hy > 40:
            continue
        out["building"].append(np.array([[cx - hx, cy - hy], [cx + hx, cy - hy], [cx + hx, cy + hy], [cx - hx, cy + hy], [cx - hx, cy - hy]]))
    return out


def lc_veg_mask(x0, y0, size):
    R = region()
    lc = R.lc_window(x0, y0, size, VEG_RES)
    return np.isin(lc, (10,))


def fake_image(x0, y0, size):
    """Si l'imagerie n'est pas joignable : image synthetique depuis la couverture du sol (vegetation sombre)."""
    R = region()
    lc = R.lc_window(x0, y0, size, VEG_RES)
    img = np.full(lc.shape + (3,), 120, np.uint8)
    img[lc == 10] = (40, 55, 35); img[lc == 20] = (70, 80, 50); img[lc == 50] = (150, 140, 130); img[lc == 80] = (40, 60, 90)
    return img


COMPASS_SD = np.radians(3.0)        # ASSUMPTION bruit du compas magnetique (rad)
COMPASS_DIST_SD = np.radians(2.0)   # ASSUMPTION perturbation locale lente (masses metalliques, epaves), correlee ~60 s
CRATER_RANGE = 8.0      # ASSUMPTION m : portee de detection d'un creux par le LiDAR 3D (segmentation du sol)


def scan_no_craters(w, x, y, yaw, p, rng):
    """Scan LiDAR horizontal : les crateres sont des CREUX, invisibles pour un balayage horizontal."""
    full, kind = w.obst, w.obst_kind
    w.obst = full[kind == 0]
    try:
        return w.lidar_scan(x, y, yaw, p.lidar_rays, p.lidar_range, p.lidar_sigma, p.lidar_dropout, rng)
    finally:
        w.obst = full


CRATER_SLOW = 0.4    # ASSUMPTION facteur de vitesse dans un cratere


def in_crater(w, x, y):
    an = w.anomalies
    if len(an) == 0:
        return -1
    d = np.hypot(an[:, 0] - x, an[:, 1] - y) - an[:, 2]
    k = int(np.argmin(d))
    return k if d[k] < 0 else -1


def contact_solid(w, x, y, r):
    """Contact avec les obstacles PLEINS seulement (troncs, murs) ; les crateres sont des creux."""
    full, kind = w.obst, w.obst_kind
    w.obst = full[kind == 0]
    try:
        return w.contact(x, y, r)
    finally:
        w.obst = full


def crater_ranges(w, x, y, yaw, p, rng):
    """Creux detectes par le LiDAR 3D (segmentation du sol) convertis en portees 'virtuelles' pour l'evitement."""
    full = w.obst
    w.obst = w.anomalies if len(w.anomalies) else np.zeros((0, 3))
    try:
        if len(w.obst) == 0:
            return None
        ang, rr = w.lidar_scan(x, y, yaw, p.lidar_rays, CRATER_RANGE, 0.05, 0.0, rng)
    finally:
        w.obst = full
    return np.where(rr >= CRATER_RANGE - 1e-6, p.lidar_range, rr)


def perceive_craters(w, nav, true_pose, est_pose, seen):
    """Le LiDAR 3D detecte les crateres proches (depression du sol) devant le robot : obstacles pour le
    planificateur, places dans le repere ESTIME. Pas utilises pour le recalage."""
    x, y, yaw = true_pose
    ex, ey, eyaw = est_pose
    an = w.anomalies
    if len(an) == 0:
        return seen
    rel = an[:, :2] - [x, y]
    d = np.hypot(*rel.T)
    bearing = _wrap(np.arctan2(rel[:, 1], rel[:, 0]) - yaw)
    new = False
    for k in np.flatnonzero(d < CRATER_RANGE):                         # LiDAR 3D 360 deg (MID-360)
        if k in seen:
            continue
        seen.add(k)
        b = eyaw + bearing[k]
        nav.planner.add_obstacle(ex + d[k] * np.cos(b), ey + d[k] * np.sin(b), an[k, 2] + 0.6)
        new = True
    if new:
        nav.replan_pending = True
    return seen


class Tile:
    def __init__(self, x0, y0, start_l, goal_l, seed, level, final, delivery):
        R = region()
        self.x0, self.y0, self.size = x0, y0, TILE_M
        lon, lat = to_ll(x0, y0)
        self.site = "tile_%d_%d" % (round(x0), round(y0))
        rf.SITES[self.site] = {"name": "tuile %s" % self.site, "sw": (float(lat), float(lon)), "size": TILE_M}
        try:
            img = rf.imagery(self.site, VEG_RES)
            self.imagery = "esri"
        except Exception:
            img = fake_image(x0, y0, TILE_M)
            self.imagery = "synthetique (hors ligne)"
        fd = float(R.front_dist(x0 + TILE_M / 2, y0 + TILE_M / 2))
        craters = 2.0 + 40.0 * float(np.clip(1 - fd / 6000.0, 0, 1))        # ASSUMPTION : bombardements pres du front
        self.front_m, self.craters_per_ha = fd, craters
        dem = R.dem_window(x0, y0, TILE_M, 1.0)
        self.w = RealWorld(self.site, start_l, goal_l, seed=seed, dem=dem, osm=tile_osm(x0, y0, TILE_M), img=img,
                           craters_per_ha=craters, lc_veg=lc_veg_mask(x0, y0, TILE_M))
        if not (final and delivery):
            self.w.trench = np.zeros((0, 2))
        self.level = level

    def prior(self, p, seed):
        return RealMap(self.w, "dem" if self.level == "none" else self.level, p, seed)

    def image_jpeg(self, max_px=600):
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(self.w.img[::-1]).resize((max_px, max_px)).save(buf, "JPEG", quality=78)
        return buf.getvalue()


class Mission(threading.Thread):
    def __init__(self, route, mode="point", level="sat", delivery=True, payload_kg=3.0, speed=10.0, seed=1,
                 target_label="", compass=True):
        super().__init__(daemon=True)
        self.compass = compass
        self.route = np.asarray(route, float)
        self.mode, self.level, self.delivery, self.seed = mode, level, delivery, seed
        self.speed = speed
        self.payload_kg = payload_kg
        self.target_label = target_label
        self.lock = threading.Lock()
        self.cmd_stop = self.cmd_pause = False
        self.camera = "suivi"
        self.state = {"status": "preparation", "phase": "PREPARATION"}
        self.events, self.vox_new, self.vox_all, self.track = [], [], [], []
        self.scan, self.particles, self.local_path = [], [], []
        self.frame = None                      # JPEG MuJoCo courant
        self.tile_img = None
        self.tile_info = None
        self.metrics = {}
        self.err_hist = []

    # ------------------------------------------------------------------ utilitaires
    def log(self, t, typ, text, xy=None):
        e = {"t": round(t, 1), "type": typ, "text": text}
        if xy is not None:
            lon, lat = to_ll(*xy)
            e.update(lon=round(float(lon), 6), lat=round(float(lat), 6))
        with self.lock:
            self.events.append(e)

    def legs(self):
        r = self.route
        seg = np.hypot(*np.diff(r, axis=0).T)
        s = np.r_[0, np.cumsum(seg)]
        n = max(1, int(np.ceil(s[-1] / LEG_M)))
        d = np.linspace(0, s[-1], n + 1)
        return np.c_[np.interp(d, s, r[:, 0]), np.interp(d, s, r[:, 1])], s[-1]

    def params(self):
        if self.mode in ("g1", "go1"):
            from talos2.mujoco_g1.run import mujoco_params
            p = mujoco_params(**({"plan_slope_max": 0.35} if self.mode == "go1" else {}))
        else:
            p = Params()
        p = replace(osm_params(p), pf_lam_calib=True, pf_inject_sd_max=3.0, pf_odom_gate=True, payload_kg=self.payload_kg, max_time=1e9)
        return p

    # ------------------------------------------------------------------ boucle
    def run(self):
        try:
            self._run()
        except Exception as e:
            traceback.print_exc()
            self.log(self.state.get("t", 0), "error", "erreur interne : %s" % e)
            with self.lock:
                self.state["status"] = "erreur"

    def _run(self):
        p = self.params()
        pts, total = self.legs()
        matching = self.level != "none"
        imu = ImuOdometry(p, self.seed)
        rng = np.random.default_rng(40_000 + self.seed)
        th = ThermalModel()
        comp_bias = 0.0
        crater_cross = 0
        t, wh, dist_true, contacts, falls = 0.0, 0.0, 0.0, 0, 0
        x, y = pts[0]                                                     # verite (repere region)
        yaw = float(np.arctan2(*(pts[1] - pts[0])[::-1]))
        ex_g, ey_g, eyaw = x, y, yaw                                      # estimee (repere region)
        old = None                                                        # (navigator, origine) du troncon precedent
        payload = self.payload_kg
        dlv = None
        wall0 = time.time()
        t_wall_ref, t_ref = time.time(), 0.0
        status = "en cours"
        self.log(0, "info", "depart : %d troncons, %.1f km, mode %s, carte %s" % (len(pts) - 1, total / 1000,
                                                                                    {"g1": "G1 MuJoCo", "go1": "quadrupede Go1 MuJoCo"}.get(self.mode, "simulateur rapide"), self.level))
        for k in range(1, len(pts)):
            final = k == len(pts) - 1
            gx, gy = pts[k]
            mid = (np.array([x, y]) + pts[k]) / 2
            x0 = float(np.clip(mid[0] - TILE_M / 2, 0, WIDTH_M - TILE_M)); y0 = float(np.clip(mid[1] - TILE_M / 2, 0, HEIGHT_M - TILE_M))
            with self.lock:
                self.state.update(status="chargement tuile %d/%d" % (k, len(pts) - 1), leg=k, legs=len(pts) - 1)
            tile = Tile(x0, y0, (x - x0, y - y0), (gx - x0, gy - y0), self.seed * 100 + k, self.level, final, self.delivery)
            w = tile.w
            prior = tile.prior(p, self.seed * 100 + k)
            self.tile_img = tile.image_jpeg()
            lon0, lat0 = to_ll(x0, y0); lon1, lat1 = to_ll(x0 + TILE_M, y0 + TILE_M)
            with self.lock:
                self.tile_info = {"id": tile.site, "k": k, "x0": x0, "y0": y0, "size": TILE_M,
                                  "bounds": [float(lon0), float(lat0), float(lon1), float(lat1)],
                                  "imagery": tile.imagery, "front_m": round(tile.front_m), "craters_per_ha": round(tile.craters_per_ha, 1),
                                  "n_trees": int(len(w.trees)), "n_walls": int(len(w.walls)), "n_craters": int(len(w.anomalies)),
                                  "obst": np.round(np.c_[w.obst[:, :2] + [x0, y0], w.obst[:, 2], w.obst_kind], 1)[::2].tolist(),
                                  "trench": (w.trench + [x0, y0]).round(1).tolist()}
            self.log(t, "tile", "tuile %d/%d chargee (%s) : %d troncs, %d murs, %d crateres, ligne a %.1f km" % (
                k, len(pts) - 1, tile.imagery, len(w.trees), len(w.walls), len(w.anomalies), tile.front_m / 1000))
            trench_order = trench_order_for(w.trench, self.seed) if (final and self.delivery) else None
            M = None
            if self.mode in ("g1", "go1"):
                from talos2.mujoco_g1.run import G1Mission, NAV_DT
                from talos2.mujoco_g1.quad import QuadMission
                full, fk = w.obst, w.obst_kind                         # crateres = creux : hors de la scene physique
                w.obst, w.obst_kind = full[fk == 0], fk[fk == 0]
                try:
                    M = (G1Mission(self.seed, p, relief_scale=0.4, world=w, prior=prior) if self.mode == "g1" else
                         QuadMission(self.seed, p, relief_scale=1.0, world=w, prior=prior, payload_kg=self.payload_kg))
                finally:
                    w.obst, w.obst_kind = full, fk
                dyaw = _wrap(M.yaw0 - yaw)
                yaw = M.yaw0
                eyaw = eyaw + dyaw
                renderer = self._make_renderer(M)
                contacts_before = contacts
            sx, sy = ex_g - x0, ey_g - y0
            nav = Navigator(prior, p, (sx, sy), eyaw, (gx - x0, gy - y0), self.seed + k, matching,
                            no_go=[trench_order] if trench_order is not None else None)
            if old is not None:                                           # transfert de l'incertitude
                onav, (ox0, oy0) = old
                off = np.array([ox0 - x0, oy0 - y0])
                if nav.pf is not None and onav.pf is not None:
                    nav.pf.s = onav.pf.s.copy(); nav.pf.s[:, :2] += off
                    if self.mode in ("g1", "go1"):
                        nav.pf.s[:, 2] += dyaw
                    nav.pf.w = onav.pf.w.copy()
                    nav.pf.travel_since_good = onav.pf.travel_since_good
                    for a in ("lam_obs", "lam_exp"):
                        if hasattr(onav.pf, a):
                            setattr(nav.pf, a, getattr(onav.pf, a))
                nav.dr.x, nav.dr.y = onav.dr.x + off[0], onav.dr.y + off[1]
                nav.dr.yaw = onav.dr.yaw + (dyaw if self.mode in ("g1", "go1") else 0.0)
            if nav.path is None:
                self.log(t, "error", "pas de chemin local sur la tuile %d" % k)
                status = "sans chemin"; break
            lx, ly = x - x0, y - y0
            prog_ref, prog_t = (lx, ly), t
            leg_done = False
            vox = {}
            seen_craters = set()
            pause = 0.0
            crossed = set()
            step = 0
            scan_every = int(round(p.scan_dt / DT))
            while not leg_done:
                while self.cmd_pause and not self.cmd_stop:
                    time.sleep(0.05); t_wall_ref, t_ref = time.time(), t
                if self.cmd_stop:
                    status = "arretee par l'operateur"; break
                v_cmd, w_cmd = nav.command(t, DT)
                if nav.arrived and dlv is None:
                    if not final:
                        leg_done = True; break
                    if self.delivery:
                        dlv = Delivery(w.trench, p.payload_kg, p, self.seed, retreat_to=nav.retreat_point(), order=trench_order,
                                       **(dict(v_approach=0.25, standoff=1.8, guard_dist=1.2, v_guard=0.3) if self.mode in ("g1", "go1") else {}))
                        self.log(t, "delivery", "arrivee a la position : approche de la tranchee", (lx + x0, ly + y0))
                    else:
                        status = "arrivee"; break
                if dlv is not None:
                    v_cmd, w_cmd = dlv.step(nav, t, v_cmd, w_cmd)
                    if dlv.state == "DONE":
                        status = "mission accomplie"; break
                # ---------------- dynamique ----------------
                if M is None:
                    if pause > 0:
                        pause -= DT; v_cmd_eff = 0.0
                    else:
                        v_cmd_eff = v_cmd
                    w_true = w_cmd * 0.95 + rng.normal(0, 0.02)
                    slope = float(w.slope_at(lx, ly)); mud = bool(w.mud_at(lx, ly))
                    slope_f = max(0.3, 1.0 - 0.8 * max(0.0, slope - p.slope_slow_from))
                    v_true = v_cmd_eff * max(0.3, slope_f * (p.mud_slip if mud else 1.0))
                    kc = in_crater(w, lx, ly)
                    if kc >= 0:                                           # creux : franchissement lent et risque
                        v_true *= CRATER_SLOW
                        if kc not in crossed:
                            crossed.add(kc); crater_cross += 1
                            self.log(t, "anomaly", "franchissement d'un cratere (ralenti, risque de chute pour un humanoide)", (lx + x0, ly + y0))
                    yaw += w_true * DT
                    nx_, ny_ = lx + v_true * np.cos(yaw) * DT, ly + v_true * np.sin(yaw) * DT
                    kk, push = contact_solid(w, nx_, ny_, p.robot_radius)
                    if kk >= 0 and np.hypot(*push) > 1e-3:
                        nx_, ny_ = nx_ + push[0] * 1.02, ny_ + push[1] * 1.02
                        if pause <= 0:
                            contacts += 1
                            self.log(t, "contact", "contact (%s)" % ("cratere" if w.obst_kind[kk] else "arbre / mur"), (nx_ + x0, ny_ + y0))
                        pause = p.obstacle_contact_pause
                    dist_true += float(np.hypot(nx_ - lx, ny_ - ly))
                    lx, ly = float(np.clip(nx_, 0.5, TILE_M - 0.5)), float(np.clip(ny_, 0.5, TILE_M - 0.5))
                    v_meas, gyro = imu.measure(v_cmd_eff * slope_f, w_true, DT)
                    nav.on_proprio(v_meas, gyro, DT)
                    v_real = v_true
                else:
                    v_fwd, yawrate, fell = M.advance(v_cmd, w_cmd, t)
                    if fell:
                        falls += 1; status = "chute"; self.log(t, "error", "chute du robot", (lx + x0, ly + y0)); break
                    v_meas, gyro, v_lat = M.proprio(v_fwd, yawrate)
                    nav.on_proprio(v_meas, gyro, DT, v_lat)
                    plx, ply = lx, ly
                    lx, ly, yaw = M.pose
                    dist_true += float(np.hypot(lx - plx, ly - ply))
                    slope = float(w.slope_at(lx, ly)); mud = bool(w.mud_at(lx, ly))
                    v_real = v_fwd
                    contacts = contacts_before + M.contacts
                pw = power_w(p, abs(v_real), slope, mud, payload)
                wh += pw * DT / 3600.0
                th.step(pw, DT)
                if dlv is not None:
                    dlv.perceive((lx, ly), nav.est[:2]); dlv.on_true_state((lx, ly)); payload = dlv.payload
                if step % scan_every == 0:
                    if M is None:
                        ang, rr = scan_no_craters(w, lx, ly, yaw, p, rng)
                        avoid = crater_ranges(w, lx, ly, yaw, p, rng)
                        gxt, gyt = w.grad_at(lx, ly)
                        along = gxt * np.cos(yaw) + gyt * np.sin(yaw) + rng.normal(0, p.imu_slope_noise)
                        across = -gxt * np.sin(yaw) + gyt * np.cos(yaw) + rng.normal(0, p.imu_slope_noise)
                    else:
                        ang, rr = M.scan2d(); along, across = M.slope_meas()
                        avoid = crater_ranges(w, lx, ly, yaw, p, rng)
                    if self.compass:
                        comp_bias += (-comp_bias / 60.0) * p.scan_dt + rng.normal(0, COMPASS_DIST_SD * np.sqrt(2 * p.scan_dt / 60.0))
                        nav.on_heading(yaw + comp_bias + rng.normal(0, COMPASS_SD), COMPASS_SD + COMPASS_DIST_SD)
                    nav.on_scan(ang, rr, t, along, across, avoid_rr=avoid)
                    ex, ey, eyaw_l = nav.est[:3]
                    seen_craters = perceive_craters(w, nav, (lx, ly, yaw), (ex, ey, eyaw_l), seen_craters)
                    vv = {}
                    _vox_add(vv, w, prior, (lx, ly), (ex, ey, eyaw_l), ang, rr, p.lidar_range, t)
                    hit = rr < p.lidar_range - 0.5
                    with self.lock:
                        for (i, j, kz), (c, tt) in vv.items():
                            key = (i, j, kz)
                            if key not in vox:
                                vox[key] = 1
                                v = [round(i * 0.5 + 0.25 + x0, 1), round(j * 0.5 + 0.25 + y0, 1), round(kz * 0.5, 1), c]
                                self.vox_all.append(v)
                        self.scan = np.round(np.c_[ex + rr[hit] * np.cos(eyaw_l + ang[hit]) + x0,
                                                   ey + rr[hit] * np.sin(eyaw_l + ang[hit]) + y0], 1).tolist()
                        if nav.pf is not None:
                            sel = nav.pf.s[::max(1, len(nav.pf.s) // 150), :2] + [x0, y0]
                            self.particles = np.round(sel, 1).tolist()
                # ---------------- etat publie ----------------
                ex, ey, eyaw_l, sigma = nav.est[:4]
                x, y = lx + x0, ly + y0
                ex_g, ey_g, eyaw = ex + x0, ey + y0, eyaw_l
                err = float(np.hypot(x - ex_g, y - ey_g))
                if step % 5 == 0:
                    rem_leg = nav.remaining_m()
                    rem = rem_leg + sum(np.hypot(*np.diff(pts[k:], axis=0).T)) if k < len(pts) - 1 else rem_leg
                    v_avg = max(0.25, dist_true / max(t, 1.0)) if t > 30 else 0.5
                    lon, lat = to_ll(x, y); elon, elat = to_ll(ex_g, ey_g)
                    with self.lock:
                        self.track.append([round(float(elon), 6), round(float(elat), 6), round(float(lon), 6), round(float(lat), 6)])
                        # t, erreur, sigma, v, batterie %, dT, ligne m, reste m, charge, pente, phase
                        self.err_hist.append([round(t, 1), round(err, 2), round(float(sigma), 2), round(float(v_real), 2),
                                              round(100 * (p.battery_wh - wh) / p.battery_wh, 1), round(th.dT, 2),
                                              round(float(region().front_dist(x, y))), round(float(rem)), payload,
                                              round(slope, 3), dlv.state if dlv else "NAV"])
                        self.local_path = (np.asarray(nav.path) + [x0, y0]).round(1)[::2].tolist() if nav.path is not None else []
                        self.state = {
                            "status": status, "t": round(t, 1), "leg": k, "legs": len(pts) - 1,
                            "true": [float(lon), float(lat)], "true_xy": [round(x, 1), round(y, 1)], "yaw": round(float(yaw), 3),
                            "est": [float(elon), float(elat)], "est_xy": [round(ex_g, 1), round(ey_g, 1)], "eyaw": round(float(eyaw), 3),
                            "sigma": round(float(sigma), 2), "err": round(err, 2), "v": round(float(v_real), 2),
                            "v_cmd": round(float(v_cmd), 2), "w_cmd": round(float(w_cmd), 2),
                            "battery_pct": round(100 * (p.battery_wh - wh) / p.battery_wh, 1), "energy_wh": round(wh, 1),
                            "dT": round(th.dT, 1), "detect_m": round(th.detection_range_m(th.dT)),
                            "payload": payload, "phase": dlv.state if dlv else "NAV",
                            "remaining_m": round(float(rem)), "eta_s": round(float(rem) / v_avg),
                            "dist_m": round(dist_true), "contacts": contacts, "replans": nav.replans,
                            "slope": round(slope, 3), "mud": mud, "front_m": round(float(region().front_dist(x, y))),
                            "speed": self.speed, "mode": self.mode, "level": self.level, "wall_s": round(time.time() - wall0),
                        }
                    for e in nav.events[len(getattr(nav, "_pub", [])):]:
                        if "x" in e:
                            self.log(t, e["type"], e["text"], (e["x"] + x0, e["y"] + y0))
                        else:
                            self.log(t, e["type"], e["text"])
                    nav._pub = list(nav.events)
                if M is not None:
                    self._render(M, renderer, t)
                if t - prog_t > 120:
                    if np.hypot(lx - prog_ref[0], ly - prog_ref[1]) < 1.0:
                        status = "bloque"; self.log(t, "error", "robot bloque (aucun progres en 2 min)", (x, y)); break
                    prog_ref, prog_t = (lx, ly), t
                t += DT; step += 1
                # cadence temps reel x speed
                if self.speed > 0:
                    lag = (t - t_ref) / self.speed - (time.time() - t_wall_ref)
                    if lag > 0:
                        time.sleep(min(lag, 0.2))
            if dlv is not None:
                for e in dlv.log:
                    self.log(e["t"], "delivery", e["text"])
            if M is not None:
                try:
                    renderer.close()
                except Exception:
                    pass
            if not leg_done:
                break
            self.log(t, "info", "troncon %d/%d termine" % (k, len(pts) - 1))
            old = (nav, (x0, y0))
        if status == "en cours":
            status = "terminee"
        lon, lat = to_ll(x, y)
        bx, by = self.route[-1]
        self.metrics = {"status": status, "time_s": round(t), "dist_m": round(dist_true), "energy_wh": round(wh, 1),
                        "final_error_to_B_m": round(float(np.hypot(x - bx, y - by)), 1), "contacts": contacts, "falls": falls,
                        "err_mean_m": round(float(np.mean([e[1] for e in self.err_hist])) if self.err_hist else 0, 2),
                        "craters_crossed": crater_cross,
                        **(dlv.metrics() if dlv else {})}
        with self.lock:
            self.state["status"] = status
            self.state["final"] = self.metrics
        self.log(t, "info", "fin de mission : %s (%.0f m de B)" % (status, self.metrics["final_error_to_B_m"]))

    # ------------------------------------------------------------------ rendu MuJoCo
    def _make_renderer(self, M):
        import mujoco
        M.m.vis.global_.offwidth, M.m.vis.global_.offheight = 960, 540
        M.m.vis.map.zfar = 300
        r = mujoco.Renderer(M.m, 540, 960)
        self._cam = mujoco.MjvCamera()
        self._opt = mujoco.MjvOption(); self._opt.geomgroup[:] = 1
        self._last_render = 0.0
        return r

    def _render(self, M, r, t):
        import mujoco
        if time.time() - self._last_render < 0.08:
            return
        self._last_render = time.time()
        cam = self._cam
        x, y, yaw = M.pose
        z = float(M.d.qpos[2])
        mode = self.camera
        if mode == "tete":
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = [x + 4 * np.cos(yaw), y + 4 * np.sin(yaw), z + 0.2]
            cam.distance, cam.azimuth, cam.elevation = 4.0, np.degrees(yaw) + 180, -3
            cam.lookat[:] = [x + 4.0 * np.cos(yaw), y + 4.0 * np.sin(yaw), z + 0.45]
        else:
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING; cam.trackbodyid = 1
            d, el, az = {"suivi": (4.5, -16, 0), "dessus": (16, -65, 20), "loin": (35, -25, 30), "cote": (5, -8, 90)}.get(mode, (4.5, -16, 0))
            if self.mode == "go1":
                d *= 0.55
            cam.distance, cam.elevation, cam.azimuth = d, el, np.degrees(yaw) + az
        r.update_scene(M.d, cam, self._opt)
        img = r.render()
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, "JPEG", quality=72)
        self.frame = buf.getvalue()

    # ------------------------------------------------------------------ lecture concurrente
    def pop_updates(self, since_event):
        with self.lock:
            vox, self.vox_new = self.vox_new, []
            return {"state": dict(self.state), "events": self.events[since_event:], "vox": vox, "scan": self.scan,
                    "particles": self.particles, "path": self.local_path, "track_len": len(self.track)}
