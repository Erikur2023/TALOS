"""Navigateur embarque, INDEPENDANT du simulateur.

Entrees (ce que le robot mesure)          Sortie
  - gyro (rad/s), odometrie jambes (m/s)    consigne (vx, wz)
  - scan LiDAR 2D (angles corps, portees)   + etat : estimee, sigma, chemin, ETA, evenements
  - pente IMU (le long / en travers, m/m)

Il ne voit jamais la verite terrain : seulement la carte a priori, la destination et ses capteurs.
Le meme objet pilote le simulateur point (sim.py) et le G1 sous MuJoCo (mujoco_g1/).
"""
import numpy as np
from .localization import ParticleFilter
from .planner import Planner
from .sensors import DeadReckoning


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def cluster_scan(ang, rng_, est_x, est_y, est_yaw, prior, max_r):
    """Groupe les rayons adjacents -> amas (centre, largeur, nb) ; ne garde que ceux hors carte."""
    n = len(ang)
    hit = rng_ < max_r - 0.5
    px = est_x + rng_ * np.cos(est_yaw + ang)
    py = est_y + rng_ * np.sin(est_yaw + ang)
    unexplained = hit & (prior.known_dist_at(px, py) > 0.7)
    clusters, cur = [], []
    for k in range(n + 1):
        i = k % n
        ok = unexplained[i] and (not cur or np.hypot(px[i] - px[cur[-1]], py[i] - py[cur[-1]]) < 2.5)
        if ok and k < n:
            cur.append(i)
        else:
            if cur:
                xs, ys = px[cur], py[cur]
                width = float(np.hypot(xs.max() - xs.min(), ys.max() - ys.min()))
                clusters.append((float(xs.mean()), float(ys.mean()), width, len(cur)))
            cur = [i] if (unexplained[i] and k < n) else []
    return clusters


class Navigator:
    def __init__(self, prior, p, start_xy, yaw0, goal_xy, seed=0, matching=True, no_go=None):
        self.prior, self.p, self.goal = prior, p, np.asarray(goal_xy, float)
        self.matching = matching
        x, y = map(float, start_xy)
        self.dr = DeadReckoning(x, y, yaw0)
        self.pf = ParticleFilter(prior, p, x, y, yaw0, seed) if matching else None
        self.planner = Planner(prior, p)
        for line in (no_go or []):                       # ex. tranchee transmise par les soldats : infranchissable
            line = np.asarray(line, float)
            for a, b in zip(line[:-1], line[1:]):
                for u in np.linspace(0, 1, max(2, int(np.hypot(*(b - a)) / 0.5) + 1)):
                    q = a + u * (b - a)
                    self.planner.add_obstacle(q[0], q[1], 2.5)   # 1,5 m + erreur de l'ordre (~1 m)
        self.path = self.planner.plan((x, y), self.goal)
        self.path_idx = 0
        self.est = (x, y, yaw0, 1.0, 1.0, 0.0)          # x, y, cap, sigma, echelle, biais gyro
        self.last_scan = None
        self.seen_cells, self.anomalies, self.events = set(), [], []
        self.paths = [] if self.path is None else [{"t": 0.0, "pts": np.round(self.path[::3], 1).tolist()}]
        self.replans = self.reflex = 0
        self.last_replan = -99.0
        self.replan_pending = False
        self.blocked_for, self.escape_until = 0.0, -1.0
        self.turn_dir, self.turn_until = 1, -1.0
        self.step_count = 0
        self.arrived = False
        self.ema_cmd, self.ema_odom, self.stalls = 0.0, 0.0, 0
        self.slope_along = None
        self.escape_mode, self.escape_heading = "back", 0.0
        self.prog_best, self.prog_t = 1e9, 0.0
        self.crumbs = [(x, y)]                           # fil d'Ariane : une position estimee par metre parcouru
        self.target = tuple(map(float, self.goal))

    # ------------------------------------------------------------------ etat
    def estimate(self):
        if self.matching:
            self.est = self.pf.estimate()
        else:
            self.est = (self.dr.x, self.dr.y, self.dr.yaw, 0.0, 1.0, 0.0)
        return self.est

    def remaining_m(self):
        if self.path is None or self.path_idx >= len(self.path) - 1:
            return 0.0
        return float(np.sum(np.hypot(*np.diff(self.path[self.path_idx:], axis=0).T)))

    def retarget(self, goal_xy, t):
        """Nouvelle destination (ex. point de repli apres le depot) : replanification avec les memes obstacles / zones interdites."""
        self.goal = np.asarray(goal_xy, float)
        self.arrived = False
        ex, ey = self.est[:2]
        self._new_path(t, "retrait : replanification vers le point de repli", ex, ey)
        self.prog_best, self.prog_t = self.remaining_m(), t

    def retreat_point(self, back_m=15.0, step_m=3.0):
        """Points deja parcourus (chemin suivi, a rebours, tous les step_m) jusqu'a ~back_m : repli apres le depot."""
        pts = self.path[:max(2, self.path_idx + 1)][::-1]
        d = np.r_[0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
        idx = [min(int(np.searchsorted(d, s)), len(pts) - 1) for s in np.arange(step_m, back_m + 1e-6, step_m)]
        return pts[sorted(set(idx))]

    def _new_path(self, t, text, ex, ey):
        newp = self.planner.plan((ex, ey), self.goal)
        if newp is not None:
            self.path, self.path_idx = newp, 0
            self.replans += 1
            self.paths.append({"t": round(t, 1), "pts": np.round(self.path[::3], 1).tolist()})
            if text:
                self.events.append({"t": round(t, 1), "type": "replan", "x": round(ex, 1), "y": round(ey, 1), "text": text})
        return newp

    # ------------------------------------------------------------- commande
    def command(self, t, dt):
        """Consigne (v, w) a partir de l'estimee courante. Met self.arrived a True a destination."""
        p = self.p
        ex, ey, eyaw = self.estimate()[:3]
        if np.hypot(ex - self.goal[0], ey - self.goal[1]) < p.goal_tolerance_est:
            self.arrived = True
            return 0.0, 0.0
        path = self.path
        look = 4.0
        seg = path[self.path_idx:self.path_idx + 40]
        self.path_idx += int(np.argmin(np.hypot(seg[:, 0] - ex, seg[:, 1] - ey)))
        j = self.path_idx
        while j < len(path) - 1 and np.hypot(path[j, 0] - ex, path[j, 1] - ey) < look:
            j += 1
        tx, ty = path[j]
        self.target = (float(tx), float(ty))
        err_h = wrap(np.arctan2(ty - ey, tx - ex) - eyaw)
        w_cmd = float(np.clip(1.5 * err_h, -p.w_cmd_max, p.w_cmd_max))
        v_cmd = p.v_cmd_max * max(0.0, np.cos(err_h)) if abs(err_h) < 1.0 else 0.0
        # vitesse selon terrain CONNU (carte)
        if self.prior.mud[int(np.clip(ey, 0, self.prior.mud.shape[0] - 1)), int(np.clip(ex, 0, self.prior.mud.shape[1] - 1))]:
            v_cmd *= 0.8
        # evitement reactif LiDAR : couloir de balayage devant le robot
        blocked_now = False
        if self.last_scan is not None:
            ang, rr = self.last_scan
            fwd, lat = rr * np.cos(ang), rr * np.sin(ang)
            corridor = (np.abs(lat) < p.corridor_half) & (fwd > 0)
            dc = float(fwd[corridor].min()) if corridor.any() else 99.0
            too_close = float(rr[np.abs(ang) < 1.75].min()) < p.close_dist and dc < 1.5
            if dc < p.stop_dist or too_close:
                blocked_now = True
                v_cmd = 0.0
                left = rr[(ang > 0.2) & (ang < 1.6)].mean()
                right = rr[(ang < -0.2) & (ang > -1.6)].mean()
                if t > self.turn_until:
                    self.turn_dir = 1 if left >= right else -1
                    self.turn_until = t + 2.0
                w_cmd = p.w_cmd_max * self.turn_dir
                self.reflex += 1 if (self.step_count % 5 == 0) else 0
            elif dc < p.slow_dist:
                v_cmd *= (dc - p.stop_dist) / (p.slow_dist - p.stop_dist)
                if v_cmd > 0:
                    v_cmd = max(v_cmd, p.v_slow_min)
        if p.slope_speed_limit and self.slope_along is not None:
            # descente : le G1 accelere seul (mesure MuJoCo, chute graine 101) -> on reduit la consigne.
            # (en montee, reduire la consigne le fait caler : on ne touche a rien)
            v_cmd *= float(np.clip(1.0 - (-self.slope_along - 0.08) / 0.2, 0.5, 1.0))
        # calage : on commande d'avancer mais l'odometrie ne bouge pas (pente + ralentissement, pied coince...)
        if p.stall_detect and self.ema_cmd > 0.05 and self.ema_odom < 0.25 * self.ema_cmd and self.escape_until <= t:
            blocked_now = True
            self.stalls += 1 if (self.step_count % 10 == 0) else 0
        self.blocked_for = self.blocked_for + dt if blocked_now else max(0.0, self.blocked_for - 0.3 * dt)
        # chien de garde de progression : le chemin restant doit diminuer (detecte les va-et-vient sans fin)
        rem = self.remaining_m()
        if rem < self.prog_best - 1.0:
            self.prog_best, self.prog_t = rem, t
        no_progress = p.progress_watchdog > 0 and t - self.prog_t > p.progress_watchdog
        if self.escape_until > t:                                     # manoeuvre de degagement en cours
            if self.escape_mode == "back":
                v_cmd, w_cmd = p.v_back, 0.0
            elif self.escape_mode == "crumb":                         # retour par ou l'on est venu
                to = self.escape_goal - np.array([ex, ey])
                if np.hypot(*to) < 0.7:
                    self.escape_until = t
                e = wrap(np.arctan2(to[1], to[0]) - eyaw)
                w_cmd = float(np.clip(1.5 * e, -p.w_cmd_max, p.w_cmd_max))
                ang, rr = self.last_scan
                v_cmd = p.v_escape if (abs(e) < 0.35 and float(rr[np.abs(ang) < 0.35].min()) > 0.3) else 0.0
            else:                                                     # vers la plus grande ouverture
                e = wrap(self.escape_heading - eyaw)
                w_cmd = float(np.clip(1.5 * e, -p.w_cmd_max, p.w_cmd_max))
                ang, rr = self.last_scan
                front = float(rr[np.abs(ang) < 0.35].min())
                v_cmd = p.v_escape if (abs(e) < 0.35 and front > p.close_dist + 0.2) else 0.0
        elif (self.blocked_for > 10.0 or no_progress) and self.last_scan is not None:
            ang, rr = self.last_scan
            rear_free = float(rr[np.abs(ang) > 2.6].min()) > 1.3
            back = [c for c in self.crumbs[::-1] if np.hypot(c[0] - ex, c[1] - ey) > 3.5]
            if rear_free and not no_progress:
                self.escape_mode = "back"
            elif back:
                self.escape_mode = "crumb"
                self.escape_goal = np.array(back[0])
            else:
                # ouverture : portee minimale sur +-12 deg, plafonnee, avec preference pour la direction du chemin
                k = max(1, int(round(np.radians(12) / (2 * np.pi / len(ang)))))
                clr = np.min(np.stack([np.roll(np.minimum(rr, 8.0), s) for s in range(-k, k + 1)]), axis=0)
                tgt = wrap(np.arctan2(self.target[1] - ey, self.target[0] - ex) - eyaw)
                score = clr + 1.0 * np.cos(ang - tgt) - 2.0 * np.cos(ang)      # on evite de refoncer tout droit
                i = int(np.argmax(np.where(clr > 1.5, score, -99)))
                self.escape_mode = "gap"
                self.escape_heading = eyaw + ang[i]
            self.escape_until = t + (12.0 if self.escape_mode == "crumb" else 5.0)
            self.planner.add_obstacle(ex + 1.4 * np.cos(eyaw), ey + 1.4 * np.sin(eyaw), 1.0)
            # le passage que le chemin voulait emprunter est declare bouche -> la replanification doit en changer
            ahead = self.path[self.path_idx:self.path_idx + 6]
            for q in ahead[np.hypot(ahead[:, 0] - ex, ahead[:, 1] - ey) > 1.0][:3]:
                self.planner.add_obstacle(q[0], q[1], 0.8)
            self.last_replan = -99.0
            self._new_path(t, None, ex, ey)
            self.events.append({"t": round(t, 1), "type": "replan", "x": round(ex, 1), "y": round(ey, 1),
                                "text": ("sans progression %.0f s" % p.progress_watchdog if no_progress else "bloque 10 s")
                                        + " : degagement (%s) et replanification" % {"back": "marche arriere", "crumb": "retour sur ses pas", "gap": "vers l'ouverture"}[self.escape_mode]})
            self.blocked_for = 0.0
            self.prog_best, self.prog_t = self.remaining_m(), t
        self.step_count += 1
        a = dt / 3.0                                                    # moyennes glissantes ~3 s
        self.ema_cmd += a * (max(v_cmd, 0.0) - self.ema_cmd)
        self.last_cmd = (float(v_cmd), float(w_cmd))
        return float(v_cmd), float(w_cmd)

    # ------------------------------------------------------------ capteurs
    def on_proprio(self, v_meas, gyro, dt, v_lat=0.0):
        """v_meas / v_lat : odometrie jambes (avant / lateral, repere corps), gyro : lacet."""
        self.dr.step(v_meas, gyro, dt, v_lat)
        ex, ey = self.est[0], self.est[1]
        if np.hypot(ex - self.crumbs[-1][0], ey - self.crumbs[-1][1]) > 1.0:
            self.crumbs.append((float(ex), float(ey)))
            del self.crumbs[:-30]
        self.ema_odom += dt / 3.0 * (max(v_meas, 0.0) - self.ema_odom)
        if self.matching:
            self.pf.predict(v_meas, gyro, dt, v_lat)

    def _odom_gate(self, t):
        """Garde de coherence (cartes peu fiables) : depuis un point d'ancrage, l'estimee ne doit pas s'ecarter de la
        prediction de l'odometrie de plus de ~5 m + 8 % de la distance parcourue (derive odometrique attendue).
        Sinon le recalage est rejete : le nuage est recentre sur la prediction odometrique."""
        dr = self.dr
        ex, ey, eyaw = self.est[:3]
        if getattr(self, "_anchor", None) is None:
            self._anchor = (ex, ey, eyaw, dr.x, dr.y, dr.yaw)
            return
        ax, ay, ayaw, dx0, dy0, dyaw0 = self._anchor
        # cap : celui du gyroscope (derive lente) ; position : odometrie + decalage mesure au point d'ancrage
        ddx, ddy = dr.x - dx0, dr.y - dy0
        px, py, pyaw = ax + ddx, ay + ddy, dr.yaw
        dist = float(np.hypot(ddx, ddy))
        dev = float(np.hypot(ex - px, ey - py))
        if dev > 5.0 + 0.08 * dist:
            pf = self.pf
            n = len(pf.s)
            pf.s[:, 0] = px + pf.rng.normal(0, 1.0, n)
            pf.s[:, 1] = py + pf.rng.normal(0, 1.0, n)
            pf.s[:, 2] = pyaw + pf.rng.normal(0, 0.03, n)
            pf.w = np.full(n, 1.0 / n)
            self.est = pf.estimate()
            self.gate_rejects = getattr(self, "gate_rejects", 0) + 1
            self.events.append({"t": round(t, 1), "type": "anomaly", "x": round(px, 1), "y": round(py, 1),
                                "text": "recalage incoherent avec l'odometrie (ecart %.0f m) : rejete" % dev})
            self._anchor = (px, py, pyaw, dr.x, dr.y, dr.yaw)
        elif dist > 15.0 and self.est[3] < 1.5:
            self._anchor = (ex, ey, eyaw, dr.x, dr.y, dr.yaw)

    def on_heading(self, yaw_meas, sigma):
        """Compas magnetique : corrige le cap du filtre et (filtre complementaire) celui de l'odometrie seule."""
        if self.matching:
            self.pf.update_heading(yaw_meas, sigma)
        d = (yaw_meas - self.dr.yaw + np.pi) % (2 * np.pi) - np.pi
        self.dr.yaw += 0.05 * d

    def on_scan(self, ang, rr, t, slope_along=None, slope_across=None, avoid_rr=None):
        """avoid_rr : portees d'obstacles vus par d'autres moyens (creux detectes par le LiDAR 3D) : utilises pour
        l'evitement, PAS pour le recalage."""
        p = self.p
        self.last_scan = (ang, rr if avoid_rr is None else np.minimum(rr, avoid_rr))
        if slope_along is not None:
            self.slope_along = float(slope_along)
        if self.matching:
            self.pf.update(ang, rr)
            if slope_along is not None:
                self.pf.update_relief(slope_along, slope_across)
            self.est = self.pf.estimate()
            if p.pf_odom_gate:
                self._odom_gate(t)
        if avoid_rr is not None:
            rr = np.minimum(rr, avoid_rr)
        ex, ey, eyaw = self.est[:3]
        clusters = cluster_scan(ang, rr, ex, ey, eyaw, self.prior, p.lidar_range)
        new_block = False
        if p.map_hits_as_obstacles:
            # carte sans objets individuels : chaque impact devient un obstacle local (grille d'occupation simple)
            hit = rr < p.lidar_range - 0.5
            hx, hy = ex + rr[hit] * np.cos(eyaw + ang[hit]), ey + rr[hit] * np.sin(eyaw + ang[hit])
            for qx, qy in zip(hx, hy):
                cell = (int(qx), int(qy), 1)
                if cell not in self.seen_cells:
                    self.seen_cells.add(cell)
                    self.planner.add_obstacle(qx, qy, p.hit_obstacle_radius)
                    new_block = True
        for cx, cy, width, nrays in clusters:
            cell = (int(cx // 2), int(cy // 2))
            if cell in self.seen_cells:
                continue
            self.seen_cells.add(cell)
            if not p.map_hits_as_obstacles:
                self.planner.add_obstacle(cx, cy, max(0.8, width / 2 + 0.9))
                new_block = True
            # anomalie = gros objet la ou la carte dit 'degage' (carte arbres : toujours le cas ici, dist > 0,7)
            if width >= 1.6 and nrays >= p.anomaly_min_points - 1 and float(self.prior.dist_at(cx, cy)) > 0.7 and \
                    all(np.hypot(cx - a["x"], cy - a["y"]) > 6 for a in self.anomalies):
                ev = {"t": round(t, 1), "type": "anomaly", "x": round(cx, 1), "y": round(cy, 1),
                      "w": round(width, 1), "text": "obstacle hors carte, largeur ~%.1f m" % width}
                self.anomalies.append(ev)
                self.events.append(ev)
        self.replan_pending = self.replan_pending or new_block
        if self.replan_pending and t - self.last_replan > 3.0:
            self.replan_pending = False
            ahead = self.path[self.path_idx:self.path_idx + 25]
            gi = np.clip(ahead[:, 1].astype(int), 0, self.planner.n - 1)
            gj = np.clip(ahead[:, 0].astype(int), 0, self.planner.n - 1)
            if self.planner.dyn_block[gi, gj].any():
                self.last_replan = t
                self._new_path(t, "replanification (obstacle sur la route)", ex, ey)
