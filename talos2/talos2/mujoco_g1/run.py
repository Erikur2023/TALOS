"""Mission G1 sous MuJoCo pilotee par le Navigator.
  python -m talos2.mujoco_g1.run --seed 100               # sans affichage (benchmark)
  python -m talos2.mujoco_g1.run --seed 100 --viewer      # demo avec la visionneuse MuJoCo
Capteurs : gyro = vitesse de lacet du pelvis + biais/bruit (ImuOdometry), odometrie = vitesse avant du pelvis
+ erreur d'echelle/bruit, LiDAR = mj_multiRay depuis la tete (groupe obstacles), pente = relief local + bruit."""
import argparse, json, time
from dataclasses import replace
import numpy as np
import mujoco
from ..params import Params
from ..world import World
from ..prior_map import PriorMap
from ..planner import Planner
from ..sensors import ImuOdometry
from ..navigator import Navigator
from .g1_policy import G1Walker, SIM_DT, gravity_orientation
from . import scene as S

NAV_DT = 0.1
STEPS_PER_NAV = int(round(NAV_DT / SIM_DT))
LIDAR_HEIGHT = 0.45            # ASSUMPTION m au-dessus du pelvis (LiDAR dans la tete du G1)


def mujoco_params(**kw):
    """Parametres de nav adaptes au domaine de la politique (mesure : calib.py, NOTES_NUIT.md)."""
    base = dict(v_cmd_max=0.6, w_cmd_max=0.6, v_back=-0.2, v_slow_min=0.25, slope_speed_limit=True,
                plan_slope_max=0.12)
    base.update(kw)
    return replace(Params(), **base)


def yaw_of(q):
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def scale_relief(w, prior, k):
    """Le monde ET la carte voient le relief attenue (sinon la nav serait faussee)."""
    if k == 1.0:
        return
    w.height = w.height * k; w.gx = w.gx * k; w.gy = w.gy * k; w.slope = w.slope * k
    prior.height = prior.height * k; prior.pgx = prior.pgx * k; prior.pgy = prior.pgy * k
    prior.slope = w.slope


class Lidar2D:
    def __init__(self, n_rays, max_range):
        self.ang = np.linspace(-np.pi, np.pi, n_rays, endpoint=False)
        self.n, self.max_range = n_rays, max_range
        self.gg = np.zeros(6, np.uint8); self.gg[S.G_OBST] = 1
        self.gid = np.zeros(n_rays, np.int32); self.dist = np.zeros(n_rays)

    def scan(self, m, d, x, y, z, yaw, sigma, dropout, rng):
        vec = np.c_[np.cos(yaw + self.ang), np.sin(yaw + self.ang), np.zeros(self.n)].ravel()
        mujoco.mj_multiRay(m, d, np.array([x, y, z]), vec, self.gg, 1, -1, self.gid, self.dist, None, self.n, self.max_range)
        r = np.where(self.dist < 0, self.max_range, np.minimum(self.dist, self.max_range))
        r = r + rng.normal(0, sigma, self.n) * (r < self.max_range)
        r[rng.random(self.n) < dropout] = self.max_range
        return self.ang.copy(), r


class G1Mission:
    """Monde MuJoCo + G1 + capteurs. Utilisable pas a pas (run() ci-dessous, ou d'autres boucles : depot, LiDAR 3D)."""
    def __init__(self, seed, p, relief_scale=1.0, world=None, prior=None):
        self.p = p
        self.w = world or World(seed)
        self.prior = prior or PriorMap(self.w, p, seed)
        scale_relief(self.w, self.prior, relief_scale)
        self.rng = np.random.default_rng(40_000 + seed)
        self.imu = ImuOdometry(p, seed)
        w = self.w
        corridor = Planner(self.prior, p).plan(w.start, w.goal)
        self.m, self.info = S.build(w, np.vstack([w.start, corridor, w.goal]))
        self.d = mujoco.MjData(self.m)
        self.walker = G1Walker()
        self.yaw0 = float(np.arctan2(*(w.goal - w.start)[::-1]))
        x0, y0 = map(float, w.start)
        d = self.d
        ring = [(x0 + 0.4 * np.cos(a), y0 + 0.4 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
        z_feet = max(float(w.height_at(x0, y0)), max(float(w.height_at(*q)) for q in ring))
        d.qpos[:3] = [x0, y0, z_feet + 0.80]                    # au-dessus du point le plus haut sous les pieds
        d.qpos[3:7] = [np.cos(self.yaw0 / 2), 0, 0, np.sin(self.yaw0 / 2)]
        self.walker.reset(d)
        mujoco.mj_forward(self.m, d)
        self.lidar = Lidar2D(p.lidar_rays, p.lidar_range)
        self.obst_set = set(self.info["obst_geom"].tolist())
        self.mech_j, self.dist_true = 0.0, 0.0
        self.contacts, self.last_contact_t = 0, -9.0
        for _ in range(int(1.0 / SIM_DT)):          # 1 s pour se stabiliser
            self.walker.step(self.m, d, [0, 0, 0])

    @property
    def pose(self):
        return float(self.d.qpos[0]), float(self.d.qpos[1]), yaw_of(self.d.qpos[3:7])

    def advance(self, vc, wc, t, dt=NAV_DT):
        """Avance de dt avec la consigne (vc, wc). Retourne (v_fwd, yawrate) vrais et un drapeau chute."""
        m, d = self.m, self.d
        xp, yp, yawp = self.pose
        for _ in range(int(round(dt / SIM_DT))):
            self.walker.step(m, d, [vc, 0.0, wc])
            self.mech_j += float(np.sum(np.maximum(0.0, d.actuator_force[:12] * d.qvel[6:18]))) * SIM_DT
        x, y, yaw = self.pose
        g = gravity_orientation(d.qpos[3:7])
        fell = d.qpos[2] - self.w.height_at(x, y) < 0.45 or g[2] > -0.5
        for c in d.contact[:d.ncon]:
            if (c.geom1 in self.obst_set) != (c.geom2 in self.obst_set):
                if t - self.last_contact_t > 1.0:
                    self.contacts += 1
                self.last_contact_t = t
                break
        dyaw = (yaw - yawp + np.pi) % (2 * np.pi) - np.pi
        dx, dy = x - xp, y - yp
        self.dist_true += float(np.hypot(dx, dy))
        ym = yawp + dyaw / 2
        v_fwd = (dx * np.cos(ym) + dy * np.sin(ym)) / dt
        self.v_lat = (-dx * np.sin(ym) + dy * np.cos(ym)) / dt
        return v_fwd, dyaw / dt, fell

    def proprio(self, v_fwd, yawrate, dt=NAV_DT):
        """Odometrie jambes (avant, lateral) avec erreur d'echelle et bruit + gyro biaise."""
        v_meas, gyro = self.imu.measure(v_fwd, yawrate, dt)
        v_lat = self.v_lat * self.imu.scale + self.imu.rng.normal(0, self.p.odom_noise)
        return v_meas, gyro, v_lat

    def slope_meas(self):
        x, y, yaw = self.pose
        gxt, gyt = self.w.grad_at(x, y)
        p, rng = self.p, self.rng
        return (gxt * np.cos(yaw) + gyt * np.sin(yaw) + rng.normal(0, p.imu_slope_noise),
                -gxt * np.sin(yaw) + gyt * np.cos(yaw) + rng.normal(0, p.imu_slope_noise))

    def scan2d(self):
        x, y, yaw = self.pose
        p = self.p
        return self.lidar.scan(self.m, self.d, x, y, float(self.d.qpos[2]) + LIDAR_HEIGHT, yaw,
                               p.lidar_sigma, p.lidar_dropout, self.rng)


def run(seed, matching=True, p=None, relief_scale=1.0, viewer=False, realtime=False, record=False,
        max_time=None, verbose=False, nav_factory=None, scan_fn=None, slope_fn=None, map_kind="trees", delivery=False,
        robot="g1"):
    if p is None:
        p = mujoco_params(**({"plan_slope_max": 0.35} if robot in ("go1", "isaac_go1_rough") else {}))
    if robot in ("unitree_mjlab", "gr00t_wbc", "isaac_go1_rough"):
        from .quad import ExtMission
        Mission = lambda *aa, **kk: ExtMission(*aa, key=robot, **kk)
    elif robot == "go1":
        from .quad import QuadMission as Mission
    else:
        Mission = G1Mission
    prior = None
    if map_kind == "osm":
        from ..osm_map import OsmMap, osm_params
        p = osm_params(p)
        world = World(seed, tracks=True)
        prior = OsmMap(world, p, seed)
        M = Mission(seed, p, relief_scale, world=world, prior=prior)
    else:
        M = Mission(seed, p, relief_scale)
    w = M.w
    x0, y0 = map(float, w.start)
    from ..delivery import trench_order_for
    trench_order = trench_order_for(w.trench, seed) if delivery else None
    nav = (nav_factory or Navigator)(M.prior, p, (x0, y0), M.yaw0, w.goal, seed, matching,
                                     no_go=[trench_order] if delivery else None)
    v = None
    if viewer:
        import mujoco.viewer
        v = mujoco.viewer.launch_passive(M.m, M.d)
        v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING; v.cam.trackbodyid = 1; v.cam.distance = 6; v.cam.elevation = -25
        v.opt.geomgroup[:] = 1                 # arbres (groupe 3) et tranchee (groupe 4) visibles
    max_time = max_time or p.max_time
    t, step = 0.0, 0
    status, falls = "timeout", 0
    errs, frames, cmd_log = [], [], []
    prog_ref, prog_t = np.array(M.pose[:2]), 0.0
    scan_every = int(round(p.scan_dt / NAV_DT))
    wall0 = time.time()
    from ..delivery import Delivery
    dlv, arrival = None, None
    while t < max_time:
        vc, wc = nav.command(t, NAV_DT)
        if nav.arrived and arrival is None:
            status = "arrived_est"
            arrival = M.pose[:2] + (t,)
            if not delivery:
                break
            dlv = Delivery(w.trench, p.payload_kg, p, seed, retreat_to=nav.retreat_point(), order=trench_order, v_approach=0.25,
                           standoff=1.8, guard_dist=1.2, v_guard=0.3)   # G1 : glisse ~0,1 m/s a l'arret en pente (calib)
        if dlv is not None:
            vc, wc = dlv.step(nav, t, vc, wc)
            if dlv.state == "DONE":
                break
        v_fwd, yawrate, fell = M.advance(vc, wc, t)
        if fell:
            falls += 1; status = "fall"; break
        if dlv is not None:
            dlv.perceive(M.pose[:2], nav.est[:2]); dlv.on_true_state(M.pose[:2])
        v_meas, gyro, v_lat = M.proprio(v_fwd, yawrate)
        nav.on_proprio(v_meas, gyro, NAV_DT, v_lat)
        if step % scan_every == 0:
            ang, rr = (scan_fn or (lambda M: M.scan2d()))(M)
            along, across = (slope_fn or (lambda M: M.slope_meas()))(M)   # inclinaison IMU : disponible aussi sans recalage (limitation de vitesse)
            nav.on_scan(ang, rr, t, along, across)
        x, y, yaw = M.pose
        ex, ey, eyaw, sigma = nav.est[:4]
        if step % 5 == 0:
            errs.append(float(np.hypot(x - ex, y - ey)))
            cmd_log.append((vc, wc, v_fwd, yawrate))
            if record:
                frames.append([round(t, 1), round(x, 2), round(y, 2), round(yaw, 3), round(ex, 2), round(ey, 2),
                               round(eyaw, 3), round(sigma, 2), round(v_fwd, 2), round(nav.remaining_m(), 0)])
        if verbose and step % 300 == 0:
            print("t=%5.0f pos=(%6.1f,%6.1f) err=%.2f v=%.2f/%.2f d_goal=%.1f" % (
                t, x, y, errs[-1] if errs else 0, v_fwd, vc, np.hypot(x - w.goal[0], y - w.goal[1])), flush=True)
        if v is not None:
            if not v.is_running():
                status = "viewer_closed"; break
            v.sync()
            if realtime:
                lag = t - (time.time() - wall0)
                if lag > 0:
                    time.sleep(lag)
        if t - prog_t > 90:
            if np.hypot(x - prog_ref[0], y - prog_ref[1]) < 1.0:
                status = "stuck"; break
            prog_ref, prog_t = np.array([x, y]), t
        t += NAV_DT; step += 1
    if v is not None:
        v.close()
    x, y = (arrival[0], arrival[1]) if arrival else M.pose[:2]
    d_goal = float(np.hypot(x - w.goal[0], y - w.goal[1]))
    e = np.array(errs) if errs else np.zeros(1)
    cl = np.array(cmd_log) if cmd_log else np.zeros((1, 4))
    moving = (cl[:, 0] > 0.2) & (np.abs(cl[:, 1]) < 0.1)
    met = {
        "seed": seed, "matching": matching, "sim": "mujoco_g1", "relief_scale": relief_scale,
        "success": bool(d_goal <= p.success_tolerance_true and status != "fall"), "status": status,
        "falls": falls, "time_s": round(t, 1), "dist_true_m": round(M.dist_true, 1),
        "final_goal_error_m": round(d_goal, 2), "err_mean_m": round(float(e.mean()), 2),
        "err_p95_m": round(float(np.percentile(e, 95)), 2), "err_max_m": round(float(e.max()), 2),
        "contacts": M.contacts, "replans": nav.replans, "anomalies_detected": len(nav.anomalies),
        "mech_positive_work_wh": round(M.mech_j / 3600, 1),
        "v_track_ratio": round(float(np.median(cl[moving, 2] / cl[moving, 0])), 2) if moving.any() else None,
        "n_obst_in_scene": M.info["n_obst"], "wall_s": round(time.time() - wall0, 1), "map": map_kind, "robot": robot,
    }
    if dlv is not None:
        met.update(dlv.metrics()); met["delivery_time_s"] = round(t - arrival[2], 1)
    if getattr(M, "lidar3d", None) is not None:
        met.update({"lidar3d_" + k: v for k, v in M.lidar3d.summary().items()})
    out = {"metrics": met}
    if record:
        out["telemetry"] = {"frames": frames, "events": nav.events + (dlv.log if dlv else []), "paths": nav.paths}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--dr", action="store_true", help="odometrie seule (pas de recalage)")
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--relief", type=float, default=1.0)
    ap.add_argument("--max-time", type=float, default=None)
    ap.add_argument("--lidar3d", action="store_true")
    ap.add_argument("--osm", action="store_true", help="carte style OSM (monde avec chemins)")
    ap.add_argument("--delivery", action="store_true", help="phase de depot a la tranchee")
    a = ap.parse_args()
    r = run(a.seed, matching=not a.dr, relief_scale=a.relief, viewer=a.viewer, realtime=a.viewer,
            max_time=a.max_time, verbose=True, delivery=a.delivery, map_kind="osm" if a.osm else "trees",
            **(__import__("talos2.mujoco_g1.lidar3d", fromlist=["x"]).attach_lidar3d() if a.lidar3d else {}))
    print(json.dumps(r["metrics"], indent=1))


if __name__ == "__main__":
    main()
