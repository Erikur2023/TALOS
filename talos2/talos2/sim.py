"""Mission complete en boucle fermee, simulateur 'point avec glissement'.
Le robot est pilote par Navigator (navigator.py), qui ne voit que les capteurs et la carte a priori."""
import os
import numpy as np
from .params import Params
from .world import World
from .prior_map import PriorMap
from .sensors import ImuOdometry
from .navigator import Navigator, cluster_scan as _cluster_scan, wrap  # noqa: F401 (compat)
from .energy import power_w

DT = 0.1
DEBUG = bool(os.environ.get('TALOS_DEBUG'))


SENSOR_H, FOV_UP, VOX = 1.2, np.tan(np.radians(7.0)), 0.5     # LiDAR 3D (MID-360 retourne, voir mujoco_g1/lidar3d.py)


def _vox_add(vox, w, prior, true_xy, est, ang, rr, rmax, t):
    """Carte 3D de ce que le robot VOIT, construite dans le repere ESTIME (les erreurs de localisation s'y voient).
    Approximation du LiDAR 3D a partir du scan : chaque impact = tronc vu du sol jusqu'a la hauteur visible
    (champ vertical +7 deg) ; le long des rayons, points de sol jusqu'a 15 m. Altitudes relatives a celle du robot,
    posees sur le MNT de la carte a la position estimee."""
    ex, ey, eyaw = est
    tx, ty = true_xy
    hz = getattr(w, "tree_h", None)
    z0 = float(prior.height[int(np.clip(ey, 0, prior.height.shape[0] - 1)), int(np.clip(ex, 0, prior.height.shape[1] - 1))])         - float(w.height_at(tx, ty))
    for k in range(0, len(ang), 2):
        r = rr[k]
        hit = r < rmax - 0.5
        c, s_ = np.cos(eyaw + ang[k]), np.sin(eyaw + ang[k])
        if k % 4 == 0:
            for d in np.arange(1.5, min(r, 15.0), 1.5):
                gx_, gy_ = ex + d * c, ey + d * s_
                gz = z0 + float(w.height_at(tx + d * c, ty + d * s_))
                vox.setdefault((int(gx_ / VOX), int(gy_ / VOX), int(np.floor(gz / VOX))), (0, t))
        if hit:
            px, py = ex + r * c, ey + r * s_
            gz = z0 + float(w.height_at(tx + r * c, ty + r * s_))
            top = SENSOR_H + r * FOV_UP
            if hz is not None:
                rel = w.obst[:, :2] - [tx + r * c, ty + r * s_]
                j = int(np.argmin(np.einsum("ij,ij->i", rel, rel)))
                top = min(top, float(hz[j]))
            for z in np.arange(0.0, top, VOX):
                vox.setdefault((int(px / VOX), int(py / VOX), int(np.floor((gz + z) / VOX))), (1, t))


def _vox_export(vox, cap=80_000):
    if not vox:
        return []
    a = np.array([[i * VOX + VOX / 2, j * VOX + VOX / 2, k * VOX, c, t] for (i, j, k), (c, t) in vox.items()])
    a = a[np.argsort(a[:, 4])][:cap]
    return np.round(a, 1).tolist()


def run_mission(seed, matching=True, p=None, record=True, world=None, prior=None, delivery=False):
    p = p or Params()
    w = world or World(seed)
    prior = prior or PriorMap(w, p, seed)
    imu = ImuOdometry(p, seed)
    rng = np.random.default_rng(40_000 + seed)

    x, y = map(float, w.start)
    yaw = float(np.arctan2(*(w.goal - w.start)[::-1]))
    from .delivery import trench_order_for
    trench_order = trench_order_for(w.trench, seed) if delivery else None
    nav = Navigator(prior, p, (x, y), yaw, w.goal, seed, matching, no_go=[trench_order] if delivery else None)
    if nav.path is None:
        return {"metrics": {"success": False, "reason": "no_path"}}
    path_len0 = float(np.sum(np.hypot(*np.diff(nav.path, axis=0).T)))

    t, pause, wh_used = 0.0, 0.0, 0.0
    contacts = 0
    dist_true = 0.0
    errs, dists = [], []
    rec = {"frames": [], "scans": [], "cloud": [], "events": []}
    vox = {}
    prog_ref, prog_t = (x, y), 0.0
    status = "timeout"
    step = 0
    scan_every = int(round(p.scan_dt / DT))
    from .thermal import ThermalModel
    from .delivery import Delivery
    th, dT_peak = ThermalModel(), 0.0
    dlv, arrival = None, None
    payload = p.payload_kg

    while t < p.max_time:
        v_cmd, w_cmd = nav.command(t, DT)
        if nav.arrived and arrival is None:
            status = "arrived_est"
            arrival = (x, y, t, wh_used)
            if not delivery:
                break
            dlv = Delivery(w.trench, p.payload_kg, p, seed, retreat_to=nav.retreat_point(), order=trench_order)
        if dlv is not None:
            v_cmd, w_cmd = dlv.step(nav, t, v_cmd, w_cmd)
            if dlv.state == "DONE":
                break
        if DEBUG and step % 10 == 0:
            print('t=%.1f pos=(%.1f,%.1f) yaw=%.2f v=%.2f w=%.2f' % (t, x, y, yaw, v_cmd, w_cmd))
        # ---------- dynamique REELLE ----------
        if pause > 0:
            pause -= DT
            v_cmd_eff = 0.0
        else:
            v_cmd_eff = v_cmd
        w_true = w_cmd * 0.95 + rng.normal(0, 0.02)
        slope = float(w.slope_at(x, y)); mud = bool(w.mud_at(x, y))
        slope_f = max(0.3, 1.0 - 0.8 * max(0.0, slope - p.slope_slow_from))
        v_true = v_cmd_eff * max(0.3, slope_f * (p.mud_slip if mud else 1.0))   # glissement boue invisible aux jambes
        yaw += w_true * DT
        nx, ny = x + v_true * np.cos(yaw) * DT, y + v_true * np.sin(yaw) * DT
        k, push = w.contact(nx, ny, p.robot_radius)
        if k >= 0 and np.hypot(*push) > 1e-3:
            nx, ny = nx + push[0] * 1.02, ny + push[1] * 1.02
            if pause <= 0:
                contacts += 1
                rec["events"].append({"t": round(t, 1), "type": "contact", "x": round(nx, 1), "y": round(ny, 1),
                                      "text": "contact obstacle (%s)" % ("anomalie" if w.obst_kind[k] else "arbre")})
            pause = p.obstacle_contact_pause
        dist_true += float(np.hypot(nx - x, ny - y))
        x, y = float(np.clip(nx, 0.5, w.size - 0.5)), float(np.clip(ny, 0.5, w.size - 0.5))
        pw = power_w(p, v_true, slope, mud, payload)
        wh_used += pw * DT / 3600.0
        dT_peak = max(dT_peak, th.step(pw, DT))
        if dlv is not None:
            dlv.perceive((x, y), nav.est[:2]); dlv.on_true_state((x, y))
            payload = dlv.payload

        # ---------- capteurs ----------
        v_meas, gyro = imu.measure(v_cmd_eff * slope_f, w_true, DT)
        nav.on_proprio(v_meas, gyro, DT)
        if step % scan_every == 0:
            ang, rr = w.lidar_scan(x, y, yaw, p.lidar_rays, p.lidar_range, p.lidar_sigma, p.lidar_dropout, rng)
            along = across = None
            if matching:
                gxt, gyt = w.grad_at(x, y)
                along = gxt * np.cos(yaw) + gyt * np.sin(yaw) + rng.normal(0, p.imu_slope_noise)
                across = -gxt * np.sin(yaw) + gyt * np.cos(yaw) + rng.normal(0, p.imu_slope_noise)
            nav.on_scan(ang, rr, t, along, across)

        # ---------- enregistrement ----------
        ex, ey, eyaw, sigma = nav.est[:4]
        err = float(np.hypot(x - ex, y - ey))
        if step % 5 == 0:
            errs.append(err); dists.append(dist_true)
            remaining = nav.remaining_m()
            v_avg = max(0.3, dist_true / max(t, 1.0)) if t > 20 else 0.7
            if record:
                rec["frames"].append([round(t, 1), round(x, 2), round(y, 2), round(yaw, 3),
                                      round(ex, 2), round(ey, 2), round(eyaw, 3), round(sigma, 2),
                                      round(v_true, 2), round(p.battery_wh - wh_used, 1),
                                      round(remaining / v_avg, 0), round(err, 2),
                                      round(nav.dr.x, 2), round(nav.dr.y, 2), int(mud), round(slope, 2), round(remaining, 0),
                                      round(th.dT, 2), payload, dlv.state if dlv else "NAV"])
        if record and step % 10 == 0 and nav.last_scan is not None:
            ang, rr = nav.last_scan
            hit = rr < p.lidar_range - 0.5
            pts = np.c_[ex + rr[hit] * np.cos(eyaw + ang[hit]), ey + rr[hit] * np.sin(eyaw + ang[hit])]
            rec["scans"].append(np.round(pts[::2], 1).tolist())
            _vox_add(vox, w, prior, (x, y), (ex, ey, eyaw), ang, rr, p.lidar_range, t)
            if matching:
                sel = nav.pf.s[::max(1, len(nav.pf.s) // 60), :2]
                rec["cloud"].append(np.round(sel, 1).tolist())
        if t - prog_t > 90:
            if np.hypot(x - prog_ref[0], y - prog_ref[1]) < 1.0:
                status = "stuck"
                break
            prog_ref, prog_t = (x, y), t
        t += DT
        step += 1

    xa, ya = (arrival[0], arrival[1]) if arrival else (x, y)
    d_goal = float(np.hypot(xa - w.goal[0], ya - w.goal[1]))
    success = d_goal <= p.success_tolerance_true
    errs_a = np.array(errs) if errs else np.zeros(1)
    usable = p.battery_wh * (1 - p.battery_reserve)
    m = {
        "seed": seed, "matching": matching, "success": bool(success), "status": status,
        "time_s": round(t, 1), "dist_true_m": round(dist_true, 1), "path_len_planned_m": round(path_len0, 1),
        "final_goal_error_m": round(d_goal, 2),
        "err_mean_m": round(float(errs_a.mean()), 2), "err_median_m": round(float(np.median(errs_a)), 2),
        "err_p95_m": round(float(np.percentile(errs_a, 95)), 2), "err_max_m": round(float(errs_a.max()), 2),
        "err_final_m": round(float(errs_a[-1]), 2),
        "contacts": contacts, "reflex_stops": nav.reflex, "replans": nav.replans,
        "anomalies_detected": len(nav.anomalies),
        "energy_wh": round(float(wh_used), 1), "battery_left_pct": round(float(100 * (p.battery_wh - wh_used) / p.battery_wh), 1),
        "within_usable_battery": bool(wh_used <= usable),
        "thermal_dT_peak_k": round(dT_peak, 1), "thermal_dT_final_k": round(th.dT, 1),
        "thermal_detect_range_peak_m": round(th.detection_range_m(dT_peak)),
    }
    if arrival is not None:
        m["nav_time_s"] = round(arrival[2], 1)
    if dlv is not None:
        m.update(dlv.metrics())
        m["delivery_time_s"] = round(t - arrival[2], 1)
        m["delivery_energy_wh"] = round(wh_used - arrival[3], 2)
    out = {"metrics": m, "err_series": {"dist": dists, "err": errs}}
    if record:
        rec["paths"] = nav.paths
        rec["events"] = sorted(rec["events"] + nav.events + (dlv.log if dlv else []), key=lambda e: e["t"])
        rec["world"] = {
            "size": w.size, "start": w.start.round(1).tolist(), "goal": w.goal.round(1).tolist(),
            "trees": np.round(w.trees, 1).tolist(), "prior_trees": np.round(prior.trees, 1).tolist(),
            "anomalies": np.round(w.anomalies, 1).tolist(), "trench": w.trench.round(1).tolist(),
            "height": np.round(w.height[::3, ::3], 1).tolist(), "mud": w.mud[::3, ::3].astype(int).tolist(),
        }
        rec["vox"] = _vox_export(vox)
        rec["params"] = {"battery_wh": p.battery_wh, "reserve": p.battery_reserve, "payload_kg": p.payload_kg}
        if dlv is not None:
            rec["delivery"] = {"trench_order": np.round(dlv.trench_order, 2).tolist()}
        out["telemetry"] = rec
    return out
