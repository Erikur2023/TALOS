"""Serveur de l'application TALOS C2 (FastAPI) : donnees region, planification, mission en direct (WebSocket),
flux video MuJoCo (MJPEG). Lance par talos_app.app (fenetre native) ou : python -m talos_app.server"""
import asyncio, json, os, time
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .datasets import BUILD
from . import region as RG
from .route import plan as plan_route
from .mission import Mission

WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
app = FastAPI(title="TALOS C2")
STATE = {"mission": None, "overlays": {}}


@app.middleware("http")
async def no_cache_web(request, call_next):
    r = await call_next(request)
    if request.url.path.startswith("/web/") and "vendor" not in request.url.path or request.url.path == "/":
        r.headers["Cache-Control"] = "no-cache"
    return r


@app.get("/")
def index():
    return FileResponse(os.path.join(WEB, "index.html"))


app.mount("/web", StaticFiles(directory=WEB), name="web")


@app.get("/api/region")
def api_region():
    R = RG.get()
    return {"summary": R.summary(), "allied": R.allied, "places": [p for p in R.places if p["k"] in ("city", "town", "village")],
            "front": {"occupied": R.front["occupied"], "grey": R.front["grey"], "line": R.front["line"], "date": R.front["date"]}}


@app.get("/api/layer/{name}")
def api_layer(name: str):
    if name not in ("roads", "tracks", "tree_rows", "water"):
        raise HTTPException(404)
    return FileResponse(os.path.join(BUILD, name + ".geojson"), media_type="application/json")


@app.get("/api/overlay/{name}.png")
def api_overlay(name: str):
    if name not in ("landcover", "hillshade", "danger"):
        raise HTTPException(404)
    cache = os.path.join(BUILD, "overlay_%s.png" % name)
    if not os.path.exists(cache):
        open(cache, "wb").write(RG.get().overlay_png(name))
    return FileResponse(cache, media_type="image/png", headers={"Cache-Control": "max-age=86400"})


@app.get("/api/point")
def api_point(lon: float, lat: float):
    """Fiche d'un point : statut, distance a la ligne, couverture du sol, altitude."""
    R = RG.get()
    x, y = RG.to_xy(lon, lat)
    from .datasets import WC
    return {"status": R.status_at(lon, lat), "front_m": round(float(R.front_dist(x, y))),
            "landcover": WC.get(int(R.landcover_at(x, y)), "?"), "alt_m": round(float(R.dem_window(x - 0.5, y - 0.5, 1, 1)[0, 0]))}


class RouteReq(BaseModel):
    a: list
    b: list
    profile: str = "equilibre"


@app.post("/api/route")
def api_route(r: RouteReq):
    t = time.time()
    out = plan_route(tuple(r.a), tuple(r.b), r.profile)
    out["compute_s"] = round(time.time() - t, 2)
    return out


class MissionReq(BaseModel):
    xy: list
    mode: str = "point"
    level: str = "sat"
    delivery: bool = True
    payload_kg: float = 3.0
    speed: float = 10.0
    seed: int = 1
    target: str = ""


@app.post("/api/mission/start")
def mission_start(r: MissionReq):
    m = STATE["mission"]
    if m is not None and m.is_alive():
        m.cmd_stop = True
        m.join(timeout=5)
    if r.payload_kg > 3.0:
        raise HTTPException(400, "charge > 3 kg (bras du G1)")
    m = Mission(r.xy, mode=r.mode, level=r.level, delivery=r.delivery, payload_kg=r.payload_kg, speed=r.speed,
                seed=r.seed, target_label=r.target)
    STATE["mission"] = m
    m.start()
    return {"ok": True}


class Ctl(BaseModel):
    action: str
    value: float | str | None = None


@app.post("/api/mission/control")
def mission_control(c: Ctl):
    m = STATE["mission"]
    if m is None:
        raise HTTPException(404, "aucune mission")
    if c.action == "stop":
        m.cmd_stop = True
    elif c.action == "pause":
        m.cmd_pause = not m.cmd_pause
    elif c.action == "speed":
        m.speed = float(c.value)
    elif c.action == "camera":
        m.camera = str(c.value)
    return {"ok": True, "paused": m.cmd_pause}


@app.get("/api/mission/tile.jpg")
def tile_img():
    m = STATE["mission"]
    if m is None or m.tile_img is None:
        raise HTTPException(404)
    return Response(m.tile_img, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/mission/report")
def report():
    m = STATE["mission"]
    if m is None:
        raise HTTPException(404)
    with m.lock:
        return {"state": m.state, "metrics": m.metrics, "events": m.events, "err_hist": m.err_hist,
                "track": m.track, "route": m.route.tolist()}


RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def _clean(o):
    """NaN / inf -> null (JSON strict)."""
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    return o


def _load(rel):
    p = os.path.join(RESULTS, rel)
    return _clean(json.load(open(p, encoding="utf-8"))) if os.path.exists(p) else None


@app.get("/api/benchmarks")
def benchmarks():
    """Tous les benchmarks reproductibles (results/), pour l'onglet BENCHMARKS."""
    out = {}
    for key, rel in (("point", "final_point/benchmark.json"), ("point_holdout", "final_point_holdout/benchmark.json"),
                     ("synth_tree", "final_tracks_treemap/benchmark.json"), ("synth_osm", "final_tracks_osm/benchmark.json"),
                     ("mujoco2d", "final_mujoco/benchmark.json"), ("mujoco3d", "final_mujoco_lidar3d/benchmark.json"),
                     ("real", "real/benchmark.json"), ("calib", "mujoco_calib.json"), ("robots", "robots/benchmark.json"),
                     ("slopes", "robots/slopes.json"), ("region", "region_bench/benchmark.json")):
        d = _load(rel)
        if d is None:
            continue
        if isinstance(d, dict) and "runs" in d and key not in ("real", "robots", "region"):
            d = {k: v for k, v in d.items() if k != "params"}
        out[key] = d
    nav = {}
    rd = os.path.join(RESULTS, "robots")
    if os.path.isdir(rd):
        for n in sorted(os.listdir(rd)):
            d = _load(os.path.join("robots", n, "benchmark.json")) if n.startswith("nav_") else None
            if d:
                nav[n[4:]] = {"robot": d.get("robot", "g1"), "relief": d.get("relief_scale"), "policy": d.get("policy"),
                              **d["with_map_matching"]}
    if nav:
        out["nav_robots"] = nav
    cpu = _load("robots/slopes_cpu.json")                    # meme protocole sous MuJoCo CPU (Windows)
    if cpu:
        rows = (out.get("slopes") or {}).get("rows", []) + cpu["rows"]
        out["slopes"] = {"rows": rows}
    return out


@app.get("/api/training")
def training():
    """Entrainements RL (results/rl/<run>/progress.json + config.json + images)."""
    base = os.path.join(RESULTS, "rl")
    runs = []
    if os.path.isdir(base):
        for r in sorted(os.listdir(base)):
            d = os.path.join(base, r)
            if not os.path.isdir(d):
                continue
            item = {"id": r, "progress": _load(os.path.join("rl", r, "progress.json")) or [],
                    "config": _load(os.path.join("rl", r, "config.json")) or {},
                    "eval": _load(os.path.join("rl", r, "eval.json")),
                    "media": sorted(f for f in os.listdir(d) if f.lower().endswith((".png", ".jpg", ".gif", ".mp4")))}
            runs.append(item)
    return {"runs": runs}


@app.get("/api/results/{path:path}")
def result_file(path: str):
    p = os.path.realpath(os.path.join(RESULTS, path))
    if not p.startswith(os.path.realpath(RESULTS)) or not p.lower().endswith((".png", ".jpg", ".gif", ".mp4", ".json")):
        raise HTTPException(404)
    if not os.path.exists(p):
        raise HTTPException(404)
    return FileResponse(p)


@app.get("/api/mujoco.mjpg")
async def mjpeg():
    async def gen():
        last = None
        while True:
            m = STATE["mission"]
            f = m.frame if m is not None else None
            if f is not None and f is not last:
                last = f
                yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(f)).encode() + b"\r\n\r\n" + f + b"\r\n"
            await asyncio.sleep(0.05)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    cur, tile_id = None, None
    ev_i = vx_i = tr_i = er_i = 0
    try:
        while True:
            m = STATE["mission"]
            if m is not cur:
                cur, tile_id = m, None
                ev_i = vx_i = tr_i = er_i = 0
                await sock.send_text(json.dumps({"type": "reset"}))
            if m is not None:
                with m.lock:
                    msg = {"type": "tick", "state": dict(m.state), "events": m.events[ev_i:], "vox": m.vox_all[vx_i:vx_i + 40000],
                           "scan": m.scan, "particles": m.particles, "path": m.local_path,
                           "track": m.track[tr_i:], "err": m.err_hist[er_i:]}
                    ev_i, tr_i, er_i = len(m.events), len(m.track), len(m.err_hist)
                    vx_i += len(msg["vox"])
                    if m.tile_info is not None and m.tile_info["id"] != tile_id:
                        if tile_id is None:                                  # 1er envoi : itineraire et parametres
                            lon, lat = RG.to_ll(m.route[:, 0], m.route[:, 1])
                            msg["route"] = {"coords": np.c_[lon, lat].round(6).tolist(), "xy": m.route.round(1).tolist(),
                                            "mode": m.mode, "level": m.level}
                        tile_id = m.tile_info["id"]
                        msg["tile"] = m.tile_info
                await sock.send_text(json.dumps(msg, default=float))
            await asyncio.sleep(0.2)
    except (WebSocketDisconnect, RuntimeError):
        return


def serve(port=8770):
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    serve()
