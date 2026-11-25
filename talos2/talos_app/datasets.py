"""Jeux de donnees de la REGION (~100 km de front, Donetsk : Kramatorsk / Sloviansk / Kostiantynivka / Siversk).

Telecharges une fois dans data/region/ puis decoupes en fichiers compacts (data/region/build/) :
  - ESA WorldCover 2021 v200, 10 m (couverture du sol : foret, cultures, bati, eau...)  ~80 Mo   CC-BY 4.0
  - Copernicus DEM GLO-30, 30 m (relief)                                               ~170 Mo  licence Copernicus
  - OpenStreetMap, extrait Ukraine de Geofabrik (routes, pistes, haies, batiments, lieux) ~1,5 Go ODbL
  - Ligne de contact : DeepStateMap (territoires occupes / statut inconnu, publics).     ~0,6 Mo
    Seuls les POLYGONES de territoire sont utilises ; les points (unites, directions) sont ignores.
Usage : python -m talos_app.datasets            (telecharge + construit, reprend si interrompu)
"""
import json, math, os, sys, time, urllib.request
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "region")
BUILD = os.path.join(ROOT, "build")
BBOX = (37.00, 48.20, 38.30, 49.10)           # lon0, lat0, lon1, lat1  (~95 km x 100 km)
UA = {"User-Agent": "talos-app-research/0.2"}

WORLDCOVER = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_N48E036_Map.tif"
DEM_TILES = ["N48_00_E037_00", "N48_00_E038_00", "N49_00_E037_00", "N49_00_E038_00"]
DEM_URL = "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_{t}_DEM/Copernicus_DSM_COG_10_{t}_DEM.tif"
OSM_PBF = "https://download.geofabrik.de/europe/ukraine-latest.osm.pbf"
DEEPSTATE = "https://deepstatemap.live/api/history/last"

# classes WorldCover
WC = {10: "foret", 20: "arbustes", 30: "prairie", 40: "cultures", 50: "bati", 60: "sol nu", 80: "eau", 90: "zone humide",
      95: "mangrove", 100: "mousses"}


def download(url, dst, force=False):
    if os.path.exists(dst) and not force:
        return dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".part"
    have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    req = urllib.request.Request(url, headers=dict(UA, **({"Range": "bytes=%d-" % have} if have else {})))
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "ab" if have else "wb") as f:
        total = int(r.headers.get("Content-Length", 0)) + have
        done, t0 = have, time.time()
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b); done += len(b)
            if time.time() - t0 > 5:
                print("  %s %.0f/%.0f Mo" % (os.path.basename(dst), done / 1e6, total / 1e6), flush=True); t0 = time.time()
    os.replace(tmp, dst)
    return dst


def fetch_all():
    download(WORLDCOVER, os.path.join(ROOT, "worldcover_N48E036.tif"))
    for t in DEM_TILES:
        download(DEM_URL.format(t=t), os.path.join(ROOT, "dem_%s.tif" % t))
    download(DEEPSTATE, os.path.join(ROOT, "deepstate_last.json"), force=True)
    download(OSM_PBF, os.path.join(ROOT, "ukraine-latest.osm.pbf"))


# ---------------------------------------------------------------- construction des couches compactes
def build_rasters():
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.merge import merge
    lon0, lat0, lon1, lat1 = BBOX
    os.makedirs(BUILD, exist_ok=True)
    with rasterio.open(os.path.join(ROOT, "worldcover_N48E036.tif")) as s:
        w = from_bounds(lon0, lat0, lon1, lat1, s.transform)
        lc = s.read(1, window=w)                                     # lignes : nord -> sud
    np.save(os.path.join(BUILD, "landcover.npy"), lc[::-1].copy())  # lignes : sud -> nord
    srcs = [rasterio.open(os.path.join(ROOT, "dem_%s.tif" % t)) for t in DEM_TILES]
    dem, _ = merge(srcs, bounds=(lon0, lat0, lon1, lat1))
    for s in srcs:
        s.close()
    np.save(os.path.join(BUILD, "dem.npy"), dem[0][::-1].astype(np.float32).copy())
    json.dump({"bbox": BBOX, "landcover_shape": lc.shape, "dem_shape": dem[0].shape},
              open(os.path.join(BUILD, "meta.json"), "w"))
    print("rasters : couverture", lc.shape, "MNT", dem[0].shape)


def build_front():
    """Territoires occupes / gris (DeepState) decoupes a la region + ligne de contact."""
    from shapely.geometry import shape, box, mapping
    from shapely.ops import unary_union
    d = json.load(open(os.path.join(ROOT, "deepstate_last.json"), encoding="utf-8"))
    feats = d["map"]["features"]
    reg = box(*BBOX).buffer(0.3)
    occ, grey = [], []
    for f in feats:
        if f["geometry"]["type"] not in ("Polygon", "MultiPolygon"):
            continue                                                   # points (unites, fleches) ignores
        n = f["properties"].get("name", "")
        g = shape(f["geometry"]).buffer(0)
        if not g.intersects(reg):
            continue
        if "Occupied" in n or "окупована" in n:
            occ.append(g)
        elif "Unknown" in n:
            grey.append(g)
    occ_u = unary_union(occ).intersection(reg)
    grey_u = unary_union(grey).intersection(reg) if grey else None
    line = occ_u.boundary.intersection(box(*BBOX))
    out = {"date": d.get("datetime"), "source": "DeepStateMap (deepstatemap.live)",
           "occupied": mapping(occ_u), "grey": mapping(grey_u) if grey_u is not None else None, "line": mapping(line)}
    json.dump(out, open(os.path.join(BUILD, "front.json"), "w"))
    print("front :", out["date"], "longueur ligne ~%.0f km" % (line.length * 85))


def build_osm():
    """Routes / pistes / haies / foret / eau / lieux dans la region -> GeoJSON compacts. Batiments -> tableau numpy."""
    import osmium
    lon0, lat0, lon1, lat1 = BBOX
    inside = lambda lon, lat: lon0 <= lon <= lon1 and lat0 <= lat <= lat1
    roads, tracks, rows, water, places, blds = [], [], [], [], [], []

    class H(osmium.SimpleHandler):
        def way(self, w):
            t = w.tags
            hw, nat, ww, b = t.get("highway"), t.get("natural"), t.get("waterway"), t.get("building")
            if not (hw or nat == "tree_row" or ww or b):
                return
            try:
                c = [(n.lon, n.lat) for n in w.nodes]
            except osmium.InvalidLocationError:
                return
            if not c or not any(inside(*p) for p in c):
                return
            c = [(round(x, 5), round(y, 5)) for x, y in c]
            if b:
                xs, ys = zip(*c); blds.append((sum(xs) / len(xs), sum(ys) / len(ys), max(xs) - min(xs), max(ys) - min(ys)))
            elif hw in ("motorway", "trunk", "primary", "secondary", "tertiary"):
                roads.append({"c": c, "k": hw, "n": t.get("name:en") or t.get("name", "")})
            elif hw in ("unclassified", "residential", "track", "service", "path"):
                tracks.append({"c": c, "k": hw})
            elif nat == "tree_row":
                rows.append({"c": c})
            elif ww in ("river", "stream", "canal"):
                water.append({"c": c, "k": ww, "n": t.get("name:en") or t.get("name", "")})

        def node(self, n):
            t = n.tags
            if t.get("place") in ("city", "town", "village", "hamlet") and inside(n.location.lon, n.location.lat):
                places.append({"lon": round(n.location.lon, 5), "lat": round(n.location.lat, 5), "k": t["place"],
                               "n": t.get("name:en") or t.get("name:uk") or t.get("name", ""),
                               "p": int(t.get("population", "0").replace(" ", "") or 0) if t.get("population", "0").replace(" ", "").isdigit() else 0})

    pbf = os.path.join(ROOT, "ukraine-latest.osm.pbf")
    t0 = time.time()
    H().apply_file(pbf, locations=True, idx="flex_mem")
    gj = lambda items, props: {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "LineString", "coordinates": it["c"]},
         "properties": {k: it[k] for k in props if k in it}} for it in items if len(it["c"]) > 1]}
    json.dump(gj(roads, ["k", "n"]), open(os.path.join(BUILD, "roads.geojson"), "w"))
    json.dump(gj(tracks, ["k"]), open(os.path.join(BUILD, "tracks.geojson"), "w"))
    json.dump(gj(rows, []), open(os.path.join(BUILD, "tree_rows.geojson"), "w"))
    json.dump(gj(water, ["k", "n"]), open(os.path.join(BUILD, "water.geojson"), "w"))
    json.dump(places, open(os.path.join(BUILD, "places.json"), "w"))
    np.save(os.path.join(BUILD, "buildings.npy"), np.array(blds, np.float32).reshape(-1, 4))
    print("osm : %d routes, %d pistes, %d haies, %d cours d'eau, %d lieux, %d batiments (%.0f s)" % (
        len(roads), len(tracks), len(rows), len(water), len(places), len(blds), time.time() - t0))


def main():
    fetch_all()
    build_rasters()
    build_front()
    build_osm()


if __name__ == "__main__":
    steps = sys.argv[1:] or ["main"]
    for s in steps:
        globals()[s]()
