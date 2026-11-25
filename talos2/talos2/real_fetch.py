"""Telechargement (avec cache) des donnees REELLES d'une zone : imagerie satellite, MNT, OpenStreetMap.

Sources (toutes publiques, sans cle) — a crediter (voir README) :
  - imagerie : Esri World Imagery (tuiles XYZ, ~0,4 m/pixel au zoom 18) — Esri, Maxar, Earthstar Geographics ;
  - MNT : AWS Terrain Tiles (format terrarium, Mapzen / Linux Foundation ; source SRTM / GMTED / ETOPO), zoom 15 ;
  - carte : OpenStreetMap via l'API Overpass (ODbL, (c) contributeurs OpenStreetMap).
Repere local : metres Est / Nord depuis le coin sud-ouest de la zone (projection equirectangulaire locale, erreur
negligeable sur < 1 km).
Usage : python -m talos2.real_fetch --site serebryansky
"""
import argparse, io, json, math, os, time, urllib.parse, urllib.request
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "real")
UA = {"User-Agent": "talos2-research-sim/0.1"}

# Zones : (lat, lon) du coin sud-ouest, cote en metres. Lieux du front cites dans la presse ; la position de la
# tranchee et des unites est FICTIVE (choisie dans la simulation), seuls le terrain et la vegetation sont reels.
SITES = {
    "serebryansky": {"name": "Foret de Serebrianka (pres de Kreminna, oblast de Louhansk)",
                     "sw": (48.985, 38.120), "size": 500.0},
    "robotyne": {"name": "Haies brise-vent pres de Robotyne (oblast de Zaporijjia)",
                 "sw": (47.440, 35.840), "size": 500.0},
}


def _get(url, data=None, tries=4):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, data=data, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as e:                                    # reseau instable : on reessaie
            if k == tries - 1:
                raise
            time.sleep(2 + 3 * k)


def deg2tile(lat, lon, z):
    n = 2 ** z
    x = (lon + 180) / 360 * n
    y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
    return x, y


def m_per_deg(lat):
    return 111_320.0 * math.cos(math.radians(lat)), 110_574.0


def bbox(site):
    s = SITES[site]
    lat0, lon0 = s["sw"]
    mx, my = m_per_deg(lat0)
    return lat0, lon0, lat0 + s["size"] / my, lon0 + s["size"] / mx


def _mosaic(site, z, url_fmt, kind, decode):
    """Mosaique de tuiles couvrant la zone, reechantillonnee sur une grille locale (res m) -> tableau [ny, nx, ...]."""
    from PIL import Image
    lat0, lon0, lat1, lon1 = bbox(site)
    xa, ya = deg2tile(lat1, lon0, z)
    xb, yb = deg2tile(lat0, lon1, z)
    tx0, tx1, ty0, ty1 = int(xa), int(xb), int(ya), int(yb)
    cache = os.path.join(DATA, site, kind)
    os.makedirs(cache, exist_ok=True)
    rows = []
    for ty in range(ty0, ty1 + 1):
        row = []
        for tx in range(tx0, tx1 + 1):
            f = os.path.join(cache, "%d_%d_%d.png" % (z, tx, ty))
            if not os.path.exists(f):
                open(f, "wb").write(_get(url_fmt.format(z=z, x=tx, y=ty)))
                time.sleep(0.2)
            row.append(np.asarray(Image.open(f).convert("RGB")))
        rows.append(np.concatenate(row, axis=1))
    mos = np.concatenate(rows, axis=0)
    return decode(mos), (tx0, ty0)


def resample(mos, origin, z, site, res, order=1):
    """Echantillonne la mosaique (pixels de tuiles) sur la grille locale ENU (lignes = Nord croissant)."""
    from scipy.ndimage import map_coordinates
    lat0, lon0, _, _ = bbox(site)
    size = SITES[site]["size"]
    mx, my = m_per_deg(lat0)
    n = int(size / res)
    c = (np.arange(n) + 0.5) * res
    E, N = np.meshgrid(c, c)
    lat, lon = lat0 + N / my, lon0 + E / mx
    n2 = 2 ** z
    px = ((lon + 180) / 360 * n2 - origin[0]) * 256
    py = ((1 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2 * n2 - origin[1]) * 256
    if mos.ndim == 2:
        return map_coordinates(mos, [py - 0.5, px - 0.5], order=order, mode="nearest")
    return np.stack([map_coordinates(mos[..., k].astype(float), [py - 0.5, px - 0.5], order=order, mode="nearest")
                     for k in range(mos.shape[2])], -1)


def imagery(site, res=0.5, z=18):
    mos, o = _mosaic(site, z, "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                     "img", lambda m: m)
    return np.clip(resample(mos, o, z, site, res), 0, 255).astype(np.uint8)


def dem(site, res=1.0, z=15):
    dec = lambda m: m[..., 0].astype(float) * 256 + m[..., 1] + m[..., 2] / 256.0 - 32768.0
    mos, o = _mosaic(site, z, "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png", "dem", dec)
    return resample(mos, o, z, site, res, order=3)


def osm(site):
    f = os.path.join(DATA, site, "osm.json")
    if not os.path.exists(f):
        lat0, lon0, lat1, lon1 = bbox(site)
        pad = 0.002
        b = "%f,%f,%f,%f" % (lat0 - pad, lon0 - pad, lat1 + pad, lon1 + pad)
        q = ("[out:json][timeout:60];(way[landuse=forest](%s);way[natural=wood](%s);way[natural=tree_row](%s);"
             "way[highway](%s);way[natural=wetland](%s);way[natural=water](%s);way[waterway](%s);way[building](%s);"
             "relation[landuse=forest](%s);relation[natural=wood](%s););out geom;") % ((b,) * 10)
        os.makedirs(os.path.dirname(f), exist_ok=True)
        open(f, "wb").write(_get("https://overpass-api.de/api/interpreter",
                                 data=urllib.parse.urlencode({"data": q}).encode()))
    return json.load(open(f, encoding="utf-8"))


def to_local(site, lat, lon):
    lat0, lon0, _, _ = bbox(site)
    mx, my = m_per_deg(lat0)
    return (np.asarray(lon) - lon0) * mx, (np.asarray(lat) - lat0) * my


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="serebryansky", choices=list(SITES))
    a = ap.parse_args()
    from PIL import Image
    img = imagery(a.site)
    Image.fromarray(img[::-1]).save(os.path.join(DATA, a.site, "imagery_local.jpg"), quality=88)
    h = dem(a.site)
    o = osm(a.site)
    print(a.site, "image", img.shape, "MNT %.0f-%.0f m" % (h.min(), h.max()), "OSM", len(o["elements"]), "objets")


if __name__ == "__main__":
    main()
