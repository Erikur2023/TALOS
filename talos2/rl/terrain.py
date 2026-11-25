"""Terrain d'entrainement 'front ukrainien' pour le heightfield 20 x 20 m de MuJoCo Playground (256 x 256).

Ingredients (ASSUMPTION, inspires des tuiles reelles observees : champs laboures, ornieres, cratere d'obus) :
  - buttes / pentes longues (jusqu'a ~20 deg localement) ;
  - labours : sillons de 6 cm, pas de 0,7 m, orientation aleatoire par parcelle ;
  - ornieres de vehicules (2 sillons paralleles, 12 cm) ;
  - crateres d'obus : cuvette (10 a 35 cm de profondeur, rayon 0,8 a 2 m) avec bourrelet ;
  - eboulis / debris : blocs de 4 a 12 cm.
La zone de depart (rayon 1,5 m autour de l'origine) est laissee plate a l'altitude 0 (le robot y apparait).
Les pentes du MNT reel de la region (< 5 %) sont trop douces pour l'entrainement : on les exagere volontairement.
"""
import numpy as np
from scipy.ndimage import gaussian_filter

N, SIZE = 256, 20.0          # pixels, metres (hfield +-10 m)


def make(seed=0, hills=0.6, craters=7, debris=180, depth=(0.10, 0.35), block=(0.04, 0.12)):
    rng = np.random.default_rng(seed)
    c = (np.arange(N) + 0.5) / N * SIZE - SIZE / 2
    X, Y = np.meshgrid(c, c)
    h = np.zeros((N, N))
    # buttes : bruit lisse (longueur ~4 m) -> pentes jusqu'a ~20 deg
    b = gaussian_filter(rng.standard_normal((N, N)), 4.0 / (SIZE / N), mode="wrap")
    h += hills * b / (np.abs(b).max() + 1e-9)
    # labours par parcelles
    for _ in range(4):
        cx, cy, r = rng.uniform(-8, 8), rng.uniform(-8, 8), rng.uniform(3, 7)
        th = rng.uniform(0, np.pi)
        u = X * np.cos(th) + Y * np.sin(th)
        m = np.hypot(X - cx, Y - cy) < r
        h += m * 0.03 * np.sin(2 * np.pi * u / 0.7)
    # ornieres
    for _ in range(2):
        th = rng.uniform(0, np.pi); off = rng.uniform(-6, 6)
        v = -X * np.sin(th) + Y * np.cos(th) - off
        h -= 0.12 * (np.exp(-((v - 0.8) / 0.18) ** 2) + np.exp(-((v + 0.8) / 0.18) ** 2))
    # crateres
    for _ in range(craters):
        cx, cy = rng.uniform(-9, 9, 2)
        if np.hypot(cx, cy) < 3.0:
            continue
        r, d = rng.uniform(0.8, 2.0), rng.uniform(*depth)
        q = np.hypot(X - cx, Y - cy) / r
        h += -d * np.clip(1 - q ** 2, 0, None) + 0.25 * d * np.exp(-((q - 1.1) / 0.25) ** 2)
    # eboulis
    for _ in range(debris):
        cx, cy = rng.uniform(-9.5, 9.5, 2)
        if np.hypot(cx, cy) < 2.0:
            continue
        s, z = rng.uniform(0.1, 0.35), rng.uniform(*block)
        h += z * ((np.abs(X - cx) < s) & (np.abs(Y - cy) < s))
    # zone de depart plate, raccord doux
    w = np.clip((np.hypot(X, Y) - 1.5) / 1.5, 0, 1)
    h = h * w
    return h                                                   # metres, h(0,0) = 0


def to_hfield(h):
    """-> (donnees normalisees 0..1 ligne par ligne, taille z, decalage du geom pour que h=0 soit en z=0)."""
    lo, hi = float(h.min()), float(h.max())
    zs = max(hi - lo, 1e-3)
    return ((h - lo) / zs).astype(np.float32), zs, lo


def slope_plane(deg):
    c = (np.arange(N) + 0.5) / N * SIZE - SIZE / 2
    X, _ = np.meshgrid(c, c)
    return np.tan(np.radians(deg)) * X


def apply(env, h, base=0.0):
    """Remplace le heightfield du modele (CPU + MJX)."""
    from mujoco import mjx
    import mujoco
    m = env._mj_model
    data, zs, lo = to_hfield(h)
    m.hfield_data[:] = data.ravel()
    m.hfield_size[0][2] = zs
    m.hfield_size[0][3] = 0.1
    for i in range(m.ngeom):
        if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_HFIELD:
            m.geom_pos[i][2] = lo + base
    env._mjx_model = mjx.put_model(m)


def add_payload(env, body, kg):
    from mujoco import mjx
    m = env._mj_model
    bid = m.body(body).id
    m.body_mass[bid] += kg
    env._mjx_model = mjx.put_model(m)


def preview(h, path, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.4))
    ls = LightSource(315, 35)
    ax[0].imshow(ls.shade(h, plt.cm.gist_earth, vert_exag=3, blend_mode="soft"), origin="lower", extent=[-10, 10, -10, 10])
    ax[0].set_title("terrain d'entrainement (ombre) " + title)
    gy, gx = np.gradient(h, SIZE / N)
    im = ax[1].imshow(np.degrees(np.arctan(np.hypot(gx, gy))), origin="lower", extent=[-10, 10, -10, 10], cmap="magma", vmax=35)
    ax[1].set_title("pente locale (deg)"); plt.colorbar(im, ax=ax[1])
    plt.tight_layout(); plt.savefig(path, dpi=80); plt.close()
