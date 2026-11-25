"""Scene MuJoCo generee depuis World : relief (hfield), arbres et anomalies (cylindres) pres du couloir, tranchee.

Groupes de geoms : 0/1 robot (fichier Unitree), 2 = sol (hfield), 3 = obstacles (arbres, anomalies), 4 = decor (tranchee, visuel).
Le LiDAR 2D (etape 2) ne voit que le groupe 3 ; le LiDAR 3D (etape 5) voit 2 + 3 et doit segmenter le sol lui-meme."""
import numpy as np
import mujoco
from .g1_policy import G1_XML

G_GROUND, G_OBST, G_DECOR = 2, 3, 4


def corridor_mask(xy, path, buffer_m):
    """Points a moins de buffer_m d'un chemin (polyligne)."""
    keep = np.zeros(len(xy), bool)
    for a, b in zip(path[:-1], path[1:]):
        ab = b - a
        L2 = float(ab @ ab) + 1e-9
        t = np.clip(((xy - a) @ ab) / L2, 0, 1)
        d = np.hypot(*(xy - (a + t[:, None] * ab)).T)
        keep |= d < buffer_m
    return keep


def build(world, corridor_path, buffer_m=30.0, relief_scale=1.0, tree_height=5.0, robot_spec=None):
    """Retourne (MjModel, infos). relief_scale < 1 aplatit le relief (le robot et la nav voient le meme relief).
    robot_spec : MjSpec d'un autre robot (ex. quadrupede, voir quad.py) a la place du G1 12 ddl."""
    if robot_spec is None:
        spec = mujoco.MjSpec.from_file(G1_XML)
        spec.option.timestep = 0.002
    else:
        spec = robot_spec
    # --- ciel / lumiere / materiaux ---
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.55, 0.65, 0.75], rgb2=[0.15, 0.2, 0.25], width=256, height=256)
    img = getattr(world, "img", None)
    if img is not None:                                  # terrain reel : imagerie satellite plaquee sur le relief
        from PIL import Image
        im = np.asarray(Image.fromarray(np.asarray(img)[::-1]).resize((1024, 1024)))
        tex = spec.add_texture(name="sat", type=mujoco.mjtTexture.mjTEXTURE_2D, width=1024, height=1024, nchannel=3)
        tex.data = im.astype(np.uint8).tobytes()
        spec.add_material(name="ground", textures=["", "sat"], texrepeat=[1, 1], texuniform=False, rgba=[1, 1, 1, 1])
    else:
        spec.add_material(name="ground", rgba=[0.33, 0.30, 0.20, 1])
    spec.add_material(name="bark", rgba=[0.30, 0.22, 0.14, 1])
    spec.add_material(name="anomaly", rgba=[0.55, 0.15, 0.10, 1])
    spec.add_material(name="trench", rgba=[0.18, 0.14, 0.10, 1])
    spec.worldbody.add_light(pos=[150, 150, 80], dir=[0.3, 0.2, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=[0.8, 0.8, 0.75])

    # --- relief : hfield. Sommet j de la grille = centre de cellule (j+0.5) m, comme World.height_at ---
    n = world.n
    h = world.height * relief_scale
    hmax = float(h.max()) + 1e-3
    hf = spec.add_hfield(name="terrain", nrow=n, ncol=n,
                         size=[(n - 1) / 2 * world.res, (n - 1) / 2 * world.res, hmax, 1.0])
    hf.userdata = (h / hmax).astype(np.float32).ravel().tolist()
    spec.worldbody.add_geom(name="terrain", type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname="terrain",
                            pos=[n / 2 * world.res, n / 2 * world.res, 0], material="ground", group=G_GROUND,
                            friction=[0.9, 0.005, 0.0001])

    def hz(x, y):
        return float(world.height_at(x, y)) * relief_scale

    # --- obstacles (arbres + anomalies) pres du couloir ---
    keep = corridor_mask(world.obst[:, :2], np.asarray(corridor_path), buffer_m)
    ids = np.flatnonzero(keep)
    first = None
    for k in ids:
        x, y, r = world.obst[k]
        anomaly = bool(world.obst_kind[k])
        hgt = 1.6 if anomaly else tree_height       # anomalie : vehicule / gravats / abri, bas
        g = spec.worldbody.add_geom(name="obst_%d" % k, type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                                    size=[r, hgt / 2 + 0.5, 0], pos=[x, y, hz(x, y) + hgt / 2 - 0.5],
                                    material="anomaly" if anomaly else "bark", group=G_OBST)
        first = first or g.name
    # --- tranchee (decor, sans collision : la nav s'arrete avant ; cf. phase de depot) ---
    tr = world.trench
    for a, b in zip(tr[:-1], tr[1:]):
        c, L = (a + b) / 2, float(np.hypot(*(b - a)))
        yaw = float(np.arctan2(b[1] - a[1], b[0] - a[0]))
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[L / 2 + 0.6, 0.6, 0.05],
                                pos=[c[0], c[1], hz(*c) + 0.03], euler=[0, 0, yaw],
                                material="trench", group=G_DECOR, contype=0, conaffinity=0)
    m = spec.compile()
    obst_geom = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "obst_%d" % k) for k in ids], int)
    info = {"obst_index": ids, "obst_geom": obst_geom, "relief_scale": relief_scale, "n_obst": len(ids), "spec": spec}
    return m, info
