"""Quadrupede Unitree Go1 sous MuJoCo (CPU), pilote par une politique de marche ENTRAINEE PAR NOUS (rl/train.py,
PPO GPU sur terrain 'front ukrainien', charge 3 kg) et par le meme Navigator que le G1.

Modele : Menagerie / MuJoCo Playground (scene 'feetonly' : seuls les pieds entrent en collision — limite connue,
le corps traverse les troncs ; les contacts pieds-obstacles restent comptes).
Observation (identique a l'entrainement, Playground Go1Joystick) : vitesse locale (3), gyro (3), gravite (3),
angles - pose de repos (12), vitesses articulaires (12), derniere action (12), consigne vx vy wz (3) = 48.
Action : position cible = pose de repos + 0,5 x tanh(sortie du reseau), PD Kp 35 / Kd 0,5, 50 Hz, physique 250 Hz."""
import json, os, re
import numpy as np
import mujoco
from ..params import Params
from ..world import World
from ..prior_map import PriorMap
from ..planner import Planner
from ..sensors import ImuOdometry
from . import scene as S
from .run import Lidar2D, scale_relief, yaw_of, G1Mission

HERE = os.path.dirname(os.path.abspath(__file__))
RL = os.path.join(HERE, "..", "..", "rl")
MODELS = os.path.join(RL, "models")
RESULTS_RL = os.path.join(HERE, "..", "..", "results", "rl")
CTRL_DT, PHYS_DT = 0.02, 0.004
ACTION_SCALE, KP, KD = 0.5, 35.0, 0.5
LIDAR_HEIGHT_Q = 0.15      # ASSUMPTION m au-dessus du tronc (LiDAR monte sur le dos)


def load_assets(robot="go1"):
    d = os.path.join(MODELS, robot, "assets")
    assets = {}
    for f in os.listdir(d):
        b = open(os.path.join(d, f), "rb").read()
        if f.endswith(".xml"):                               # chemins Menagerie -> fichiers a plat (VFS)
            b = re.sub(rb'meshdir="[^"]*"', b'meshdir=""', b)
            b = re.sub(rb'texturedir="[^"]*"', b'texturedir=""', b)
            b = re.sub(rb'file="[^"]*/([^"/]+)"', lambda mm: b'file="' + mm.group(1) + b'"', b)
        assets[f] = b
    return assets, open(os.path.join(MODELS, robot, "scene.xml")).read()


class NpPolicy:
    """MLP exporte (rl/train.py:export_policy) : normalisation -> couches denses (swish) -> tanh(moyenne)."""
    def __init__(self, path):
        z = np.load(path)
        self.mean, self.std = z["obs_mean"], z["obs_std"]
        k = 0
        self.W = []
        while "W%d" % k in z:
            self.W.append((z["W%d" % k], z["b%d" % k])); k += 1

    def __call__(self, obs):
        x = (obs - self.mean) / np.maximum(self.std, 1e-6)
        for i, (W, b) in enumerate(self.W):
            x = x @ W + b
            if i < len(self.W) - 1:
                x = x / (1 + np.exp(-x))                       # swish
        na = x.shape[-1] // 2
        return np.tanh(x[..., :na])


def robot_spec(payload_kg=3.0):
    assets, xml = load_assets("go1")
    flat = os.path.join(MODELS, "go1", "flat")                 # fichiers a plat sur disque (chemins Menagerie reecrits)
    os.makedirs(flat, exist_ok=True)
    if not os.path.exists(os.path.join(flat, "scene.xml")):     # une seule fois (processus paralleles : ecriture atomique)
        for k, v in assets.items():
            tmp = os.path.join(flat, k + ".%d.tmp" % os.getpid())
            open(tmp, "wb").write(v); os.replace(tmp, os.path.join(flat, k))
        tmp = os.path.join(flat, "scene.xml.%d.tmp" % os.getpid())
        open(tmp, "w").write(xml); os.replace(tmp, os.path.join(flat, "scene.xml"))
    spec = mujoco.MjSpec.from_file(os.path.join(flat, "scene.xml"))
    for g in list(spec.geoms):
        if g.name == "floor":
            spec.delete(g)
    for g in spec.geoms:                                    # groupes du robot hors des groupes sol (2) / obstacles (3)
        if g.group in (2, 3):
            g.group = 1 if g.group == 2 else 5
    trunk = spec.body("trunk")
    trunk.mass = trunk.mass + payload_kg
    spec.option.timestep = PHYS_DT
    # Le XML Playground fixe iterations=1 / ls_iterations=5 (reglage MJX pour la vitesse GPU). Sous MuJoCo CPU sur un
    # heightfield, les pieds traversent alors le sol au bout de quelques minutes : on remonte la precision du solveur.
    spec.option.iterations = 30
    spec.option.ls_iterations = 30
    return spec


class Go1Walker:
    def __init__(self, m, policy_path):
        self.pi = NpPolicy(policy_path)
        key = m.key("home")
        self.default = key.qpos[7:19].copy()
        self.home_z = float(key.qpos[2])
        self.last_act = np.zeros(12)
        self.cmd = np.zeros(3)
        self.s_gyro = m.sensor("gyro").adr[0]
        self.s_lin = m.sensor("local_linvel").adr[0]
        self.s_up = m.sensor("upvector").adr[0]
        self.imu = m.site("imu").id
        m.dof_damping[6:18] = KD
        m.actuator_gainprm[:, 0] = KP
        m.actuator_biasprm[:, 1] = -KP

    def reset(self, d):
        d.qpos[7:19] = self.default
        d.ctrl[:] = self.default
        self.last_act[:] = 0

    def obs(self, d):
        grav = d.site_xmat[self.imu].reshape(3, 3).T @ np.array([0, 0, -1.0])
        return np.concatenate([d.sensordata[self.s_lin:self.s_lin + 3], d.sensordata[self.s_gyro:self.s_gyro + 3], grav,
                               d.qpos[7:19] - self.default, d.qvel[6:18], self.last_act, self.cmd])

    def control_step(self, m, d, cmd):
        self.cmd = np.asarray(cmd, float)
        a = self.pi(self.obs(d))
        self.last_act = a
        d.ctrl[:] = self.default + ACTION_SCALE * a
        for _ in range(int(round(CTRL_DT / PHYS_DT))):
            mujoco.mj_step(m, d)

    def upright(self, d):
        return float(d.sensordata[self.s_up + 2])


def default_policy():
    if os.environ.get("TALOS_GO1_POLICY"):
        return os.environ["TALOS_GO1_POLICY"]
    for run in ("go1_terrain_ua", "go1_sol_plat"):
        p = os.path.join(RESULTS_RL, run, "policy.npz")
        if os.path.exists(p):
            return p
    raise FileNotFoundError("aucune politique Go1 entrainee (results/rl/go1_*/policy.npz) : lancer rl/train_all.sh")


class QuadMission(G1Mission):
    """Meme interface que G1Mission (pose, advance, proprio, slope_meas, scan2d), robot = Go1."""
    def __init__(self, seed, p, relief_scale=1.0, world=None, prior=None, policy_path=None, payload_kg=3.0):
        self.p = p
        self.w = world or World(seed)
        self.prior = prior or PriorMap(self.w, p, seed)
        scale_relief(self.w, self.prior, relief_scale)
        self.rng = np.random.default_rng(40_000 + seed)
        self.imu = ImuOdometry(p, seed)
        w = self.w
        corridor = Planner(self.prior, p).plan(w.start, w.goal)
        self.m, self.info = S.build(w, np.vstack([w.start, corridor, w.goal]), robot_spec=robot_spec(payload_kg))
        self.d = mujoco.MjData(self.m)
        self.walker = Go1Walker(self.m, policy_path or default_policy())
        self.yaw0 = float(np.arctan2(*(w.goal - w.start)[::-1]))
        x0, y0 = map(float, w.start)
        ring = [(x0 + 0.35 * np.cos(a), y0 + 0.35 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
        z = max(float(w.height_at(x0, y0)), max(float(w.height_at(*q)) for q in ring))
        d = self.d
        d.qpos[:3] = [x0, y0, z + self.walker.home_z + 0.05]
        d.qpos[3:7] = [np.cos(self.yaw0 / 2), 0, 0, np.sin(self.yaw0 / 2)]
        self.walker.reset(d)
        mujoco.mj_forward(self.m, d)
        self.lidar = Lidar2D(p.lidar_rays, p.lidar_range)
        self.obst_set = set(self.info["obst_geom"].tolist())
        self.mech_j, self.dist_true = 0.0, 0.0
        self.contacts, self.last_contact_t = 0, -9.0
        for _ in range(25):                                   # 0,5 s de stabilisation
            self.walker.control_step(self.m, d, [0, 0, 0])

    def advance(self, vc, wc, t, dt=0.1):
        m, d = self.m, self.d
        xp, yp, yawp = self.pose
        for _ in range(int(round(dt / CTRL_DT))):
            self.walker.control_step(m, d, [vc, 0.0, wc])
            tau = d.actuator_force[:12] if self.m.nu else d.qfrc_applied[6:18]   # ext : couples appliques directement
            self.mech_j += float(np.sum(np.maximum(0.0, tau * d.qvel[6:6 + len(tau)]))) * CTRL_DT
        x, y, yaw = self.pose
        # chute = tronc incline de plus de ~60 deg (la hauteur au-dessus du relief, echantillonne a 1 m, n'est pas
        # assez precise pour un robot de 30 cm en pente : fausses chutes)
        fell = self.walker.upright(d) < 0.5
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

    def scan2d(self):
        x, y, yaw = self.pose
        p = self.p
        return self.lidar.scan(self.m, self.d, x, y, float(self.d.qpos[2]) + LIDAR_HEIGHT_Q, yaw,
                               p.lidar_sigma, p.lidar_dropout, self.rng)


class ExtMission(QuadMission):
    """Humanoide G1 29 ddl pilote par une politique pre-entrainee tierce (ext_policies.py), meme Navigator."""
    def __init__(self, seed, p, relief_scale=1.0, world=None, prior=None, key="unitree_mjlab", payload_kg=3.0):
        from . import ext_policies as E
        self.p = p
        self.w = world or World(seed)
        self.prior = prior or PriorMap(self.w, p, seed)
        scale_relief(self.w, self.prior, relief_scale)
        self.rng = np.random.default_rng(40_000 + seed)
        self.imu = ImuOdometry(p, seed)
        w = self.w
        spec_fn, Walker = E.REGISTRY[key]
        corridor = Planner(self.prior, p).plan(w.start, w.goal)
        self.m, self.info = S.build(w, np.vstack([w.start, corridor, w.goal]), robot_spec=spec_fn(payload_kg))
        self.d = d = mujoco.MjData(self.m)
        self.walker = Walker(self.m)
        self.yaw0 = float(np.arctan2(*(w.goal - w.start)[::-1]))
        x0, y0 = map(float, w.start)
        ring = [(x0 + 0.35 * np.cos(a), y0 + 0.35 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
        z = max(float(w.height_at(x0, y0)), max(float(w.height_at(*q)) for q in ring))
        self.walker.reset(d)
        d.qpos[:3] = [x0, y0, z + self.walker.home_z + 0.03]
        d.qpos[3:7] = [np.cos(self.yaw0 / 2), 0, 0, np.sin(self.yaw0 / 2)]
        mujoco.mj_forward(self.m, d)
        self.lidar = Lidar2D(p.lidar_rays, p.lidar_range)
        self.obst_set = set(self.info["obst_geom"].tolist())
        self.mech_j, self.dist_true = 0.0, 0.0
        self.contacts, self.last_contact_t = 0, -9.0
        for _ in range(50):
            self.walker.control_step(self.m, d, [0, 0, 0])

    def scan2d(self):
        return G1Mission.scan2d(self)
