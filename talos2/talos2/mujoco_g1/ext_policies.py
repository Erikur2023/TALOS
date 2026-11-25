"""Politiques de marche G1 PRE-ENTRAINEES PAR DES TIERS, rejouees dans notre banc MuJoCo CPU (meme interface que
quad.Go1Walker : reset / control_step / upright / home_z), pour les comparer a nos propres entrainements.

- "unitree_mjlab" : politique officielle Unitree G1 29 ddl, suivi de vitesse (unitree_rl_mjlab, Apache-2.0,
  (c) Unitree Robotics) : deploy/robots/g1/config/policy/velocity/v0. Obs 98 = gyro bassin, gravite projetee,
  consigne, phase de marche (periode 0,6 s), angles - defaut, vitesses, derniere action ; physique 5 ms, 50 Hz.
- "gr00t_wbc" : NVIDIA GR00T-WholeBodyControl, contrôleur decouple (jambes + taille 15 ddl, politiques Balance / Walk ;
  code Apache-2.0, POIDS sous NVIDIA Open Model License). Obs 86 x historique 6, physique 5 ms, 50 Hz.

Les depots sont clones dans ../ext (hors depot TALOS) : variable d'environnement TALOS_EXT pour un autre chemin."""
import collections, os
import numpy as np
import mujoco
import onnxruntime as ort
import yaml

EXT = os.environ.get("TALOS_EXT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "ext"))
MJLAB = os.path.join(EXT, "unitree_rl_mjlab")
GR00T = os.path.join(EXT, "GR00T-WholeBodyControl", "decoupled_wbc", "sim2mujoco", "resources", "robots", "g1")
CTRL_DT, PHYS_DT = 0.02, 0.005


def _sess(path):
    o = ort.SessionOptions(); o.intra_op_num_threads = 1; o.inter_op_num_threads = 1
    return ort.InferenceSession(path, o, providers=["CPUExecutionProvider"])


def _grav(quat):
    w, x, y, z = quat
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    return R.T @ np.array([0, 0, -1.0])


def available():
    out = []
    if os.path.exists(os.path.join(MJLAB, "deploy", "robots", "g1", "config", "policy", "velocity", "v0", "exported", "policy.onnx")):
        out.append("unitree_mjlab")
    if os.path.exists(os.path.join(GR00T, "policy", "GR00T-WholeBodyControl-Walk.onnx")):
        out.append("gr00t_wbc")
    return out


# ------------------------------------------------------------------ Unitree mjlab (29 ddl)
def unitree_spec(payload_kg=3.0):
    spec = mujoco.MjSpec.from_file(os.path.join(MJLAB, "src", "assets", "robots", "unitree_g1", "xmls", "g1.xml"))
    for g in spec.geoms:
        if g.name.endswith("_collision"):
            g.contype, g.conaffinity = 0, 1                # pas d'auto-collision (FULL_COLLISION_WITHOUT_SELF)
            if "foot" in g.name:
                g.condim, g.priority, g.friction = 3, 1, [0.6, 0.005, 0.0001]
            else:
                g.condim = 1
        elif g.contype or g.conaffinity:
            g.contype = g.conaffinity = 0
    spec.body("torso_link").mass += payload_kg
    spec.option.timestep = PHYS_DT
    spec.option.iterations, spec.option.ls_iterations = 10, 20
    return spec


class UnitreeMjlabWalker:
    label = "G1 29 ddl — Unitree officiel (unitree_rl_mjlab)"

    def __init__(self, m):
        base = os.path.join(MJLAB, "deploy", "robots", "g1", "config", "policy", "velocity", "v0")
        c = yaml.safe_load(open(os.path.join(base, "params", "deploy.yaml")))
        self.pi = _sess(os.path.join(base, "exported", "policy.onnx"))
        self.kp = np.array(c["stiffness"]); self.kd = np.array(c["damping"])
        self.q0 = np.array(c["default_joint_pos"])
        self.scale = np.array(c["actions"]["JointPositionAction"]["scale"])
        self.offset = np.array(c["actions"]["JointPositionAction"]["offset"])
        self.tau_max = self.scale * self.kp / 0.25           # G1_ACTION_SCALE = 0,25 x effort / raideur
        self.period = c["observations"]["gait_phase"]["params"]["period"]
        m.dof_armature[6:] = self.kp / (2 * np.pi * 10.0) ** 2   # raideur = armature x (2 pi 10 Hz)^2 (g1_constants.py)
        m.dof_damping[6:] = 0.0
        self.gyro = m.sensor("imu_ang_vel").adr[0]
        self.home_z, self.t = 0.78, 0.0
        self.act = np.zeros(29); self.target = self.q0.copy()

    def reset(self, d):
        d.qpos[7:36] = self.q0; d.qvel[:] = 0
        self.act[:] = 0; self.target = self.q0.copy(); self.t = 0.0

    def control_step(self, m, d, cmd):
        cmd = np.asarray(cmd, float)
        self.t += CTRL_DT
        ph = (self.t % self.period) / self.period
        phase = np.zeros(2) if np.linalg.norm(cmd) < 0.1 else np.array([np.sin(2 * np.pi * ph), np.cos(2 * np.pi * ph)])
        obs = np.concatenate([d.sensordata[self.gyro:self.gyro + 3], _grav(d.qpos[3:7]), cmd, phase,
                              d.qpos[7:36] - self.q0, d.qvel[6:35], self.act]).astype(np.float32)[None]
        self.act = self.pi.run(None, {"obs": obs})[0][0].astype(float)
        self.target = self.offset + self.scale * self.act
        for _ in range(int(round(CTRL_DT / PHYS_DT))):
            tau = self.kp * (self.target - d.qpos[7:36]) - self.kd * d.qvel[6:35]
            d.qfrc_applied[6:35] = np.clip(tau, -self.tau_max, self.tau_max)
            mujoco.mj_step(m, d)

    def upright(self, d):
        return float(-_grav(d.qpos[3:7])[2])


# ------------------------------------------------------------------ NVIDIA GR00T decoupled WBC
def gr00t_spec(payload_kg=3.0):
    spec = mujoco.MjSpec.from_file(os.path.join(GR00T, "g1_gear_wbc.xml"))
    for g in list(spec.geoms):
        if g.name == "floor":
            spec.delete(g)
    spec.body("torso_link").mass += payload_kg
    spec.option.timestep = PHYS_DT
    return spec


class Gr00tWalker:
    label = "G1 29 ddl — NVIDIA GR00T WBC (Balance/Walk)"
    HIST = 6

    def __init__(self, m):
        c = yaml.safe_load(open(os.path.join(GR00T, "g1_gear_wbc.yaml")))
        self.c = c
        self.bal = _sess(os.path.join(GR00T, "policy", "GR00T-WholeBodyControl-Balance.onnx"))
        self.walk = _sess(os.path.join(GR00T, "policy", "GR00T-WholeBodyControl-Walk.onnx"))
        self.kp = np.array(c["kps"]); self.kd = np.array(c["kds"]); self.q0 = np.array(c["default_angles"])
        self.torso = m.body("torso_link").id
        self.home_z = 0.79
        self.act = np.zeros(15); self.target = self.q0.copy()
        self.hist = collections.deque([np.zeros(86, np.float32)] * self.HIST, maxlen=self.HIST)

    def reset(self, d):
        d.qpos[7:22] = self.q0; d.qpos[22:36] = 0; d.qvel[:] = 0
        self.act[:] = 0; self.target = self.q0.copy()
        self.hist = collections.deque([np.zeros(86, np.float32)] * self.HIST, maxlen=self.HIST)

    def _obs(self, d, cmd):
        c = self.c
        o = np.zeros(86, np.float32)
        o[0:3] = np.asarray(cmd) * np.array(c["cmd_scale"]); o[3] = c["height_cmd"]; o[4:7] = c["rpy_cmd"]
        o[7:10] = d.qvel[3:6] * c["ang_vel_scale"]
        o[10:13] = _grav(d.qpos[3:7])
        q0 = np.zeros(29); q0[:15] = self.q0
        o[13:42] = (d.qpos[7:36] - q0) * c["dof_pos_scale"]
        o[42:71] = d.qvel[6:35] * c["dof_vel_scale"]
        o[71:86] = self.act
        return o

    def control_step(self, m, d, cmd):
        for _ in range(int(round(CTRL_DT / PHYS_DT))):
            leg = self.kp * (self.target - d.qpos[7:22]) - self.kd * d.qvel[6:21]
            arm = 100.0 * (0 - d.qpos[22:36]) - 0.5 * d.qvel[21:35]
            d.ctrl[:15] = leg; d.ctrl[15:29] = arm
            mujoco.mj_step(m, d)
        self.hist.append(self._obs(d, cmd))
        x = np.concatenate(list(self.hist))[None].astype(np.float32)
        pol = self.bal if np.linalg.norm(cmd) <= 0.05 else self.walk
        self.act = pol.run(None, {"input": x})[0][0].astype(float)
        self.target = self.act * self.c["action_scale"] + self.q0

    def upright(self, d):
        return float(-_grav(d.qpos[3:7])[2])


REGISTRY = {"unitree_mjlab": (unitree_spec, UnitreeMjlabWalker), "gr00t_wbc": (gr00t_spec, Gr00tWalker)}
