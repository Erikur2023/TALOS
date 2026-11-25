"""Entrainement PPO massivement parallele (GPU, MuJoCo MJX + Brax) d'une politique de marche.
  bash rl/run.sh train.py --env Go1JoystickRoughTerrain --run go1_ua --steps 100e6 --terrain ua --payload 3
Sorties : results/rl/<run>/{progress.json, config.json, params.pkl, policy.npz, terrain.png}
Base : MuJoCo Playground 0.0.5 (Google DeepMind, Apache-2.0) — environnements, recompenses, randomisation de domaine,
hyperparametres PPO publies ; ajouts TALOS : terrain 'front ukrainien', charge portee, export numpy."""
import argparse, functools, json, os, pickle, time
import numpy as np
import jax
from mujoco_playground import registry, wrapper
from mujoco_playground.config import locomotion_params
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
import terrain as T

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "rl")
PAYLOAD_BODY = {"G1": "torso_link", "Go1": "trunk", "T1": "Trunk", "Berkeley": "torso"}


# Mise en forme de la recompense (TALOS) : avec les poids par defaut, nos entrainements convergent vers l'optimum
# local 'rester debout' (suivi de vitesse ~0). On renforce le suivi de vitesse et on allege les penalites de pieds.
SHAPING = {"tracking_lin_vel": 2.0, "tracking_ang_vel": 0.8, "feet_clearance": -0.5, "pose": 0.2, "stand_still": -0.5}


def make_env(name, terrain, payload, seed, cfg=None, shaping=True):
    cfg = cfg or registry.get_default_config(name)
    if shaping:
        for k, v in SHAPING.items():
            if k in cfg.reward_config.scales:
                cfg.reward_config.scales[k] = v
    env = registry.load(name, config=cfg)
    if terrain == "ua":
        T.apply(env, T.make(seed))
    elif terrain == "ua_doux":                     # etape intermediaire du curriculum
        T.apply(env, T.make(seed, hills=0.2, craters=4, debris=60, depth=(0.05, 0.15), block=(0.02, 0.06)))
    if payload:
        body = [v for k, v in PAYLOAD_BODY.items() if name.startswith(k)][0]
        T.add_payload(env, body, payload)
    return nan_guard(env)


def nan_guard(env):
    """G1 + randomisation de domaine Playground : ~1 robot sur 128 diverge (NaN) en quelques centaines de pas, ce qui
    empoisonne tout le lot PPO. On termine l'episode du robot concerne (recompense 0, obs nettoyees) ; l'auto-reset
    de Brax le reinitialise."""
    import jax.numpy as jp
    step0 = env.step

    def step(state, action):
        s = step0(state, action)
        bad = jp.isnan(s.data.qpos).any() | jp.isnan(s.data.qvel).any() | jp.isnan(s.reward)
        obs = jax.tree_util.tree_map(lambda o: jp.nan_to_num(o, nan=0.0, posinf=0.0, neginf=0.0), s.obs)
        data = s.data.replace(qpos=jp.where(bad, state.data.qpos, s.data.qpos),
                              qvel=jp.where(bad, jp.zeros_like(s.data.qvel), s.data.qvel))
        metrics = {k: jp.nan_to_num(v) for k, v in s.metrics.items()}
        return s.replace(obs=obs, data=data, metrics=metrics, reward=jp.where(bad, 0.0, s.reward),
                         done=jp.where(bad, 1.0, s.done))
    env.step = step
    return env


def export_policy(params, path):
    """Politique deterministe en numpy : normalisation -> MLP (swish) -> tanh(moyenne)."""
    norm, pol = params[0], params[1]
    mean = norm.mean["state"] if isinstance(norm.mean, dict) else norm.mean
    std = norm.std["state"] if isinstance(norm.std, dict) else norm.std
    layers = pol["params"]
    names = sorted(layers.keys(), key=lambda s: int(s.split("_")[-1]))
    arrs = {"obs_mean": np.asarray(mean), "obs_std": np.asarray(std)}
    for i, n in enumerate(names):
        arrs["W%d" % i] = np.asarray(layers[n]["kernel"]); arrs["b%d" % i] = np.asarray(layers[n]["bias"])
    np.savez(path, **arrs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="Go1JoystickRoughTerrain")
    ap.add_argument("--run", default=None)
    ap.add_argument("--steps", type=float, default=100e6)
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--terrain", default="ua", choices=["ua", "ua_doux", "default"])
    ap.add_argument("--restore", default=None, help="run precedent dont on reprend les parametres (curriculum)")
    ap.add_argument("--payload", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--evals", type=int, default=12)
    ap.add_argument("--no-shaping", action="store_true")
    a = ap.parse_args()
    run = a.run or "%s_%s" % (a.env, a.terrain)
    out = os.path.join(OUT, run)
    os.makedirs(out, exist_ok=True)
    env = make_env(a.env, a.terrain, a.payload, a.seed, shaping=not a.no_shaping)
    eval_env = make_env(a.env, a.terrain, a.payload, a.seed, shaping=not a.no_shaping)
    if a.terrain == "ua":
        T.preview(T.make(a.seed), os.path.join(out, "terrain.png"), "(%s)" % a.env)
    if a.terrain == "ua_doux":
        T.preview(T.make(a.seed, hills=0.2, craters=4, debris=60, depth=(0.05, 0.15), block=(0.02, 0.06)), os.path.join(out, "terrain.png"), "doux (%s)" % a.env)
    pp = locomotion_params.brax_ppo_config(a.env)
    tp = dict(pp)
    nf = ppo_networks.make_ppo_networks
    if "network_factory" in pp:
        del tp["network_factory"]
        nf = functools.partial(ppo_networks.make_ppo_networks, **pp.network_factory)
    tp.update(num_timesteps=int(a.steps), num_envs=a.num_envs, num_evals=a.evals, seed=a.seed)
    cfg = {"env": a.env, "terrain": a.terrain, "payload_kg": a.payload, "gpu": str(jax.devices()[0]),
           "reward_shaping": None if a.no_shaping else SHAPING,
           **{k: (list(v) if isinstance(v, tuple) else v) for k, v in tp.items() if isinstance(v, (int, float, str, bool, tuple))},
           "policy_layers": list(pp.network_factory.policy_hidden_layer_sizes) if "network_factory" in pp else None}
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
    prog, t0 = [], time.time()

    def progress(step, metrics):
        row = {"step": int(step), "wall_s": round(time.time() - t0, 1),
               "reward": float(metrics.get("eval/episode_reward", np.nan)),
               "reward_std": float(metrics.get("eval/episode_reward_std", 0.0))}
        for k, v in metrics.items():
            if k.startswith("eval/episode_reward/") and any(s in k for s in ("tracking_lin_vel", "tracking_ang_vel", "orientation", "termination", "feet_slip")):
                row[k.split("/")[-1]] = float(v)
            if k == "eval/avg_episode_length":
                row["episode_length"] = float(v)
        if len(prog) and step > prog[-1]["step"]:
            row["sps"] = round((step - prog[-1]["step"]) / max(1e-6, row["wall_s"] - prog[-1]["wall_s"]))
        prog.append(row)
        json.dump(prog, open(os.path.join(out, "progress.json"), "w"), indent=1)
        print("[%s] %5.1fM pas  R=%.2f  %s  %.0f s" % (run, step / 1e6, row["reward"], ("%d pas/s" % row["sps"]) if "sps" in row else "", row["wall_s"]), flush=True)

    if a.restore:
        rp = a.restore if a.restore.endswith(".pkl") else os.path.join(OUT, a.restore, "params.pkl")
        tp["restore_params"] = pickle.load(open(rp, "rb"))
        cfg["restore"] = a.restore; json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
    def checkpoint(step, make_policy, params):     # reprise possible apres une coupure (veille du PC, etc.)
        p = jax.device_get(params)
        pickle.dump(p, open(os.path.join(out, "checkpoint.pkl.tmp"), "wb"))
        os.replace(os.path.join(out, "checkpoint.pkl.tmp"), os.path.join(out, "checkpoint.pkl"))
        json.dump({"step": int(step)}, open(os.path.join(out, "checkpoint.json"), "w"))

    train_fn = functools.partial(ppo.train, **tp, network_factory=nf, progress_fn=progress, policy_params_fn=checkpoint,
                                 randomization_fn=registry.get_domain_randomizer(a.env))
    make_inf, params, _ = train_fn(environment=env, eval_env=eval_env, wrap_env_fn=wrapper.wrap_for_brax_training)
    pickle.dump(jax.device_get(params), open(os.path.join(out, "params.pkl"), "wb"))
    export_policy(jax.device_get(params), os.path.join(out, "policy.npz"))
    cfg["train_wall_s"] = round(time.time() - t0); json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
    print("termine :", out)


if __name__ == "__main__":
    main()
