"""Evaluation comparative des politiques entrainees (GPU, MJX, lots de robots en parallele). Meme protocole pour tous :
  A. PENTES : plan incline de 0 a 25 deg, robot face a la pente, consigne vx = 0,5 m/s, 15 s, 128 robots par pente.
     Mesures : vitesse de progression vers le haut (m/s), taux de chute.
  B. TERRAIN UA INEDIT : terrains 'front ukrainien' de graines 101 et 102 (jamais vues a l'entrainement, graine 0),
     consignes aleatoires vx 0,3-0,8 m/s, wz +-0,3, 20 s, 256 robots par terrain.
     Mesures : chutes, erreur de suivi de vitesse, vitesse moyenne ; 'reussite' = pas de chute et erreur < 0,25 m/s.
Les politiques 'sol plat' sont evaluees dans le modele du terrain accidente (memes observations).
  bash rl/run.sh evaluate.py
Sorties : results/robots/slopes.json, results/robots/benchmark.json, results/rl/<run>/eval.json"""
import functools, json, os, pickle, time
import numpy as np
import jax, jax.numpy as jp
from mujoco import mjx
from mujoco_playground import registry
from mujoco_playground.config import locomotion_params
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics
import terrain as T

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
RUNS = [("go1_ua_direct", "Go1JoystickRoughTerrain", "Go1 — UA direct, récompense par défaut"),
        ("go1_plat_sans_mise_en_forme", "Go1JoystickRoughTerrain", "Go1 — plat, récompense par défaut"),
        ("go1_sol_plat", "Go1JoystickRoughTerrain", "Go1 — sol plat"),
        ("go1_terrain_ua", "Go1JoystickRoughTerrain", "Go1 — curriculum plat→UA"),
        ("g1_sol_plat", "G1JoystickRoughTerrain", "G1 29 ddl — sol plat"),
        ("g1_terrain_ua", "G1JoystickRoughTerrain", "G1 29 ddl — curriculum plat→UA")]
BODY = {"Go1": "trunk", "G1": "torso_link"}


def policy_for(run, env_name, env):
    params = pickle.load(open(os.path.join(ROOT, "rl", run, "params.pkl"), "rb"))
    pp = locomotion_params.brax_ppo_config(env_name)
    nets = ppo_networks.make_ppo_networks(env.observation_size, env.action_size,
                                          preprocess_observations_fn=running_statistics.normalize, **pp.network_factory)
    return jax.jit(ppo_networks.make_inference_fn(nets)(params[:2], deterministic=True))


def local_v(data):
    """Vitesse lineaire dans le repere de cap du robot (avant, lateral)."""
    q = np.asarray(data.qpos[:, 3:7]); v = np.asarray(data.qvel[:, 0:3])
    w, x, y, z = q.T
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.c_[np.cos(yaw) * v[:, 0] + np.sin(yaw) * v[:, 1], -np.sin(yaw) * v[:, 0] + np.cos(yaw) * v[:, 1]]


def rollout(env, pol, n, seconds, cmd_fn, rng, face_x=False):
    steps = int(seconds / env.dt)
    keys = jax.random.split(rng, n)
    state = jax.jit(jax.vmap(env.reset))(keys)
    if face_x:                                   # robot face a +x (vers le haut de la pente), en (0,0)
        q = state.data.qpos.at[:, 0:2].set(0.0).at[:, 3:7].set(jp.array([1.0, 0, 0, 0]))
        d = jax.vmap(lambda dd, qq: mjx.forward(env.mjx_model, dd.replace(qpos=qq)))(state.data, q)
        state = state.replace(data=d)
    cmds = cmd_fn(jax.random.split(rng, n))
    step = jax.jit(jax.vmap(env.step))
    act = jax.jit(jax.vmap(lambda o, k: pol(o, k)[0]))
    x0 = np.asarray(state.data.qpos[:, 0])
    alive = np.ones(n, bool); fell_t = np.full(n, np.inf)
    err = np.zeros(n); cnt = np.zeros(n)
    for k in range(steps):
        info = dict(state.info); info["command"] = cmds; info["steps_until_next_cmd"] = jp.full((n,), 10_000, jp.int32)
        state = state.replace(info=info)
        a = act(state.obs, jax.random.split(jax.random.PRNGKey(k), n))
        state = step(state, a)
        done = np.asarray(state.done) > 0
        newly = done & alive
        fell_t[newly] = k * env.dt
        alive &= ~done
        if k > 50:
            v = local_v(state.data)
            e = np.hypot(*(v - np.asarray(cmds[:, :2])).T)
            err += e * alive; cnt += alive
    x1 = np.asarray(state.data.qpos[:, 0])
    return {"alive": alive, "fell_t": fell_t, "dx": x1 - x0, "err": err / np.maximum(cnt, 1)}


def main():
    os.makedirs(os.path.join(ROOT, "robots"), exist_ok=True)
    slopes_rows, summary = [], []
    for run, env_name, label in RUNS:
        if not os.path.exists(os.path.join(ROOT, "rl", run, "params.pkl")):
            print("absent :", run); continue
        t0 = time.time()
        env = registry.load(env_name)
        T.add_payload(env, BODY[env_name.split("Joystick")[0]], 3.0)
        try:
            pol = policy_for(run, env_name, env)
        except AttributeError as e:   # parametres ecrits par une autre version de Brax (go1_ua_direct : Brax 0.14)
            print("illisible (version de Brax) :", run, e); continue
        ev = {"rows": []}
        # A. pentes
        for deg in (0, 5, 10, 15, 20, 25):
            T.apply(env, T.slope_plane(deg))
            r = rollout(env, pol, 128, 15.0, lambda k: jp.tile(jp.array([0.5, 0.0, 0.0]), (k.shape[0], 1)), jax.random.PRNGKey(deg), face_x=True)
            spd = r["dx"] / 15.0
            for i in range(128):
                slopes_rows.append({"robot": label, "slope_deg": deg, "speed": round(float(spd[i]), 3), "fell": bool(not r["alive"][i])})
            ev["rows"].append({"test": "pente %d deg" % deg, "vitesse_m_s": float(np.mean(np.where(r["alive"], spd, 0))),
                               "chutes_pct": float(100 * (1 - r["alive"].mean()))})
            print(label, deg, "deg : v=%.2f  chutes=%.0f%%" % (ev["rows"][-1]["vitesse_m_s"], ev["rows"][-1]["chutes_pct"]), flush=True)
        # B. terrain UA inedit
        ok = falls = n = 0; spds = []
        for seed in (101, 102):
            T.apply(env, T.make(seed))
            def cmd(k):
                u = jax.vmap(lambda kk: jax.random.uniform(kk, (3,)))(k)
                return jp.stack([0.3 + 0.5 * u[:, 0], jp.zeros(k.shape[0]), 0.6 * u[:, 2] - 0.3], axis=1)
            r = rollout(env, pol, 256, 20.0, cmd, jax.random.PRNGKey(seed))
            n += 256; falls += int((~r["alive"]).sum()); ok += int((r["alive"] & (r["err"] < 0.25)).sum())
            spds.append(np.abs(r["dx"]) / 20.0)
            ev["rows"].append({"test": "terrain UA graine %d" % seed, "vitesse_m_s": float(np.mean(np.abs(r["dx"]) / 20.0)),
                               "chutes_pct": float(100 * (1 - r["alive"].mean())), "erreur_suivi_m_s": float(np.mean(r["err"][r["alive"]])) if r["alive"].any() else None})
        summary.append({"label": label, "run": run, "n": n, "success": ok, "falls": falls, "speed_mean": round(float(np.mean(np.concatenate(spds))), 3)})
        ev["wall_s"] = round(time.time() - t0)
        json.dump(ev, open(os.path.join(ROOT, "rl", run, "eval.json"), "w"), indent=1)
        print(label, "terrain UA :", summary[-1], flush=True)
    json.dump({"rows": slopes_rows, "protocol": __doc__}, open(os.path.join(ROOT, "robots", "slopes.json"), "w"))
    json.dump({"summary": summary, "note": "terrain UA inedit (graines 101-102), 512 essais de 20 s par robot, charge 3 kg ; reussite = sans chute et erreur de suivi < 0,25 m/s"},
              open(os.path.join(ROOT, "robots", "benchmark.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
