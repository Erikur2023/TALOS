"""Images de la scene MuJoCo (G1 dans la foret generee) pour l'IHM et le dossier.
  python -m talos2.mujoco_g1.snapshots --seed 105 --out ihm/mujoco"""
import argparse, os
import numpy as np
import mujoco
from .run import G1Mission, mujoco_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=105)
    ap.add_argument("--out", default="ihm/mujoco")
    ap.add_argument("--walk", type=float, default=20.0, help="secondes de marche avant les images")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    M = G1Mission(a.seed, mujoco_params(), relief_scale=0.4)
    for k in range(int(a.walk / 0.1)):
        M.advance(0.5, 0.0, k * 0.1)
    M.m.vis.global_.offwidth, M.m.vis.global_.offheight = 800, 480
    M.m.vis.map.zfar = 200
    r = mujoco.Renderer(M.m, 480, 800)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = 1
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 1                    # obstacles (groupe 3) et decor (groupe 4) visibles
    from PIL import Image
    out = []
    for name, dist, elev, az in (("suivi", 4.5, -18, 200), ("dessus", 14, -60, 160), ("loin", 30, -25, 230)):
        cam.distance, cam.elevation, cam.azimuth = dist, elev, az
        r.update_scene(M.d, cam, opt)
        img = r.render()
        path = os.path.join(a.out, "g1_%s.jpg" % name)
        Image.fromarray(img).save(path, quality=85)
        out.append(path)
        print(path)
    return out


if __name__ == "__main__":
    main()
