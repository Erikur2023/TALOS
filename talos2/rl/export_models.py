"""Copie les modeles MJCF + maillages des robots de Playground (Menagerie, Apache-2.0 / BSD) vers rl/models/<robot>/
pour les utiliser sous Windows (MuJoCo CPU) dans les missions TALOS."""
import os, json
from mujoco_playground._src.locomotion.go1 import base as go1base, go1_constants as C1
from mujoco_playground._src.locomotion.g1 import base as g1base, g1_constants as C2

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
for name, base, xml in (("go1", go1base, C1.FEET_ONLY_FLAT_TERRAIN_XML), ("g1pg", g1base, C2.FEET_ONLY_FLAT_TERRAIN_XML)):
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    assets = base.get_assets()
    for k, v in assets.items():
        p = os.path.join(d, "assets", k)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "wb").write(v)
    open(os.path.join(d, "scene.xml"), "w").write(xml.read_text())
    json.dump({"scene": os.path.basename(str(xml)), "n_assets": len(assets)}, open(os.path.join(d, "info.json"), "w"))
    print(name, len(assets), "fichiers", xml)
