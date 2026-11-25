import inspect
from brax.training.agents.ppo import train
sig = inspect.signature(train.train)
print([p for p in sig.parameters if "restore" in p or "curriculum" in p or "checkpoint" in p])
import json, os
for r in ("go1_sol_plat",):
    f = os.path.join(os.path.dirname(__file__), "..", "results", "rl", r, "progress.json")
    if os.path.exists(f):
        print(r, [(x["step"] // 10**6, round(x["reward"], 1)) for x in json.load(open(f))])
