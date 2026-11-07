"""Build Teleop-Fix units: for each (unit, profile) record the 500-demo dataset, train the policy with the recipe and
find up to 3 failed episodes (on seeds 7_000_000 + 100*unit + k, at most 40 tries) for the agent's task folder.
usage: build_units.py UNITS_JSON   ([{unit, profile}, ...]) -> vault/units/u<unit>/{dataset.npz, meta.json,
policy.npz, failed_episodes/*.json}. Skips units already built."""
import json
import sys
from pathlib import Path

import numpy as np

from tf import operator as op, policy, sim
from tf.service import RECIPE

ROOT = Path(__file__).resolve().parent.parent


def failures(path, unit, want=3, tries=40):
    m = policy.load(path)
    out = []
    for k in range(tries):
        ok, X, A, xe = sim.rollout(policy.Runner(m), sim.Spec(7_000_000 + 100 * unit + k), record=True)
        if not ok:
            out.append(dict(observations=np.round(X, 5).tolist(), actions=np.round(A, 5).tolist(),
                            final_object_to_goal_m=float(np.linalg.norm(xe[3:6] - xe[25:28]))))
            if len(out) == want:
                break
    return out, k + 1


if __name__ == "__main__":
    from tf import seeds
    for u in json.load(open(sys.argv[1])):
        d = ROOT / "vault" / "units" / f"u{u['unit']}"
        if (d / "meta.json").exists():
            continue
        d.mkdir(parents=True, exist_ok=True)
        data = op.record(op.PROFILES[u["profile"]], seeds.demos(u["unit"], 500), workers=8)
        np.savez_compressed(d / "dataset.npz", X=data["X"], A=data["A"])
        m = policy.train(data["X"], data["A"], seed=u["unit"], steps=RECIPE["steps"], lr=RECIPE["lr"], threads=8)
        policy.save(m, d / "policy.npz")
        fe, tried = failures(d / "policy.npz", u["unit"])
        (d / "failed_episodes").mkdir(exist_ok=True)
        for i, e in enumerate(fe):
            json.dump(e, open(d / "failed_episodes" / f"failure_{i + 1}.json", "w"))
        json.dump(dict(unit=u["unit"], profile=data["profile"], demo_meta=data["meta"], failures_found=len(fe),
                       failure_tries=tried), open(d / "meta.json", "w"), indent=1)
        print(u, "failures", len(fe), "of", tried, flush=True)
