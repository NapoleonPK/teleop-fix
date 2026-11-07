"""Non-agent arms for each built unit. Writes one weights file per (unit, arm) to vault/units/u<unit>/arms/.
usage: baselines.py UNIT [UNIT ...]   (skips arms already done)

Arms (frozen in the protocol):
  demos_K (K = 10, 25, 50): the same operator records K new demos (seeds.extra_demos), in the far region if the
      operator's habit is the comfort zone (targeted at the failure), else over the whole table; fine-tune the broken
      policy on old + new demos (2000 steps, lr 1e-4).
  filter: drop demos flagged by heuristic_flags() and retrain from scratch with the recipe.
  longer: continue training the broken policy for another full recipe run (8000 steps, lr 3e-4) on the same data.
  search: 4 random 80% subsets, retrain each from scratch, 5 rollouts each on seeds.search_rollout, keep the best
      (ties -> lowest subset index). 20 rollouts in total.
"""
import json
import sys
from pathlib import Path

import numpy as np

from tf import operator as op, policy, seeds, sim
from tf.service import RECIPE

ROOT = Path(__file__).resolve().parent.parent


def heuristic_flags(X, A):
    """Frozen day-one data-cleaning rule, using only X and A. A demo is dropped if it
    (a) pauses: an 8-step window in which the gripper's net displacement is under 1 cm while the fingers are commanded
        open throughout and the block is more than 5 cm from the goal (net displacement, so hand tremor is ignored);
    (b) re-opens the gripper after closing it while the block is still far from the goal (a retry);
    (c) fails: block not within 5 cm of the goal at the last step."""
    flags = []
    for x, a in zip(X, A):
        far = np.linalg.norm(x[:, 3:6] - x[:, 25:28], axis=1) > 0.05
        W = 8
        pause = any(np.linalg.norm(x[t + W, 0:3] - x[t, 0:3]) < 0.01 and np.all(a[t:t + W, 3] > 0) and np.all(far[t:t + W + 1])
                    for t in range(len(x) - W))
        closed = np.where(a[:, 3] < 0)[0]
        reopen = bool(len(closed)) and bool(np.any((a[closed[0]:, 3] > 0) & far[closed[0]:]))
        fail = not bool(np.linalg.norm(x[-1, 3:6] - x[-1, 25:28]) < sim.SUCCESS_M)
        flags.append(dict(pause=bool(pause), retry=reopen, fail=fail))
    return flags


def ft(m0, X, A, seed):
    return policy.train(X, A, seed=seed, steps=RECIPE["finetune_steps"], lr=RECIPE["finetune_lr"], init=m0, threads=8)


def scratch(X, A, seed):
    return policy.train(X, A, seed=seed, steps=RECIPE["steps"], lr=RECIPE["lr"], threads=8)


def run(unit):
    d = ROOT / "vault" / "units" / f"u{unit}"
    meta = json.load(open(d / "meta.json"))
    prof = op.Profile(**{k: tuple(v) if isinstance(v, list) else v for k, v in meta["profile"].items()})
    with np.load(d / "dataset.npz") as z:
        X, A = z["X"], z["A"]
    arms = d / "arms"; arms.mkdir(exist_ok=True)
    log = json.load(open(arms / "log.json")) if (arms / "log.json").exists() else {}

    def done(name, m, **info):
        policy.save(m, arms / f"{name}.npz"); log[name] = info
        json.dump(log, open(arms / "log.json", "w"), indent=1); print(unit, name, info, flush=True)

    target = "far" if prof.comfort == "near" else "whole"
    from dataclasses import replace
    for K in (10, 25, 50):
        if f"demos_{K}" in log:
            continue
        extra = op.record(replace(prof, comfort=target), seeds.extra_demos(unit, K), workers=8)
        m = ft(policy.load(d / "policy.npz"), np.concatenate([X, extra["X"]]), np.concatenate([A, extra["A"]]), unit * 10 + K)
        done(f"demos_{K}", m, operator_minutes=K * op.OPERATOR_MIN_PER_DEMO, region=target,
             extra_demo_success=float(np.mean([e["success"] for e in extra["meta"]])))
    if "filter" not in log:
        f = heuristic_flags(X, A)
        keep = [i for i, g in enumerate(f) if not any(g.values())]
        if len(keep) < 50:                       # rule floor: never train on fewer than 50 demos
            keep = list(range(len(X)))
        done("filter", scratch(X[keep], A[keep], unit * 10 + 1), kept=len(keep),
             dropped=dict(pause=sum(g["pause"] for g in f), retry=sum(g["retry"] for g in f), fail=sum(g["fail"] for g in f)))
    if "longer" not in log:
        m = policy.train(X, A, seed=unit * 10 + 2, steps=RECIPE["steps"], lr=RECIPE["lr"], init=policy.load(d / "policy.npz"), threads=8)
        done("longer", m)
    if "search" not in log:
        rng = np.random.default_rng(unit * 10 + 3)
        best, scores = None, []
        for j in range(4):
            sub = np.sort(rng.choice(len(X), int(0.8 * len(X)), replace=False))
            m = scratch(X[sub], A[sub], unit * 10 + 30 + j)
            s = sum(sim.rollout(policy.Runner(m), sim.Spec(seeds.search_rollout(unit, 5 * j + k))) for k in range(5))
            scores.append(int(s))
            if best is None or s > best[0]:
                best = (s, m)
        done("search", best[1], subset_successes_of_5=scores)


if __name__ == "__main__":
    for u in sys.argv[1:]:
        run(int(u))
