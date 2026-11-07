import numpy as np
from tf import operator as op
from scripts.baselines import heuristic_flags
if __name__ == "__main__":
    for n in ["clean", "hesitant", "desync", "comfort", "early_grip"]:
        d = op.record(op.PROFILES[n], range(1_500_000, 1_500_060), workers=4)
        f = heuristic_flags(d["X"], d["A"])
        true_h = [m["hesitation_steps"] > 0 for m in d["meta"]]
        print(n, {k: sum(g[k] for g in f) for k in ("pause", "retry", "fail")}, "any", sum(any(g.values()) for g in f), "/60",
              "| true hes", sum(true_h), "caught", sum(a and b["pause"] for a, b in zip(true_h, f)), flush=True)
