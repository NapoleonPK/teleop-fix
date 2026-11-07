import sys, time, numpy as np
from tf import operator as op, policy, evaluate
if __name__ == "__main__":
    prof, n, steps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    t = time.time(); d = op.record(op.PROFILES[prof], range(1_000_000, 1_000_000 + n), workers=8); print("record", round(time.time()-t), "s", flush=True)
    t = time.time(); m = policy.train(d["X"], d["A"], steps=steps, threads=8, log=lambda s: print(s, flush=True)); print("train", round(time.time()-t), "s params", policy.n_params(m), flush=True)
    policy.save(m, f"runs/smoke_{prof}.npz")
    for reg in ["whole", "far"]:
        t = time.time(); print(reg, evaluate.success(f"runs/smoke_{prof}.npz", range(9_000_000, 9_000_100), reg), round(time.time()-t), "s", flush=True)
