import sys, numpy as np
from tf import operator as op
if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    for name, prof in op.PROFILES.items():
        d = op.record(prof, range(1_000_000, 1_000_000 + n), workers=8)
        m = d["meta"]
        print(f"{name:11s} demo success {np.mean([x['success'] for x in m]):.2f}  hes {np.mean([x['hesitation_steps']>0 for x in m]):.2f}"
              f"  retry {np.mean([x['retry'] for x in m]):.2f}  n {len(m)}", flush=True)
