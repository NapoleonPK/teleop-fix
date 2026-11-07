import numpy as np
from dataclasses import replace
from tf import operator as op
if __name__ == "__main__":
    for lag, gain in [(1,1.0),(2,1.0),(2,0.6),(3,0.5),(3,0.35),(4,0.35)]:
        prof = replace(op.PROFILES["laggy"], lag=lag, gain=gain)
        d = op.record(prof, range(1_000_000, 1_000_040), workers=8)
        print(lag, gain, np.mean([m["success"] for m in d["meta"]]), flush=True)
