"""Calibration on dev seeds: one policy per operator profile (500 demos, recipe), success on DEV_EVAL by region.
usage: calibrate.py OUT_JSON PROFILE[,PROFILE...] [UNIT=900]   (units 900+ are calibration-only datasets)"""
import json, sys, time
from dataclasses import replace
from pathlib import Path
import numpy as np
from tf import operator as op, policy, evaluate, seeds
from tf.service import RECIPE

def run(name, prof, unit):
    d = op.record(prof, seeds.demos(unit, 500), workers=8)
    m = policy.train(d["X"], d["A"], seed=unit, steps=RECIPE["steps"], lr=RECIPE["lr"], threads=8)
    p = Path(f"runs/calib/{name}_u{unit}.npz"); p.parent.mkdir(parents=True, exist_ok=True); policy.save(m, p)
    r = dict(profile=name, unit=unit, demo_success=float(np.mean([x["success"] for x in d["meta"]])))
    for reg in ["whole", "near", "far"]:
        r[reg] = evaluate.success(p, seeds.DEV_EVAL, reg)
    return r

if __name__ == "__main__":
    out, names = sys.argv[1], sys.argv[2].split(",")
    unit = int(sys.argv[3]) if len(sys.argv) > 3 else 900
    res = json.load(open(out)) if Path(out).exists() else []
    for n in names:
        if ":" in n:                       # name:field=value;field=value  (ad-hoc variant)
            base, kv = n.split(":", 1)
            prof = replace(op.PROFILES[base], name=n, **{k: type(getattr(op.PROFILES[base], k))(eval(v)) for k, v in (x.split("=") for x in kv.split(";"))})
        else:
            prof = op.PROFILES[n]
        t = time.time(); r = run(n, prof, unit); r["minutes"] = round((time.time() - t) / 60, 1)
        res.append(r); json.dump(res, open(out, "w"), indent=1); print(r, flush=True)
