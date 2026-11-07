"""Evaluate every arm of the given units on a seed set. usage: evaluate_all.py dev|heldout OUT_JSON UNIT [UNIT ...]
Arms: broken (the unit's policy), each file in vault/units/u<unit>/arms/*.npz, and agent:<session id> for every
session of the unit (its work/submission/policy.npz; a missing or invalid submission scores as the broken policy and
is marked invalid). Regions: whole, near (x <= -0.02), far (x > 0.05). 200 seeds each (seed i of a region is the
same seed number in every arm). Resumable: arms already in OUT_JSON are skipped."""
import json
import sys
from pathlib import Path

import numpy as np

from tf import evaluate, seeds

ROOT = Path(__file__).resolve().parent.parent
REGIONS = ("whole", "near", "far")


def valid(path, ref):
    try:
        with np.load(path) as a, np.load(ref) as b:
            return sorted(a.files) == sorted(b.files) and all(a[k].shape == b[k].shape and np.all(np.isfinite(a[k])) for k in a.files)
    except Exception:
        return False


if __name__ == "__main__":
    which, out, units = sys.argv[1], Path(sys.argv[2]), [int(u) for u in sys.argv[3:]]
    S = {"dev": seeds.DEV_EVAL, "heldout": seeds.HELDOUT}[which]
    res = json.load(open(out)) if out.exists() else {}
    sess = json.load(open(ROOT / "vault" / "sessions.json")) if (ROOT / "vault" / "sessions.json").exists() else []
    for u in units:
        d = ROOT / "vault" / "units" / f"u{u}"
        arms = {"broken": d / "policy.npz", **{p.stem: p for p in sorted((d / "arms").glob("*.npz"))}}
        notes = {}
        for s in sess:
            if s["unit"] == u:
                sub = ROOT / "vault" / "sessions" / s["id"] / "work" / "submission" / "policy.npz"
                ok = sub.exists() and valid(sub, d / "policy.npz")
                arms[f"agent:{s['id']}"] = sub if ok else d / "policy.npz"
                notes[f"agent:{s['id']}"] = "ok" if ok else "invalid or missing -> broken policy"
        r = res.setdefault(str(u), {})
        for name, path in arms.items():
            if name in r:
                continue
            r[name] = {reg: evaluate.success(path, S, reg) for reg in REGIONS}
            if name in notes:
                r[name]["submission"] = notes[name]
            json.dump(res, open(out, "w"), indent=1)
            print(u, name, r[name], flush=True)
