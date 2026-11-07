"""Fix kind of each finished agent session, from the service log and the submission (no judgement).
usage: classify.py -> vault/fix_kinds.json {session: {...}}
kind:
  unchanged      submission parameters identical to the given policy
  hand_edit      submission not produced by any training job (direct weight edits, possibly on top of one)
  data_fix       produced by a training job that used demo selection / reweighting / relabelling, no extra demos
  synthetic      produced by a training job that used extra (constructed) demos
  retrain_same   produced by a training job on all original demos with no data change
plus: init of that job (given / scratch / submitted), jobs used, rollouts used, fraction of parameters changed."""
import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def params_sha_file(path):
    with np.load(path) as z:
        h = hashlib.sha256()
        for k in sorted(z.files):
            h.update(k.encode()); h.update(np.ascontiguousarray(z[k], np.float32).tobytes())
    return h.hexdigest()


def frac_changed(a, b):
    with np.load(a) as x, np.load(b) as y:
        n = sum(x[k].size for k in x.files)
        return float(sum(np.sum(np.abs(x[k].astype(np.float64) - y[k]) > 1e-6) for k in x.files) / n)


if __name__ == "__main__":
    out = {}
    units = ROOT / "vault" / "units"
    for s in json.load(open(ROOT / "vault" / "sessions.json")):
        v = ROOT / "vault" / "sessions" / s["id"]
        sub = v / "work" / "submission" / "policy.npz"
        if not (v / "manifest.json").exists():
            continue
        given = units / f"u{s['unit']}" / "policy.npz"
        svc = [json.loads(l) for l in open(v / "service.jsonl")]
        jobs = [r for r in svc if r.get("endpoint") == "/train" and "k" in r]
        rec = dict(unit=s["unit"], jobs=len(jobs), rollouts=sum(1 for r in svc if r.get("endpoint") == "/rollout" and "k" in r))
        if not sub.exists():
            rec["kind"] = "missing"
        else:
            sh = params_sha_file(sub)
            rec["frac_changed_vs_given"] = round(frac_changed(sub, given), 4)
            src = next((r for r in jobs if r.get("out_params_sha") == sh), None)
            if sh == params_sha_file(given):
                rec["kind"] = "unchanged"
            elif src is None:
                rec["kind"] = "hand_edit"
            else:
                rec["job_init"] = src["init"]
                if src.get("extra"):
                    rec["kind"] = "synthetic"
                elif src.get("relabel") or src.get("reweighted") or src.get("demos_sha") != hashlib.sha256(b"null").hexdigest()[:16]:
                    rec["kind"] = "data_fix"
                else:
                    rec["kind"] = "retrain_same"
        out[s["id"]] = rec
    json.dump(out, open(ROOT / "vault" / "fix_kinds.json", "w"), indent=1)
    print(json.dumps(out, indent=1))
