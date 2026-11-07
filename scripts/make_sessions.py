"""Make agent session configs for built units. usage: make_sessions.py PHASE UNIT [UNIT ...]
Writes runs/<id>.json per session and appends {id, unit, profile, phase, session_index} to vault/sessions.json.
Ids are random hex (a folder name reveals nothing); session_index = position in vault/sessions.json + 1."""
import json
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    phase, units = sys.argv[1], [int(u) for u in sys.argv[2:]]
    sp = ROOT / "vault" / "sessions.json"
    S = json.load(open(sp)) if sp.exists() else []
    for u in units:
        d = ROOT / "vault" / "units" / f"u{u}"
        meta = json.load(open(d / "meta.json"))
        sid = "t" + secrets.token_hex(4)
        idx = len(S) + 1
        cfg = dict(id=sid, task_card=str(ROOT / "prereg" / "cards" / "fix.md"),
                   files={"policy.npz": str(d / "policy.npz"), "dataset.npz": str(d / "dataset.npz"),
                          "failed_episodes": str(d / "failed_episodes")},
                   dataset=str(d / "dataset.npz"), policy=str(d / "policy.npz"),
                   rollout_budget=20, train_budget=6, max_turns=150, max_hours=3, session_index=idx)
        json.dump(cfg, open(ROOT / "runs" / f"{sid}.json", "w"), indent=1)
        S.append(dict(id=sid, unit=u, profile=meta["profile"]["name"], phase=phase, session_index=idx))
        print(sid, u, meta["profile"]["name"])
    json.dump(S, open(sp, "w"), indent=1)
