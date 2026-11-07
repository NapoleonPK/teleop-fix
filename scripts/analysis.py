"""Teleop-Fix analysis: run once on the frozen result files. usage: analysis.py EVAL_JSON OUT_DIR [PHASE=test]
Inputs: vault/sessions.json, EVAL_JSON (evaluate_all.py output), vault/diagnosis_scores.json, vault/fix_kinds.json,
vault/audit_verdicts.json (required for every flagged session; a confirmed violation scores as the broken policy and 0
diagnosis). Outputs OUT_DIR/teleopfix.json and OUT_DIR/teleopfix.md. Every number in the report comes from here.

Gain of an arm on a unit = its whole-table success minus the broken policy's (held-out seeds). Per failure type, gains
are averaged over its units. Hypotheses (prereg):
  H1: on >= 2 of the 3 failure types, agent mean gain >= demos_25 mean gain - 0.05.
  H2: on >= 2 of the 3 failure types, agent mean gain >= filter mean gain + 0.10.
  H3: diagnosis score 1 on >= 6 of the 9 broken units.
  H4: on the clean controls, agent mean change >= -0.05.
Exploratory: demos-equivalent of the agent (linear interpolation of the agent's mean gain on the K = 0, 10, 25, 50
curve, ">50" above it); collateral (near-region change for comfort units; per-region table for all); fix kinds."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TYPES = ["hesitant", "desync", "comfort"]
ARMS = ["demos_10", "demos_25", "demos_50", "filter", "longer", "search", "agent"]


def load(p, default):
    return json.load(open(p)) if Path(p).exists() else default


def main():
    ev_path, out = Path(sys.argv[1]), Path(sys.argv[2])
    phase = sys.argv[3] if len(sys.argv) > 3 else "test"
    out.mkdir(parents=True, exist_ok=True)
    ev = json.load(open(ev_path))
    S = [s for s in json.load(open(ROOT / "vault" / "sessions.json")) if s["phase"] == phase]
    diag = load(ROOT / "vault" / "diagnosis_scores.json", {})
    kinds = load(ROOT / "vault" / "fix_kinds.json", {})
    verdicts = load(ROOT / "vault" / "audit_verdicts.json", {})
    rows = []
    for s in S:
        au = load(ROOT / "vault" / "sessions" / s["id"] / "audit.json", {"n_flags": 0})
        if au.get("n_flags") and s["id"] not in verdicts:
            raise SystemExit(f"{s['id']} is flagged and has no hand verdict")
        viol = verdicts.get(s["id"], "clean").startswith("violation")
        e = ev[str(s["unit"])]
        agent = e["broken"] if viol else e[f"agent:{s['id']}"]
        r = dict(session=s["id"], unit=s["unit"], profile=s["profile"], violation=viol,
                 broken={k: e["broken"][k] for k in ("whole", "near", "far")},
                 arms={a: {k: e[a][k] for k in ("whole", "near", "far")} for a in ARMS[:-1] if a in e},
                 agent={k: agent[k] for k in ("whole", "near", "far")},
                 diagnosis=0 if viol else diag.get(s["id"]), kind=kinds.get(s["id"], {}).get("kind"))
        r["gain"] = {a: round(r["arms"][a]["whole"] - r["broken"]["whole"], 4) for a in r["arms"]}
        r["gain"]["agent"] = round(r["agent"]["whole"] - r["broken"]["whole"], 4)
        rows.append(r)

    def mean_gain(t, a):
        g = [r["gain"][a] for r in rows if r["profile"] == t and a in r["gain"]]
        return float(np.mean(g)) if g else None

    per_type = {t: {a: mean_gain(t, a) for a in ARMS} for t in TYPES}
    h1_types = [t for t in TYPES if per_type[t]["agent"] is not None and per_type[t]["demos_25"] is not None
                and per_type[t]["agent"] >= per_type[t]["demos_25"] - 0.05]
    h2_types = [t for t in TYPES if per_type[t]["agent"] is not None and per_type[t]["filter"] is not None
                and per_type[t]["agent"] >= per_type[t]["filter"] + 0.10]
    broken_units = [r for r in rows if r["profile"] in TYPES]
    n_diag1 = sum(r["diagnosis"] == 1 for r in broken_units)
    clean = [r["gain"]["agent"] for r in rows if r["profile"] == "clean"]
    h4_change = float(np.mean(clean)) if clean else None

    def demos_equiv(t):
        ks, gs = [0, 10, 25, 50], [0.0] + [per_type[t][f"demos_{k}"] for k in (10, 25, 50)]
        a = per_type[t]["agent"]
        if a is None or None in gs:
            return None
        if a > max(gs):
            return ">50"
        for i in range(3):
            lo, hi = sorted((gs[i], gs[i + 1]))
            if lo <= a <= hi and gs[i + 1] != gs[i]:
                return round(ks[i] + (a - gs[i]) / (gs[i + 1] - gs[i]) * (ks[i + 1] - ks[i]), 1)
        return "<=0" if a <= 0 else None

    res = dict(
        rows=rows, per_type=per_type,
        H1=dict(passed=len(h1_types) >= 2, types=h1_types),
        H2=dict(passed=len(h2_types) >= 2, types=h2_types),
        H3=dict(passed=n_diag1 >= 6, n_score1=n_diag1, n_units=len(broken_units)),
        H4=dict(passed=h4_change is not None and h4_change >= -0.05, mean_change=h4_change),
        demos_equivalent={t: demos_equiv(t) for t in TYPES},
        collateral_near_comfort=[round(r["agent"]["near"] - r["broken"]["near"], 4) for r in rows if r["profile"] == "comfort"],
        fix_kinds={r["session"]: r["kind"] for r in rows},
    )
    json.dump(res, open(out / "teleopfix.json", "w"), indent=1)
    yes = lambda b: "passed" if b else "did not pass"
    pct = lambda v: "n/a" if v is None else f"{100 * v:+.1f}"
    L = ["# Teleop-Fix results", "", f"- H1 (agent >= 25 targeted demos - 5 pts, >= 2 of 3 types): {yes(res['H1']['passed'])} "
         f"({', '.join(h1_types) or 'none'})",
         f"- H2 (agent >= heuristic filter + 10 pts, >= 2 of 3 types): {yes(res['H2']['passed'])} ({', '.join(h2_types) or 'none'})",
         f"- H3 (correct diagnosis on >= 6 of 9): {yes(res['H3']['passed'])} ({n_diag1} of {len(broken_units)})",
         f"- H4 (clean controls lose <= 5 pts): {yes(res['H4']['passed'])} (mean change {pct(h4_change)} pts)", "",
         "## Mean gain over the broken policy, whole table, points", "",
         "| Failure type | " + " | ".join(ARMS) + " |", "|" + " --- |" * (len(ARMS) + 1)]
    L += [f"| {t} | " + " | ".join(pct(per_type[t][a]) for a in ARMS) + " |" for t in TYPES]
    L += ["", "Agent worth in demos (interpolated on the K curve): " +
          ", ".join(f"{t} {res['demos_equivalent'][t]}" for t in TYPES), "",
          "## Per unit (whole / near / far success)", "", "| Unit | Type | Broken | Agent | Kind | Diagnosis |", "| --- | --- | --- | --- | --- | --- |"]
    f3 = lambda d: f"{d['whole']:.3f} / {d['near']:.3f} / {d['far']:.3f}"
    L += [f"| u{r['unit']} | {r['profile']} | {f3(r['broken'])} | {f3(r['agent'])} | {r['kind']} | {r['diagnosis']} |" for r in rows]
    (out / "teleopfix.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:6]))


if __name__ == "__main__":
    main()
