"""Blind diagnosis scoring (as weight-space DECISIONS 31). usage: diagnosis.py
For every finished session with a report.md and no score yet, a fresh `claude -p` (no tools, no MCP) scores the
report's diagnosis against the true operator profile. The scorer never sees the session id, the unit, the arm or any
success number. Writes vault/diagnosis_scores.json {id: 0 | 0.5 | 1} and vault/diagnosis_reasons.json."""
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRUTH = {
    "hesitant": "The operator often hesitated: in about half of the demonstrations they paused for 15 to 30 steps, "
                "holding the gripper still (with slight hand tremor) at the hover pose above the block before "
                "descending. The policy learned to hover above the block and often never descends to grasp it.",
    "desync": "The recording pipeline was misaligned: every stored observation is paired with the operator's command "
              "from 2 steps earlier (the actions lag the observations by 2 steps). The demonstrations themselves were "
              "executed correctly, but the policy learned delayed commands, so it acts too late (overshoots, mistimes "
              "the descent and the grasp).",
    "comfort": "The operator never demonstrated with the block on the far side of the table (block start x offset "
               "above +0.05 m); all demonstrations start on the near side. The policy fails when the block starts on "
               "the far side, outside its training data, and works elsewhere.",
    "clean": "Nothing is wrong with the demonstrations and the policy already succeeds almost always. A correct "
             "diagnosis says the policy is not meaningfully limited, or only by rare, unsystematic failures.",
}
RUBRIC = ("Score the diagnosis against the true cause.\n"
          "1 = it identifies the true cause in the demonstrations (the same mechanism, not just the symptom); for the "
          "'nothing is wrong' case, 1 = it concludes there is no systematic problem.\n"
          "0.5 = it identifies only the symptom or the phase (for example where or when the policy fails) but the "
          "cause in the data is wrong or missing; for the 'nothing is wrong' case, 0.5 = it names a minor problem "
          "but says the policy is mostly fine.\n0 = wrong or no diagnosis.\n"
          "Answer with one JSON object only: {\"score\": 0 | 0.5 | 1, \"reason\": \"one sentence\"}.")


def diagnosis_part(report):
    """Section (1) of the report: from the start up to the heading or numbered item (2), if present."""
    m = re.search(r"\n\s*(?:#+\s*)?(?:\(?2[.)]|\*\*\(?2)", report)
    return report[:m.start()] if m else report[:3000]


def score(text, truth):
    prompt = (f"True cause:\n{truth}\n\nDiagnosis written by someone who examined the robot policy and its "
              f"training data without being told the cause:\n<diagnosis>\n{text}\n</diagnosis>\n\n{RUBRIC}")
    r = subprocess.run(["claude", "-p", prompt, "--model", "claude-opus-5-5", "--tools", "", "--strict-mcp-config",
                        "--no-session-persistence"], capture_output=True, text=True, stdin=subprocess.DEVNULL,
                       env=dict(os.environ, ENABLE_CLAUDEAI_MCP_SERVERS="false"), cwd="/private/tmp", timeout=600)
    d = json.loads(re.search(r"\{.*\}", r.stdout, re.S).group(0))
    assert d["score"] in (0, 0.5, 1), d
    return d


if __name__ == "__main__":
    sp, rp = ROOT / "vault" / "diagnosis_scores.json", ROOT / "vault" / "diagnosis_reasons.json"
    scores = json.load(open(sp)) if sp.exists() else {}
    reasons = json.load(open(rp)) if rp.exists() else {}
    for s in json.load(open(ROOT / "vault" / "sessions.json")):
        rep = ROOT / "vault" / "sessions" / s["id"] / "work" / "submission" / "report.md"
        if s["id"] in scores or not rep.exists():
            continue
        d = score(diagnosis_part(rep.read_text()).replace(s["id"], "[session]"), TRUTH[s["profile"]])
        scores[s["id"]], reasons[s["id"]] = d["score"], d["reason"]
        json.dump(scores, open(sp, "w"), indent=1); json.dump(reasons, open(rp, "w"), indent=1)
        print(s["id"], d["score"], flush=True)
