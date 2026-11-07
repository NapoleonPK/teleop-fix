"""Fill the frozen-file hashes into prereg/teleopfix.md (once, before any test session)."""
import hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
FILES = ["prereg/cards/fix.md", "tf/sim.py", "tf/operator.py", "tf/policy.py", "tf/policy_np.py", "tf/service.py",
         "tf/client.py", "tf/seeds.py", "tf/evaluate.py", "scripts/run_session.py", "scripts/run_one.sh",
         "scripts/supervisor.py", "scripts/build_units.py", "scripts/baselines.py", "scripts/make_sessions.py",
         "scripts/evaluate_all.py", "scripts/diagnosis.py", "scripts/classify.py", "scripts/audit.py",
         "scripts/analysis.py", "vault/units.json"]
FILES += sorted(str(p.relative_to(ROOT)) for u in range(1, 13) for p in (ROOT / f"vault/units/u{u}").glob("*.npz"))
if __name__ == "__main__":
    p = ROOT / "prereg/teleopfix.md"; s = p.read_text()
    assert "@HASHES@" in s, "already frozen"
    lines = "\n".join(f"- `{f}` {hashlib.sha256((ROOT / f).read_bytes()).hexdigest()}" for f in FILES)
    p.write_text(s.replace("@HASHES@", lines))
    print(hashlib.sha256(p.read_bytes()).hexdigest())
