"""Unattended runner for the Teleop-Fix test phase (launchd agent com.teleopfix.supervisor; adapted from the weight-space
supervisor). Every 2 minutes, when no session is running:
  1. usage limit (a transcript saying "hit your"): move the folder to vault/failed_usage_limit/<id>_<n>, wait until
     the quoted reset + 5 min (30 min if unparsed), rerun;
  2. harness crash (folder without manifest.json): move to vault/failed_crash/, rerun once; a second crash = missing;
  3. otherwise launch the next test session without audit.json (one at a time, fixed random order).
When every test session is done (or final): wait for the baseline arms, then run diagnosis.py (retried after limits),
classify.py and evaluate_all.py heldout on all test units, write runs/ALL_DONE and exit 0.
Log runs/supervisor.log; state runs/supervisor_state.json."""
import datetime as dt
import json
import random
import re
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V, RUNS = ROOT / "vault", ROOT / "runs"
SUBJECT = Path.home() / "tf-subject" / "sessions"
PY = str(ROOT / ".venv" / "bin" / "python")
ENV = {"PYTHONPATH": str(ROOT), "PATH": "/Users/teo/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin", "HOME": str(Path.home())}


def log(m):
    with open(RUNS / "supervisor.log", "a") as f:
        f.write(f"{dt.datetime.now():%F %T} {m}\n")


def state():
    p = RUNS / "supervisor_state.json"
    return json.load(open(p)) if p.exists() else {"crashes": {}, "wait_until": None}


def save(st):
    json.dump(st, open(RUNS / "supervisor_state.json", "w"), indent=1)


def busy(pattern):
    return bool(subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True).stdout.strip())


def tests():
    S = [s for s in json.load(open(V / "sessions.json")) if s["phase"] == "test"]
    random.Random(11).shuffle(S)
    return S


def move(sid, root):
    root.mkdir(exist_ok=True)
    k = 1
    while (root / f"{sid}_{k}").exists():
        k += 1
    dst = root / f"{sid}_{k}"
    if (V / "sessions" / sid).exists():
        shutil.move(str(V / "sessions" / sid), str(dst))
    else:
        dst.mkdir()
    if (SUBJECT / sid).exists():
        shutil.move(str(SUBJECT / sid), str(dst / "subject_folder"))
    if (RUNS / f"{sid}.out").exists():
        shutil.move(str(RUNS / f"{sid}.out"), str(dst / "run.out"))
    return dst.name


def reset_time(txt):
    m = re.search(r"resets (\d{1,2})(?::(\d{2}))?\s*([ap]m)", txt)
    if not m:
        return None
    h = int(m.group(1)) % 12 + (12 if m.group(3) == "pm" else 0)
    now = dt.datetime.now()
    r = now.replace(hour=h, minute=int(m.group(2) or 0), second=0, microsecond=0)
    return r + dt.timedelta(days=1) if r < now - dt.timedelta(minutes=1) else r


def sweep(st):
    """Handle unfinished folders. Returns a datetime to wait for if a usage limit was seen."""
    wait = None
    for s in tests():
        sid = s["id"]; d = V / "sessions" / sid
        if (d / "audit.json").exists() or st["crashes"].get(sid, 0) >= 2 or not (d.exists() or (SUBJECT / sid).exists()):
            continue
        t = d / "transcript.jsonl"
        txt = t.read_text(errors="ignore") if t.exists() else ""
        if "hit your" in txt:
            r = reset_time(txt) or dt.datetime.now() + dt.timedelta(minutes=25)
            wait = max(wait, r) if wait else r
            log(f"usage limit: {sid} -> failed_usage_limit/{move(sid, V / 'failed_usage_limit')}")
        elif (d / "manifest.json").exists():
            subprocess.run([PY, "scripts/audit.py", sid], cwd=ROOT, env=ENV, capture_output=True)
            log(f"audited {sid}")
        else:
            st["crashes"][sid] = st["crashes"].get(sid, 0) + 1
            if st["crashes"][sid] >= 2:
                log(f"crash #2: {sid} final (scores as missing)")
            else:
                log(f"crash #1: {sid} -> failed_crash/{move(sid, V / 'failed_crash')}, rerun once")
    return wait + dt.timedelta(minutes=5) if wait else None


def finish():
    while busy("scripts/baselines.py|scripts/build_units.py"):
        time.sleep(120)
    for _ in range(20):                                   # the blind scorer also uses the account's usage
        r = subprocess.run([PY, "scripts/diagnosis.py"], cwd=ROOT, env=ENV, capture_output=True, text=True)
        log(f"diagnosis rc {r.returncode}: {r.stdout.strip()[-200:]}")
        if r.returncode == 0:
            break
        time.sleep(1800)
    subprocess.run([PY, "scripts/classify.py"], cwd=ROOT, env=ENV, capture_output=True)
    units = sorted({str(s["unit"]) for s in tests()}, key=int)
    r = subprocess.run([PY, "scripts/evaluate_all.py", "heldout", "vault/eval_heldout.json", *units], cwd=ROOT, env=ENV,
                       capture_output=True, text=True)
    log(f"evaluate_all heldout rc {r.returncode}")
    (RUNS / "ALL_DONE").write_text(f"{dt.datetime.now():%F %T}\n")
    log("all done; supervisor exits")


if __name__ == "__main__":
    log("supervisor started")
    while True:
        st = state()
        if busy("scripts/run_session.py"):
            time.sleep(120); continue
        if st.get("wait_until") and dt.datetime.now() < dt.datetime.fromisoformat(st["wait_until"]):
            time.sleep(120); continue
        w = sweep(st)
        st["wait_until"] = w.isoformat(timespec="seconds") if w else None
        save(st)
        if w:
            log(f"waiting for the usage limit until {w:%F %T}"); continue
        left = [s for s in tests() if not (V / "sessions" / s["id"] / "audit.json").exists() and st["crashes"].get(s["id"], 0) < 2]
        if not left:
            finish(); break
        sid = left[0]["id"]
        log(f"launch {sid} (unit {left[0]['unit']}); {len(left)} left")
        subprocess.Popen(["/bin/bash", "scripts/run_one.sh", sid], cwd=ROOT, env=ENV, stdout=open(RUNS / "queue.log", "a"),
                         stderr=subprocess.STDOUT, start_new_session=True)
        time.sleep(120)
