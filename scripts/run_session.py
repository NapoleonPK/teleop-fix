"""Teleop-Fix agent session (adapted from weightspace/scripts/run_session.py): fresh folder outside ~/Developer, its own
rollout + training service, `claude -p` under --restricted with
the sandbox, limits enforced here (turns counted from the event stream; wall clock). Writes the transcript, the
service log and a manifest to vault/sessions/<id>/; the session folder keeps only what the subject saw and wrote.

usage: run_session.py SESSION_JSON
SESSION_JSON: {id, task_card, files {name: source path}, dataset, policy, rollout_budget, train_budget,
               max_turns, max_hours, session_index}
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUBJECT = Path.home() / "tf-subject"
PYENV = Path.home() / "ws-subject" / "py"   # numpy-only interpreter shared with the weight-space subjects
MODEL, EFFORT = "claude-opus-5-5", "high"


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def other_sessions(sid):
    """Every other session folder name, present or scheduled (deny rules cannot carve an exception out of a parent)."""
    names = {p.name for p in (SUBJECT / "sessions").iterdir()} if (SUBJECT / "sessions").exists() else set()
    for c in (ROOT / "runs").glob("*.json"):
        try:
            d = json.load(open(c))
        except ValueError:
            continue
        if isinstance(d, dict) and "id" in d:
            names.add(d["id"])
    return sorted(names - {sid})


def settings(session_dir):
    py = str(PYENV / "bin")
    others = [f"Read(//{str(SUBJECT / 'sessions' / n).lstrip('/')}/**)" for n in other_sessions(session_dir.name)]
    return {
        "autoMemoryEnabled": False,
        "env": {"CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1", "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
                "PATH": f"{py}:/usr/bin:/bin:/usr/sbin:/sbin"},
        "permissions": {
            "disableBypassPermissionsMode": "disable",
            "deny": ["Read(//Users/teo/Developer/**)", "Read(//Users/teo/.claude/**)", "Read(//Users/teo/Library/**)",
                     "Read(//Users/teo/Documents/**)", "Read(//Users/teo/Desktop/**)", "Read(//Users/teo/Downloads/**)",
                     "Read(//Users/teo/ws-subject/sessions/**)",
                     *others, "WebFetch", "WebSearch"],
            "allow": [f"Read(//{str(session_dir).lstrip('/')}/**)", f"Edit(//{str(session_dir).lstrip('/')}/**)",
                      f"Write(//{str(session_dir).lstrip('/')}/**)", "Bash", "Glob", "Grep"],
        },
        "sandbox": {
            "enabled": True, "failIfUnavailable": True, "allowUnsandboxedCommands": False, "autoAllowBashIfSandboxed": True,
            "network": {"allowedDomains": [], "allowLocalBinding": True},
            "filesystem": {"denyRead": ["~/Developer", "~/.claude", "~/Library", "~/Documents", "~/Desktop", "~/Downloads",
                                        str(SUBJECT / "sessions"), str(Path.home() / "ws-subject" / "sessions")],
                           "allowRead": [str(session_dir), str(PYENV)]},
        },
    }


def main():
    cfg = json.load(open(sys.argv[1]))
    sid = cfg["id"]
    sdir = SUBJECT / "sessions" / sid
    vdir = ROOT / "vault" / "sessions" / sid
    if sdir.exists() or vdir.exists():
        sys.exit(f"session {sid} already exists")
    (sdir / "task").mkdir(parents=True); (sdir / "work" / "submission").mkdir(parents=True); vdir.mkdir(parents=True)
    port = free_port()
    for name, src in cfg.get("files", {}).items():
        dst = sdir / "task" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if Path(src).is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy(src, dst)
    (sdir / "task" / "svc.py").write_text((ROOT / "tf" / "client.py").read_text().replace("@PORT@", str(port)))
    shutil.copy(ROOT / "tf" / "policy_np.py", sdir / "task" / "policy_np.py")
    card = (Path(cfg["task_card"]).read_text().replace("@SESSION_DIR@", str(sdir))
            .replace("@ROLLOUTS@", str(cfg["rollout_budget"])).replace("@TRAINS@", str(cfg["train_budget"])))
    (sdir / "task" / "TASK.md").write_text(card)
    for p in (sdir / "task").rglob("*"):
        if p.is_file():
            p.chmod(0o444)
    task_hashes = {str(p.relative_to(sdir)): sha(p) for p in (sdir / "task").rglob("*") if p.is_file()}
    set_path = vdir / "settings.json"
    json.dump(settings(sdir), open(set_path, "w"), indent=1)
    svc_cfg = dict(port=port, rollout_budget=cfg["rollout_budget"], train_budget=cfg["train_budget"],
                   dataset=cfg["dataset"], policy=cfg["policy"], log=str(vdir / "service.jsonl"),
                   session_index=cfg["session_index"])
    json.dump(svc_cfg, open(vdir / "service.json", "w"), indent=1)
    svc = subprocess.Popen([str(ROOT / ".venv/bin/python"), "-m", "tf.service", str(vdir / "service.json")], cwd=ROOT,
                           env={**os.environ, "PYTHONPATH": str(ROOT)}, stdout=subprocess.DEVNULL, stderr=open(vdir / "service.err", "w"))
    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.5).close(); break
        except OSError:
            time.sleep(0.3)
    else:
        svc.kill(); sys.exit("rollout service did not start; see service.err")
    version = subprocess.run(["claude", "--version"], capture_output=True, text=True).stdout.strip()
    prompt = f"Your task is described in {sdir}/task/TASK.md. Read it and complete it. Work only inside {sdir}."
    cmd = ["claude", "-p", prompt, "--restricted", "--settings", str(set_path), "--permission-mode", "dontAsk",
           "--model", MODEL, "--effort", EFFORT, "--tools", "Bash,Read,Write,Edit,Glob,Grep",
           "--allowedTools", "Bash,Read,Write,Edit,Glob,Grep", "--disallowedTools", "WebFetch", "WebSearch", "Agent", "mcp__*",
           "--strict-mcp-config", "--output-format", "stream-json", "--verbose", "--no-session-persistence"]
    t0 = time.time()
    turns, stop, seen = 0, None, set()   # a turn = one assistant API message (several stream events share its id)
    with open(vdir / "transcript.jsonl", "w") as tr:
        proc = subprocess.Popen(cmd, cwd=sdir, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=open(vdir / "claude.err", "w"),
                                text=True, start_new_session=True, env={**os.environ, "ENABLE_CLAUDEAI_MCP_SERVERS": "false"})
        for line in proc.stdout:
            tr.write(line); tr.flush()
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if ev.get("type") == "assistant":
                mid = ev.get("message", {}).get("id")
                if mid not in seen:
                    seen.add(mid); turns += 1
            if turns >= cfg.get("max_turns", 150):
                stop = "max_turns"
            if time.time() - t0 > cfg.get("max_hours", 3) * 3600:
                stop = "max_hours"
            if stop:
                os.killpg(proc.pid, signal.SIGTERM); break
        proc.wait()
    svc.send_signal(signal.SIGTERM); svc.wait()
    # copy the subject's folder into the vault record (the session folder itself is left in place)
    shutil.copytree(sdir / "work", vdir / "work")
    shutil.move(str(sdir), str(vdir / "folder"))   # finished folders leave ws-subject, so later sessions cannot see them
    fdir = vdir / "folder"
    tampered = {k: v for k, v in task_hashes.items() if not (fdir / k).exists() or sha(fdir / k) != v}
    man = dict(id=sid, model=MODEL, effort=EFFORT, claude_version=version, started=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
               minutes=round((time.time() - t0) / 60, 1), turns=turns, stopped_by=stop, exit_code=proc.returncode,
               port=port, task_card_sha256=sha(cfg["task_card"]), task_files_sha256=task_hashes, task_files_changed=tampered,
               submission=sorted(str(p.relative_to(vdir / "work")) for p in (vdir / "work").rglob("*") if p.is_file()),
               cmd=cmd)
    json.dump(man, open(vdir / "manifest.json", "w"), indent=1)
    print(json.dumps({k: man[k] for k in ("id", "minutes", "turns", "stopped_by", "exit_code", "task_files_changed", "submission")}))


if __name__ == "__main__":
    main()
