# Teleop-Fix plan (status file; newest at the bottom)

Started 2026-11-07 ~10:00 at Téo's request; weight-space Exp 1/2 paused for it (see ~/Developer/weightspace/PLAN.md).
Design: docs/PROTOCOL.md (draft). Mac mini only, 24/7 allowed. Long-lived jobs go under launchd, not the Bash tool.

- [x] repo, venv (torch 2.15, gymnasium-robotics 1.4.2, mujoco 3.3.7), sim copied (tf/sim.py)
- [x] operator model tf/operator.py: clean, hesitant, laggy (lag 3, gain 0.5), comfort (near only), early_grip
- [x] ACT-lite tf/policy.py (0.6M params, chunk 10, temporal ensembling); ~57 ms/step on CPU
- [x] recipe: 500 demos, 8000 steps, no dropout -> clean 98.5-100% (200 demos/3000 steps gave 36%)
- [x] calibration (decisions 3, 5): hesitant 21%, desync k=2 33.5%, comfort 52.5% (far 1%), clean 98.5%
- [~] units building (launchd com.teleopfix.build, runs/build.log): dev u51 desync, u52 comfort DONE; test u1-12
- [~] baselines queued after the build (com.teleopfix.baselines, runs/baselines.log): demos_10/25/50, filter,
      longer, search -> vault/units/u*/arms/
- [x] harness: tf/service.py (rollout + training), tf/client.py, scripts/run_session.py (~/tf-subject), card
      prereg/cards/fix.md, audit.py, classify.py, diagnosis.py, evaluate_all.py
- [~] 2 dev sessions running (t58d959ab u51 desync, t897f9566 u52 comfort; launchd com.teleopfix.dev_*)
- [ ] prereg freeze -> 12 test sessions + baselines (launchd supervisor) -> analysis -> write-up

Launch helper: scripts/bg.sh LABEL LOG CMD... (one-off launchd job, plist removes itself when done).

**2026-11-07 14:10.** Dev sessions hit the usage limit at 10:35 (both had found the cause: 2-step lag relabel; mirror
augmentation). Desync padding tell fixed (decision 8): u4-6, u51 rebuilding (com.teleopfix.rebuild: u51 -> dev
t58d959ab rerun -> remaining units -> all baselines). Dev t897f9566 rerunning. Test supervisor written
(scripts/supervisor.py, not installed yet: install after the prereg freeze).

**2026-11-07 15:15. Prereg FROZEN** (prereg/teleopfix.md, commit 16748d6, tag `prereg`, sha in runs/prereg_sha.txt).
12 test sessions created (vault/sessions.json). Supervisor installed as launchd KeepAlive agent
**com.teleopfix.supervisor** (~/Library/LaunchAgents; log runs/supervisor.log): one session at a time, seed-11 order,
usage-limit / login-fault reruns, then diagnosis -> classify -> evaluate_all heldout -> runs/ALL_DONE.
First launch failed ("Not logged in": env had no USER) -> deviations 1-2 (commit 5c40299; also desync truth text
k=3). Restarted 15:08; t069263b0 (u3) running normally. Baselines (com.teleopfix.rebuild2) done for u51/52/1-3,
u4-12 continuing; finish() waits for them.
- [~] 12 test sessions (supervisor)
- [~] baselines u4-12
- [ ] hand-audit flagged sessions -> vault/audit_verdicts.json (flat {id: "clean" | "violation: ..."})
- [ ] after runs/ALL_DONE: `.venv/bin/python scripts/analysis.py vault/eval_heldout.json results/` once; write-up; email Téo
Stop: `launchctl bootout gui/501/com.teleopfix.supervisor`.

**2026-11-08 00:01. STOPPED at Téo's request** ("stop using any claude code token on this mac until i give u my
approval"). Supervisor booted out before its 00:05 relaunch; plist moved to scripts/com.teleopfix.supervisor.plist.paused.
Heartbeat cron cancelled. 9 of 12 test sessions done (no audit flags); left: u12 (t46339627, cut by the usage limit,
in vault/failed_usage_limit/) and 2 more. Baselines complete for all units. Resume (only with Téo's approval):
`cp scripts/com.teleopfix.supervisor.plist.paused ~/Library/LaunchAgents/com.teleopfix.supervisor.plist &&
launchctl bootstrap gui/501 ~/Library/LaunchAgents/com.teleopfix.supervisor.plist`.

**2026-11-08 11:34. RESUMED with Téo's approval**; supervisor relaunched (u12 rerun, then u8, u9). Interim, exploratory
dev-seed (not held-out) eval of finished agent submissions: runs/interim_eval.py -> runs/eval_dev_interim.json
(9/9 agents at 99.5-100%; baseline arms for u4-12 still evaluating). Not a prereg result.

**2026-11-08 16:35. TEST PHASE COMPLETE.** ALL_DONE 12:35 (12/12 sessions, 0 audit flags, diagnosis + held-out eval
done). scripts/analysis.py run once -> results/teleopfix.{md,json}: H1, H2, H3 (9/9), H4 (+0.0) all passed.
Next: write-up for Téo (nothing public without review). Possible follow-up: HF pretrained policy + weights-only tuning
(proposed 2026-11-08, waiting for Téo).
