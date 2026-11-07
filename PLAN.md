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
