# Teleop-Fix plan (status file; newest at the bottom)

Started 2026-11-07 ~10:00 at Téo's request; weight-space Exp 1/2 paused for it (see ~/Developer/weightspace/PLAN.md).
Design: docs/PROTOCOL.md (draft). Mac mini only, 24/7 allowed. Long-lived jobs go under launchd, not the Bash tool.

- [x] repo, venv (torch 2.15, gymnasium-robotics 1.4.2, mujoco 3.3.7), sim copied (tf/sim.py)
- [x] operator model tf/operator.py: clean, hesitant, laggy (lag 3, gain 0.5), comfort (near only), early_grip
- [x] ACT-lite tf/policy.py (0.6M params, chunk 10, temporal ensembling); ~57 ms/step on CPU
- [~] calibration: clean policy must be strong (>= 90% whole) before artifacts can be measured.
      200 demos / 3000 steps -> 36% (mostly pushes, rarely grasps). Trying 500 demos / 8000 steps, no dropout.
- [ ] calibrate artifact strengths (each broken policy -30..-80 points vs clean on DEV_EVAL)
- [ ] baselines: more demos K, heuristic filter, train longer, random data search
- [ ] agent harness: rollout + training services, task card, sandbox (adapt weightspace run_session.py)
- [ ] 2 dev agent sessions; blind diagnosis scorer
- [ ] prereg freeze -> 12 test sessions + baselines (launchd supervisor) -> analysis -> write-up
