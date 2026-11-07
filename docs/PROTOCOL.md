# Teleop-Fix: can an agent fix a teleop-trained policy more cheaply than more demonstrations? (protocol draft)

Status: DRAFT, 2026-11-07. Becomes `prereg/teleopfix.md` (committed, SHA-256 recorded) after calibration and the dev
sessions, before any test session. Everything runs on one Mac mini (CPU) plus Claude Code sessions.

## Question

A policy trained on teleoperated demonstrations fails because of how the operator demonstrated (pauses, latency,
gripper habits, where they demonstrated). Can an AI agent with the policy, the dataset and a small budget of test
runs find the cause and fix it, matching what new demonstrations would buy, at zero operator time?

## World (fully simulated)

- Robot and task: Fetch pick-and-place in MuJoCo (FetchPickAndPlace-v4), 100-step episodes, success = block within
  5 cm of the goal at the end. Same simulator as the weight-space experiments (tf/sim.py).
- Teleoperation: a scripted expert driven through an operator profile (tf/operator.py). Recorded pair = (follower
  observation, operator command), as in leader-follower recording. Cost of a demo: 0.5 operator-minutes (10 s
  episode + 20 s reset).
- Operator profiles (artifacts): clean; hesitant (half the demos pause 15-30 steps above the block, with tremor);
  laggy (reacts to 3-step-old observations, slows to half speed); comfort-zone (never demonstrates on the far side of
  the table); early-grip (40% of demos close too high, miss, reopen and retry; exploratory). Final strengths are
  calibrated on dev seeds so each broken policy loses 30-80 points against clean, then frozen.
- Policy: ACT-lite (tf/policy.py), a small action-chunking transformer on the state (8 field tokens, chunk 10,
  3 layers, d 128, ~0.6M parameters), L1 loss, temporal ensembling. 200 demos per dataset; ~3 min training on CPU.

## Units

Per failure type (hesitant, laggy, comfort): 3 policy seeds = 9 broken policies. Plus 3 clean-operator policies as
controls (an agent that "fixes" a healthy policy must not make it worse). 12 units.

## Arms (same budget of 20 test rollouts per unit wherever rollouts are used)

| Arm | What it gets and does |
|---|---|
| Agent (Claude Code, opus, high effort, 3 h, 150 turns) | The weights, the 200-demo dataset (no artifact labels), the training recipe, 3 recorded failed episodes, a rollout service (20 runs) and a training service (budget below). May filter, reweight or relabel demos, change the data mix, launch fine-tunes or retrains, or edit weights directly. Must submit weights + a short diagnosis. |
| More demos K = 10, 25, 50 | The same operator records K new demos, targeted at the failing region when the failure has one; fine-tune on old + new. Cost K x 0.5 operator-minutes. |
| Heuristic filter | Drop demos with a pause over 8 steps, a gripper reopen after closing, or a failed outcome; retrain. Frozen rule, no rollouts. |
| Train longer | Continue training 2x on the same data. |
| Random data search | 4 random 80% subsets, retrain each, keep the best by 5 rollouts each (20 runs). |

Training-service budget for the agent: at most 6 training jobs, each at most the recipe's steps (no more compute than
the random-search arm's 4 retrains + 2 fine-tunes).

## Measures

- Primary: success on 200 held-out evaluation seeds (whole table), never visible to any arm.
- Collateral: success on the regions where the broken policy already worked; for the clean controls, change in success.
- Cost: operator-minutes (demo arms), test rollouts, training jobs.
- Diagnosis: a blind scorer (fresh `claude -p`, no tools) compares the agent's diagnosis with the true profile.
- Edit kind: data-level fix, fine-tune, direct weight edit, or full retrain from scratch; reported separately.

## Hypotheses (thresholds frozen at pre-registration)

- H1: the agent's mean gain matches or beats K = 25 targeted demos (gain within 5 points or above) on at least 2 of
  3 failure types, with zero operator time.
- H2: the agent beats the heuristic filter by at least 10 points on at least 2 of 3 failure types.
- H3: the diagnosis names the true artifact (score 1) on at least 6 of 9 broken units.
- H4 (safety): on the 3 clean controls the agent loses at most 5 points on average.
- Exploratory: early-grip profile; how many demos the agent is "worth" (interpolated on the K curve).

## Rules

Pre-registration before any test session; held-out evaluation seeds and test units never seen in development;
every flagged session audited by hand; a harness crash rerun once; no hand-typed numbers in the report; deviations
logged with date and reason in DECISIONS.md.
