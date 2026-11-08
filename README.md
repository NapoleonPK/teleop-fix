# Teleop-Fix: can an AI agent fix a robot policy trained on flawed teleoperation data?

A behaviour-cloning policy trained on teleoperated demonstrations often fails because of *how the demonstrations were
recorded*: an operator who hesitates, a recording pipeline that misaligns commands and observations, an operator who
never covers part of the workspace. The usual remedy is to record more demonstrations. This study asks whether an AI
agent (Claude Code, `claude-opus-5-5`) given the trained policy, its dataset, 20 test rollouts and 6 training jobs can
find the cause and fix it, matching or beating what new demonstrations would buy, at zero operator time.

Everything runs in simulation on one CPU machine. The study was **pre-registered** before any test session
(`prereg/teleopfix.md`, git tag `prereg`, SHA-256 `1d6a1266236f8438db399f6e60def2d3b8da400da7eebfbb81e697b0daf6684e`); deviations are listed at the end of that file.
Every number below is produced by one run of `scripts/analysis.py` on the frozen result files.

Data, weights and full agent transcripts: **https://huggingface.co/datasets/PAOP123/teleop-fix**

## Setup

- **Task:** Fetch pick-and-place (MuJoCo `FetchPickAndPlace-v4`), 100-step episodes, success = block within 5 cm of
  the goal at the last step.
- **Demonstrations:** a scripted expert seen through a simulated operator (`tf/operator.py`), 500 demos per dataset.
  Failure types:
  - *hesitant*: in about half the demos the operator pauses 15-30 steps above the block, with hand tremor;
  - *desync*: the recording pairs each observation with the command of 3 steps earlier (leader/follower stream
    misalignment);
  - *comfort*: the operator never demonstrates with the block on the far side of the table;
  - *clean*: no flaw (controls).
- **Policy:** ACT-lite (`tf/policy.py`), a 0.6M-parameter transformer with action chunking (10) and temporal
  ensembling.
- **Units:** 12 test units (3 per failure type, plus 3 clean controls), each with its own dataset and policy seed.
- **Agent arm:** one Claude Code session per unit with the task card `prereg/cards/fix.md`: the policy weights, the
  dataset, 3 failed episodes, a rollout service (20 rollouts) and a training service (6 jobs: subset selection,
  reweighting, relabelling, up to 500 synthetic demos, init from scratch/given/submitted weights). The agent may also
  edit weights directly. Sandboxed, no network; every session audited (`scripts/audit.py`).
- **Baselines:** `demos_K` (K = 10, 25, 50 new demos from the same operator, targeted at the far side for comfort
  units, then fine-tuning), a frozen heuristic cleaning `filter`, training `longer`, and random-subset `search`.
- **Measures:** success on 200 held-out seeds never used before the freeze; blind diagnosis scoring by a fresh model
  with no tools (`scripts/diagnosis.py`); fix kind from the training-service log (`scripts/classify.py`).

## Results

- H1 (agent >= 25 targeted demos - 5 pts, >= 2 of 3 types): passed (hesitant, desync, comfort)
- H2 (agent >= heuristic filter + 10 pts, >= 2 of 3 types): passed (desync, comfort)
- H3 (correct diagnosis on >= 6 of 9): passed (9 of 9)
- H4 (clean controls lose <= 5 pts): passed (mean change +0.0 pts)

### Mean gain over the broken policy, whole table, points

| Failure type | demos_10 | demos_25 | demos_50 | filter | longer | search | agent |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hesitant | +12.8 | +7.7 | +5.0 | +68.0 | -7.3 | +23.8 | +68.2 |
| desync | +1.0 | +1.8 | -0.8 | +0.2 | +1.0 | -10.3 | +50.7 |
| comfort | +12.3 | +24.0 | +27.3 | -11.5 | +1.0 | -5.2 | +31.2 |

Agent worth in demos (interpolated on the K curve): hesitant >50, desync >50, comfort >50

### Per unit (whole / near / far success)

| Unit | Type | Broken | Agent | Kind | Diagnosis |
| --- | --- | --- | --- | --- | --- |
| u1 | hesitant | 0.385 / 0.345 / 0.555 | 1.000 / 1.000 / 1.000 | synthetic | 1 |
| u2 | hesitant | 0.065 / 0.055 / 0.150 | 1.000 / 1.000 / 1.000 | synthetic | 1 |
| u3 | hesitant | 0.505 / 0.470 / 0.415 | 1.000 / 1.000 / 1.000 | synthetic | 1 |
| u4 | desync | 0.480 / 0.435 / 0.475 | 1.000 / 1.000 / 1.000 | synthetic | 1 |
| u5 | desync | 0.615 / 0.525 / 0.645 | 1.000 / 1.000 / 1.000 | data_fix | 1 |
| u6 | desync | 0.385 / 0.315 / 0.270 | 1.000 / 1.000 / 1.000 | synthetic | 1 |
| u7 | comfort | 0.665 / 1.000 / 0.000 | 1.000 / 1.000 / 1.000 | synthetic | 1 |
| u8 | comfort | 0.680 / 1.000 / 0.025 | 1.000 / 1.000 / 1.000 | synthetic | 1 |
| u9 | comfort | 0.720 / 1.000 / 0.010 | 1.000 / 1.000 / 1.000 | synthetic | 1 |
| u10 | clean | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | synthetic | 0 |
| u11 | clean | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | synthetic | 0.5 |
| u12 | clean | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | synthetic | 0 |

Notes:
- The gain columns are average success-rate changes over the broken policy, in percentage points.
- "Agent worth in demos" interpolates the agent's gain on the 0/10/25/50-demo curve; ">50" means above the largest
  tested batch.
- *Kind*: `synthetic` = trained with agent-generated demonstrations; `data_fix` = relabelled data only.
- *Diagnosis*: 1 = true cause, 0.5 = symptom only, 0 = wrong. On the clean controls, 1 means the agent concluded
  nothing was wrong.

## Observations

- More demonstrations from the *same* operator barely help when the flaw is a habit or a pipeline bug: the new demos
  carry the same flaw. The heuristic filter fixes the hesitant operator, but not the lag or the coverage gap.
- Every agent fixed the problem through the data (cutting pauses, shifting labels by the measured lag, mirroring
  demonstrations to the uncovered side, generating recovery demos) and retrained, rather than editing weights by hand.
- On the healthy controls the agents still reported a problem and retrained. Success did not drop, but a tool built on
  this needs an explicit "leave it alone" check.

## Limits

The flaws are clean, single and injected by design into a scripted operator, on one simulated task with a small policy.
This shows the approach works when the cause is present and detectable. It does not show how it behaves on real
teleoperation data with several mixed, partial flaws, larger visuomotor policies, or real hardware.

## Repository

- `tf/`: simulator wrapper, operator model, policy (torch and numpy), services, seeds
- `scripts/`: unit building, baselines, session runner, audit, diagnosis scoring, classification, evaluation,
  analysis, supervisor
- `prereg/`: frozen pre-registration, task card, deviation hashes
- `docs/PROTOCOL.md`: design draft; `DECISIONS.md`: dated judgement calls; `PLAN.md`: running log
- `results/`: analysis output (`teleopfix.md`, `teleopfix.json`)
- `vault/*.json`: sessions, held-out evaluation, diagnosis scores and reasons, fix kinds

## Reproduce

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
export PYTHONPATH=$PWD
.venv/bin/python scripts/build_units.py vault/units.json      # datasets + broken policies
.venv/bin/python scripts/baselines.py 1 2 3 4 5 6 7 8 9 10 11 12
.venv/bin/python scripts/make_sessions.py test 1 2 3 4 5 6 7 8 9 10 11 12
.venv/bin/python scripts/supervisor.py                          # agent sessions, scoring, held-out eval
.venv/bin/python scripts/analysis.py vault/eval_heldout.json results/
```

Agent sessions need Claude Code (`claude`) logged in. To skip rebuilding, download the units and sessions from
Hugging Face into `vault/`.

## License

Code under MIT (see `LICENSE`). Data and weights on Hugging Face under CC BY 4.0.
