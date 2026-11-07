# Teleop-Fix pre-registration

Frozen before any test session. Design and reasons: docs/PROTOCOL.md (draft of 2026-11-07 morning) and
DECISIONS.md 1-10. Changes after the commit go in Deviations with date and reason.

## Question

A behaviour-cloning policy trained on teleoperated demonstrations fails because of how the demonstrations were
recorded. Can an AI agent with the policy, the dataset, 20 test rollouts and 6 training jobs find the cause and fix
it, matching what new demonstrations would buy, at zero operator time?

## World

- Fetch pick-and-place (MuJoCo, FetchPickAndPlace-v4), 100-step episodes, success = block within 5 cm of the goal
  at the last step (tf/sim.py).
- Demonstrations: scripted expert through an operator profile (tf/operator.py), 500 per dataset; a demo costs 0.5
  operator-minutes. Profiles: hesitant (half the demos pause 15-30 steps above the block, with tremor); desync (the
  recording pairs each observation with the command of 3 steps earlier, first 3 commands at rest); comfort (no demo
  with the block on the far side, x > 0.05 m); clean.
- Policy: ACT-lite (tf/policy.py), 0.6M parameters, chunk 10, temporal ensembling; recipe 8000 steps, batch 256,
  lr 3e-4, L1 (tf/service.py RECIPE).

## Units (vault/units.json; built by scripts/build_units.py)

Test: u1-3 hesitant, u4-6 desync, u7-9 comfort, u10-12 clean (one dataset seed range and policy seed per unit).
Dev (never reported as results): u51 desync (k = 2, built before decision 10), u52 comfort.

## Arms

| Arm | Rule |
|---|---|
| agent | One Claude Code session per test unit (claude-opus-5-5, effort high, card prereg/cards/fix.md, 150 turns, 3 h, 20 rollouts, 6 training jobs; sandbox as scripts/run_session.py). Submission work/submission/policy.npz; missing or invalid = the broken policy. |
| demos_K, K = 10, 25, 50 | The same operator records K new demos (far region for comfort units, else whole table); fine-tune the broken policy on old + new (2000 steps, lr 1e-4). |
| filter | Drop demos flagged by the frozen rule (pause: 8-step net gripper displacement < 1 cm with fingers open and the block > 5 cm from goal; re-open after closing while the block is far; failed demo), retrain with the recipe. |
| longer | Another full recipe run from the broken policy on the same data. |
| search | 4 random 80% subsets retrained, best of 5 rollouts each kept. |

## Measures

- Success on 200 held-out seeds (tf/seeds.py HELDOUT, 9_500_000-9_500_199) for the whole table, and the near and
  far regions (scripts/evaluate_all.py), evaluated once after all sessions.
- Gain = arm success minus broken success (whole table), averaged per failure type over its 3 units.
- Diagnosis: blind scorer (scripts/diagnosis.py; fresh claude -p, no tools) scores section (1) of report.md against
  the true profile: 1 = true cause, 0.5 = symptom only, 0 = wrong.
- Fix kind: scripts/classify.py (unchanged, hand_edit, data_fix, synthetic, retrain_same) from the service log.

## Hypotheses (scripts/analysis.py)

- **H1 (primary).** On at least 2 of the 3 failure types, the agent's mean gain is at least demos_25's mean gain
  minus 5 points.
- **H2.** On at least 2 of the 3 failure types, the agent's mean gain is at least the filter's plus 10 points.
- **H3.** The diagnosis scores 1 on at least 6 of the 9 broken units.
- **H4 (safety).** On the 3 clean controls, the agent's mean change is at least -5 points.
- Exploratory: the agent's worth in demonstrations (interpolated on the K = 0, 10, 25, 50 curve); near-side change
  on comfort units; fix kinds; rollouts and jobs used; per-unit table.

## What was seen before the freeze (disclosed)

- Both dev sessions found the true cause within 16 minutes and reached 100% on 200 dev seeds (desync: relabelled the
  2-step lag and retrained; comfort: mirrored the demos as extra data, then a weight edit plus fine-tune). Blind
  diagnosis 1 and 1.
- Dev-seed (DEV_EVAL, not held-out) success of the test units' broken policies (hesitant 43.5, 13.5, 56.5%; desync
  k = 3 47.5, 64.5, 30%; comfort 57.5, 57.5, 64.5%; clean 100%) and of the non-agent arms on u1-3 and u52: the filter fixes hesitant units (99.5-100%)
  and does nothing for comfort; extra demos from the hesitant operator do not help (they carry the same habit);
  25 targeted demos fix the comfort unit (98.5%). The hypotheses and thresholds were written in the morning draft
  before any of this and are unchanged.

## Predictions (Claude, after the dev sessions)

H1 passes 80% (demos from the same operator cannot remove a habit or a pipeline bug; comfort is a near-tie with 25
targeted demos). H2 passes 75% (the filter cannot touch desync or comfort; it already fixes hesitant, so H2 can only
pass on the other two). H3 passes 80%. H4 passes 75% (the risk is an agent retraining a healthy policy and losing a
few points).

## Rules

The test sessions run one at a time in a fixed random order (scripts/supervisor.py, seed 11). A session that hits
the account's usage limit is moved to vault/failed_usage_limit/ and rerun (the quota is external, as in the
weight-space study); a harness crash is rerun once. Audit (scripts/audit.py) flags paths outside the session,
network calls and task-file changes; every flagged session is read by hand and the verdict written to
vault/audit_verdicts.json before the analysis; a confirmed violation scores as the broken policy and diagnosis 0.
The analysis runs once on the frozen result files; every reported number comes from it.

## Frozen files (SHA-256)

- `prereg/cards/fix.md` 97a55b743fb9c0055acbf668f5aa379e11e26d526f7f7cc5eac31ab83c95a74f
- `tf/sim.py` 1f20e0942b1a46dc3943230135e2a11dd1633cfb41dc91b489c79dae5800b1e2
- `tf/operator.py` 90df07eef21ede83f4fce0e54aa9531184b97a6e496cbaee459fbf626e9f8cfd
- `tf/policy.py` 006665a449fb999760d8cc60c01abb48ca29f05ac92ede52869e1dbce8e37708
- `tf/policy_np.py` 64c24a8c10e0487cf7e41be72fef97f81f9fcba49dd53cd18ad7e8e7251d5a59
- `tf/service.py` de09aa56a7bfa3b560bba581bb38f9132000c027dad8f67af9fdcd4c1f98b66d
- `tf/client.py` 46a496017629f2aa04fd8bde3f7d82b0b2bf9ab6d6bca74b9329d231812b8a4d
- `tf/seeds.py` 104926065027893f1ab29da009d84ce0b1d0d0290ef90820d2a04f1addedb023
- `tf/evaluate.py` bb1746ece98e2f1bf8d977e8dc06205672a19c50b8fa589160bbd4c660499603
- `scripts/run_session.py` 9593caac424ef028ea8466fa308810366952125f6ba86e5e5d0d75c361fdbd3a
- `scripts/run_one.sh` e46fe5d697e0f2774345f248951d413610c78e24028c37af0694a17cd0cbfb8e
- `scripts/supervisor.py` ded1588fbe770048de932b279d077ce55d27caeb459267c5e5c2a153eb9e1203
- `scripts/build_units.py` 2126d7c263530658bee35da6d563214a772728a77da29c64616f5ea5d18a5847
- `scripts/baselines.py` 49bcecf41f2f34a3825a5e33c450a52736abd39e913047c9bb6577a5f1edcd79
- `scripts/make_sessions.py` df182c8af03552c2faf37f13db5dfe97a08087b9b55112c8525401e6daba82f9
- `scripts/evaluate_all.py` 01e6d44b9f2c70a2a6c7b13af67690e00f4ecc3302b06b772a22314dac6c1fd2
- `scripts/diagnosis.py` 6442f809f2a03fcd47d44115e7510ebff0f3187859685e668bb30e0c27569f54
- `scripts/classify.py` c55ac60ffc9653d8ce40e5014b2d2d8cae8ea59b3a129627917190f0895c72fa
- `scripts/audit.py` 1d2992481e6cb35a5d8fce45cae5d60bd16f3c1971f20b61634c68467877d481
- `scripts/analysis.py` a93a131325442d1caee1534106025261250a35ec8edd355f092592d313ceb383
- `vault/units.json` b30a3330e720018d979e7f46181cfabc364f29fbf30e8ef4d116bec0678201ae
- `vault/units/u1/dataset.npz` 5bfd71df673dd1ee6492c108840aa732350c4b2cf706d80e48782c939b91424b
- `vault/units/u1/policy.npz` 3fa789bd8d4750acc28477977bbf0c7e101d78557c67aa3b6a15beaf27aa1aba
- `vault/units/u10/dataset.npz` e5668c67f34881246e21a19b1b341c3a7f989d374284f6779ce4e6bb62dc2903
- `vault/units/u10/policy.npz` 130caea20e33efaa7b9e91f4101beab376c48b9b775123aa543b5b8a30ed7b68
- `vault/units/u11/dataset.npz` 3c55d20bd80c8c05b0e3bf452aa955afddcbedd693776cfcae586b368d566e68
- `vault/units/u11/policy.npz` c0fdd566f8f8a7badeb0ae859948f6d35bb07f44f57e31f1f4498abc0b19f69d
- `vault/units/u12/dataset.npz` b13cfee44b373a0ef1c77f3746c37e40e2dd67d8032bc0043a0dc763c1871f82
- `vault/units/u12/policy.npz` b977779b875d95834f5ea484024def8e8942ea1e0a022d82a3ba9daa1b1714bf
- `vault/units/u2/dataset.npz` 548ebb45fe18a004a123df613d8124e4afa25bcdffc6452ecaa550203d70f2e4
- `vault/units/u2/policy.npz` cba797643eb9bc11e05cbe4b260140ee83f5dd174b0501bc970a66dd52267112
- `vault/units/u3/dataset.npz` 8994d612738cc7618a8b705b7ccfc48a268c60b5eb8fda98d63f03709b141f3c
- `vault/units/u3/policy.npz` e291c5ba12e482039923df4696d2c78f3d6f567e9d1043247f4f82aa820284a0
- `vault/units/u4/dataset.npz` 23ecb27fe19dd166fccbdf71cc25e46a6ffcb13cb9bec1b02b6b52004ae6779c
- `vault/units/u4/policy.npz` d8ca6d3493e846a609f4e23d39dbba7afb7e5d483aa3feb36fe6a24a217e21d6
- `vault/units/u5/dataset.npz` 608c0f7dd1b0c2870abb0c947281aa96f68860707ac47babce203428ad53f03d
- `vault/units/u5/policy.npz` 7edea8f38611c9bedbfe68378f9a2e1323fe56ca87a18e077450de805dc2c114
- `vault/units/u6/dataset.npz` 14d1d57c908f090e149292f061d5086dd1671e87b6cdaaa526403dc6825caf8c
- `vault/units/u6/policy.npz` bb25235025c9a3680ffb12273f2eef3b50f662cedddd958685a88fb39065c0dd
- `vault/units/u7/dataset.npz` 76e4b863914fed5e19e31169d23317a669e0cd9107e740e53acc89ab71231a2f
- `vault/units/u7/policy.npz` 4578fb226a74a2f4ff8bb79fdeb3a672eec660861b6c723b320bbf91f18904b5
- `vault/units/u8/dataset.npz` 0f7ba36ba40a2fed8be1bf2f8ba15be2d059fa801a12374f243bd106b8b00d5b
- `vault/units/u8/policy.npz` 3785a4252068deda0e99b4f0afdb1fb339e41cfb51090cc186071cece3d6cc0c
- `vault/units/u9/dataset.npz` 03a65129f6e58ae4fcf5ab157e29d5586134e24e80030cd73cb4e6f8515bd887
- `vault/units/u9/policy.npz` 66b691df8962712bcf6000d79e032084384daf85435e7ff4ddb7031ecd5027ef

## Deviations

(none yet)
