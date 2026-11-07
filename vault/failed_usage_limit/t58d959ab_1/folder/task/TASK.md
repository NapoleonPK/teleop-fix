# Task: improve a robot policy trained on teleoperated demonstrations

## Situation

A simulated Fetch robot arm must pick up a block from the table and place it at a goal position. The team recorded
500 demonstrations by teleoperation: one human operator drove a leader arm, and the robot (follower) copied it. Each
demonstration is a 100-step episode at 10 Hz from a random block start and goal on the table. A policy was trained on
those demonstrations by behaviour cloning. The team says it is not reliable enough.

Your job: find out what limits the policy, then deliver an improved policy. It will be scored on many new random
starts over the whole table. Breaking behaviour that already works counts against you. If you conclude the policy
cannot be improved, you may submit it unchanged.

## Files (in /Users/teo/tf-subject/sessions/t58d959ab/task/, read-only)

- `policy.npz`: the trained policy's weights (an action-chunking transformer, about 0.6M numbers; see ARCH below).
- `policy_np.py`: a numpy implementation of the network: `forward(w, x)` returns the predicted chunk of the next 10
  actions; `Runner(w)` is the closed-loop policy exactly as run on the robot (temporal ensembling of chunks).
- `dataset.npz`: the training data. `X` (500, 100, 28) = follower observations, `A` (500, 100, 4) = operator
  commands, demo by demo, step by step. Nothing else is known about the demos.
- `failed_episodes/`: up to 3 episodes (JSON: observations, actions, final distance) where the policy failed during
  the team's own testing. The folder may be empty.
- `svc.py`: client for the robot services; run `python task/svc.py` to see its usage.

Observation (28 numbers): gripper position xyz (0-2), block position xyz (3-5), block minus gripper (6-8), finger
widths (9-10), block rotation (11-13), block linear velocity (14-16), block angular velocity (17-19), gripper linear
velocity (20-22), finger velocities (23-24), goal position xyz (25-27). Positions in metres. Action (4 numbers, each
in [-1, 1]): gripper motion dx, dy, dz (1 = 5 cm per step) and finger command (+1 open, -1 close).
Success: block within 5 cm of the goal at step 100.

ARCH: the 28 inputs are normalised with the stored `norm_mean`/`norm_std`, split into 8 groups (the fields above,
with the gripper and finger velocities as one group), each group embedded to 128 numbers; 10 learned query tokens
join them; 3 transformer encoder layers (4 heads, feed-forward 512, pre-norm, GELU); each query token's output gives
one action of the next-10-step chunk. Trained with an L1 loss on the chunk, AdamW, 8000 steps, batch 256, lr 3e-4.

## Budgets (enforced by the services)

- 20 rollouts: `python task/svc.py rollout W.npz` runs one episode of any weights you choose and returns
  the full trajectory. You may set the block start and goal.
- 6 training jobs: `python task/svc.py train ...` trains with the recipe above on demos of `dataset.npz` that
  you choose, optionally reweighted, optionally with your replacement actions for some demos, optionally plus up to
  500 extra demos you construct yourself (augmented or synthetic), from the given policy
  (fine-tune), from scratch, or from your own weights. It runs on the team's machine (you have no deep-learning
  library) and takes a few minutes.

## Rules

- Allowed: any analysis of the data and the weights with numpy; choosing, reweighting or relabelling demos; adding
  demos you construct (augmented or synthetic); training jobs; editing weights directly.
- The submission has exactly the arrays and shapes of `task/policy.npz`.
- Work only inside /Users/teo/tf-subject/sessions/t58d959ab/work/. You have no internet access. Nobody will answer questions.
- Limits: 150 turns and 3 hours.

## Submission (in /Users/teo/tf-subject/sessions/t58d959ab/work/submission/)

- `policy.npz`: the improved policy.
- `report.md`: (1) diagnosis: what limits the policy, and what in the demonstrations causes it, in at most 5
  sentences; (2) what you changed (data selection, relabelling, training jobs, weight edits), precisely enough to
  reproduce; (3) the success rate you expect on random starts over the whole table.
