# Teleop-Fix decisions (judgement calls, dated)

1. **Simulated teleop = scripted expert + operator profile (2026-11-07).** Recorded pair is (follower observation,
   operator command). The laggy operator at full speed fails its own demos (lag 3: 5% demo success; lag 2: 20%), which
   no real operator would keep doing; real operators slow down under latency, so laggy = lag 3 with xyz gain 0.5
   (97.5% demo success on 40 dev demos).
2. **Policy = ACT-lite on state (2026-11-07).** Closest cheap stand-in for the action-chunking policies used on real
   leader-follower arms (ACT); no images and no CVAE, to stay trainable on a CPU in minutes.
3. **Operator lag does not break the policy; replaced by stream desync (2026-11-07).** Calibration (unit 900, 500
   demos, DEV_EVAL): clean 98.5% whole; hesitant 21% (far 9.5%); comfort-zone 52.5% (near 99.5%, far 1%); early-grip
   79% (exploratory only: below the 30-point loss target); laggy (lag 3, gain 0.5) 99%: a slow, late operator
   produces slow but consistent demos and the policy copes. The latency failure seen in leader-follower recording is a
   pipeline one: the leader (command) and follower (observation) streams are misaligned, so each stored observation is
   paired with an older command. Profile "desync" models that (the demo executes correctly, the stored pairs are
   shifted by k steps). The first laggy calibration run used gain 1.0 by mistake (0.6% demo success); discarded.
4. **Agent may add constructed demos (2026-11-07).** Without it the comfort-zone unit cannot be fixed by any data
   action (no far-side demo exists), while an engineer would try augmentation (e.g. mirroring near-side demos). The
   training service accepts up to 500 extra demos (X, A) per job; using them is reported as its own fix kind
   ("synthetic data"), like rewrites in the weight-space study.
5. **Strengths frozen for the three main profiles (2026-11-07, unit 900, DEV_EVAL).** hesitant (p 0.5, 15-30 steps):
   21% vs clean 98.5%; desync k = 2: 33.5% (k = 3: 23.5%; k = 2 chosen as the milder one inside the 30-80 point
   window); comfort-zone (near only): 52.5% whole, far 1%. Units: dev u51 desync, u52 comfort; test u1-3 hesitant,
   u4-6 desync, u7-9 comfort, u10-12 clean.
6. **Heuristic filter rule set on dev demos (2026-11-07).** A per-step stillness rule (< 2 mm/step) missed every
   hesitation because of the hand tremor; the frozen rule uses net gripper displacement under 1 cm over an 8-step
   window (fingers open, block > 5 cm from goal). On 60 dev demos per profile (seeds 1.5M+): hesitant 23/23 caught,
   0 false flags on clean, desync and comfort; early-grip retries 21/21 caught. This deliberately makes the filter a
   strong day-one baseline (an engineer would look for pauses and retries first).
7. **Agent services (2026-11-07).** 20 rollouts and 6 training jobs per session; a job fine-tunes (<= 2000 steps,
   lr 1e-4) or retrains (<= 8000 steps, the original recipe); the service logs a hash of every returned policy so the
   submission's origin (training job vs hand edit) is classified mechanically (scripts/classify.py).
8. **Desync padding gave the artifact away (2026-11-07 14:05, from dev session t58d959ab).** The first desync build
   filled the k missing commands at the start of each demo with copies of the first command, so every demo began with
   k identical actions; the dev agent used that as its first clue (it then confirmed the exact 2-step lag from
   commands vs motion). Real misaligned streams start with the leader at rest, so the padding is now zero motion with
   the gripper open. Units u4-6 and u51 rebuilt (old ones in vault/rebuilt_desync_padding/); their baseline arms are
   recomputed. Before any test session.
9. **Dev sessions hit the usage limit (2026-11-07 10:35).** Both dev sessions stopped after 11.5 min with "You've hit
   your 5-hour limit"; folders in vault/failed_usage_limit/; rerun with the same ids after the 14:00 reset (u51's on
   the rebuilt unit). Before the limit: t58d959ab found the 2-step lag and fixed it by relabelling (10/10 random
   starts); t897f9566 trained on the demos plus mirrored copies and fixed the far side it tested.
10. **Desync strength re-frozen at k = 3 (2026-11-07 14:55).** With the realistic zero-motion padding (decision 8),
    k = 2 lost only 40-41 points on calibration units 900/901 (57.5%, 59%) and 25 on dev unit u51 (74.5%), close to
    the 30-point floor; k = 3 lost 63 (35%, 36%). Test units u4-6 rebuilt with k = 3 (the k = 2 builds are in
    vault/rebuilt_desync_k2/). Dev unit u51 and its session stay at k = 2 (dev only, same mechanism).
