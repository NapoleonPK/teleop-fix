# Teleop-Fix decisions (judgement calls, dated)

1. **Simulated teleop = scripted expert + operator profile (2026-11-07).** Recorded pair is (follower observation,
   operator command). The laggy operator at full speed fails its own demos (lag 3: 5% demo success; lag 2: 20%), which
   no real operator would keep doing; real operators slow down under latency, so laggy = lag 3 with xyz gain 0.5
   (97.5% demo success on 40 dev demos).
2. **Policy = ACT-lite on state (2026-11-07).** Closest cheap stand-in for the action-chunking policies used on real
   leader-follower arms (ACT); no images and no CVAE, to stay trainable on a CPU in minutes.
