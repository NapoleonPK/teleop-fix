# Teleop-Fix results

- H1 (agent >= 25 targeted demos - 5 pts, >= 2 of 3 types): passed (hesitant, desync, comfort)
- H2 (agent >= heuristic filter + 10 pts, >= 2 of 3 types): passed (desync, comfort)
- H3 (correct diagnosis on >= 6 of 9): passed (9 of 9)
- H4 (clean controls lose <= 5 pts): passed (mean change +0.0 pts)

## Mean gain over the broken policy, whole table, points

| Failure type | demos_10 | demos_25 | demos_50 | filter | longer | search | agent |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hesitant | +12.8 | +7.7 | +5.0 | +68.0 | -7.3 | +23.8 | +68.2 |
| desync | +1.0 | +1.8 | -0.8 | +0.2 | +1.0 | -10.3 | +50.7 |
| comfort | +12.3 | +24.0 | +27.3 | -11.5 | +1.0 | -5.2 | +31.2 |

Agent worth in demos (interpolated on the K curve): hesitant >50, desync >50, comfort >50

## Per unit (whole / near / far success)

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
