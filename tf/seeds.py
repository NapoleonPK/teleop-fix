"""Seed ranges for Teleop-Fix. Held-out evaluation seeds are never used in development (the smoke tests of
2026-11-07 used 9_000_000+, so held-out starts at 9_500_000)."""
DEV_EVAL = range(8_000_000, 8_000_200)        # development / calibration evaluation
HELDOUT = range(9_500_000, 9_500_200)         # final evaluation only, after the pre-registration


def demos(unit: int, n: int):                 # original dataset of a unit
    return range(1_000_000 + 10_000 * unit, 1_000_000 + 10_000 * unit + n)


def extra_demos(unit: int, n: int):           # "more demos" arm
    return range(2_000_000 + 10_000 * unit, 2_000_000 + 10_000 * unit + n)


def agent_rollout(session_index: int, k: int):
    return 5_000_000 + 1_000 * session_index + k


def search_rollout(unit: int, k: int):        # random data-search arm
    return 6_000_000 + 1_000 * unit + k
