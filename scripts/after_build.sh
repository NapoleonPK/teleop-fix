#!/bin/bash
# wait for the unit build to finish, then run every non-agent arm for all built units
cd ~/Developer/teleopfix
while pgrep -f "scripts/build_units.py" > /dev/null; do sleep 60; done
PYTHONPATH=. .venv/bin/python scripts/baselines.py 51 52 1 2 3 4 5 6 7 8 9 10 11 12
