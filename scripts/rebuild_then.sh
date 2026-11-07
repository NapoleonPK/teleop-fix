#!/bin/bash
# rebuild missing units (u51 first), start the u51 dev session, then all baselines
cd ~/Developer/teleopfix
echo '[{"unit":51,"profile":"desync"}]' > runs/units_51.json
PYTHONPATH=. .venv/bin/python scripts/build_units.py runs/units_51.json
scripts/bg.sh dev_t58d959ab $PWD/runs/dev_t58d959ab.log /bin/bash scripts/run_one.sh t58d959ab
PYTHONPATH=. .venv/bin/python scripts/build_units.py vault/units.json
PYTHONPATH=. .venv/bin/python scripts/baselines.py 51 52 1 2 3 4 5 6 7 8 9 10 11 12
