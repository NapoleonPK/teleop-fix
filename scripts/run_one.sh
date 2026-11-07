#!/bin/bash
# usage: run_one.sh SID  - one agent session, then audit (host side)
cd ~/Developer/teleopfix
PYTHONPATH=. .venv/bin/python -I scripts/run_session.py runs/$1.json > runs/$1.out 2>&1
PYTHONPATH=. .venv/bin/python scripts/audit.py $1 >> runs/$1.out 2>&1
echo "$1 done $(date '+%F %T')"
