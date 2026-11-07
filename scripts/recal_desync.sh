#!/bin/bash
cd ~/Developer/teleopfix
PYTHONPATH=. .venv/bin/python scripts/calibrate.py runs/calib_desync2.json "desync:desync=2,desync:desync=3" 900
PYTHONPATH=. .venv/bin/python scripts/calibrate.py runs/calib_desync2.json "desync:desync=2,desync:desync=3" 901
