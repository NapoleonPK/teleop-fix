#!/bin/zsh
# usage: ro.sh W NAME ODX ODY GDX GDY
python ../task/svc.py rollout $1 --object $3 $4 --goal $5 $6 --out ro/$2.json > ro/$2.log 2>&1
python -c "
import json,numpy as np,sys
d=json.load(open('ro/$2.json')); 
o=np.array(d['observations']) if 'observations' in d else None
print('$2', {k:v for k,v in d.items() if not isinstance(v,list)}, 'obj0',o[0,3:5].round(3),'goal',o[0,25:27].round(3),'objend',o[-1,3:6].round(3))
"
