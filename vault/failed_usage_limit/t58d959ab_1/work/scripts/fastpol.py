import sys, math, numpy as np
sys.path.insert(0, '/Users/teo/tf-subject/sessions/t58d959ab/task')
import policy_np as P

def erf(x):
    s = np.sign(x); x = np.abs(x)
    t = 1/(1+0.3275911*x)
    y = 1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t*np.exp(-x*x)
    return s*y

P.gelu = lambda x: 0.5*x*(1+erf(x/math.sqrt(2)))
load, forward, Runner = P.load, P.forward, P.Runner
