import sys, numpy as np, torch
from multiprocessing import get_context
from tf import policy, sim
class First:
    def __init__(s,m): s.m=m
    @torch.no_grad()
    def __call__(s,x): return np.clip(s.m(torch.as_tensor(x,dtype=torch.float32)[None])[0,0].numpy(),-1,1)
class Open:
    def __init__(s,m,k): s.m,s.k,s.q=m,k,[]
    @torch.no_grad()
    def __call__(s,x):
        if not s.q: s.q=list(s.m(torch.as_tensor(x,dtype=torch.float32)[None])[0,:s.k].numpy())
        return np.clip(s.q.pop(0),-1,1)
def ep(a):
    path,seed,mode=a; torch.set_num_threads(1); m=policy.load(path)
    pol={"first":lambda:First(m),"ens0.1":lambda:policy.Runner(m,0.1),"ens0.01":lambda:policy.Runner(m,0.01),"open5":lambda:Open(m,5)}[mode]()
    ok,X,A,xe=sim.rollout(pol,sim.Spec(seed),record=True)
    return ok, float(np.linalg.norm(xe[3:6]-xe[25:28])), float((X[:,9:11].sum(1)<0.03).any()), float(X[:,5].max()-X[0,5])
if __name__=="__main__":
    path=sys.argv[1]
    with get_context("spawn").Pool(8) as p:
        for mode in ["first","ens0.1","ens0.01","open5"]:
            r=np.array(p.map(ep,[(path,s,mode) for s in range(9_000_000,9_000_050)]))
            print(mode,"succ",r[:,0].mean(),"final dist",np.median(r[:,1]).round(3),"lifted>3cm",(r[:,3]>0.03).mean(),flush=True)
