import numpy as np, pandas as pd
from scipy import stats, special, optimize
rng=np.random.default_rng(7)
C=pd.read_csv('out/pairs_chirps.csv')
def mle(th):
    m=np.arctan2(np.sin(th).mean(),np.cos(th).mean()); R=np.hypot(np.cos(th).mean(),np.sin(th).mean())
    f=lambda k: special.i1e(k)/special.i0e(k)-R; k=optimize.brentq(f,1e-6,500); return m,k
def U2(th,m,k):
    u=np.sort(stats.vonmises.cdf(th,k,loc=m)%1); n=len(u); i=np.arange(1,n+1)
    return np.sum((u-(2*i-1)/(2*n))**2)-n*(u.mean()-.5)**2+1/(12*n)
out=[]
for s,g in C.groupby('site'):
    th=g.theta_rad.values; m,k=mle(th); u=U2(th,m,k); bs=[]
    for b in range(1000):
        x=stats.vonmises.rvs(k,loc=m,size=len(th),random_state=rng); mm,kk=mle(x); bs.append(U2(x,mm,kk))
    out.append((s,round(u,3),round(np.mean(np.array(bs)>=u),3),round(np.percentile(bs,95),3)))
print(pd.DataFrame(out,columns=['site','U2','boot_p','crit95']).to_string(index=False))
