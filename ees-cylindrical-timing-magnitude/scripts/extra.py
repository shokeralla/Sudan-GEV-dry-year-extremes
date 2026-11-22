import pandas as pd, numpy as np
from scipy import stats
C = pd.read_csv('out/pairs_chirps.csv'); H = pd.read_csv('out/homogeneity.csv')
rows=[]
for s,g in C.groupby('site'):
    th=g.theta_rad.values; n=len(th); Cc,Ss=np.cos(th).mean(),np.sin(th).mean(); R=np.hypot(Cc,Ss)
    m=np.arctan2(Ss,Cc)%(2*np.pi); pR=np.exp(np.sqrt(1+4*n+4*(n**2-(n*R)**2))-(1+2*n))
    rows.append(dict(site=s,lat=g.lat.iloc[0],lon=g.lon.iloc[0],mean_total=g.annual_total_mm.mean(),median_max=g.max_mm.median(),
      iqr_lo=g.max_mm.quantile(.25),iqr_hi=g.max_mm.quantile(.75),max_max=g.max_mm.max(),
      mean_doy=m/(2*np.pi)*365.25,R=R,csd_days=np.sqrt(-2*np.log(R))/(2*np.pi)*365.25,rayleigh_p=pR,
      earliest=g.doy.min(),latest=g.doy.max(),ties=(g.ties>1).sum()))
S=pd.DataFrame(rows).sort_values('lat',ascending=False); S.to_csv('out/chirps_summary.csv',index=False)
pd.set_option('display.width',250); print(S.round(3).to_string(index=False))
# date conversion helper
for d in S.mean_doy: print(round(d,1),(pd.Timestamp(2001,1,1)+pd.Timedelta(days=d-1)).strftime('%d %b'))
# correlation between products of annual max and timing difference, by period
P=pd.read_csv('out/pairs_power.csv'); m=C.merge(P[['site','year','max_mm','doy']],on=['site','year'],suffixes=('_ch','_pw'))
for per,mm in (('81-14',m[m.year<=2014]),('15-24',m[m.year>=2015])):
    print(per,'spearman max', stats.spearmanr(mm.max_mm_ch,mm.max_mm_pw)[0].round(2),'ratio pw/ch',(mm.max_mm_pw/mm.max_mm_ch).median().round(2))
dd=((m.doy_ch-m.doy_pw+182)%365)-182; print('same-pentad share',(dd.abs()<=2).mean().round(3),'within 10d',(dd.abs()<=10).mean().round(3))
print(m.groupby('site').apply(lambda g: stats.spearmanr(g.max_mm_ch,g.max_mm_pw)[0]).round(2).to_string())
