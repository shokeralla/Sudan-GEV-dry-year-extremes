import pandas as pd, numpy as np, glob, re, sys
from scipy import stats
names={(15.60,32.53):'Khartoum',(13.63,25.35):'El Fasher',(12.05,24.88):'Nyala',(11.01,29.72):'Kadugli',
(13.18,30.22):'El Obeid',(14.40,33.52):'Wad Madani',(15.45,36.40):'Kassala',(13.55,33.63):'Sennar',
(11.79,34.36):'Ad Damazin',(14.03,35.38):'Gedaref'}
def idx(path):
    rows=[]
    for line in open(path):
        p=line.split()
        if len(p)==13 and re.fullmatch(r'\d{4}',p[0]):
            rows+= [(int(p[0]),m+1,float(v)) for m,v in enumerate(p[1:])]
    d=pd.DataFrame(rows,columns=['year','month','v']); d.loc[d.v<-90,'v']=np.nan
    return d[d.month.between(7,9)].groupby('year').v.mean()
# excel single column -> dump text
nino=idx('data/indices/nino34_monthly_psl.txt'); dmi=idx('data/indices/dmi_monthly_psl.txt')
out=[];summ=[]
for f in sorted(glob.glob('data/power/*.csv')):
    m=re.search(r'_(\d+)d(\d+)N_(\d+)d(\d+)E',f); lat=float(f'{m[1]}.{m[2]}'); lon=float(f'{m[3]}.{m[4]}')
    site=names[(lat,lon)]
    d=pd.read_csv(f,skiprows=9); d.columns=['YEAR','DOY','P']
    assert len(d)==16071, len(d)
    nmiss=(d.P<0).sum(); d.loc[d.P<0,'P']=np.nan
    for y,g in d.groupby('YEAR'):
        i=g.P.idxmax(); ndays=len(g)
        doy=int(g.loc[i,'DOY']); th=2*np.pi*(doy-0.5)/ndays
        out.append(dict(site=site,lat=lat,lon=lon,year=y,annual_total_mm=round(g.P.sum(),2),
            max_mm=g.loc[i,'P'],doy=doy,date=(pd.Timestamp(y,1,1)+pd.Timedelta(days=doy-1)).date(),
            theta_rad=round(th,4),wet_days_ge1mm=int((g.P>=1).sum()),nino34_JAS=nino.get(y),dmi_JAS=dmi.get(y)))
    s=pd.DataFrame([o for o in out if o['site']==site])
    th=s.theta_rad.values; C,S=np.cos(th).mean(),np.sin(th).mean(); R=np.hypot(C,S)
    mdir=np.arctan2(S,C)%(2*np.pi); n=len(th)
    Z=n*R**2; pR=np.exp(np.sqrt(1+4*n+4*(n**2-(n*R)**2))-(1+2*n))  # Rayleigh (Zar)
    # circular-linear corr timing vs year (Mardia)
    t=s.year.values; rxc=stats.pearsonr(t,np.cos(th))[0]; rxs=stats.pearsonr(t,np.sin(th))[0]; rcs=stats.pearsonr(np.cos(th),np.sin(th))[0]
    R2=(rxc**2+rxs**2-2*rxc*rxs*rcs)/(1-rcs**2); pCL=stats.chi2.sf(n*R2,2)
    tau,pMK=stats.kendalltau(t,s.max_mm)
    # timing-magnitude dependence (circular-linear)
    x=s.max_mm.values; a=stats.pearsonr(x,np.cos(th))[0]; b=stats.pearsonr(x,np.sin(th))[0]
    R2m=(a**2+b**2-2*a*b*rcs)/(1-rcs**2); pTM=stats.chi2.sf(n*R2m,2)
    # early vs late shift of mean DOY: compare 1981-2002 vs 2003-2024
    def md(v): return (np.degrees(np.arctan2(np.sin(v).mean(),np.cos(v).mean()))%360)/360*365.25
    summ.append(dict(site=site,lat=lat,lon=lon,missing_days=int(nmiss),
        years_zero_rain=int((s.annual_total_mm==0).sum()),min_annual_max_mm=round(s.max_mm.min(),2),
        median_annual_max_mm=round(s.max_mm.median(),2),max_annual_max_mm=round(s.max_mm.max(),2),
        mean_annual_total_mm=round(s.annual_total_mm.mean(),1),
        circ_mean_DOY=round(mdir/(2*np.pi)*365.25,1),mean_resultant_R=round(R,3),
        rayleigh_p=float(f'{pR:.2g}'),meanDOY_1981_2002=round(md(th[t<=2002]),1),meanDOY_2003_2024=round(md(th[t>2002]),1),
        timing_vs_year_R2=round(R2,3),timing_trend_p=round(pCL,3),
        MK_tau_magnitude=round(tau,3),MK_p_magnitude=round(pMK,3),
        timing_magnitude_R2=round(R2m,3),timing_magnitude_p=round(pTM,3)))
pairs=pd.DataFrame(out); S=pd.DataFrame(summ).sort_values('lat')
pairs.to_csv('out/pairs_power.csv',index=False); S.to_csv('out/summary_power.csv',index=False)
pd.set_option('display.width',250); print(S.drop(columns=['lon']).to_string(index=False))
print(nino.loc[1981:2024].isna().sum(), dmi.loc[1981:2024].isna().sum(), nino.loc[2020:2024].round(2).tolist(), dmi.loc[2020:2024].round(2).tolist())
