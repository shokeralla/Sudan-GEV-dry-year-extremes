import pandas as pd, numpy as np, glob
from scipy import stats
import pymannkendall as mk
P0 = pd.read_csv('out/pairs_power.csv')  # NASA POWER pairs with indices
lat = P0.groupby('site')[['lat','lon']].first()
out = []
for f in sorted(glob.glob('chirps/*.csv')):
    d = pd.read_csv(f); site = [s for s in lat.index if s.replace(' ','') == f.split('/')[-1][:-4]][0]
    d['date'] = pd.to_datetime(d[['year','month','day']]); d['doy'] = d.date.dt.dayofyear
    assert d.prcp.isna().sum() == 0 and (d.prcp >= 0).all()
    for y, g in d.groupby('year'):
        i = g.prcp.idxmax(); n = len(g); doy = int(g.loc[i, 'doy'])
        ties = int((g.prcp == g.prcp.max()).sum())
        out.append(dict(site=site, lat=lat.loc[site,'lat'], lon=lat.loc[site,'lon'], year=y, annual_total_mm=round(g.prcp.sum(),2),
                        max_mm=round(g.loc[i,'prcp'],3), doy=doy, date=g.loc[i,'date'].date(), theta_rad=round(2*np.pi*(doy-0.5)/n, 5),
                        wet_days_ge1mm=int((g.prcp >= 1).sum()), ties=ties))
C = pd.DataFrame(out).merge(P0[['site','year','nino34_JAS','dmi_JAS']], on=['site','year'])
C.to_csv('out/pairs_chirps.csv', index=False)
def pettitt(x):
    n = len(x); U = [sum(np.sign(x[j]-x[i]) for i in range(k+1) for j in range(k+1, n)) for k in range(n-1)]
    K = np.max(np.abs(U)); k = int(np.argmax(np.abs(U))); p = 2*np.exp(-6*K**2/(n**3+n**2)); return k, min(p, 1)
def snht(x):
    z = (x - x.mean())/x.std(ddof=1); n = len(x)
    T = [k*z[:k].mean()**2 + (n-k)*z[k:].mean()**2 for k in range(1, n)]
    return int(np.argmax(T))+1, max(T)
rows = []
for src, D in (('POWER', P0), ('CHIRPS', C)):
    for s, g in D.groupby('site'):
        g = g.sort_values('year'); x = g.max_mm.values; t = g.annual_total_mm.values; yrs = g.year.values
        k, p = pettitt(x); k2, p2 = pettitt(t); ks, Ts = snht(x)
        rows.append(dict(source=src, site=s, median_max_81_14=np.median(x[yrs <= 2014]), median_max_15_24=np.median(x[yrs >= 2015]),
                         ratio=np.median(x[yrs >= 2015])/np.median(x[yrs <= 2014]),
                         pettitt_year_max=yrs[k+1], pettitt_p_max=p, snht_year_max=yrs[ks], snht_T_max=Ts,
                         pettitt_year_total=yrs[k2+1], pettitt_p_total=p2, MK_p_max=mk.original_test(x).p, MK_tau=mk.original_test(x).Tau))
H = pd.DataFrame(rows); H.to_csv('out/homogeneity.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.float_format', lambda v: f'{v:.3f}')
print(H.to_string(index=False))
m = C.merge(P0[['site','year','max_mm','doy']], on=['site','year'], suffixes=('_ch','_pw'))
dd = ((m.doy_ch - m.doy_pw + 182) % 365) - 182
print('timing agreement: |diff|<=5d', (dd.abs() <= 5).mean().round(3), 'median abs diff', dd.abs().median())
print('ties >1:', (C.ties > 1).sum())
print(C.groupby('site').max_mm.describe().round(1))
