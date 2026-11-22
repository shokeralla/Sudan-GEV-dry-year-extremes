"""Non-stationary cylindrical extreme value models for annual maximum rainfall timing and magnitude.
Joint density: f(theta, x | t) = vM(theta; mu_theta(t), kappa) * GEV(x; mu(t, theta), sigma, xi)
"""
import numpy as np, pandas as pd, json, warnings
from scipy import optimize, stats, special
import pymannkendall as mk
warnings.filterwarnings('ignore')
rng = np.random.default_rng(20261121)
import sys
SRC = sys.argv[1]; PFX = sys.argv[2]; B = int(sys.argv[3]) if len(sys.argv)>3 else 200
P = pd.read_csv(SRC)
SITES = P.groupby('site').lat.first().sort_values().index.tolist()

def gev_nll(x, mu, sig, xi):
    if np.any(sig <= 0): return 1e10
    z = (x - mu) / sig
    if abs(xi) < 1e-6:
        return np.sum(np.log(sig) + z + np.exp(-z))
    w = 1 + xi * z
    if np.any(w <= 0): return 1e10
    return np.sum(np.log(sig) + (1 + 1/xi) * np.log(w) + w ** (-1/xi))

def vm_nll(th, m, k):
    if k <= 0: return 1e10
    return np.sum(-k*np.cos(th - m) + np.log(2*np.pi*special.i0e(k)) + k)

def circmean(th): return np.arctan2(np.sin(th).mean(), np.cos(th).mean())

# ---- model definitions -------------------------------------------------
# s = (year-2002.5)/10 (decades); lat = sin(theta-thbar) (lateness), off = 1-cos(theta-thbar)
def design(df, thbar):
    s = (df.year.values - 2002.5) / 10
    d = np.angle(np.exp(1j*(df.theta_rad.values - thbar)))
    return s, np.sin(d), 1 - np.cos(d)

MODELS = {
 'M0': dict(trend=False, dep=False, ttrend=False),
 'M1': dict(trend=True,  dep=False, ttrend=False),
 'M2': dict(trend=True,  dep=True,  ttrend=False),
 'M3': dict(trend=True,  dep=True,  ttrend=True),
 'MD': dict(trend=False, dep=True,  ttrend=False),  # dependence only (appendix)
}

def unpack(p, spec):
    i = 0; out = {}
    out['mu0'] = p[i]; i += 1
    out['mu1'] = p[i] if spec['trend'] else 0.0; i += spec['trend']
    out['b1'], out['b2'] = (p[i], p[i+1]) if spec['dep'] else (0.0, 0.0); i += 2*spec['dep']
    out['lsig'] = p[i]; out['xi'] = p[i+1]; i += 2
    out['m0'] = p[i]; out['lk'] = p[i+1]; i += 2
    out['g'] = p[i] if spec['ttrend'] else 0.0
    return out

def nll(p, spec, x, th, s, la, of):
    q = unpack(p, spec)
    mu = q['mu0'] + q['mu1']*s + q['b1']*la + q['b2']*of
    mth = q['m0'] + 2*np.arctan(q['g']*s)          # tan-half link for circular mean trend
    return gev_nll(x, mu, np.exp(q['lsig']), q['xi']) + vm_nll(th, mth, np.exp(q['lk']))

def npar(spec): return 5 + spec['trend'] + 2*spec['dep'] + spec['ttrend']

def fit(df, name, thbar=None, start=None, fast=False):
    spec = MODELS[name]
    x = df.max_mm.values; th = df.theta_rad.values
    thbar = circmean(th) if thbar is None else thbar
    s, la, of = design(df, thbar)
    R = np.hypot(np.cos(th).mean(), np.sin(th).mean())
    k0 = R*(2 - R**2)/(1 - R**2)
    g = stats.genextreme.fit(x); xi0 = -g[0]
    base = [g[1]] + [0.0]*spec['trend'] + [0.0, 0.0]*spec['dep'] + [np.log(g[2]), max(min(xi0, .4), -.2), thbar, np.log(k0)] + [0.0]*spec['ttrend']
    best = None
    for st in ([start] if start is not None else []) + ([] if fast else [base]):
        for meth in (['Nelder-Mead'] if fast else ['Nelder-Mead', 'Powell']):
            r = optimize.minimize(nll, st, args=(spec, x, th, s, la, of), method=meth,
                                  options=dict(maxiter=40000, maxfev=40000, xatol=1e-7, fatol=1e-9) if meth=='Nelder-Mead' else dict(maxiter=40000))
            r = optimize.minimize(nll, r.x, args=(spec, x, th, s, la, of), method='Nelder-Mead', options=dict(maxiter=40000, maxfev=40000, xatol=1e-8, fatol=1e-10))
            if best is None or r.fun < best.fun: best = r
    p = best.x
    # numerical Hessian -> SEs
    if fast:
        return dict(p=best.x, ll=-best.fun, q=unpack(best.x, spec), thbar=thbar, spec=spec)
    H = hessian(lambda q: nll(q, spec, x, th, s, la, of), p)
    try:
        cov = np.linalg.inv(H); se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        cov = None; se = np.full_like(p, np.nan)
    k = npar(spec); n = len(x); ll = -best.fun
    return dict(p=p, se=se, cov=cov, ll=ll, k=k, aic=2*k - 2*ll, aicc=2*k - 2*ll + 2*k*(k+1)/(n-k-1),
                bic=k*np.log(n) - 2*ll, thbar=thbar, spec=spec, q=unpack(p, spec), name=name)

def hessian(f, p, h=1e-4):
    k = len(p); H = np.zeros((k, k))
    for i in range(k):
        for j in range(i, k):
            ei = np.zeros(k); ej = np.zeros(k); ei[i] = h*max(1, abs(p[i])); ej[j] = h*max(1, abs(p[j]))
            H[i, j] = H[j, i] = (f(p+ei+ej) - f(p+ei-ej) - f(p-ei+ej) + f(p-ei-ej)) / (4*ei[i]*ej[j])
    return H

def se_of(fr, key):
    names = ['mu0'] + (['mu1'] if fr['spec']['trend'] else []) + (['b1','b2'] if fr['spec']['dep'] else []) + ['lsig','xi','m0','lk'] + (['g'] if fr['spec']['ttrend'] else [])
    return fr['se'][names.index(key)] if key in names else np.nan

def gev_q(T, mu, sig, xi):
    y = -np.log(1 - 1/T)
    return mu + sig/xi*(y**(-xi) - 1) if abs(xi) > 1e-6 else mu - sig*np.log(y)

def simulate(fr, df):
    q = fr['q']; s, _, _ = design(df, fr['thbar'])
    mth = q['m0'] + 2*np.arctan(q['g']*s)
    th = (stats.vonmises.rvs(np.exp(q['lk']), loc=mth, random_state=rng)) % (2*np.pi)
    d = np.angle(np.exp(1j*(th - fr['thbar'])))
    mu = q['mu0'] + q['mu1']*s + q['b1']*np.sin(d) + q['b2']*(1-np.cos(d))
    x = stats.genextreme.rvs(-q['xi'], loc=mu, scale=np.exp(q['lsig']), random_state=rng)
    out = df.copy(); out['theta_rad'] = th; out['max_mm'] = x
    return out

# ---- run ---------------------------------------------------------------
res = {}; rows = []; fits = {}
for site in SITES:
    df = P[P.site == site].sort_values('year').reset_index(drop=True)
    f = {m: fit(df, m) for m in MODELS}
    fits[site] = f
    lr = lambda a, b: 2*(f[b]['ll'] - f[a]['ll'])
    best = min(['M0','M1','M2','M3'], key=lambda m: f[m]['aicc'])
    q1 = f['M1']['q']; q2 = f['M2']['q']; q3 = f['M3']['q']
    rows.append(dict(site=site,
        AICc_M0=f['M0']['aicc'], AICc_M1=f['M1']['aicc'], AICc_M2=f['M2']['aicc'], AICc_M3=f['M3']['aicc'], best=best,
        LR_M1vM0=lr('M0','M1'), p_M1vM0=stats.chi2.sf(lr('M0','M1'), 1),
        LR_M2vM1=lr('M1','M2'), p_M2vM1=stats.chi2.sf(lr('M1','M2'), 2),
        LR_M3vM2=lr('M2','M3'), p_M3vM2=stats.chi2.sf(lr('M2','M3'), 1),
        mu0=q1['mu0'], mu1=q1['mu1'], se_mu1=se_of(f['M1'],'mu1'), sigma=np.exp(q1['lsig']), xi=q1['xi'], se_xi=se_of(f['M1'],'xi'),
        kappa=np.exp(q1['lk']), thbar_doy=(f['M1']['thbar'] % (2*np.pi))/(2*np.pi)*365.25,
        b1=q2['b1'], se_b1=se_of(f['M2'],'b1'), b2=q2['b2'], se_b2=se_of(f['M2'],'b2'),
        g=q3['g'], se_g=se_of(f['M3'],'g'),
        shift_days_per_decade=2*np.arctan(q3['g'])/(2*np.pi)*365.25))
T = pd.DataFrame(rows)

# return levels (M1): effective 50-yr RL in 1985 and 2024 + parametric bootstrap
rl_rows = []
for site in SITES:
    df = P[P.site == site].sort_values('year').reset_index(drop=True)
    fr = fits[site]['M1']; q = fr['q']
    def rls(qq):
        out = {}
        for yr in (1985, 2024):
            s = (yr - 2002.5)/10
            for Tn in (10, 50, 100):
                out[(yr, Tn)] = gev_q(Tn, qq['mu0'] + qq['mu1']*s, np.exp(qq['lsig']), qq['xi'])
        return out
    pt = rls(q); boot = []
    for b in range(B):
        sim = simulate(fr, df)
        try:
            boot.append(rls(fit(sim, 'M1', thbar=fr['thbar'], start=fr['p'], fast=True)['q']))
        except Exception: pass
    for Tn in (10, 50, 100):
        a = np.array([[bb[(1985, Tn)], bb[(2024, Tn)]] for bb in boot])
        d = a[:, 1] - a[:, 0]
        rl_rows.append(dict(site=site, T=Tn, RL1985=pt[(1985, Tn)], RL2024=pt[(2024, Tn)],
            change=pt[(2024, Tn)] - pt[(1985, Tn)], change_pct=100*(pt[(2024, Tn)]/pt[(1985, Tn)] - 1),
            RL1985_lo=np.percentile(a[:,0], 2.5), RL1985_hi=np.percentile(a[:,0], 97.5),
            RL2024_lo=np.percentile(a[:,1], 2.5), RL2024_hi=np.percentile(a[:,1], 97.5),
            change_lo=np.percentile(d, 2.5), change_hi=np.percentile(d, 97.5)))
RL = pd.DataFrame(rl_rows)

# conditional 50-yr RL by timing (M2), at thbar -/+ 1 circular SD, year 2024
cond = []
for site in SITES:
    fr = fits[site]['M2']; q = fr['q']; th = P[P.site == site].theta_rad.values
    R = np.hypot(np.cos(th).mean(), np.sin(th).mean()); csd = np.sqrt(-2*np.log(R))
    s = (2024 - 2002.5)/10
    v = {}
    for lab, d in (('early', -csd), ('peak', 0.0), ('late', csd)):
        v[lab] = gev_q(50, q['mu0'] + q['mu1']*s + q['b1']*np.sin(d) + q['b2']*(1-np.cos(d)), np.exp(q['lsig']), q['xi'])
    cond.append(dict(site=site, circ_sd_days=csd/(2*np.pi)*365.25, RL50_early=v['early'], RL50_peak=v['peak'], RL50_late=v['late']))
C = pd.DataFrame(cond)

# regional pooling: (a) DerSimonian-Laird random effects of relative trend mu1/mu0 (%/decade)
def dl(est, se):
    w = 1/se**2; fe = np.sum(w*est)/w.sum(); Q = np.sum(w*(est-fe)**2); k = len(est)
    tau2 = max(0, (Q-(k-1))/(w.sum() - (w**2).sum()/w.sum()))
    ws = 1/(se**2 + tau2); m = np.sum(ws*est)/ws.sum(); s = np.sqrt(1/ws.sum())
    return dict(mean=m, se=s, lo=m-1.96*s, hi=m+1.96*s, tau2=tau2, Q=Q, pQ=stats.chi2.sf(Q, k-1), I2=max(0, (Q-(k-1))/Q)*100 if Q > 0 else 0)
rel = 100*T.mu1/T.mu0; rel_se = 100*T.se_mu1/T.mu0
REG = {'trend_pct_per_decade_DL': dl(rel.values, rel_se.values),
       'b1_DL': dl(T.b1.values, T.se_b1.values), 'b2_DL': dl(T.b2.values, T.se_b2.values),
       'g_shift_DL': dl(T.g.values, T.se_g.values)}
# (b) year-block bootstrap of the regional mean relative trend (keeps cross-site dependence)
years = np.arange(1981, 2025); W = P.pivot(index='year', columns='site', values='max_mm')
def regtrend(Wm):
    vals = []
    for c in Wm.columns:
        y = Wm[c].values; s = (Wm.index.values - 2002.5)/10
        sl = np.polyfit(s, y, 1)[0]; vals.append(100*sl/np.mean(y))
    return np.mean(vals)
obs = regtrend(W)
# block bootstrap of residual years under null? -> use moving-block bootstrap of detrended + CI of slope via paired-year resampling
bs = []
for b in range(2000):
    idx = rng.choice(len(years), len(years), replace=True)
    Wb = W.iloc[idx]; bs.append(regtrend(Wb))
REG['trend_pct_per_decade_yearboot'] = dict(mean=obs, lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5))
# permutation test (permute years jointly across sites) for regional trend
perm = []
for b in range(5000):
    Wp = W.copy(); Wp.index = rng.permutation(W.index.values); Wp = Wp.sort_index(); perm.append(regtrend(Wp))
REG['trend_perm_p'] = float(np.mean(np.abs(perm) >= abs(obs)))
# regional field significance for timing dependence: joint permutation of timing across years within site
# robustness: modified MK (Hamed-Rao), Sen slope, drop largest value
rob = []
for site in SITES:
    df = P[P.site == site].sort_values('year'); x = df.max_mm.values
    a = mk.original_test(x); h = mk.hamed_rao_modification_test(x)
    j = np.argmax(x); x2 = np.delete(x, j); a2 = mk.original_test(x2)
    lag1 = stats.pearsonr(x[:-1], x[1:])[0]
    fr = fits[site]['M1']; q = fr['q']; s, _, _ = design(df, fr['thbar'])
    z = (x - (q['mu0'] + q['mu1']*s))/np.exp(q['lsig'])
    g = np.log(1 + q['xi']*z)/q['xi'] if abs(q['xi']) > 1e-6 else z   # -> standard Gumbel
    ad = stats.anderson(g, dist='gumbel_r')
    ks = stats.kstest(g, 'gumbel_r')
    th = df.theta_rad.values; mth = fr['q']['m0']; kap = np.exp(fr['q']['lk'])
    u = np.sort(stats.vonmises.cdf(th, kap, loc=mth) % 1)
    n = len(u); i = np.arange(1, n+1)
    U2 = np.sum((u - (2*i-1)/(2*n))**2) - n*(u.mean()-0.5)**2 + 1/(12*n)
    rob.append(dict(site=site, MK_tau=a.Tau, MK_p=a.p, MK_HR_p=h.p, sen_slope_mm_per_decade=a.slope*10,
                    MK_p_drop_max=a2.p, lag1_r=lag1, KS_p_gumbel_resid=ks.pvalue, Watson_U2=U2))
ROB = pd.DataFrame(rob)

with pd.ExcelWriter(f'out/{PFX}_model_results.xlsx') as w:
    T.to_excel(w, 'fits', index=False); RL.to_excel(w, 'return_levels', index=False)
    C.to_excel(w, 'conditional_RL', index=False); ROB.to_excel(w, 'robustness', index=False)
json.dump(REG, open(f'out/{PFX}_regional.json', 'w'), indent=1, default=float)
pd.set_option('display.width', 260); pd.set_option('display.float_format', lambda v: f'{v:.3f}')
print(T.to_string(index=False)); print(RL[RL['T'] == 50].to_string(index=False)); print(C.to_string(index=False))
print(ROB.to_string(index=False)); print(json.dumps(REG, indent=1, default=float))
import pickle; pickle.dump({s: {m: {k: v for k, v in f.items() if k != 'spec'} for m, f in d.items()} for s, d in fits.items()}, open(f'out/{PFX}_fits.pkl', 'wb'))
