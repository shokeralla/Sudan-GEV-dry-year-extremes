"""Empirical analysis: data preparation, four GEV formulations, return levels, threshold sensitivity,
full-model goodness of fit, regional homogeneity and record-event influence analysis.
Run from the code/ directory:  python emp.py
"""
import os, sys, pickle, json
import numpy as np, pandas as pd
from scipy import stats
from scipy.optimize import minimize, minimize_scalar
sys.path.insert(0, os.path.dirname(__file__))
from evt import *

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data"); OUT = os.environ.get("EVT_OUT", os.path.join(HERE, "..", "out"))
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20261012)
C = 1.0
RP = [2, 10, 25, 50, 100]
MODELS = ["std", "cens", "twopart", "tptrunc"]
log = open(os.path.join(OUT, "emp_log.txt"), "w")
def say(*a):
    s = " ".join(str(v) for v in a); print(s, flush=True); log.write(s + "\n"); log.flush()

# ------------------------------------------------------------------ 1. data
d = pd.read_csv(os.path.join(DATA, "NASA_POWER_Sudan_points_daily_clean_1981_2024.csv"), parse_dates=["Date"])
n_raw = len(d)
d = d.drop_duplicates(subset=["Point", "Date"])
d = d[d.PRECTOTCORR >= 0]
say("daily rows", n_raw, "-> after removing duplicates", len(d))
pts = sorted(d.Point.unique(), key=lambda s: (float(s[:5]), float(s[7:12])))
ann = d.groupby(["Point", "YEAR"]).PRECTOTCORR.agg(AMAX="max", TOT="sum").reset_index()
am_user = pd.read_csv(os.path.join(DATA, "NASA_POWER_Sudan_points_EVT_annual_maxima_1981_2024.csv"))
chk = ann.merge(am_user, on=["Point", "YEAR"])
say("max |AMAX - user annual max| =", float(np.abs(chk.AMAX - chk.annual_max_rainfall).max()))
X = {p: ann[ann.Point == p].sort_values("YEAR") for p in pts}
data = {p: X[p].AMAX.values.astype(float) for p in pts}
years = {p: X[p].YEAR.values for p in pts}
lab = lambda p: f"{int(float(p[:5]))}°N {int(float(p[7:12]))}°E"
arid = [p for p in pts if float(p[:5]) >= 18]

rows = []
for p in pts:
    x = data[p]; l1, l2, t3, t4 = sample_lmoments(x)
    rows.append(dict(Point=p, Lat=float(p[:5]), Lon=float(p[7:12]), n=len(x), MAP=X[p].TOT.mean(), Mean=x.mean(),
                     SD=x.std(ddof=1), CV=x.std(ddof=1) / x.mean(), Min=x.min(), Max=x.max(),
                     Year_max=int(years[p][np.argmax(x)]), L_CV=l2 / l1, t3=t3, t4=t4,
                     n_lt05=int((x < 0.5).sum()), n_lt1=int((x < 1).sum()), n_lt2=int((x < 2).sum()), n_zero=int((x == 0).sum())))
T1 = pd.DataFrame(rows); T1.to_csv(os.path.join(OUT, "T1_descriptive.csv"), index=False)
say(T1.round(3).to_string())

# --------------------------------------------- 2. four formulations at c = 1 mm
fits = {}
rows = []
for p in pts:
    x = data[p]; fits[p] = {}
    for m in MODELS:
        F = fit(m, x, C)
        fits[p][m] = F
        lo, hi = profile_xi_ci(F)
        F.xi_ci = (lo, hi)
        k = F.k; n = len(x)
        rows.append(dict(Point=p, model=m, mu=F.mu, sigma=F.sigma, xi=F.xi, xi_lo=lo, xi_hi=hi, p0=F.p0,
                         n_c=int((x < C).sum()) if m != "std" else 0, LogLik=F.ll,
                         AICc=2 * k - 2 * F.ll + 2 * k * (k + 1) / (n - k - 1)))
    say(p, [(m, round(fits[p][m].xi, 3), round(fits[p][m].rl(100), 1)) for m in MODELS])
T2 = pd.DataFrame(rows)
# LR test of the censoring restriction p0 = G(c) (censored nested in truncated two-part), arid sites
lr_rows = []
for p in arid:
    a, b = fits[p]["cens"], fits[p]["tptrunc"]
    lr = max(2 * (b.ll - a.ll), 0.0)
    Gc = float(gev_cdf(C, *a.theta))
    lr_rows.append(dict(Point=p, n_c=int((data[p] < C).sum()), p0_obs=b.p0, Gc_cens=Gc, LR=lr, p_LR=1 - stats.chi2.cdf(lr, 1)))
TLR = pd.DataFrame(lr_rows); TLR.to_csv(os.path.join(OUT, "T2b_LR_censoring_restriction.csv"), index=False)
T2.to_csv(os.path.join(OUT, "T2_four_models.csv"), index=False)
say(T2.round(3).to_string()); say(TLR.round(4).to_string())

# ------------------------------------------- 3. return levels + profile CIs
rows = []
for p in pts:
    for m in MODELS:
        F = fits[p][m]
        same_as_std = (m != "std") and (data[p] < C).sum() == 0 and m != "tptrunc"
        for T in RP:
            z = F.rl(T)
            if same_as_std:
                prev = [r for r in rows if r["Point"] == p and r["model"] == "std" and r["T"] == T][0]
                lo, hi = prev["lo"], prev["hi"]
            elif m == "tptrunc" and (data[p] < C).sum() == 0:
                lo, hi = np.nan, np.nan      # identical to std within rounding; not repeated
            else:
                lo, hi = profile_rl_ci(F, T)
            rows.append(dict(Point=p, model=m, T=T, RL=z, lo=lo, hi=hi))
    say("RL done", p)
T3 = pd.DataFrame(rows); T3.to_csv(os.path.join(OUT, "T3_return_levels.csv"), index=False)
L = []
for p in pts:
    q = fit_lmom(data[p]); L.append(dict(Point=p, xi_LM=q[2], RL100_LM=return_level("lmom", q, 100), RL10_LM=return_level("lmom", q, 10)))
pd.DataFrame(L).to_csv(os.path.join(OUT, "T3b_lmoments.csv"), index=False)

# ---------------------------------------- 4. threshold sensitivity (arid sites)
rows = []
for p in arid:
    x = data[p]
    for c in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
        for m in ["cens", "twopart", "tptrunc"]:
            F = fit(m, x, c)
            lo, hi = profile_xi_ci(F)
            if c in (0.5, 1.0, 2.0, 3.0) and m in ("cens", "twopart"):
                rlo, rhi = profile_rl_ci(F, 100)
            else:
                rlo, rhi = np.nan, np.nan
            rows.append(dict(Point=p, c=c, model=m, n_c=int((x < c).sum()), xi=F.xi, xi_lo=lo, xi_hi=hi,
                             RL100=F.rl(100), RL100_lo=rlo, RL100_hi=rhi, RL10=F.rl(10)))
    say("threshold done", p)
T4 = pd.DataFrame(rows); T4.to_csv(os.path.join(OUT, "T4_threshold.csv"), index=False)
# antimode of log10 annual maxima (all arid sites pooled): kernel density
xa = np.concatenate([data[p] for p in arid]); xa = xa[xa > 0]
kde = stats.gaussian_kde(np.log10(xa), bw_method=0.25)
g = np.linspace(-1.5, 1.5, 601); dens = kde(g)
mins = [g[i] for i in range(1, len(g) - 1) if dens[i] < dens[i - 1] and dens[i] < dens[i + 1]]
json.dump({"kde_grid": g.tolist(), "kde": dens.tolist(), "antimodes_mm": [10 ** m for m in mins]},
          open(os.path.join(OUT, "kde_arid.json"), "w"))
say("antimodes (mm):", [round(10 ** m, 2) for m in mins])

# ------------------------------------------------ 5. full-model goodness of fit
def gof(F, x, B=500):
    def stat(Fm, xx, r):
        A = np.mean([ad_uniform(randomized_pit(Fm, xx, r)) for _ in range(10)])
        K = np.mean([ks_uniform(randomized_pit(Fm, xx, r)) for _ in range(10)])
        return A, K
    A0, K0 = stat(F, x, rng)
    Ab, Kb, fail = [], [], 0
    for b in range(B):
        xb = simulate(F, len(x), rng)
        Fb = fit(F.model, xb, F.c, init=F.q, fast=True)
        if not Fb.ok: fail += 1; continue
        a, k = stat(Fb, xb, rng); Ab.append(a); Kb.append(k)
    return A0, np.mean(np.array(Ab) >= A0), K0, np.mean(np.array(Kb) >= K0), fail

rows = []
for p in pts:
    x = data[p]
    for m in (MODELS if (x < C).sum() > 0 else ["std"]):
        F = fits[p][m]
        A, pA, K, pK, fail = gof(F, x)
        # dry-year frequency check (only informative for std / cens)
        nc = int((x < C).sum())
        if m in ("std", "cens"):
            Gc = float(gev_cdf(C, *F.theta)); p_dry = stats.binomtest(nc, len(x), min(max(Gc, 1e-9), 1 - 1e-9)).pvalue
        else:
            Gc, p_dry = F.p0, 1.0
        # upper-tail check: exceedances of the fitted 5- and 10-year levels
        e5 = int((x > F.rl(5)).sum()); e10 = int((x > F.rl(10)).sum())
        p_tail = stats.binomtest(e5, len(x), 0.2).pvalue
        rows.append(dict(Point=p, model=m, AD=A, AD_p=pA, KS=K, KS_p=pK, boot_fail=fail, n_c=nc, P_dry_model=Gc,
                         p_dry=p_dry, exc5=e5, exc10=e10, p_tail5=p_tail))
    say("gof done", p)
T5 = pd.DataFrame(rows); T5.to_csv(os.path.join(OUT, "T5_gof_full.csv"), index=False)
say(T5.round(3).to_string())

pickle.dump(dict(pts=pts, data=data, years=years, fits={p: {m: (fits[p][m].q, fits[p][m].ll, fits[p][m].xi_ci) for m in MODELS} for p in pts}),
            open(os.path.join(OUT, "emp_stage1.pkl"), "wb"))
say("stage 1 complete")
