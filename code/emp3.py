"""Stage 3 checks requested in review: coherent TP-C diagnostics, bootstrap-calibrated full-model GOF
(including a refit-based tail-exceedance test) and a bootstrap-calibrated LR test of p0 = G(c)."""
import os, sys, pickle
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(__file__))
from evt import *

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("EVT_OUT", os.path.join(HERE, "..", "out"))
S1 = pickle.load(open(os.path.join(OUT, "emp_stage1.pkl"), "rb"))
pts, data = S1["pts"], S1["data"]
rng = np.random.default_rng(777); C = 1.0
B = int(os.environ.get("EVT_B", "500"))
arid = [p for p in pts if float(p[:5]) >= 18]
FITS = {p: {m: Fit(m, q, -ll + (tp_bernoulli_ll(data[p], C) if m in ("twopart", "tptrunc") else 0), data[p], C)
            for m, (q, ll, _) in S1["fits"][p].items()} for p in pts}

# 1. TP-C probability excess below the threshold
rows = []
for p in arid:
    F = FITS[p]["twopart"]
    rows.append(dict(Point=p, p0=F.p0, G_c=float(gev_cdf(C, *F.theta)), excess=cpa_jump(F), F_c_plus=F.p0 + cpa_jump(F)))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "T2c_tpc_threshold_excess.csv"), index=False)
print(pd.DataFrame(rows).round(4))

# 2. full-model GOF with refit-based tail test
def stat(F, x, r):
    A = np.mean([ad_uniform(randomized_pit(F, x, r)) for _ in range(10)])
    K = np.mean([ks_uniform(randomized_pit(F, x, r)) for _ in range(10)])
    return A, K

rows = []
for p in pts:
    x = data[p]
    for m in (["std", "cens", "twopart", "tptrunc"] if (x < C).sum() > 0 else ["std"]):
        F = FITS[p][m]
        A0, K0 = stat(F, x, rng); e0 = int((x > F.rl(5)).sum())
        Ab, Kb, Eb, fail = [], [], [], 0
        for b in range(B):
            xb = simulate(F, len(x), rng)
            Fb = fit(m, xb, C, init=F.q, fast=True)
            if not Fb.ok:
                fail += 1; continue
            a, k = stat(Fb, xb, rng); Ab.append(a); Kb.append(k); Eb.append(int((xb > Fb.rl(5)).sum()))
        Ab, Kb, Eb = map(np.array, (Ab, Kb, Eb))
        nc = int((x < C).sum())
        if m in ("std", "cens"):
            Gc = float(gev_cdf(C, *F.theta)); p_dry = stats.binomtest(nc, len(x), min(max(Gc, 1e-9), 1 - 1e-9)).pvalue
        elif m == "twopart":
            Gc = F.p0 + cpa_jump(F); p_dry = stats.binomtest(nc, len(x), Gc).pvalue
        else:
            Gc, p_dry = F.p0, np.nan
        rows.append(dict(Point=p, model=m, AD=A0, AD_p=np.mean(Ab >= A0), KS=K0, KS_p=np.mean(Kb >= K0), boot_fail=fail,
                         n_c=nc, P_below_c_model=Gc, p_dry=p_dry, exc5=e0, exc5_boot_mean=Eb.mean(),
                         p_tail5_boot=np.mean(np.abs(Eb - Eb.mean()) >= abs(e0 - Eb.mean()) - 1e-9)))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(os.path.join(OUT, "T5_gof_full.csv"), index=False)

# 3. bootstrap-calibrated LR test of the censoring restriction (H0: censored GEV)
rows = []
for p in arid:
    x = data[p]; Fc, Ft = FITS[p]["cens"], FITS[p]["tptrunc"]
    LR0 = max(2 * (Ft.ll - Fc.ll), 0.0)
    LRb = []
    for b in range(B):
        xb = simulate(Fc, len(x), rng)
        a = fit("cens", xb, C, init=Fc.q, fast=True)
        t1 = fit("tptrunc", xb, C, init=a.q, fast=True)
        t2 = fit("tptrunc", xb, C, init=Ft.q, fast=True)
        t = t1 if t1.ll >= t2.ll else t2
        if a.ok:
            LRb.append(max(2 * (t.ll - a.ll), 0.0))
    LRb = np.array(LRb)
    rows.append(dict(Point=p, LR=LR0, p_chi2=1 - stats.chi2.cdf(LR0, 1), p_boot=np.mean(LRb >= LR0), B_used=len(LRb),
                     boot_LR_95=np.percentile(LRb, 95)))
    print(rows[-1], flush=True)
T = pd.read_csv(os.path.join(OUT, "T2b_LR_censoring_restriction.csv"))
T = T.merge(pd.DataFrame(rows)[["Point", "p_boot", "B_used", "boot_LR_95"]], on="Point")
T.to_csv(os.path.join(OUT, "T2b_LR_censoring_restriction.csv"), index=False)
print(T.round(4))
