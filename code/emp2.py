"""Empirical stage 2: regional homogeneity (bootstrap-calibrated LR, profile CI of common xi,
Hosking-Wallis), record-event influence and prospective analysis."""
import os, sys, pickle
import numpy as np, pandas as pd
from scipy import stats
from scipy.optimize import minimize, minimize_scalar
sys.path.insert(0, os.path.dirname(__file__))
from evt import *
import lmoments3.distr as ld

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.environ.get("EVT_OUT", os.path.join(HERE, "..", "out"))
S1 = pickle.load(open(os.path.join(OUT, "emp_stage1.pkl"), "rb"))
pts, data, years = S1["pts"], S1["data"], S1["years"]
rng = np.random.default_rng(31); C = 1.0
south = [p for p in pts if float(p[:5]) <= 15]; north = [p for p in pts if float(p[:5]) >= 18]


def ll_fixed(x, xi, start=None):
    g = lambda r: nll_cens([r[0], r[1], xi], x, C)
    mu0, ls0, med = (lambda s: s)(start) if start is not None else (None, None, None)
    best = None
    xs = x[x >= C]; sd = np.std(xs, ddof=1); b = np.log(max(np.sqrt(6) * sd / np.pi, 1e-3))
    for st in ([start] if start is not None else []) + [[np.mean(xs) - 0.5772 * np.exp(b), b], [np.median(x), b - 0.3]]:
        r = minimize(g, st, method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-9, "maxiter": 6000})
        if best is None or r.fun < best.fun: best = r
    return -best.fun


def common_fit(xs):
    r = minimize_scalar(lambda xi: -sum(ll_fixed(x, xi) for x in xs), bounds=(-0.6, 2.0), method="bounded",
                        options={"xatol": 1e-4})
    return r.x, -r.fun


def lr_test(xs, B=200):
    llf = sum(fit("cens", x, C).ll for x in xs)
    xic, llc = common_fit(xs)
    LR = 2 * (llf - llc)
    # profile CI for common xi
    crit = stats.chi2.ppf(0.95, 1) / 2
    lo = hi = np.nan
    for d, store in [(-1, "lo"), (1, "hi")]:
        for xi in xic + d * np.arange(0.01, 1.5, 0.01):
            if llc - sum(ll_fixed(x, xi) for x in xs) > crit:
                if d < 0: lo = xi
                else: hi = xi
                break
    # parametric bootstrap under H0 (common xi, site-specific mu, sigma)
    H0 = []
    for x in xs:
        g = lambda r: nll_cens([r[0], r[1], xic], x, C)
        F = fit("cens", x, C)
        r = minimize(g, [F.q[0], F.q[1]], method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-9})
        H0.append(r.x)
    LRb = []
    for b in range(B):
        xb = []
        for x, (mu, ls) in zip(xs, H0):
            v = gev_rvs(mu, np.exp(ls), xic, len(x), rng); v[v < C] = rng.uniform(0, C, (v < C).sum()); xb.append(v)
        f1 = sum(fit("cens", v, C, fast=True).ll for v in xb)
        r = minimize_scalar(lambda xi: -sum(ll_fixed(v, xi) for v in xb), bounds=(xic - 0.6, xic + 0.6),
                            method="bounded", options={"xatol": 1e-3})
        LRb.append(max(2 * (f1 + r.fun), 0))
    LRb = np.array(LRb)
    return dict(k=len(xs), xi_common=xic, xi_lo=lo, xi_hi=hi, LR=LR, df=len(xs) - 1,
                p_chi2=1 - stats.chi2.cdf(LR, len(xs) - 1), p_boot=np.mean(LRb >= LR))


def hosking_wallis(xs, nsim=500):
    L = np.array([sample_lmoments(x) for x in xs]); n = np.array([len(x) for x in xs])
    u = np.column_stack([L[:, 1] / L[:, 0], L[:, 2], L[:, 3]]); N = len(xs)
    ub = u.mean(0); A = (u - ub).T @ (u - ub)
    D = [N / 3 * (u[i] - ub) @ np.linalg.solve(A, u[i] - ub) for i in range(N)] if N > 3 else [np.nan] * N
    tR = np.sum(n[:, None] * u, 0) / n.sum()
    V = np.sqrt(np.sum(n * (u[:, 0] - tR[0]) ** 2) / n.sum())
    kap = ld.kap.lmom_fit(lmom_ratios=[1.0, tR[0], tR[1], tR[2]])
    Vs = []
    for s in range(nsim):
        us = np.array([(lambda l: l[1] / l[0])(sample_lmoments(ld.kap.rvs(size=ni, random_state=rng, **kap))) for ni in n])
        tb = np.sum(n * us) / n.sum(); Vs.append(np.sqrt(np.sum(n * (us - tb) ** 2) / n.sum()))
    return (V - np.mean(Vs)) / np.std(Vs, ddof=1), tR, D


rows = []
for name, grp in [("All sites", pts), ("Sahelian-savanna (10-15N)", south), ("Arid (18-20N)", north)]:
    xs = [data[p] for p in grp]
    r = lr_test(xs); H1, tR, D = hosking_wallis(xs)
    r.update(Group=name, H1=H1, tR_LCV=tR[0], tR_t3=tR[1], tR_t4=tR[2])
    rows.append(r); print(r, flush=True)
    if name == "All sites":
        pd.Series(D, index=grp, name="D").to_csv(os.path.join(OUT, "T6b_discordancy.csv"))
# standard (uncensored) likelihood LR for comparison
llf = sum(fit("std", data[p], C).ll for p in pts)
def ll_fixed_std(x, xi):
    g = lambda r: nll_std([r[0], r[1], xi], x)
    F = fit("std", x, C)
    return -minimize(g, [F.q[0], F.q[1]], method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-9}).fun
r = minimize_scalar(lambda xi: -sum(ll_fixed_std(data[p], xi) for p in pts), bounds=(-0.5, 2), method="bounded")
LRs = 2 * (llf + r.fun)
rows.insert(0, dict(Group="All sites (standard GEV)", k=10, xi_common=r.x, LR=LRs, df=9, p_chi2=1 - stats.chi2.cdf(LRs, 9)))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "T6_homogeneity.csv"), index=False)

# ---------------- record events: leave-record-out influence and prospective (pre-event) fits
rows = []
for p in pts:
    x, yr = data[p], years[p]
    k = int(np.argmax(x)); rec, ry = x[k], int(yr[k])
    Ff = fit("cens", x, C)
    keep = np.arange(len(x)) != k
    Fi = fit("cens", x[keep], C)
    Fp = fit("cens", x[yr < ry], C) if (yr < ry).sum() >= 30 else None
    T = lambda F: 1 / max(1 - float(gev_cdf(rec, *F.theta)), 1e-15)
    rows.append(dict(Point=p, Record=rec, Year=ry, T_full=T(Ff), xi_full=Ff.xi, T_leave_out=T(Fi), xi_leave_out=Fi.xi,
                     n_prospective=int((yr < ry).sum()) if Fp else np.nan,
                     T_prospective=T(Fp) if Fp else np.nan, xi_prospective=Fp.xi if Fp else np.nan,
                     RL100_full=Ff.rl(100), RL100_leave_out=Fi.rl(100)))
TR = pd.DataFrame(rows); TR.to_csv(os.path.join(OUT, "T12_records.csv"), index=False)
print(TR.round(2).to_string())
