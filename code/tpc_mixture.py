"""Coherent maximum-likelihood version of the conventional two-part (TP-C) procedure.
Mixture: P(Z<c) = pi + (1-pi) G(c), wet-year GEV untruncated; likelihood
  l(pi, theta) = n_c log[pi + (1-pi) G(c)] + n_w log(1-pi) + sum_{z>=c} log g(z),  pi >= 0.
Substituting p0 = pi + (1-pi) G(c) gives the TP-T likelihood restricted to p0 >= G(c); pi = 0 is the censored GEV.
Also reports the weight-only correction of TP-C (GEV parameters held fixed) for comparison."""
import os, sys, pickle
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evt import *

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("EVT_OUT", os.path.join(HERE, "..", "out"))
S = pickle.load(open(os.path.join(OUT, "emp_stage1.pkl"), "rb")); C = 1.0


def nll(par, x):
    mu, ls, xi, a = par
    if not (-0.9 < xi < 1.5):
        return 1e10
    pi = 1 / (1 + np.exp(-a)); th = (mu, np.exp(ls), xi)
    w = x[x >= C]; nc = int((x < C).sum())
    lg = gev_logpdf(w, *th)
    if not np.all(np.isfinite(lg)):
        return 1e10
    P = pi + (1 - pi) * float(gev_cdf(C, *th))
    if P <= 0 or pi >= 1:
        return 1e10
    return -(nc * np.log(P) + len(w) * np.log1p(-pi) + lg.sum())


rows = []
for p in [q for q in S["pts"] if float(q[:5]) >= 18]:
    x = S["data"][p]
    F = {m: Fit(m, *S["fits"][p][m][:1], -S["fits"][p][m][1], x, C) for m in ("cens", "twopart", "tptrunc")}
    best = None
    for m in ("cens", "twopart", "tptrunc"):
        for xi0 in (F[m].xi, 0.1, 0.5):
            for a0 in (-12, -4, -2, -1, 0, 1, 2):
                st = [F[m].theta[0], np.log(F[m].theta[1]), float(np.clip(xi0, -0.85, 1.45)), a0]
                r = minimize(nll, st, args=(x,), method="Nelder-Mead",
                             options=dict(maxiter=40000, maxfev=40000, xatol=1e-9, fatol=1e-11))
                if best is None or r.fun < best.fun:
                    best = r
    mu, ls, xi, a = best.x; pi = 1 / (1 + np.exp(-a)); th = (mu, np.exp(ls), xi)
    ll_mix = -best.fun
    ll_cens = -nll([F["cens"].theta[0], np.log(F["cens"].theta[1]), F["cens"].xi, -50], x)
    LRb = max(2 * (ll_mix - ll_cens), 0.0)
    p_bound = 0.5 * (1 - stats.chi2.cdf(LRb, 1)) if LRb > 1e-6 else 1.0  # 50:50 chi2_0 / chi2_1 (Self & Liang 1987)
    # TP-T constraint check and weight-only correction of TP-C
    GcT = float(gev_cdf(C, *F["tptrunc"].theta))
    T = F["twopart"]; GcC = float(gev_cdf(C, *T.theta)); piw = (T.p0 - GcC) / (1 - GcC)
    rl_tpc = T.rl(100)
    rl_w = float(gev_ppf(1 - 1 / (100 * (1 - piw)), *T.theta))
    rl_w0 = float(gev_ppf(1 - 1 / (100 * (1 - max(piw, 0.0))), *T.theta))
    rl_mix = float(gev_ppf(1 - 1 / (100 * (1 - pi)), *th))
    rows.append(dict(Point=p, p0_obs=float(np.mean(x < C)), TPT_p0=F["tptrunc"].p0, TPT_Gc=GcT, TPT_constraint_ok=F["tptrunc"].p0 >= GcT,
                     pi_weight_only=piw, x100_TPC=rl_tpc, x100_weight_only=rl_w, x100_weight_only_pi0=rl_w0,
                     pct_weight_only=100 * (rl_w / rl_tpc - 1), pct_weight_only_pi0=100 * (rl_w0 / rl_tpc - 1),
                     pi_mix=pi, xi_mix=xi, ll_mix=ll_mix, ll_cens=ll_cens, LR_pi0=LRb, p_pi0_boundary=p_bound,
                     x100_mix=rl_mix, x100_cens=F["cens"].rl(100), pct_mix_vs_TPC=100 * (rl_mix / rl_tpc - 1)))
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
pd.DataFrame(rows).to_csv(os.path.join(OUT, "T2d_tpc_coherent_mixture.csv"), index=False)
