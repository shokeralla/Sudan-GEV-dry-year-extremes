"""Non-stationary censored-GEV models with explicit nesting, influence checks and bootstrap CIs.
Shape constrained to (-0.9, 1.5) to exclude degenerate solutions. M0 stationary; M1 mu ~ t; M2 mu, log sigma ~ t; M3 step (2015) in mu, log sigma; M4 = M2 + M3.
Nesting: M0 < M1 < M2 < M4 and M0 < M3 < M4."""
import os, sys
import numpy as np, pandas as pd
from scipy import stats
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(__file__))
from evt import *

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data"); OUT = os.environ.get("EVT_OUT", os.path.join(HERE, "..", "out"))
rng = np.random.default_rng(99); C = 1.0
d = pd.read_csv(os.path.join(DATA, "NASA_POWER_Sudan_points_daily_clean_1981_2024.csv"))
d = d.drop_duplicates(subset=["Point", "Date"])
pts = sorted(d.Point.unique(), key=lambda s: (float(s[:5]), float(s[7:12])))
ann = d.groupby(["Point", "YEAR"]).PRECTOTCORR.max().reset_index()
K = {"M0": 3, "M1": 4, "M2": 5, "M3": 5, "M4": 7}


def unpack(m, q, t, S):
    if m == "M0": return q[0], np.exp(q[1]), q[2]
    if m == "M1": return q[0] + q[1] * t, np.exp(q[2]), q[3]
    if m == "M2": return q[0] + q[1] * t, np.exp(q[2] + q[3] * t), q[4]
    if m == "M3": return q[0] + q[1] * S, np.exp(q[2] + q[3] * S), q[4]
    return q[0] + q[1] * t + q[2] * S, np.exp(q[3] + q[4] * t + q[5] * S), q[6]


def nll(m, q, x, t, S):
    mu, s, xi = unpack(m, q, t, S)
    if not (-0.9 < xi < 1.5):          # admissible range for the shape parameter
        return BIG
    mu = np.broadcast_to(mu, x.shape); s = np.broadcast_to(s, x.shape)
    obs = x >= C
    z = (x[obs] - mu[obs]) / s[obs]
    a = 1 + xi * z
    if np.any(a <= 0) or np.any(s <= 0): return BIG
    v = np.sum(np.log(s[obs]) + (1 + 1 / xi) * np.log(a) + a ** (-1 / xi))
    if (~obs).any():
        zc = (C - mu[~obs]) / s[~obs]; ac = 1 + xi * zc
        if np.any(ac <= 0): return BIG
        v += np.sum(ac ** (-1 / xi))
    return v


def base(m, q0):
    mu, ls, xi = q0
    return {"M0": [mu, ls, xi], "M1": [mu, 0, ls, xi], "M2": [mu, 0, ls, 0, xi], "M3": [mu, 0, ls, 0, xi],
            "M4": [mu, 0, 0, ls, 0, 0, xi]}[m]


def fit_ns(m, x, t, S, q0, extra=()):
    f = lambda q: nll(m, q, x, t, S)
    best = None
    trials = [np.array(base(m, q0), float)] + [np.array(e, float) for e in extra]
    for sc in [-0.3, 0.3]:
        b = np.array(base(m, q0), float)
        if m != "M0": b[1] = sc * np.exp(q0[1]); trials.append(b)
    for q in trials:
        if f(q) >= BIG / 10: continue
        r = minimize(f, q, method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-10, "maxiter": 60000, "maxfev": 120000})
        r = minimize(f, r.x, method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-11, "maxiter": 60000, "maxfev": 120000})
        if best is None or r.fun < best.fun: best = r
    return best.x, -best.fun


def fit_all(x, yr):
    t = (yr - 2002.5) / 10.0; S = (yr >= 2015).astype(float)
    q0 = fit("cens", x, C).q
    res = {}
    for m in ["M0", "M1", "M2", "M3"]:
        res[m] = fit_ns(m, x, t, S, q0)
    q2, q3 = res["M2"][0], res["M3"][0]
    ext = [[q2[0], q2[1], 0, q2[2], q2[3], 0, q2[4]], [q3[0], 0, q3[1], q3[2], 0, q3[3], q3[4]]]
    res["M4"] = fit_ns("M4", x, t, S, q0, ext)
    # enforce nesting monotonicity of the maximised log-likelihood
    for small, big in [("M0", "M1"), ("M1", "M2"), ("M2", "M4"), ("M0", "M3"), ("M3", "M4")]:
        if res[big][1] < res[small][1] - 1e-6:
            raise RuntimeError(f"nesting violated {small}->{big}")
    return res, t, S


rows, store = [], {}
for p in pts:
    a = ann[ann.Point == p].sort_values("YEAR"); x = a.PRECTOTCORR.values.astype(float); yr = a.YEAR.values
    res, t, S = fit_all(x, yr)
    n = len(x)
    aicc = {m: 2 * K[m] - 2 * res[m][1] + 2 * K[m] * (K[m] + 1) / (n - K[m] - 1) for m in res}
    lr = lambda big, small, df: (2 * (res[big][1] - res[small][1]), 1 - stats.chi2.cdf(max(2 * (res[big][1] - res[small][1]), 0), df))
    r = dict(Point=p, **{f"AICc_{m}": aicc[m] for m in res}, Best=min(aicc, key=aicc.get))
    for (big, small, df) in [("M1", "M0", 1), ("M2", "M1", 1), ("M3", "M0", 2), ("M4", "M2", 2), ("M4", "M3", 2)]:
        L_, P_ = lr(big, small, df); r[f"LR_{big}v{small}"] = L_; r[f"p_{big}v{small}"] = P_
    r["mu1_M1"] = res["M1"][0][1]; r["xi_M1"] = res["M1"][0][3]
    r["mu1_M2"], r["phi1_M2"], r["xi_M2"] = res["M2"][0][1], res["M2"][0][3], res["M2"][0][4]
    r["dmu_M3"], r["dphi_M3"] = res["M3"][0][1], res["M3"][0][3]
    rows.append(r); store[p] = (x, yr, res, t, S)
    print(p, r["Best"], {m: round(aicc[m] - aicc["M0"], 2) for m in res}, flush=True)
TN = pd.DataFrame(rows); TN.to_csv(os.path.join(OUT, "T11_nonstationary.csv"), index=False)

# ---------------- influence analysis and bootstrap CIs for the selected non-stationary model
rows, eff = [], []
for p in pts:
    best = TN.set_index("Point").loc[p, "Best"]
    if best == "M0": continue
    x, yr, res, t, S = store[p]
    q = res[best][0]
    q0 = fit("cens", x, C).q
    # leave-one-year-out
    loo = []
    for i in range(len(x)):
        keep = np.arange(len(x)) != i
        qi, _ = fit_ns(best, x[keep], t[keep], S[keep], q0, [q])
        loo.append(qi)
    loo = np.array(loo)
    # drop the two largest maxima
    keep = np.argsort(x)[:-2]; keep.sort()
    q2drop, _ = fit_ns(best, x[keep], t[keep], S[keep], q0, [q])
    names = {"M2": ["mu0", "mu1", "phi0", "phi1", "xi"], "M3": ["mu0", "dmu", "phi0", "dphi", "xi"],
             "M1": ["mu0", "mu1", "phi0", "xi"], "M4": ["mu0", "mu1", "dmu", "phi0", "phi1", "dphi", "xi"]}[best]
    for j, nm in enumerate(names):
        rows.append(dict(Point=p, model=best, param=nm, est=q[j], loo_min=loo[:, j].min(), loo_max=loo[:, j].max(), drop2=q2drop[j]))
    # effective (year-specific) return levels with parametric bootstrap
    B = 150; bs = []
    for b in range(B):
        mu, s, xi = unpack(best, q, t, S)
        xb = gev_rvs(0, 1, xi, len(x), rng) * s + mu
        xb = np.where(xb < C, rng.uniform(0, C, len(x)), xb)
        try:
            qb, _ = fit_ns(best, xb, t, S, q0, [q])
            bs.append(qb)
        except Exception:
            pass
    bs = np.array(bs)
    for Tr in [2, 20, 100]:
        for iy, Y in [(0, 1981), (len(x) - 1, 2024)]:
            mu, s, xi = unpack(best, q, t, S)
            z = gev_ppf(1 - 1 / Tr, np.broadcast_to(mu, x.shape)[iy], np.broadcast_to(s, x.shape)[iy], xi)
            zb = []
            for qb in bs:
                mb, sb, xib = unpack(best, qb, t, S)
                zb.append(gev_ppf(1 - 1 / Tr, np.broadcast_to(mb, x.shape)[iy], np.broadcast_to(sb, x.shape)[iy], xib))
            eff.append(dict(Point=p, model=best, T=Tr, Year=Y, RL=float(z), lo=np.percentile(zb, 2.5), hi=np.percentile(zb, 97.5)))
    print("influence/bootstrap done", p, flush=True)
pd.DataFrame(rows).to_csv(os.path.join(OUT, "T11b_ns_influence.csv"), index=False)
pd.DataFrame(eff).to_csv(os.path.join(OUT, "T11c_effective_RL.csv"), index=False)
import pickle; pickle.dump({p: (store[p][0], store[p][1], {m: store[p][2][m][0] for m in store[p][2]}) for p in pts},
                           open(os.path.join(OUT, "ns_store.pkl"), "wb"))
print("done")
