"""Monte Carlo study: standard, censored and two-part GEV estimators under two dry-year mechanisms.

Mechanism A (two-part world): with prob. p0 a year is dry (maximum ~ U(0, c)); otherwise the
maximum is GEV(mu=4, sigma=3, xi) conditioned on exceeding c.
Mechanism B (censored world): the maximum is GEV(mu_B, 3, xi) with mu_B chosen so that G(c) = p0;
values below c are replaced by U(0, c) trace amounts.
Usage: python mc.py [reps]   (writes ../out/mc_raw.csv; resumable by cell)
"""
import sys, os, itertools, time
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.optimize import brentq
sys.path.insert(0, os.path.dirname(__file__))
from evt import *

C = 1.0; SIG = 3.0; MU_A = 4.0; T = 100
OUT = os.environ.get("EVT_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out"))
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 1000


def true_x100(mech, xi, p0):
    if mech == "A":
        Gc = float(gev_cdf(C, MU_A, SIG, xi))
        u = 1 - (1 - Gc) / (T * (1 - p0))
        return float(gev_ppf(u, MU_A, SIG, xi)), (MU_A, SIG, xi)
    mu = C - SIG * ((-np.log(p0)) ** (-xi) - 1) / xi
    return float(gev_ppf(1 - 1 / T, mu, SIG, xi)), (mu, SIG, xi)


def gen(mech, n, xi, p0, rng):
    if mech == "A":
        nd = rng.binomial(n, p0)
        Gc = float(gev_cdf(C, MU_A, SIG, xi))
        wet = gev_ppf(rng.uniform(Gc, 1, n - nd), MU_A, SIG, xi)
        return np.concatenate([rng.uniform(0, C, nd), wet])
    _, (mu, s, xi_) = true_x100("B", xi, p0)
    x = gev_rvs(mu, s, xi_, n, rng)
    low = x < C
    x[low] = rng.uniform(0, C, low.sum())
    return x


def run_cell(args):
    mech, n, xi, p0, seed = args
    rng = np.random.default_rng(seed)
    x100, par = true_x100(mech, xi, p0)
    init = [par[0], np.log(par[1]), par[2]]
    rows = []
    reps = REPS if n == 44 else REPS // 2
    for r in range(reps):
        x = gen(mech, n, xi, p0, rng)
        for m in ["std", "cens", "twopart", "tptrunc"]:
            if m != "std" and (x >= C).sum() < 8:
                rows.append((mech, n, xi, p0, r, m, np.nan, np.nan, np.nan, np.nan, 0)); continue
            F = fit(m, x, C, fast=True)
            if not F.ok:
                rows.append((mech, n, xi, p0, r, m, np.nan, np.nan, np.nan, np.nan, 0)); continue
            z = F.rl(T)
            lo, hi = wald_log_ci(F, T)
            rows.append((mech, n, xi, p0, r, m, F.xi, z, lo, hi, 1))
        q = fit_lmom(x)
        rows.append((mech, n, xi, p0, r, "lmom", q[2], return_level("lmom", q, T), np.nan, np.nan, 1))
    df = pd.DataFrame(rows, columns=["mech", "n", "xi", "p0", "rep", "est", "xi_hat", "x100_hat", "lo", "hi", "ok"])
    df["x100_true"] = x100
    return df


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "mc_raw.csv")
    done = set()
    if os.path.exists(path):
        d0 = pd.read_csv(path, usecols=["mech", "n", "xi", "p0"]).drop_duplicates()
        done = set(map(tuple, d0.values.tolist()))
    cells = []
    seed = 1000
    for mech, n, xi, p0 in itertools.product(["A", "B"], [30, 44, 60, 100], [0.1, 0.3, 0.5], [0.0, 0.15, 0.30, 0.45]):
        seed += 1
        if mech == "B" and p0 == 0.0:
            continue
        if (mech, n, xi, p0) in done:
            continue
        cells.append((mech, n, xi, p0, seed))
    print(len(cells), "cells to run", flush=True)
    t0 = time.time()
    with Pool(2) as pool:
        for i, df in enumerate(pool.imap_unordered(run_cell, cells)):
            df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
            print(f"{i+1}/{len(cells)} done, {time.time()-t0:.0f}s", flush=True)
