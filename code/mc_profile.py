"""Coverage of 95% profile-likelihood intervals for x100 (subset of the Monte Carlo design: n = 44, xi = 0.3;
censored and TP-C estimators)."""
import os, sys, time
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
from evt import *
from mc import gen, true_x100, C

OUT = os.environ.get("EVT_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out"))
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 200


def cell(args):
    mech, xi, p0, seed = args
    rng = np.random.default_rng(seed); x100, _ = true_x100(mech, xi, p0); rows = []
    for r in range(REPS):
        x = gen(mech, 44, xi, p0, rng)
        for m in ["cens", "twopart"]:
            F = fit(m, x, C, fast=True)
            if not F.ok:
                rows.append((mech, xi, p0, r, m, np.nan, np.nan)); continue
            lo, hi = profile_rl_ci(F, 100)
            rows.append((mech, xi, p0, r, m, lo, hi))
    df = pd.DataFrame(rows, columns=["mech", "xi", "p0", "rep", "est", "lo", "hi"]); df["x100_true"] = x100
    return df


if __name__ == "__main__":
    cells = [(mech, 0.3, p0, 500 + i) for i, (mech, p0) in enumerate(
        [("A", 0.15), ("A", 0.30), ("A", 0.45), ("B", 0.15), ("B", 0.30), ("B", 0.45)])]
    path = os.path.join(OUT, "mc_profile_raw.csv")
    if os.path.exists(path): os.remove(path)
    t0 = time.time()
    with Pool(2) as pool:
        for i, df in enumerate(pool.imap_unordered(cell, cells)):
            df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
            print(i + 1, "cells", round(time.time() - t0), flush=True)
