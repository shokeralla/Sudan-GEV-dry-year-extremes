"""Check the reduced (fast) optimiser used in the Monte Carlo against the full multi-start optimiser."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from evt import *
from mc import gen, C
OUT = os.environ.get("EVT_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out"))
rng = np.random.default_rng(404); rows = []
for mech, xi, p0 in [("A", 0.3, 0.30), ("B", 0.3, 0.30), ("A", 0.5, 0.45), ("B", 0.1, 0.15)]:
    for r in range(40):
        x = gen(mech, 44, xi, p0, rng)
        for m in ["std", "cens", "twopart", "tptrunc"]:
            a = fit(m, x, C, fast=True); b = fit(m, x, C)
            rows.append(dict(mech=mech, xi=xi, p0=p0, est=m, dll=b.ll - a.ll, dxi=abs(b.xi - a.xi),
                             drl=abs(b.rl(100) / a.rl(100) - 1) if a.ok and b.ok else np.nan))
D = pd.DataFrame(rows)
S = D.groupby("est").agg(n=("dll", "size"), share_dll_gt_0_01=("dll", lambda v: np.mean(v > 0.01)),
                         max_dll=("dll", "max"), share_dxi_gt_0_01=("dxi", lambda v: np.mean(v > 0.01)),
                         median_drl=("drl", "median"), p95_drl=("drl", lambda v: np.nanpercentile(v, 95))).reset_index()
S.to_csv(os.path.join(OUT, "T13c_optimizer_check.csv"), index=False); print(S.round(4).to_string())
