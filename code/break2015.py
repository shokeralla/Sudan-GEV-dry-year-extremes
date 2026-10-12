"""A-priori break test at 2015 (documented MERRA-2 correction-background change), regional mean of
site-standardised indices, with a moving-block bootstrap of whole years (keeps spatial dependence)."""
import os
import numpy as np, pandas as pd
OUT = os.environ.get("EVT_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out"))
E = pd.read_csv(os.path.join(OUT, "indices_annual.csv"))
rng = np.random.default_rng(15); yrs = np.arange(1981, 2025); ny = len(yrs); BL = 5
rows = []
for ix in ["PRCPTOT", "WetDays", "Rx1day", "Rx5day", "SDII", "R20mm", "R95pTOT", "Share", "CWD", "CDD"]:
    M = E.pivot(index="YEAR", columns="Point", values=ix).loc[yrs]
    Z = ((M - M.mean()) / M.std(ddof=1).replace(0, np.nan)).mean(axis=1).values
    post = yrs >= 2015
    D = Z[post].mean() - Z[~post].mean()
    Db = []
    for b in range(5000):
        idx = []
        while len(idx) < ny:
            s = rng.integers(0, ny - BL + 1); idx.extend(range(s, s + BL))
        zb = Z[np.array(idx[:ny])]
        Db.append(zb[post].mean() - zb[~post].mean())
    rows.append(dict(Index=ix, diff_z=D, p_boot=np.mean(np.abs(Db) >= abs(D))))
T = pd.DataFrame(rows); T.to_csv(os.path.join(OUT, "T10b_break2015_regional.csv"), index=False)
print(T.round(3))
