"""Figure: regional standardized anomalies of MERRA-2 (NASA POWER) and CHIRPS indices and their difference, with the 2015 boundary."""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("EVT_OUT", os.path.join(HERE, "..", "out")); FIG = os.environ.get("EVT_FIG", os.path.join(HERE, "..", "fig"))
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "axes.linewidth": 0.7, "grid.color": "#e4e3df", "grid.linewidth": 0.5,
                     "savefig.dpi": 300, "savefig.bbox": "tight", "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})
E = pd.read_csv(os.path.join(OUT, "chirps_indices_annual.csv"))
B = pd.read_csv(os.path.join(OUT, "T16_chirps_break2015.csv")).set_index(["Index", "series"])
yrs = np.arange(1981, 2025); pts = sorted(E.Point.unique()); post = yrs >= 2015

def z(prod, ix):
    M = E[E["product"] == prod].pivot(index="YEAR", columns="Point", values=ix).loc[yrs, pts]
    return ((M - M.mean()) / M.std(ddof=1).replace(0, np.nan)).mean(axis=1).values

COL = {"MERRA-2": "#2a78d6", "CHIRPS": "#eb6834", "MERRA-2 minus CHIRPS": "#0b0b0b"}
panels = [("Rx1day", "(a) Rx1day"), ("SDII", "(b) SDII"), ("WetDays", "(c) Wet days"), ("PRCPTOT", "(d) PRCPTOT")]
fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), sharex=True)
for a, (ix, title) in zip(axes.flat, panels):
    zm, zc = z("MERRA-2", ix), z("CHIRPS", ix)
    for tag, s, ls in [("MERRA-2", zm, "-"), ("CHIRPS", zc, "-"), ("MERRA-2 minus CHIRPS", zm - zc, ":")]:
        a.plot(yrs, s, color=COL[tag], lw=1.1 if ls == "-" else 0.9, ls=ls, label=tag)
        for m in (~post, post):
            a.hlines(np.nanmean(s[m]), yrs[m][0], yrs[m][-1], color=COL[tag], lw=2.0, alpha=0.55)
    a.axvline(2014.5, color="#8a8984", lw=0.8, ls="--"); a.axhline(0, color="#8a8984", lw=0.5)
    pm, pc, pd_ = (B.loc[(ix, t), "p_boot"] for t in ["MERRA-2", "CHIRPS", "MERRA-2 minus CHIRPS"])
    a.set_title(f"{title}   step p: M {pm:.3f} | C {pc:.3f} | diff {pd_:.3f}", fontsize=7.6, loc="left")
    a.grid(True)
for a in axes[:, 0]: a.set_ylabel("Mean standardized anomaly")
for a in axes[1]: a.set_xlabel("Year")
axes[0, 0].legend(frameon=False, fontsize=6.6, loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "Fig8_chirps_2015.png")); plt.close()
print("chirps figure done")
