"""Monte Carlo figures."""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.environ.get("EVT_OUT", os.path.join(HERE, "..", "out")); FIG = os.environ.get("EVT_FIG", os.path.join(HERE, "..", "fig"))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "axes.linewidth": 0.7, "grid.color": "#e4e3df", "grid.linewidth": 0.5,
                     "savefig.dpi": 300, "savefig.bbox": "tight", "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})
COL = {"std": "#eb6834", "cens": "#2a78d6", "twopart": "#eda100", "tptrunc": "#1baf7a", "lmom": "#0b0b0b"}
NAME = {"std": "Standard", "cens": "Censored", "twopart": "Two-part (CPA)", "tptrunc": "Two-part (truncated)", "lmom": "L-moments"}
MK = {"std": "o", "cens": "s", "twopart": "^", "tptrunc": "D", "lmom": "v"}
M = pd.read_csv(os.path.join(OUT, "T13_mc_summary.csv"))

# Fig 8: n = 44, xi = 0.3
sub = M[(M.n == 44) & (M.xi == 0.3)]
fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8))
for i, mech in enumerate(["A", "B"]):
    s = sub[sub.mech == mech]
    for e in ["std", "cens", "twopart", "tptrunc", "lmom"]:
        d = s[s.est == e].sort_values("p0")
        axes[i, 0].plot(d.p0, d.bias_xi, marker=MK[e], color=COL[e], lw=1.2, ms=4, label=NAME[e])
        axes[i, 1].plot(d.p0, d.rmse_log_x100, marker=MK[e], color=COL[e], lw=1.2, ms=4)
        if e != "lmom":
            axes[i, 2].plot(d.p0, d.coverage_wald_log, marker=MK[e], color=COL[e], lw=1.2, ms=4)
    axes[i, 0].axhline(0, color="#8a8984", lw=0.6); axes[i, 2].axhline(0.95, color="#8a8984", lw=0.6, ls="--")
    axes[i, 0].set_ylabel(f"Mechanism {mech}\n" + "Bias of ξ̂"); axes[i, 1].set_ylabel("RMSE of log(x̂₁₀₀/x₁₀₀)")
    axes[i, 2].set_ylabel("Coverage of 95% CI"); axes[i, 2].set_ylim(0.3, 1.0)
    for a in axes[i]: a.grid(True); a.set_xticks([0, 0.15, 0.3, 0.45])
for a in axes[1]: a.set_xlabel("Proportion of dry years p₀")
axes[0, 0].legend(frameon=False, fontsize=6.4, loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "Fig4_montecarlo.png")); plt.close()

# Fig S1: effect of sample size (mechanism A, xi = 0.3, p0 = 0.30) and (mechanism B)
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
for mech, ls in [("A", "-"), ("B", "--")]:
    s = M[(M.mech == mech) & (M.xi == 0.3) & (M.p0 == 0.30)]
    for e in ["std", "cens", "twopart", "tptrunc", "lmom"]:
        d = s[s.est == e].sort_values("n")
        axes[0].plot(d.n, d.bias_xi, marker=MK[e], color=COL[e], ls=ls, lw=1.1, ms=3.5, label=f"{NAME[e]} ({mech})" if mech == "A" else None)
        axes[1].plot(d.n, d.rmse_log_x100, marker=MK[e], color=COL[e], ls=ls, lw=1.1, ms=3.5)
        if e != "lmom": axes[2].plot(d.n, d.coverage_wald_log, marker=MK[e], color=COL[e], ls=ls, lw=1.1, ms=3.5)
axes[0].axhline(0, color="#8a8984", lw=0.6); axes[2].axhline(0.95, color="#8a8984", lw=0.6, ls="--")
axes[0].set_ylabel("Bias of ξ̂"); axes[1].set_ylabel("RMSE log x̂₁₀₀"); axes[2].set_ylabel("Coverage")
for a in axes: a.set_xlabel("Sample size n (years)"); a.grid(True)
axes[0].legend(frameon=False, fontsize=5.6)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "FigS1_mc_samplesize.png")); plt.close()
print("mc figures done")
