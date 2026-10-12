"""Figures for the revised manuscript (300 dpi PNG)."""
import os, sys, json, pickle
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.ticker as mt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
sys.path.insert(0, os.path.dirname(__file__))
from evt import *

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = os.environ.get("EVT_OUT", os.path.join(HERE, "..", "out")); FIG = os.environ.get("EVT_FIG", os.path.join(HERE, "..", "fig"))
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 8.5, "axes.labelsize": 8.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.2, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#52514e", "axes.linewidth": 0.7, "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "grid.color": "#e4e3df", "grid.linewidth": 0.5, "savefig.dpi": 300,
                     "savefig.bbox": "tight"})
BLUE, ORANGE, AQUA, YELLOW, GREY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8984", "#0b0b0b"
MCOL = {"std": ORANGE, "cens": BLUE, "twopart": YELLOW, "tptrunc": AQUA, "lmom": INK}
MNAME = {"std": "Standard GEV", "cens": "Censored GEV", "twopart": "Two-part GEV (CPA)",
         "tptrunc": "Two-part GEV (truncated)", "lmom": "GEV, L-moments"}
S1 = pickle.load(open(os.path.join(OUT, "emp_stage1.pkl"), "rb"))
pts, data, years = S1["pts"], S1["data"], S1["years"]
T1 = pd.read_csv(os.path.join(OUT, "T1_descriptive.csv")).set_index("Point")
lab = lambda p: f"{int(float(p[:5]))}°N {int(float(p[7:12]))}°E"
grp = lambda p: BLUE if float(p[:5]) <= 15 else ORANGE
arid = [p for p in pts if float(p[:5]) >= 18]
C = 1.0
FITS = {p: {m: Fit(m, q, -ll + (tp_bernoulli_ll(data[p], C) if m in ("twopart", "tptrunc") else 0), data[p], C)
            for m, (q, ll, _) in S1["fits"][p].items()} for p in pts}

# ---- Fig 1 study area
gj = json.load(open(os.path.join(DATA, "ne_50m_admin_0_countries.geojson")))
fig, ax = plt.subplots(figsize=(6.3, 5.0))
def draw(g, **kw):
    for poly in (g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]):
        r = np.array(poly[0]); ax.fill(r[:, 0], r[:, 1], **kw)
for f in gj["features"]:
    sd = f["properties"]["ADMIN"] == "Sudan"
    draw(f["geometry"], facecolor="#f3f1ea" if sd else "#fbfbf9", edgecolor="#52514e" if sd else "#b9b8b2",
         linewidth=1.0 if sd else 0.5, zorder=2 if sd else 1)
for nm, (x0, y0) in {"EGYPT": (29.5, 23.3), "LIBYA": (22.0, 22.8), "CHAD": (20.2, 14.0), "SOUTH SUDAN": (30.0, 7.4),
                     "ETHIOPIA": (38.4, 11.6), "ERITREA": (39.2, 15.2), "CAR": (21.5, 7.6)}.items():
    ax.text(x0, y0, nm, color=GREY, fontsize=7, ha="center", style="italic", zorder=3)
ax.text(27.6, 12.6, "SUDAN", color="#52514e", fontsize=9, ha="center", fontweight="bold", zorder=3)
cmap = LinearSegmentedColormap.from_list("b", ["#cfe0f6", "#2a78d6", "#0d2f5c"])
sc = ax.scatter(T1.Lon, T1.Lat, c=T1.MAP, cmap=cmap, norm=LogNorm(5, 2000), s=95, edgecolors="white", linewidths=1.6, zorder=5)
for p in pts:
    ax.annotate(f"{lab(p)}\n{T1.loc[p,'MAP']:.0f} mm", (T1.loc[p, "Lon"], T1.loc[p, "Lat"]), xytext=(7, -2),
                textcoords="offset points", fontsize=6.6, zorder=6, va="top")
cb = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02); cb.set_label("Mean annual precipitation, 1981–2024 (mm)"); cb.outline.set_linewidth(0.5)
ax.axhline(16.5, color=GREY, lw=0.6, ls=(0, (4, 3)), zorder=4)
ax.text(18.2, 16.8, "arid–hyper-arid", fontsize=6.8, color=ORANGE); ax.text(18.2, 15.9, "Sahelian–savanna", fontsize=6.8, color=BLUE)
ax.set_xlim(18, 40); ax.set_ylim(5.5, 24.5); ax.set_aspect("equal"); ax.grid(True, zorder=0)
ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
fig.savefig(os.path.join(FIG, "Fig1_study_area.png")); plt.close()

# ---- Fig 2 annual maxima on a log scale (dry-year cluster)
fig, ax = plt.subplots(figsize=(6.8, 3.4))
for i, p in enumerate(pts):
    x = data[p].copy(); x[x <= 0] = 0.005
    jit = (np.random.default_rng(i).uniform(-0.18, 0.18, len(x)))
    ax.scatter(np.full(len(x), i) + jit, x, s=10, color=grp(p), alpha=0.8, edgecolor="none")
for c, ls in [(0.5, ":"), (1.0, "--"), (2.0, ":")]:
    ax.axhline(c, color=GREY, lw=0.8, ls=ls)
ax.text(5.5, 1.1, "c = 1 mm (dotted: 0.5 and 2 mm)", fontsize=6.8, color="#52514e", ha="center")
ax.set_yscale("log"); ax.set_ylim(0.004, 300)
ax.set_xticks(range(len(pts))); ax.set_xticklabels([lab(p).replace(" ", "\n") for p in pts], fontsize=7)
ax.set_ylabel("Annual maximum daily rainfall (mm)"); ax.grid(True, axis="y")
ax.text(7, 0.0062, "exact zero shown at 0.005 mm", fontsize=6.3, color=GREY, ha="center")
fig.savefig(os.path.join(FIG, "Fig2_annual_maxima.png")); plt.close()

# ---- Fig 3 return-level curves at arid sites, four formulations
Tg = np.logspace(np.log10(1.1), np.log10(200), 160)
fig, axes = plt.subplots(2, 2, figsize=(6.9, 5.6))
for ax, p in zip(axes.flatten(), arid):
    x = data[p]; xs = np.sort(x); n = len(xs)
    Te = 1 / (1 - (np.arange(1, n + 1) - 0.44) / (n + 0.12))
    ax.scatter(Te, np.maximum(xs, 0.005), s=10, color=INK, zorder=5, label="Observed (Gringorten)")
    for m, ls in [("std", "-"), ("twopart", "-."), ("tptrunc", "--"), ("cens", "-")]:
        F = FITS[p][m]
        ax.plot(Tg, [F.rl(T) for T in Tg], color=MCOL[m], lw=1.4 if m in ("std", "cens") else 1.1, ls=ls,
                label=f"{MNAME[m]} (ξ = {F.xi:.2f})")
    ax.axhline(C, color=GREY, lw=0.6, ls=":")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(0.004, 2000)
    ax.set_xticks([1, 2, 5, 10, 25, 50, 100, 200]); ax.xaxis.set_major_formatter(mt.ScalarFormatter())
    ax.set_title(f"{lab(p)} — {int((x < C).sum())} of {len(x)} maxima < 1 mm", loc="left")
    ax.set_xlabel("Return period (years)"); ax.set_ylabel("Return level (mm/day)"); ax.grid(True)
    ax.legend(frameon=False, loc="lower right", fontsize=5.9)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "Fig3_return_levels_arid.png")); plt.close()

# ---- Fig 4 threshold sensitivity
T4 = pd.read_csv(os.path.join(OUT, "T4_threshold.csv"))
fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.0), sharex=True)
for j, p in enumerate(arid):
    for m, mk in [("cens", "s"), ("twopart", "^")]:
        t = T4[(T4.Point == p) & (T4.model == m)]
        off = 0.97 if m == "cens" else 1.03
        axes[0, j].errorbar(t.c * off, t.xi, yerr=[t.xi - t.xi_lo, t.xi_hi - t.xi], fmt=mk, color=MCOL[m], ms=3.5, lw=0.8,
                            capsize=1.5, label=MNAME[m])
        tt = t.dropna(subset=["RL100_lo"])
        axes[1, j].errorbar(tt.c * off, tt.RL100, yerr=[tt.RL100 - tt.RL100_lo, tt.RL100_hi - tt.RL100], fmt=mk, color=MCOL[m],
                            ms=3.5, lw=0.8, capsize=1.5)
        axes[1, j].plot(t.c * off, t.RL100, color=MCOL[m], lw=0.6, alpha=0.6)
    Fs = FITS[p]["std"]
    axes[0, j].axhline(Fs.xi, color=ORANGE, lw=0.9, ls="--"); axes[1, j].axhline(Fs.rl(100), color=ORANGE, lw=0.9, ls="--")
    axes[0, j].axvline(1.0, color=GREY, lw=0.6, ls=":"); axes[1, j].axvline(1.0, color=GREY, lw=0.6, ls=":")
    axes[0, j].set_title(lab(p), loc="left"); axes[1, j].set_yscale("log")
    axes[1, j].set_xticks([0.5, 1, 2, 3]); axes[1, j].xaxis.set_major_formatter(mt.FormatStrFormatter("%g"))
    axes[1, j].set_xlabel("Threshold c (mm)"); axes[0, j].grid(True); axes[1, j].grid(True)
axes[0, 0].set_ylabel("Shape ξ (95% profile CI)"); axes[1, 0].set_ylabel("100-year level (mm/day)")
axes[0, 0].plot([], [], color=ORANGE, ls="--", label="Standard GEV (no threshold)")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=7, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(os.path.join(FIG, "Fig5_threshold.png")); plt.close()

# ---- Fig 6 shape vs aridity
T2 = pd.read_csv(os.path.join(OUT, "T2_four_models.csv")); LM = pd.read_csv(os.path.join(OUT, "T3b_lmoments.csv")).set_index("Point")
fig, ax = plt.subplots(figsize=(5.6, 3.9))
for p in pts:
    mp = T1.loc[p, "MAP"]
    for m, mk, off in [("std", "o", 0.93), ("cens", "s", 1.0)]:
        r = T2[(T2.Point == p) & (T2.model == m)].iloc[0]
        ax.errorbar(mp * off, r.xi, yerr=[[r.xi - r.xi_lo], [r.xi_hi - r.xi]], fmt=mk, color=MCOL[m], ms=4.5, lw=0.8, capsize=2)
    ax.scatter(mp * 1.07, LM.loc[p, "xi_LM"], marker="^", color=AQUA, s=26, zorder=5)
ax.errorbar([], [], fmt="o", color=ORANGE, label="Standard GEV (95% profile CI)")
ax.errorbar([], [], fmt="s", color=BLUE, label="Censored GEV, c = 1 mm (95% profile CI)")
ax.scatter([], [], marker="^", color=AQUA, label="GEV, L-moments")
ax.axhline(0, color=GREY, lw=0.7, ls=":"); ax.set_xscale("log")
ax.set_xlabel("Mean annual precipitation (mm, log scale)"); ax.set_ylabel("GEV shape parameter ξ")
ax.legend(frameon=False, loc="upper right", fontsize=6.8); ax.grid(True)
fig.savefig(os.path.join(FIG, "Fig6_xi_vs_aridity.png")); plt.close()

# ---- Fig 7 regional anomalies & 2015 break
E = pd.read_csv(os.path.join(OUT, "indices_annual.csv"))
fig, axes = plt.subplots(2, 1, figsize=(6.8, 4.6), sharex=True)
for ax, items in zip(axes, [[("Rx1day", BLUE), ("PRCPTOT", AQUA), ("WetDays", GREY)], [("Share", ORANGE), ("CDD", INK)]]):
    for ix, col in items:
        M = E.pivot(index="YEAR", columns="Point", values=ix)
        Z = ((M - M.mean()) / M.std(ddof=1)).mean(axis=1)
        ax.plot(Z.index, Z.values, color=col, lw=1.2, marker="o", ms=2.2, label=ix if ix != "Share" else "Max-day share of 5-day total")
        for a, b in [(1981, 2014), (2015, 2024)]:
            seg = Z.loc[a:b]; ax.plot([a, b], [seg.mean()] * 2, color=col, lw=0.8, ls="--", alpha=0.8)
    ax.axvline(2014.5, color=GREY, lw=0.8, ls=(0, (4, 3))); ax.axhline(0, color=GREY, lw=0.5)
    ax.set_ylabel("Mean standardized anomaly"); ax.grid(True, axis="y"); ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=6.8)
axes[0].set_title("(a) Totals, frequency and intensity", loc="left"); axes[1].set_title("(b) Within-pentad concentration and dry spells", loc="left")
axes[1].set_xlabel("Year")
axes[0].text(2014.7, axes[0].get_ylim()[0] + 0.1, "2015: correction background\nMERRA → GEOS-5 FP-IT", fontsize=6.2, color="#52514e")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "Fig7_indices_2015.png")); plt.close()
print("main figures done")
