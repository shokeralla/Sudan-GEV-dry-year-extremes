"""Secondary analysis: ETCCDI indices, FDR-controlled trends, block-bootstrap field significance,
regional change points, and a diagnostic for the documented 2015 change in the MERRA-2
precipitation-correction background (Reichle et al., 2017)."""
import os, sys, json
import numpy as np, pandas as pd
from scipy import stats
import pymannkendall as mk
sys.path.insert(0, os.path.dirname(__file__))
from evt import pettitt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data"); OUT = os.environ.get("EVT_OUT", os.path.join(HERE, "..", "out"))
rng = np.random.default_rng(7)
d = pd.read_csv(os.path.join(DATA, "NASA_POWER_Sudan_points_daily_clean_1981_2024.csv"), parse_dates=["Date"])
d = d.drop_duplicates(subset=["Point", "Date"]).sort_values(["Point", "Date"])
pts = sorted(d.Point.unique(), key=lambda s: (float(s[:5]), float(s[7:12])))
MINBASE = 30   # minimum number of 1981-2010 wet days required to define R95p threshold


def spell(b):
    m = c = 0
    for v in b:
        c = c + 1 if v else 0; m = max(m, c)
    return m


rows, qual = [], []
for p in pts:
    s = d[d.Point == p]
    base = s[(s.YEAR <= 2010) & (s.PRECTOTCORR >= 1)].PRECTOTCORR
    nb = len(base); q95 = np.percentile(base, 95) if nb >= MINBASE else np.nan
    qual.append(dict(Point=p, base_wet_days=nb, base_wet_days_per_year=nb / 30, q95=q95,
                     q95_boot_sd=np.std([np.percentile(rng.choice(base, nb), 95) for _ in range(500)]) if nb >= MINBASE else np.nan))
    for y, g in s.groupby("YEAR"):
        pr = g.PRECTOTCORR.values; wet = pr >= 1
        r5 = pd.Series(pr).rolling(5, center=True, min_periods=1).sum()
        k = int(np.argmax(pr))
        rows.append(dict(Point=p, YEAR=y, PRCPTOT=pr[wet].sum(), WetDays=int(wet.sum()), Rx1day=pr.max(),
                         Rx5day=pd.Series(pr).rolling(5).sum().max(), SDII=pr[wet].mean() if wet.any() else np.nan,
                         R20mm=int((pr >= 20).sum()), R95pTOT=pr[pr > q95].sum() if np.isfinite(q95) else np.nan,
                         CWD=spell(wet), CDD=spell(~wet),
                         Share=pr[k] / r5.iloc[k] if r5.iloc[k] > 0 else np.nan))
E = pd.DataFrame(rows); E.to_csv(os.path.join(OUT, "indices_annual.csv"), index=False)
Q = pd.DataFrame(qual); Q.to_csv(os.path.join(OUT, "T7_index_quality.csv"), index=False)
IDX = ["PRCPTOT", "WetDays", "Rx1day", "Rx5day", "SDII", "R20mm", "R95pTOT", "CWD", "CDD"]


def mkt(v):
    v = np.asarray(v, float)
    if np.all(np.isnan(v)) or np.nanstd(v) == 0:
        return np.nan, np.nan, np.nan
    try:
        r = mk.hamed_rao_modification_test(v)
    except Exception:
        r = mk.original_test(v)
    return r.Tau, r.p, r.slope


rows = []
for p in pts:
    e = E[E.Point == p].sort_values("YEAR")
    for ix in IDX:
        tau, pv, sl = mkt(e[ix].values)
        m = np.nanmean(e[ix].values)
        rows.append(dict(Point=p, Index=ix, tau=tau, p=pv, Sen_pct_decade=100 * 10 * sl / m if m else np.nan))
TT = pd.DataFrame(rows)
# Benjamini-Hochberg over the whole family of site-level tests that could be computed
ok = TT.p.notna()
pv = TT.loc[ok, "p"].values; order = np.argsort(pv); m_ = len(pv)
adj = np.empty(m_); prev = 1.0
for rank, i in list(enumerate(order))[::-1]:
    prev = min(prev, pv[i] * m_ / (rank + 1)); adj[i] = prev
TT.loc[ok, "p_BH"] = adj
TT.to_csv(os.path.join(OUT, "T8_site_trends.csv"), index=False)
print("site tests:", ok.sum(), "raw p<0.05:", (TT.p < 0.05).sum(), "BH q<0.05:", (TT.p_BH < 0.05).sum())

# --------------- field significance and regional change points (moving-block bootstrap of years)
yrs = np.arange(1981, 2025); ny = len(yrs); BL = 5
W = {ix: E.pivot(index="YEAR", columns="Point", values=ix).loc[yrs, pts] for ix in IDX}


def zscore(M):
    return (M - M.mean()) / M.std(ddof=1).replace(0, np.nan)


def block_idx():
    idx = []
    while len(idx) < ny:
        s = rng.integers(0, ny - BL + 1); idx.extend(range(s, s + BL))
    return np.array(idx[:ny])


rows = []
NB = 1000
for ix in IDX:
    M = W[ix]
    count = sum(1 for p in pts if (lambda r: np.isfinite(r[1]) and r[1] < 0.05)(mkt(M[p].values)))
    Z = zscore(M).mean(axis=1).values
    k, K, p_asym = pettitt(Z)
    tau, p_reg, sl = mkt(Z)
    # bootstrap null: resample whole years (all sites jointly) in blocks -> destroys trend, keeps spatial dependence
    cnt_b, K_b = [], []
    for b in range(NB):
        ii = block_idx(); Mb = M.iloc[ii].reset_index(drop=True)
        cnt_b.append(sum(1 for p in pts if (lambda r: np.isfinite(r[1]) and r[1] < 0.05)(mkt(Mb[p].values))))
        Zb = zscore(Mb).mean(axis=1).values
        K_b.append(pettitt(Zb)[1])
    rows.append(dict(Index=ix, n_sig_sites=count, p_field=np.mean(np.array(cnt_b) >= count),
                     regional_tau=tau, regional_p_MK=p_reg, Pettitt_year=int(yrs[k + 1]), Pettitt_p_asym=p_asym,
                     Pettitt_p_boot=np.mean(np.array(K_b) >= K)))
    print(rows[-1], flush=True)
TR = pd.DataFrame(rows); TR.to_csv(os.path.join(OUT, "T9_regional.csv"), index=False)

# inter-site dependence
C = W["Rx1day"].corr(method="spearman").values
iu = np.triu_indices(len(pts), 1)
json.dump({"mean_spearman_Rx1day": float(np.mean(C[iu])), "min": float(np.min(C[iu])), "max": float(np.max(C[iu]))},
          open(os.path.join(OUT, "intersite_corr.json"), "w"))
print("inter-site Rx1day Spearman mean", np.mean(C[iu]).round(3))

# --------------- diagnostic: documented background change (MERRA through 2014, GEOS-5 FP-IT from 2015)
rows = []
for p in pts:
    e = E[E.Point == p]
    a, b = e[e.YEAR <= 2014], e[e.YEAR >= 2015]
    for ix in ["PRCPTOT", "WetDays", "Rx1day", "SDII", "Share"]:
        va, vb = a[ix].dropna().values, b[ix].dropna().values
        if len(va) > 3 and len(vb) > 3 and np.std(np.r_[va, vb]) > 0:
            pmw = stats.mannwhitneyu(va, vb).pvalue
        else:
            pmw = np.nan
        rows.append(dict(Point=p, Index=ix, mean_1981_2014=va.mean(), mean_2015_2024=vb.mean(),
                         change_pct=100 * (vb.mean() / va.mean() - 1) if va.mean() else np.nan, p_MW=pmw))
TD = pd.DataFrame(rows); TD.to_csv(os.path.join(OUT, "T10_background_diagnostic.csv"), index=False)
print(TD.pivot(index="Point", columns="Index", values="change_pct").round(0))
print(TD.pivot(index="Point", columns="Index", values="p_MW").round(3))
