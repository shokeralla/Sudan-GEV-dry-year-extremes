"""Independent check with CHIRPS v2.0 daily precipitation (0.05 deg) at the same ten points:
agreement of annual maxima, replication of the dry-year effect on GEV fits, and the 2015 step."""
import os, sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(__file__))
from evt import *

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data"); OUT = os.environ.get("EVT_OUT", os.path.join(HERE, "..", "out"))
rng = np.random.default_rng(2015); C = 1.0
ch = pd.read_csv(os.path.join(DATA, "CHIRPS_Sudan_10pts_daily_1981_2024.csv"), parse_dates=["date"]).rename(columns={"first": "pr"})
ch["Point"] = ch.apply(lambda r: f"{r.lat:.2f}N_{r.lon:.2f}E", axis=1)
ch["YEAR"] = ch.date.dt.year
me = pd.read_csv(os.path.join(DATA, "NASA_POWER_Sudan_points_daily_clean_1981_2024.csv"), parse_dates=["Date"])
me = me.drop_duplicates(subset=["Point", "Date"]).rename(columns={"Date": "date", "PRECTOTCORR": "pr"})
pts = sorted(me.Point.unique(), key=lambda s: (float(s[:5]), float(s[7:12])))
assert set(pts) == set(ch.Point.unique())
print("CHIRPS rows", len(ch), "missing", ch.pr.isna().sum(), "negative", (ch.pr < 0).sum())


def spell(b):
    m = c = 0
    for v in b:
        c = c + 1 if v else 0; m = max(m, c)
    return m


def indices(df, tag):
    rows = []
    for p in pts:
        s = df[df.Point == p].sort_values("date")
        for y, g in s.groupby(s.date.dt.year):
            pr = g.pr.values; wet = pr >= 1
            r5 = pd.Series(pr).rolling(5, center=True, min_periods=1).sum(); k = int(np.argmax(pr))
            rows.append(dict(product=tag, Point=p, YEAR=y, Rx1day=pr.max(), PRCPTOT=pr[wet].sum(), WetDays=int(wet.sum()),
                             SDII=pr[wet].mean() if wet.any() else np.nan, R20mm=int((pr >= 20).sum()),
                             Share=pr[k] / r5.iloc[k] if r5.iloc[k] > 0 else np.nan, CDD=spell(~wet), CWD=spell(wet),
                             TOT=pr.sum()))
    return pd.DataFrame(rows)

E = pd.concat([indices(me, "MERRA-2"), indices(ch, "CHIRPS")])
E.to_csv(os.path.join(OUT, "chirps_indices_annual.csv"), index=False)

# ---- 1. agreement of annual series
rows = []
for p in pts:
    a = E[(E["product"] == "MERRA-2") & (E.Point == p)].set_index("YEAR"); b = E[(E["product"] == "CHIRPS") & (E.Point == p)].set_index("YEAR")
    rows.append(dict(Point=p, MAP_merra=a.TOT.mean(), MAP_chirps=b.TOT.mean(),
                     r_total=stats.spearmanr(a.TOT, b.TOT).statistic, r_rx1=stats.spearmanr(a.Rx1day, b.Rx1day).statistic,
                     mean_rx1_merra=a.Rx1day.mean(), mean_rx1_chirps=b.Rx1day.mean(),
                     max_rx1_merra=a.Rx1day.max(), max_rx1_chirps=b.Rx1day.max(),
                     n_lt1_merra=int((a.Rx1day < 1).sum()), n_lt1_chirps=int((b.Rx1day < 1).sum()),
                     n_zero_chirps=int((b.Rx1day == 0).sum())))
TA = pd.DataFrame(rows); TA.to_csv(os.path.join(OUT, "T14_chirps_agreement.csv"), index=False); print(TA.round(2).to_string())

# ---- 2. dry-year effect replicated on CHIRPS
rows = []
for p in pts:
    x = E[(E["product"] == "CHIRPS") & (E.Point == p)].sort_values("YEAR").Rx1day.values.astype(float)
    Fs = fit("std", x, C); Fc = fit("cens", x, C); q = fit_lmom(x)
    lo, hi = profile_rl_ci(Fc, 100) if (x >= C).sum() >= 10 else (np.nan, np.nan)
    rows.append(dict(Point=p, n_lt1=int((x < C).sum()), xi_std=Fs.xi, RL100_std=Fs.rl(100), xi_cens=Fc.xi, RL100_cens=Fc.rl(100),
                     RL100_cens_lo=lo, RL100_cens_hi=hi, xi_LM=q[2], RL100_LM=return_level("lmom", q, 100), max_obs=x.max()))
    print(rows[-1], flush=True)
TG = pd.DataFrame(rows)
MAP = TA.set_index("Point").MAP_chirps
for k in ["xi_std", "xi_cens", "xi_LM"]:
    r = stats.spearmanr(MAP.loc[TG.Point].values, TG[k].values)
    TG.loc[0, f"rho_{k}_vs_MAP"] = r.statistic; TG.loc[0, f"p_{k}_vs_MAP"] = r.pvalue
TG.to_csv(os.path.join(OUT, "T15_chirps_gev.csv"), index=False)

# ---- 3. 2015 step: each product and their difference (block bootstrap of whole years)
yrs = np.arange(1981, 2025); ny = len(yrs); BL = 5; post = yrs >= 2015
def zmean(prod, ix):
    M = E[E["product"] == prod].pivot(index="YEAR", columns="Point", values=ix).loc[yrs, pts]
    return ((M - M.mean()) / M.std(ddof=1).replace(0, np.nan)).mean(axis=1).values
def blocks():
    idx = []
    while len(idx) < ny:
        s = rng.integers(0, ny - BL + 1); idx.extend(range(s, s + BL))
    return np.array(idx[:ny])
rows = []
for ix in ["Rx1day", "SDII", "R20mm", "PRCPTOT", "WetDays", "Share", "CDD", "CWD"]:
    zm, zc = zmean("MERRA-2", ix), zmean("CHIRPS", ix)
    for tag, z in [("MERRA-2", zm), ("CHIRPS", zc), ("MERRA-2 minus CHIRPS", zm - zc)]:
        D = np.nanmean(z[post]) - np.nanmean(z[~post])
        Db = [np.nanmean(z[i][post]) - np.nanmean(z[i][~post]) for i in (blocks() for _ in range(5000))]
        k, K, pa = pettitt(np.nan_to_num(z))
        rows.append(dict(Index=ix, series=tag, step_2015_SD=D, p_boot=np.mean(np.abs(Db) >= abs(D)), Pettitt_year=int(yrs[k + 1]), Pettitt_p_asym=pa))
TB = pd.DataFrame(rows); TB.to_csv(os.path.join(OUT, "T16_chirps_break2015.csv"), index=False); print(TB.round(3).to_string())

# site-level percent change 1981-2014 -> 2015-2024, both products
rows = []
for p in pts:
    for prod in ["MERRA-2", "CHIRPS"]:
        e = E[(E["product"] == prod) & (E.Point == p)]
        a, b = e[e.YEAR <= 2014], e[e.YEAR >= 2015]
        d = dict(Point=p, product=prod)
        for ix in ["Rx1day", "PRCPTOT", "WetDays", "SDII", "Share"]:
            d[ix] = 100 * (b[ix].mean() / a[ix].mean() - 1) if a[ix].mean() else np.nan
            d[ix + "_p"] = stats.mannwhitneyu(a[ix].dropna(), b[ix].dropna()).pvalue if b[ix].dropna().std() > 0 or a[ix].dropna().std() > 0 else np.nan
        rows.append(d)
TS = pd.DataFrame(rows); TS.to_csv(os.path.join(OUT, "T17_chirps_site_change.csv"), index=False); print(TS.round(1).to_string())
