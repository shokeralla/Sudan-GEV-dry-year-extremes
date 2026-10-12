"""Summarise Monte Carlo output: bias/RMSE of xi, relative bias/RMSE of x100 (unclipped),
RMSE of log ratio, Wald-log coverage, failure rates; also profile coverage subset if available."""
import os
import numpy as np, pandas as pd
OUT = os.environ.get("EVT_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out"))
R = pd.read_csv(os.path.join(OUT, "mc_raw.csv"))
R["exi"] = R.xi_hat - R.xi
R["rel"] = R.x100_hat / R.x100_true - 1
R["lr"] = np.log(R.x100_hat / R.x100_true)
R["cov"] = (R.lo <= R.x100_true) & (R.x100_true <= R.hi)
R["has_ci"] = R.lo.notna() & R.hi.notna()
g = R.groupby(["mech", "n", "xi", "p0", "est"])
S = pd.DataFrame({
    "reps": g.size(),
    "fail_rate": 1 - g.ok.mean(),
    "bias_xi": g.exi.mean(),
    "rmse_xi": g.apply(lambda d: np.sqrt(np.mean(d.exi.dropna() ** 2))),
    "mean_rel_bias_x100": g.rel.mean(),
    "median_rel_bias_x100": g.rel.median(),
    "rmse_rel_x100": g.apply(lambda d: np.sqrt(np.mean(d.rel.dropna() ** 2))),
    "rmse_log_x100": g.apply(lambda d: np.sqrt(np.mean(d.lr.dropna() ** 2))),
    "p_rel_gt_1": g.apply(lambda d: np.mean(d.rel.dropna() > 1)),
    "coverage_wald_log": g.apply(lambda d: d.loc[d.has_ci, "cov"].mean() if d.has_ci.any() else np.nan),
    "ci_available": g.has_ci.mean(),
}).reset_index()
S.to_csv(os.path.join(OUT, "T13_mc_summary.csv"), index=False)
pd.set_option("display.width", 250)
print(S[S.n == 44].round(3).to_string())
p = os.path.join(OUT, "mc_profile_raw.csv")
if os.path.exists(p):
    P = pd.read_csv(p)
    P["cov"] = (P.lo <= P.x100_true) & (P.x100_true <= P.hi)
    P["upper_inf"] = ~np.isfinite(P.hi)
    Q = P.dropna(subset=["lo"]).groupby(["mech", "p0", "est"]).agg(reps=("cov", "size"), coverage=("cov", "mean"),
                                                                    upper_unbounded=("upper_inf", "mean")).reset_index()
    Q.to_csv(os.path.join(OUT, "T13b_profile_coverage.csv"), index=False)
    print(Q.round(3).to_string())
