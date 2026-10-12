"""Core extreme-value tools: standard, left-censored and two-part GEV models.

Parameter vector q = (mu, log sigma, xi); Coles (2001) sign convention (xi > 0 heavy tail).
"""
import numpy as np
from scipy import stats
from scipy.optimize import minimize, brentq

EPS = 1e-6
BIG = 1e12


# ----------------------------------------------------------------- GEV basics
def gev_logpdf(x, mu, sigma, xi):
    z = (np.asarray(x, float) - mu) / sigma
    if abs(xi) < EPS:
        return -np.log(sigma) - z - np.exp(-z)
    a = 1.0 + xi * z
    out = np.full(z.shape, -np.inf)
    ok = a > 0
    la = np.log(a[ok])
    out[ok] = -np.log(sigma) - (1 + 1 / xi) * la - np.exp(-la / xi)
    return out


def gev_cdf(x, mu, sigma, xi):
    z = (np.asarray(x, float) - mu) / sigma
    if abs(xi) < EPS:
        return np.exp(-np.exp(-z))
    a = 1.0 + xi * z
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        out = np.exp(-np.power(np.maximum(a, 1e-300), -1 / xi))
    out = np.where(a <= 0, 0.0 if xi > 0 else 1.0, out)
    return out


def gev_ppf(u, mu, sigma, xi):
    y = -np.log(np.asarray(u, float))
    if abs(xi) < EPS:
        return mu - sigma * np.log(y)
    return mu + sigma / xi * (y ** (-xi) - 1)


def gev_rvs(mu, sigma, xi, n, rng):
    return gev_ppf(rng.uniform(size=n), mu, sigma, xi)


# ------------------------------------------------------------ log-likelihoods
def nll_std(q, x):
    mu, ls, xi = q
    lp = gev_logpdf(x, mu, np.exp(ls), xi)
    return BIG if not np.all(np.isfinite(lp)) else -np.sum(lp)


def nll_cens(q, x, c):
    """Left-censored GEV: maxima below c contribute log G(c)."""
    mu, ls, xi = q
    s = np.exp(ls)
    obs = x >= c
    lp = gev_logpdf(x[obs], mu, s, xi)
    if not np.all(np.isfinite(lp)):
        return BIG
    v = -np.sum(lp)
    nc = int((~obs).sum())
    if nc:
        Gc = float(gev_cdf(c, mu, s, xi))
        if Gc <= 0:
            return BIG
        v -= nc * np.log(Gc)
    return v


def nll_tp_gev(q, x, c):
    """Two-part (conditional-probability) model, wet-year component: GEV fitted to maxima >= c
    (Jennings & Benson 1969 adjustment); dry-year probability p0 is estimated separately."""
    mu, ls, xi = q
    lp = gev_logpdf(x[x >= c], mu, np.exp(ls), xi)
    return BIG if not np.all(np.isfinite(lp)) else -np.sum(lp)


def nll_tp_trunc(q, x, c):
    """Variant: wet-year GEV truncated below at c (used only as a sensitivity check)."""
    mu, ls, xi = q
    s = np.exp(ls)
    xo = x[x >= c]
    lp = gev_logpdf(xo, mu, s, xi)
    if not np.all(np.isfinite(lp)):
        return BIG
    Gc = float(gev_cdf(c, mu, s, xi))
    if Gc >= 1 - 1e-12:
        return BIG
    return -(np.sum(lp) - len(xo) * np.log1p(-Gc))


def tp_p0(x, c):
    return float(np.mean(x < c))


def tp_bernoulli_ll(x, c):
    n = len(x); nc = int((x < c).sum()); p = nc / n
    out = 0.0
    if nc: out += nc * np.log(p)
    if n - nc: out += (n - nc) * np.log(1 - p)
    return out


# ------------------------------------------------------------------- fitting
def _starts(x):
    xs = x[x > 0] if np.sum(x > 0) > 3 else x
    sd = np.std(xs, ddof=1)
    beta = max(np.sqrt(6) * sd / np.pi, 1e-3)
    return np.mean(xs) - 0.5772 * beta, np.log(beta), np.median(xs)


def _fit(f, x, init=None, fast=False):
    mu0, ls0, med = _starts(x)
    inits = [] if init is None else [np.array(init, float)]
    xis = (0.1, 0.6) if fast else (0.0, 0.15, 0.3, 0.5, 0.8, 1.1, -0.2)
    mus = (mu0,) if fast else (mu0, med, 0.6 * mu0)
    for xi0 in xis:
        for m in mus:
            inits.append(np.array([m, ls0, xi0]))
    best = None
    for p0 in inits:
        if f(p0) >= BIG / 10:
            continue
        r = minimize(f, p0, method="Nelder-Mead",
                     options={"xatol": 1e-7 if fast else 1e-9, "fatol": 1e-8 if fast else 1e-10,
                              "maxiter": 3000 if fast else 20000, "maxfev": 4000 if fast else 40000})
        if best is None or r.fun < best.fun:
            best = r
    if best is None:
        return None
    if not fast:
        r2 = minimize(f, best.x, method="BFGS")
        if np.isfinite(r2.fun) and r2.fun < best.fun - 1e-10:
            best = r2
    return best


class Fit:
    def __init__(self, model, q, nll, x, c, ok=True):
        self.model, self.q, self.nll, self.x, self.c, self.ok = model, np.asarray(q), nll, x, c, ok
        self.mu, self.sigma, self.xi = q[0], float(np.exp(q[1])), q[2]
        self.theta = np.array([self.mu, self.sigma, self.xi])
        tp = model in ("twopart", "tptrunc")
        self.p0 = tp_p0(x, c) if tp else None
        self.k = 4 if tp else 3
        self.ll = -nll + (tp_bernoulli_ll(x, c) if tp else 0.0)

    def rl(self, T):
        return return_level(self.model, self.q, T, self.c, self.p0)


def fit(model, x, c=1.0, init=None, fast=False):
    x = np.asarray(x, float)
    if model == "std":
        f = lambda q: nll_std(q, x)
    elif model == "cens":
        f = lambda q: nll_cens(q, x, c)
    elif model == "twopart":
        f = lambda q: nll_tp_gev(q, x, c)
    elif model == "tptrunc":
        f = lambda q: nll_tp_trunc(q, x, c)
    else:
        raise ValueError(model)
    r = _fit(f, x, init, fast)
    if r is None:
        return Fit(model, [np.nan] * 3, np.inf, x, c, ok=False)
    ok = bool(np.isfinite(r.fun) and r.fun < BIG / 10 and -0.95 < r.x[2] < 3.0)
    return Fit(model, r.x, r.fun, x, c, ok)


def nll_fun(model, x, c):
    if model == "std":
        return lambda q: nll_std(q, x)
    if model == "cens":
        return lambda q: nll_cens(q, x, c)
    if model == "tptrunc":
        return lambda q: nll_tp_trunc(q, x, c)
    return lambda q: nll_tp_gev(q, x, c)


# -------------------------------------------------------------- return levels
def return_level(model, q, T, c=1.0, p0=None):
    mu, ls, xi = q
    s = np.exp(ls)
    if model in ("std", "cens", "lmom"):
        return float(gev_ppf(1 - 1 / T, mu, s, xi))
    if T * (1 - p0) <= 1:
        return float(c)
    if model == "twopart":   # F_w(z_T) = 1 - 1/(T(1-p0))
        return float(gev_ppf(1 - 1 / (T * (1 - p0)), mu, s, xi))
    # truncated variant: p0 + (1-p0) * (G(z)-G(c))/(1-G(c)) = 1 - 1/T
    Gc = float(gev_cdf(c, mu, s, xi))
    u = 1 - (1 - Gc) / (T * (1 - p0))
    return float(gev_ppf(u, mu, s, xi))


def annual_cdf(model, q, z, c=1.0, p0=None):
    mu, ls, xi = q
    s = np.exp(ls)
    if model in ("std", "cens"):
        return gev_cdf(z, mu, s, xi)
    if model == "twopart":
        return np.where(np.asarray(z) < c, p0, p0 + (1 - p0) * gev_cdf(z, mu, s, xi))
    Gc = float(gev_cdf(c, mu, s, xi))
    Fw = (gev_cdf(z, mu, s, xi) - Gc) / (1 - Gc)
    return np.where(np.asarray(z) < c, p0, p0 + (1 - p0) * np.clip(Fw, 0, 1))


# ------------------------------------------------------- Hessian / Wald on log
def hessian(f, p, rel=1e-4):
    p = np.asarray(p, float); k = len(p); H = np.zeros((k, k))
    h = rel * np.maximum(1.0, np.abs(p))
    for i in range(k):
        for j in range(i, k):
            ei = np.zeros(k); ej = np.zeros(k); ei[i] = h[i]; ej[j] = h[j]
            H[i, j] = H[j, i] = (f(p + ei + ej) - f(p + ei - ej) - f(p - ei + ej) + f(p - ei - ej)) / (4 * h[i] * h[j])
    return H


def wald_log_ci(F, T, level=0.95):
    """Delta-method CI for log z_T (back-transformed); for two-part adds p0 sampling variance."""
    f = nll_fun(F.model, F.x, F.c)
    try:
        cov = np.linalg.inv(hessian(f, F.q))
        if np.any(np.diag(cov) < 0):
            return np.nan, np.nan
    except np.linalg.LinAlgError:
        return np.nan, np.nan
    g = lambda q: np.log(max(return_level(F.model, q, T, F.c, F.p0), 1e-9))
    h = 1e-5
    grad = np.array([(g(F.q + h * e) - g(F.q - h * e)) / (2 * h) for e in np.eye(3)])
    var = grad @ cov @ grad
    if F.model == "twopart" and 0 < F.p0 < 1:
        n = len(F.x)
        gp = (np.log(return_level(F.model, F.q, T, F.c, F.p0 + 1e-5)) - np.log(return_level(F.model, F.q, T, F.c, F.p0 - 1e-5))) / 2e-5
        var += gp ** 2 * F.p0 * (1 - F.p0) / n
    if not np.isfinite(var) or var < 0:
        return np.nan, np.nan
    zc = stats.norm.ppf(0.5 + level / 2); m = g(F.q)
    return float(np.exp(m - zc * np.sqrt(var))), float(np.exp(m + zc * np.sqrt(var)))


def profile_rl_ci(F, T, level=0.95, max_mult=100.0, quick=False):
    """Profile-likelihood CI for z_T: for fixed z, maximise over (log sigma, xi) with mu implied."""
    crit = stats.chi2.ppf(level, 1) / 2
    f = nll_fun(F.model, F.x, F.c)
    zhat = F.rl(T)

    def mu_given(z, ls, xi):
        if F.model in ("std", "cens", "twopart"):
            pe = 1 / T if F.model != "twopart" else 1 / (T * (1 - F.p0))
            y = -np.log(1 - pe)
            return z - np.exp(ls) * (y ** (-xi) - 1) / xi if abs(xi) > EPS else z + np.exp(ls) * np.log(y)
        # two-part: solve return_level(mu) = z by root finding in mu
        g = lambda mu: return_level("tptrunc", [mu, ls, xi], T, F.c, F.p0) - z
        lo, hi = F.mu - 50 * np.exp(ls) - abs(z), F.mu + 50 * np.exp(ls) + abs(z)
        try:
            return brentq(g, lo, hi, xtol=1e-8)
        except Exception:
            return np.nan

    def prof(z, start):
        def obj(r):
            ls, xi = r
            mu = mu_given(z, ls, xi)
            return BIG if not np.isfinite(mu) else f([mu, ls, xi])
        best = None
        sts = [start, [F.q[1], F.q[2]]] if quick else [start, [F.q[1], F.q[2]], [F.q[1], F.q[2] + 0.25], [F.q[1], F.q[2] - 0.25]]
        for st in sts:
            r = minimize(obj, st, method="Nelder-Mead", options={"xatol": 1e-6 if quick else 1e-7, "fatol": 1e-7 if quick else 1e-8, "maxiter": 4000})
            if best is None or r.fun < best.fun:
                best = r
        return best.fun, best.x

    def search(d):
        start = [F.q[1], F.q[2]]; prev = zhat; z = zhat
        step = (0.05 if quick else 0.03) * abs(zhat) + 1e-3
        for _ in range(300):
            z = z + d * step
            if d < 0 and z <= 1e-6:
                return 0.0
            if d > 0 and z > max_mult * zhat:
                return np.inf
            v, start = prof(z, start)
            if v - F.nll > crit:
                lo, hi = prev, z
                for _ in range(16 if quick else 30):
                    mid = 0.5 * (lo + hi); vm, _ = prof(mid, start)
                    if vm - F.nll > crit: hi = mid
                    else: lo = mid
                return 0.5 * (lo + hi)
            prev = z; step *= (1.2 if quick else 1.15)
        return np.inf if d > 0 else 0.0

    return search(-1), search(+1)


def profile_xi_ci(F, level=0.95, span=1.6, ngrid=161):
    crit = stats.chi2.ppf(level, 1) / 2
    f = nll_fun(F.model, F.x, F.c)

    def prof(xi, st):
        g = lambda r: f([r[0], r[1], xi])
        best = None
        for s0 in [st, [F.q[0], F.q[1]]]:
            r = minimize(g, s0, method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-9, "maxiter": 6000})
            if best is None or r.fun < best.fun: best = r
        return best.fun, best.x
    out = []
    for d in (-1, 1):
        st = [F.q[0], F.q[1]]; res = np.nan
        for xi in F.xi + d * np.linspace(0, span, ngrid)[1:]:
            v, st = prof(xi, st)
            if v - F.nll > crit:
                res = xi; break
        out.append(res)
    return tuple(out)


# ----------------------------------------------------------------- L-moments
def sample_lmoments(x):
    x = np.sort(np.asarray(x, float)); n = len(x); j = np.arange(1, n + 1)
    b0 = x.mean()
    b1 = np.sum((j - 1) / (n - 1) * x) / n
    b2 = np.sum((j - 1) * (j - 2) / ((n - 1) * (n - 2)) * x) / n
    b3 = np.sum((j - 1) * (j - 2) * (j - 3) / ((n - 1) * (n - 2) * (n - 3)) * x) / n
    l1, l2 = b0, 2 * b1 - b0
    l3 = 6 * b2 - 6 * b1 + b0
    l4 = 20 * b3 - 30 * b2 + 12 * b1 - b0
    return l1, l2, l3 / l2, l4 / l2


def fit_lmom(x):
    from scipy.special import gamma
    l1, l2, t3, _ = sample_lmoments(x)
    cc = 2 / (3 + t3) - np.log(2) / np.log(3)
    k = 7.8590 * cc + 2.9554 * cc ** 2
    if abs(k) < 1e-6:
        s = l2 / np.log(2); return np.array([l1 - 0.5772 * s, np.log(s), 0.0])
    s = l2 * k / ((1 - 2 ** (-k)) * gamma(1 + k))
    return np.array([l1 - s * (1 - gamma(1 + k)) / k, np.log(s), -k])


# ----------------------------------------------------------- GOF (full model)
def ad_uniform(u):
    u = np.clip(np.sort(u), 1e-12, 1 - 1e-12); n = len(u); i = np.arange(1, n + 1)
    return -n - np.mean((2 * i - 1) * (np.log(u) + np.log(1 - u[::-1])))


def ks_uniform(u):
    u = np.sort(u); n = len(u); i = np.arange(1, n + 1)
    return max(np.max(i / n - u), np.max(u - (i - 1) / n))


def randomized_pit(F, x, rng):
    """PIT of the full annual model; censored/dry years get U(0, F(c))."""
    x = np.asarray(x, float)
    if F.model == "std":
        return np.clip(gev_cdf(x, *F.theta), 0, 1)
    u = np.empty(len(x))
    obs = x >= F.c
    if F.model == "cens":
        Gc = float(gev_cdf(F.c, *F.theta))
        u[obs] = gev_cdf(x[obs], *F.theta)
        u[~obs] = rng.uniform(0, Gc, (~obs).sum())
    elif F.model == "twopart":
        # TP-C as a mixture: dry point mass p0 plus an untruncated wet-year GEV, so
        # P(Z < c) = p0 + (1 - p0) G(c) under the fitted model
        Gc = float(gev_cdf(F.c, *F.theta))
        u[obs] = annual_cdf(F.model, F.q, x[obs], F.c, F.p0)
        u[~obs] = rng.uniform(0, F.p0 + (1 - F.p0) * Gc, (~obs).sum())
    else:
        u[obs] = annual_cdf(F.model, F.q, x[obs], F.c, F.p0)
        u[~obs] = rng.uniform(0, F.p0, (~obs).sum()) if F.p0 > 0 else 0
    return u


def cpa_jump(F):
    """Probability that the fitted TP-C mixture assigns below c in excess of p0: (1 - p0) G(c)."""
    return (1 - F.p0) * float(gev_cdf(F.c, *F.theta))


def simulate(F, n, rng):
    """Simulate annual maxima from a fitted model (dry years drawn as U(0, c))."""
    if F.model == "std":
        return gev_rvs(*F.theta, n, rng)
    if F.model == "cens":
        x = gev_rvs(*F.theta, n, rng)
        low = x < F.c
        x[low] = rng.uniform(0, F.c, low.sum())
        return x
    nd = rng.binomial(n, F.p0)
    if F.model == "twopart":
        # TP-C mixture: wet years drawn from the untruncated GEV (values below c stay sub-threshold)
        wet = gev_rvs(*F.theta, n - nd, rng)
        neg = wet < 0
        wet[neg] = rng.uniform(0, F.c, neg.sum())
        return np.concatenate([rng.uniform(0, F.c, nd), wet])
    Gc = float(gev_cdf(F.c, *F.theta))   # TP-T wet years: GEV conditioned on exceeding c
    u = rng.uniform(Gc, 1, n - nd)
    return np.concatenate([rng.uniform(0, F.c, nd), gev_ppf(u, *F.theta)])


# ------------------------------------------------------------- trend tests
def pettitt(x):
    x = np.asarray(x, float); n = len(x)
    U = np.array([np.sum(np.sign(x[t + 1:, None] - x[None, :t + 1])) for t in range(n - 1)])
    k = int(np.argmax(np.abs(U))); K = float(np.abs(U[k]))
    return k, K, min(1.0, 2 * np.exp(-6 * K ** 2 / (n ** 3 + n ** 2)))
