"""C5 shared statistics (E072, E073). Frozen plan: docs/C5/C5_Research_Protocol.md.

Pure functions on arrays; no file or test-score access here.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp


# ------------------------------------------------------------------ A1: dispersion and conditioned nulls
def phi(k: np.ndarray, E: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Pearson dispersion ratio over units: sum (k-E)^2 / sum V. Works on (..., F) arrays."""
    num = ((k - E) ** 2).sum(axis=-1)
    den = V.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def stratum_rates(y: np.ndarray, g: np.ndarray, G: int) -> np.ndarray:
    """Per-capture fitted probability = rejection rate of the capture's stratum. y (..., n) 0/1, g (n,) ints."""
    n_g = np.bincount(g, minlength=G).astype(float)
    if y.ndim == 1:
        s = np.bincount(g, weights=y.astype(float), minlength=G)
        return (s / np.maximum(n_g, 1))[g]
    D = y.shape[0]
    s = np.zeros((D, G))
    np.add.at(s, (np.repeat(np.arange(D), y.shape[1]), np.tile(g, D)), y.ravel().astype(float))
    return (s / np.maximum(n_g, 1))[:, g]


def unit_sums(x: np.ndarray, unit: np.ndarray, U: int) -> np.ndarray:
    """Sum capture-level x (..., n) into units (..., U)."""
    if x.ndim == 1:
        return np.bincount(unit, weights=x.astype(float), minlength=U)
    M = np.zeros((x.shape[-1], U))
    M[np.arange(x.shape[-1]), unit] = 1.0
    return x @ M


def phi_conditioned(y: np.ndarray, g: np.ndarray, G: int, unit: np.ndarray, U: int,
                    keep: np.ndarray | None = None) -> np.ndarray:
    """Phi of observed or simulated decisions y under the stratum model fitted to y itself.
    keep: optional boolean mask over units (S-A1b)."""
    pi = stratum_rates(y, g, G)
    k = unit_sums(y, unit, U); E = unit_sums(pi, unit, U); V = unit_sums(pi * (1 - pi), unit, U)
    if keep is not None:
        k, E, V = k[..., keep], E[..., keep], V[..., keep]
    return phi(k, E, V)


def simulate_null(pi: np.ndarray, D: int, rng: np.random.Generator) -> np.ndarray:
    """D iid Bernoulli draws of every capture with probabilities pi (n,) -> (D, n) uint8."""
    return (rng.random((D, len(pi))) < pi[None, :]).astype(np.uint8)


# ------------------------------------------------------------------ A2: estimators of p^K
def unbiased_pK(k: np.ndarray, n: np.ndarray, K: int) -> np.ndarray:
    """C(k,K)/C(n,K) (unbiased for p^K under binomial sampling); NaN where n < K."""
    k = np.asarray(k, float); n = np.asarray(n, float)
    out = np.ones(np.broadcast(k, n).shape)
    for j in range(K):
        with np.errstate(divide="ignore", invalid="ignore"):
            out = out * np.clip(k - j, 0, None) / (n - j)
    return np.where(n >= K, out, np.nan)


def bb_moments(k: np.ndarray, n: np.ndarray, w: np.ndarray | None = None):
    """Beta-binomial moment estimator (ANOVA form). k, n (F,) ; w (..., F) unit weights (bootstrap).
    Returns (a, b, m, rho) with leading shape of w."""
    k = np.asarray(k, float); n = np.asarray(n, float)
    if w is None:
        w = np.ones_like(k)
    W = np.asarray(w, float)
    m = (W * k).sum(-1) / (W * n).sum(-1)
    m = np.clip(m, 1e-9, 1 - 1e-9)
    m_ = m[..., None]
    S = (W * (k - n * m_) ** 2 / n).sum(-1)
    base = m * (1 - m)
    num = S - base * W.sum(-1)
    den = base * (W * (n - 1)).sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = np.where(den > 0, num / den, 1e-6)
    rho = np.clip(rho, 1e-6, 0.999)
    t = 1.0 / rho - 1.0
    return m * t, (1 - m) * t, m, rho


def eb_pK(k: np.ndarray, n: np.ndarray, a, b, K: int) -> np.ndarray:
    """Posterior mean of p^K under Beta(a,b) prior: prod_{j<K} (a+k+j)/(a+b+n+j). a,b scalars or (..., 1)."""
    out = 1.0
    for j in range(K):
        out = out * (a + k + j) / (a + b + n + j)
    return out


def eb_mean(k, n, a, b):
    return (a + k) / (a + b + n)


# ------------------------------------------------------------------ B: weighted distribution endpoints
def _sorted_desc(x: np.ndarray, W: np.ndarray):
    order = np.argsort(-x, kind="mergesort")
    return x[order], W[:, order]


def wtail_mean(x: np.ndarray, W: np.ndarray, alpha: float) -> np.ndarray:
    """CVaR_alpha (upper tail): weighted mean of the highest (1-alpha) share of weight, fractional boundary.
    x (U,), W (R, U) -> (R,)."""
    xs, Ws = _sorted_desc(x, W)
    tot = Ws.sum(1)
    t = (1 - alpha) * tot
    cw = np.cumsum(Ws, 1); cm = np.cumsum(Ws * xs[None, :], 1)
    j = np.minimum((cw < t[:, None] - 1e-12).sum(1), len(x) - 1)
    r = np.arange(W.shape[0])
    cwp = np.where(j > 0, cw[r, np.maximum(j - 1, 0)], 0.0)
    cmp_ = np.where(j > 0, cm[r, np.maximum(j - 1, 0)], 0.0)
    mass = cmp_ + (t - cwp) * xs[j]
    return mass / t


def wtop_share(x: np.ndarray, W: np.ndarray, frac: float) -> np.ndarray:
    """Share of total weighted burden carried by the top `frac` of weight (concentration)."""
    tail = wtail_mean(x, W, 1 - frac) * frac * W.sum(1)
    tot = (W * x[None, :]).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(tot > 0, tail / tot, np.nan)


def wquantile(x: np.ndarray, W: np.ndarray, q: float) -> np.ndarray:
    """Smallest x with cumulative weight share >= q (ascending)."""
    order = np.argsort(x, kind="mergesort")
    xs = x[order]; Ws = W[:, order]
    c = np.cumsum(Ws, 1) / Ws.sum(1, keepdims=True)
    j = np.minimum((c < q - 1e-12).sum(1), len(x) - 1)
    return xs[j]


def wmean_rows(x: np.ndarray, W: np.ndarray) -> np.ndarray:
    return (W @ x) / W.sum(1)


def wshare(cond: np.ndarray, W: np.ndarray) -> np.ndarray:
    return (W @ cond.astype(float)) / W.sum(1)


def subject_matrix(subj_of_cred: np.ndarray, U: int, valid: np.ndarray | None = None) -> np.ndarray:
    """A (U, F): credential-average within subject over valid credentials (NaN-safe use: x filled with 0)."""
    F = len(subj_of_cred)
    v = np.ones(F, bool) if valid is None else valid
    A = np.zeros((U, F))
    A[subj_of_cred[v], np.flatnonzero(v)] = 1.0
    cnt = A.sum(1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        A = np.where(cnt > 0, A / cnt, 0.0)
    return A


def loo_range(x: np.ndarray, fn) -> tuple[float, float]:
    """Leave-one-unit-out range of fn(x_minus_i) with unit weights."""
    vals = []
    for i in range(len(x)):
        m = np.ones(len(x), bool); m[i] = False
        vals.append(float(fn(x[m], np.ones((1, m.sum())))[0]))
    return min(vals), max(vals)


# ------------------------------------------------------------------ C-S3
def session_phi(r: np.ndarray, u: np.ndarray, K: int, phi_fail: float):
    """M0 session with a secondary check that a legitimate claimant fails with probability phi_fail
    (failure ends the session in denial): L' = r^K + phi*G; G, N unchanged."""
    geo = sum(r ** j for j in range(K))
    G = geo * u
    return r ** K + phi_fail * G, G, geo


# ------------------------------------------------------------------ A3: nested random-intercept logit
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(15)       # probabilists' Hermite: weight exp(-x^2/2)
GH_LW = np.log(GH_W / math.sqrt(2 * math.pi))


def glmm_negll(theta, X, y, cred, subj_of_cred, U, use_s=True, use_c=True):
    """Negative marginal log-likelihood. theta = [beta (p), log sig_s, log sig_c]."""
    p = X.shape[1]
    beta = theta[:p]
    ss = math.exp(theta[p]) if use_s else 0.0
    sc = math.exp(theta[p + 1]) if use_c else 0.0
    eta = X @ beta                                         # (n,)
    nodes_s = ss * GH_X if use_s else np.zeros(1)
    lw_s = GH_LW if use_s else np.zeros(1)
    nodes_c = sc * GH_X if use_c else np.zeros(1)
    lw_c = GH_LW if use_c else np.zeros(1)
    # capture loglik at each (a, b): (n, A, Bn)
    z = eta[:, None, None] + nodes_s[None, :, None] + nodes_c[None, None, :]
    ll = np.where(y[:, None, None] > 0, -np.logaddexp(0, -z), -np.logaddexp(0, z))
    F = len(subj_of_cred)
    llc = np.zeros((F, ll.shape[1], ll.shape[2]))
    np.add.at(llc, cred, ll)                               # (F, A, Bn)
    lc = logsumexp(llc + lw_c[None, None, :], axis=2)      # (F, A)
    ls = np.zeros((U, lc.shape[1]))
    np.add.at(ls, subj_of_cred, lc)                        # (U, A)
    return -float(logsumexp(ls + lw_s[None, :], axis=1).sum())


def glmm_fit(X, y, cred, subj_of_cred, U, use_s=True, use_c=True, x0=None):
    p = X.shape[1]
    th0 = np.zeros(p + 2) if x0 is None else np.array(x0, float)
    if x0 is None:
        m = np.clip(y.mean(), 1e-3, 1 - 1e-3)
        th0[0] = math.log(m / (1 - m)); th0[p] = th0[p + 1] = math.log(0.5)
    free = [i for i in range(p + 2) if i < p or (i == p and use_s) or (i == p + 1 and use_c)]

    def f(t):
        th = th0.copy(); th[free] = t
        return glmm_negll(th, X, y, cred, subj_of_cred, U, use_s, use_c)
    bounds = [(-15, 15)] * p + [(-6, 3)] * (len(free) - p)
    res = minimize(f, th0[free], method="L-BFGS-B", bounds=bounds)
    th = th0.copy(); th[free] = res.x
    return {"ok": bool(res.success), "negll": float(res.fun), "theta": th,
            "sig_s": math.exp(th[p]) if use_s else 0.0, "sig_c": math.exp(th[p + 1]) if use_c else 0.0}


def lrt_boundary_p(stat: float) -> float:
    """p-value for one variance component on the boundary: 0.5*chi2_0 + 0.5*chi2_1."""
    from scipy.stats import chi2
    if stat <= 0:
        return 1.0
    return 0.5 * float(chi2.sf(stat, 1))


def design(deg_type: np.ndarray, deg_sev: np.ndarray, extra: np.ndarray | None = None):
    """Intercept + type dummies (ref = first) + severity dummies (ref = 1) [+ extra column]."""
    types = sorted(set(deg_type.tolist())); sevs = sorted(set(deg_sev.tolist()))
    cols = [np.ones(len(deg_type))]
    cols += [(deg_type == t).astype(float) for t in types[1:]]
    cols += [(deg_sev == s).astype(float) for s in sevs[1:]]
    if extra is not None:
        cols.append(extra.astype(float))
    return np.stack(cols, 1)
