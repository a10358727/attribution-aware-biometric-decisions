"""G-S synthetic unit tests for experiments/c5_lib/stats.py (C5 protocol §5). No real data."""
import itertools
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.c5_lib import stats as S  # noqa: E402


def test_phi_hand():
    k = np.array([2., 0.]); E = np.array([1., 1.]); V = np.array([0.5, 0.5])
    assert abs(S.phi(k, E, V) - 2.0) < 1e-12


def test_phi_homogeneous_near_one():
    rng = np.random.default_rng(1)
    F, nf = 400, 8
    unit = np.repeat(np.arange(F), nf); g = rng.integers(0, 6, F * nf)
    pi_g = np.linspace(0.05, 0.4, 6)
    vals = []
    for _ in range(50):
        y = (rng.random(F * nf) < pi_g[g]).astype(np.uint8)
        vals.append(S.phi_conditioned(y, g, 6, unit, F))
    assert 0.9 < np.mean(vals) < 1.05


def test_phi_detects_heterogeneity():
    rng = np.random.default_rng(2)
    F, nf = 300, 8
    unit = np.repeat(np.arange(F), nf); g = np.zeros(F * nf, int)
    p_f = rng.beta(1, 4, F)
    y = (rng.random(F * nf) < p_f[unit]).astype(np.uint8)
    assert S.phi_conditioned(y, g, 1, unit, F) > 1.5


def test_stratum_rates_batch():
    y = np.array([[1, 0, 1, 1], [0, 0, 0, 1]]); g = np.array([0, 0, 1, 1])
    out = S.stratum_rates(y, g, 2)
    assert np.allclose(out, [[.5, .5, 1, 1], [0, 0, .5, .5]])
    assert np.allclose(S.stratum_rates(y[0], g, 2), out[0])


def test_unbiased_pK_exact():
    for n in range(3, 8):
        for p in (0.1, 0.37, 0.8):
            e = sum(math.comb(n, k) * p ** k * (1 - p) ** (n - k) * S.unbiased_pK(np.array([k]), np.array([n]), 3)[0]
                    for k in range(n + 1))
            assert abs(e - p ** 3) < 1e-12
    assert np.isnan(S.unbiased_pK(np.array([1]), np.array([2]), 3)[0])


def test_bb_moments_recovery():
    rng = np.random.default_rng(3)
    a, b = 0.8, 6.0
    n = rng.integers(3, 12, 20000); p = rng.beta(a, b, 20000); k = rng.binomial(n, p)
    ah, bh, m, rho = S.bb_moments(k, n)
    assert abs(m - a / (a + b)) < 0.005 and abs(rho - 1 / (a + b + 1)) < 0.01


def test_bb_moments_weights_rows():
    rng = np.random.default_rng(4)
    n = rng.integers(2, 9, 50); k = rng.binomial(n, 0.2)
    W = np.vstack([np.ones(50), rng.integers(0, 3, 50)])
    a, b, m, r = S.bb_moments(k, n, W)
    a0, b0, m0, r0 = S.bb_moments(k, n)
    assert a.shape == (2,) and abs(a[0] - a0) < 1e-12


def test_eb_pK_numeric():
    from scipy.integrate import quad
    from scipy.stats import beta as B
    a, b, k, n = 0.7, 4.0, 2, 5
    num = quad(lambda p: p ** 3 * B.pdf(p, a + k, b + n - k), 0, 1)[0]
    assert abs(S.eb_pK(k, n, a, b, 3) - num) < 1e-9


def test_cvar_fractional():
    x = np.array([1., 2., 3., 4., 10.])
    W = np.ones((1, 5))
    assert abs(S.wtail_mean(x, W, 0.8)[0] - 10.0) < 1e-12          # top 1 of 5
    assert abs(S.wtail_mean(x, W, 0.7)[0] - (10 + 0.5 * 4) / 1.5) < 1e-12
    assert abs(S.wtail_mean(x, W, 0.0)[0] - x.mean()) < 1e-12
    W2 = np.array([[2., 1, 1, 1, 0]])                               # weights (bootstrap multiplicities)
    assert abs(S.wtail_mean(x, W2, 0.8)[0] - 4.0) < 1e-12           # top 20% of 5 weight = 1 -> x=4
    assert abs(S.wtop_share(x, W, 0.2)[0] - 10 / 20) < 1e-12


def test_wquantile():
    x = np.array([3., 1., 2., 4.]); W = np.ones((1, 4))
    assert S.wquantile(x, W, 0.5)[0] == 2.0 and S.wquantile(x, W, 0.75)[0] == 3.0


def test_subject_matrix():
    A = S.subject_matrix(np.array([0, 0, 1]), 2)
    assert np.allclose(A @ np.array([1., 3., 5.]), [2., 5.])
    A = S.subject_matrix(np.array([0, 0, 1]), 2, valid=np.array([True, False, True]))
    assert np.allclose(A @ np.array([1., 0., 5.]), [1., 5.])


def test_session_phi_bruteforce():
    r, u, K, ph = 0.3, 0.1, 3, 0.2

    def denial(j):                     # probability of ending in denial from attempt j onward
        if j == K:
            return 1.0
        return r * denial(j + 1) + u * ph

    L, G, N = S.session_phi(np.array([r]), np.array([u]), K, ph)
    assert abs(L[0] - denial(0)) < 1e-12
    assert abs(G[0] - u * (1 + r + r ** 2)) < 1e-12 and abs(N[0] - (1 + r + r ** 2)) < 1e-12


def test_glmm_recovery():
    rng = np.random.default_rng(5)
    U, Fper, nf = 60, 6, 8
    sub = np.repeat(np.arange(U), Fper); F = len(sub)
    cred = np.repeat(np.arange(F), nf)
    us = rng.normal(0, 1.0, U); vc = rng.normal(0, 1.0, F)
    X = np.ones((F * nf, 1))
    eta = -1.0 + us[sub][cred] + vc[cred]
    y = (rng.random(F * nf) < 1 / (1 + np.exp(-eta))).astype(float)
    fit = S.glmm_fit(X, y, cred, sub, U)
    assert fit["ok"] and abs(fit["sig_s"] - 1) < 0.35 and abs(fit["sig_c"] - 1) < 0.35
    f0 = S.glmm_fit(X, y, cred, sub, U, use_c=False)
    assert f0["negll"] >= fit["negll"] - 1e-6
