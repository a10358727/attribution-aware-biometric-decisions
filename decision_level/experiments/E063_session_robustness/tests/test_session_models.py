"""Gate G-S (synthetic inputs only; no score file is read)."""
import itertools
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE.parents[3]))
import session_models as S                      # noqa: E402
from experiments.tcss_lib import core as C      # noqa: E402


def test_m0_equals_core():
    rng = np.random.default_rng(1)
    r = rng.random(50); u = rng.random(50) * (1 - r)
    for k in (1, 3, 5):
        for a, b in zip(S.m0(r, u, k), C.session_legit(r, u, k)):
            assert np.allclose(a, b, atol=0, rtol=0)


def test_m1_brute_force_all_small_multisets():
    for n in range(1, 7):
        for nA, nU in itertools.product(range(n + 1), repeat=2):
            nR = n - nA - nU
            if nR < 0:
                continue
            outs = ["A"] * nA + ["U"] * nU + ["R"] * nR
            for K in (1, 2, 3):
                L, G, N, el = S.m1(np.array([nR]), np.array([nU]), np.array([n]), K)
                if n < K:
                    assert not el[0] and np.isnan(L[0]); continue
                bl, bg, bn = S.brute_force_m1(outs, K)
                assert abs(L[0] - bl) < 1e-12 and abs(G[0] - bg) < 1e-12 and abs(N[0] - bn) < 1e-12, (outs, K)


def test_m1_without_stepup_equals_e060():
    rng = np.random.default_rng(2)
    n = rng.integers(1, 12, 200); R = np.array([rng.integers(0, x + 1) for x in n])
    for K in (3, 5):
        L, G, N, el = S.m1(R, np.zeros_like(R), n, K)
        den, att, el2 = C.without_replacement(R, n, K)
        assert (el == el2).all()
        assert np.allclose(L[el], den[el], atol=1e-14) and np.allclose(N[el], att[el], atol=1e-14)
        assert np.allclose(G[el], 0)


def test_m1_large_n_approaches_m0():
    n = np.array([100000]); R = np.array([30000]); U = np.array([10000])
    L, G, N, _ = S.m1(R, U, n, 3)
    l0, g0, n0 = S.m0(R / n, U / n, 3)
    assert abs(L[0] - l0[0]) < 1e-4 and abs(G[0] - g0[0]) < 1e-4 and abs(N[0] - n0[0]) < 1e-4


def test_session_probabilities_sum_to_one():
    # P(accept) + P(step-up) + P(denial) = 1 under both models
    rng = np.random.default_rng(3)
    n = rng.integers(3, 10, 300); U = np.array([rng.integers(0, x + 1) for x in n])
    R = np.array([rng.integers(0, n[i] - U[i] + 1) for i in range(len(n))])
    for K in (1, 3):
        L, G, N, el = S.m1(R, U, n, K)
        # accept probability via brute force identity: sum_j a_j * A/(n-j)
        A = n - R - U; a = np.ones(len(n)); Pacc = np.zeros(len(n))
        for j in range(K):
            Pacc += a * A / (n - j); a = a * np.clip(R - j, 0, None) / (n - j)
        assert np.allclose(Pacc + G + L, 1.0, atol=1e-12)
        r, u = R / n, U / n
        l0, g0, _ = S.m0(r, u, K)
        p0 = sum(r ** j for j in range(K)) * (1 - r - u)
        assert np.allclose(p0 + g0 + l0, 1.0, atol=1e-12)


def test_counts_integer():
    cred = np.array([0, 0, 1, 2, 2, 2])
    assert (S.counts(np.array([1, 0, 1, 1, 1, 0]), cred, 3) == [1, 1, 2]).all()
