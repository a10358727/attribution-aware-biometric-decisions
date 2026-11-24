"""E064 gate G-S2 (synthetic inputs only; no score file is read)."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE.parents[3]))
import session_m2 as M                           # noqa: E402
from experiments.tcss_lib import core as C       # noqa: E402


def _ru(n=60, seed=0):
    rng = np.random.default_rng(seed)
    r = rng.random(n); u = rng.random(n) * (1 - r)
    r[:3] = [0.0, 1.0, 0.5]; u[:3] = [0.0, 0.0, 0.5]
    return r, u


def test_rho0_equals_m0_core():
    r, u = _ru()
    for K in M.K_GRID:
        for a, b in zip(M.m2(r, u, K, 0.0), C.session_legit(r, u, K)):
            assert np.allclose(a, b, rtol=0, atol=1e-14)


def test_exact_enumeration():
    r, u = _ru(25, 1)
    for K in M.K_GRID:
        for rho in M.RHO_GRID + (0.1, 0.9):
            L, G, N = M.m2(r, u, K, rho)
            for i in range(len(r)):
                e = M.enumerate_m2(r[i], u[i], K, rho)
                assert abs(L[i] - e[0]) < 1e-12 and abs(G[i] - e[1]) < 1e-12 and abs(N[i] - e[2]) < 1e-12


def test_rho1_closed_form_and_probabilities():
    r, u = _ru(40, 2)
    for K in M.K_GRID:
        L, G, N = M.m2(r, u, K, 1.0)
        assert np.allclose(L, r) and np.allclose(G, u) and np.allclose(N, 1 + (K - 1) * r)
        for rho in M.RHO_GRID:
            L, G, N = M.m2(r, u, K, rho)
            acc = 1 - L - G                       # P(session ends in ACCEPT)
            assert np.all(acc > -1e-12) and np.all(L >= -1e-15) and np.all(N <= K + 1e-12)


def test_monotone_in_rho():
    r, u = _ru(40, 3)
    for K in (2, 3, 5):
        Ls = [M.m2(r, u, K, rho)[0] for rho in M.RHO_GRID]
        assert all(np.all(b >= a - 1e-15) for a, b in zip(Ls, Ls[1:]))


def test_monte_carlo_crosscheck():
    rng = np.random.default_rng(4)
    r, u, K, rho, n = 0.3, 0.2, 3, 0.5, 400_000
    p = r + rho * (1 - r); su = u / (1 - r)
    L = G = N = 0
    first = rng.random(n); later = rng.random((n, K)); split = rng.random((n, K))
    for i in range(n):
        for j in range(K):
            N += 1
            if j == 0:
                o = "R" if first[i] < r else ("U" if first[i] < r + u else "A")
            else:
                o = "R" if later[i, j] < p else ("U" if split[i, j] < su else "A")
            if o != "R":
                G += o == "U"; break
        else:
            L += 1
    eL, eG, eN = M.m2(np.array([r]), np.array([u]), K, rho)
    assert abs(L / n - eL[0]) < 4e-3 and abs(G / n - eG[0]) < 4e-3 and abs(N / n - eN[0]) < 6e-3


def test_attackers():
    P = np.array([0.1, 0.0, 0.3]); V = np.array([0.2, 0.0, 0.1]); q = 0.9
    assert np.allclose(M.pmp_fresh(P, V, q, 1), M.pmp_same(P, V, q))
    assert np.all(M.pmp_fresh(P, V, q, 3) >= M.pmp_same(P, V, q) - 1e-15)
    # pi4 identity used by G-STRUCT: moving mass w from ACCEPT to STEP-UP changes fresh PMP by -q*w*sum_{j<K} s^j
    w = np.array([0.05, 0.0, 0.1]); s = 1 - P - V
    for K in (1, 3, 5):
        d = M.pmp_fresh(P - w, V + w, q, K) - M.pmp_fresh(P, V, q, K)
        assert np.allclose(d, -q * w * sum(s ** j for j in range(K)))
