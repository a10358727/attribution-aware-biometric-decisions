"""E063 legitimate-session retry models (post hoc; frozen in docs/experiments/E063_session_robustness.md).

Per credential f with n_f captures, of which R_f are REJECT and U_f are STEP-UP under a given policy, detector seed
and frozen parameters (the rest ACCEPT). A session makes up to K attempts and ends at the first non-REJECT outcome
(ACCEPT, or STEP-UP -> secondary check), or in denial after K REJECTs.

M0  (E061 primary): each attempt draws a capture uniformly WITH replacement -> r = R/n, u = U/n,
    L = r^K, G = sum_{j<K} r^j u, N = sum_{j<K} r^j   (identical to tcss_lib.core.session_legit).
M0|elig (new control): M0 restricted to credentials with n_f >= K (the population M1 admits).
M1  (E060 S1-M1 extended to step-up): attempts draw WITHOUT replacement over a uniformly random ordering of f's
    captures; credentials with n_f >= K only. With a_j = P(first j draws all REJECT) = prod_{i<j} (R-i)/(n-i):
    L = a_K, G = sum_{j<K} a_j * U/(n-j), N = sum_{j<K} a_j.  With U = 0 this equals
    tcss_lib.core.without_replacement (E060).
"""
from __future__ import annotations

import numpy as np


def m0(r: np.ndarray, u: np.ndarray, K: int):
    geo = sum(r ** j for j in range(K))
    return r ** K, geo * u, geo


def m1(R: np.ndarray, U: np.ndarray, n: np.ndarray, K: int):
    """Closed-form M1 per credential. R, U, n: integer counts (F,). Returns L, G, N, eligible mask.
    Ineligible credentials (n < K) get NaN."""
    R = np.asarray(R, np.float64); U = np.asarray(U, np.float64); n = np.asarray(n, np.float64)
    elig = n >= K
    a = np.ones_like(R)
    L = np.full_like(R, np.nan); G = np.zeros_like(R); N = np.zeros_like(R)
    with np.errstate(divide="ignore", invalid="ignore"):
        for j in range(K):
            G += a * np.where(n - j > 0, U / (n - j), 0.0)
            N += a
            a = a * np.where(n - j > 0, np.clip(R - j, 0, None) / (n - j), 0.0)
    L = np.where(elig, a, np.nan); G = np.where(elig, G, np.nan); N = np.where(elig, N, np.nan)
    return L, G, N, elig


def counts(x_cap: np.ndarray, cred_of_capture: np.ndarray, F: int) -> np.ndarray:
    """Integer per-credential counts of a 0/1 capture-level indicator."""
    c = np.bincount(cred_of_capture, weights=x_cap.astype(np.float64), minlength=F)
    r = np.rint(c)
    assert np.max(np.abs(c - r)) < 1e-9
    return r


def brute_force_m1(outcomes: list[str], K: int):
    """Exhaustive enumeration over all orderings (test oracle; small n only)."""
    import itertools
    perms = list(itertools.permutations(range(len(outcomes))))
    L = G = N = 0.0
    for p in perms:
        att = 0
        for j in range(K):
            att += 1
            o = outcomes[p[j]]
            if o == "A":
                break
            if o == "U":
                G += 1
                break
        else:
            L += 1
        N += att
    m = len(perms)
    return L / m, G / m, N / m
