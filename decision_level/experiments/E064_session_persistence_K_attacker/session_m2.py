"""E064 session models beyond E063 (post hoc; protocol docs/experiments/E064_session_persistence_K_attacker.md).

PLANNED, NOT FROZEN. Synthetic-input tests only (tests/). No score file is read by this module.

M2 (first-order persistence of the legitimate failure cause). Per credential, r = R/n and u = U/n as in M0.
The first attempt draws REJECT / STEP-UP / ACCEPT with probabilities r / u / 1-r-u. After a REJECT, the next attempt
is REJECT with probability p = r + rho*(1-r); otherwise it is STEP-UP or ACCEPT in the M0 proportion u/(1-r).
rho = 0 reproduces M0 exactly; rho = 1 means a rejected claimant is rejected on every retry (retry has no value).
rho is NOT identifiable from the corpus (captures are not sessions); it is a sensitivity parameter on a fixed grid.

    L = r p^(K-1)
    N = 1 + r * sum_{j=0}^{K-2} p^j
    G = u + r * sum_{j=1}^{K-1} p^(j-1) (1-p) * u/(1-r)        (u/(1-r) := 0 when r = 1)

Attacker models (synthetic decisions only; identical under every legitimate model):
    fresh   : a new synthetic image per attempt from the pooled unseen generators (E061/E063 primary)
              PMP = sum_{j<K} (1-P-V)^j [P + V(1-q)]
    same    : one image reused on every attempt: PMP = P + V(1-q)
    worstgen: fresh draws within ONE unseen generator per (cell, K), chosen inside each bootstrap replicate as the
              argmax of pi0's fresh PMP averaged over the two arms, then used for every arm and policy (protocol v4;
              implemented in run_e064.worst_idx). A worst case over a pre-listed set, not an adaptive attacker.
with P = synthetic ACCEPT rate, V = synthetic STEP-UP rate, q = step-up check's attacker stop probability.
"""
from __future__ import annotations

import itertools

import numpy as np

RHO_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
K_GRID = (1, 2, 3, 4, 5)


def m2(r, u, K: int, rho: float):
    r = np.asarray(r, np.float64); u = np.asarray(u, np.float64)
    if not (0.0 <= rho <= 1.0) or K < 1:
        raise ValueError("rho in [0,1], K >= 1")
    p = r + rho * (1.0 - r)
    share_u = np.divide(u, 1.0 - r, out=np.zeros_like(u), where=(1.0 - r) > 0)
    L = r * p ** (K - 1)
    N = 1.0 + r * sum(p ** j for j in range(K - 1)) if K > 1 else np.ones_like(r)
    G = u + r * sum(p ** (j - 1) * (1.0 - p) * share_u for j in range(1, K)) if K > 1 else u.copy()
    return L, G, N


def pmp_fresh(P, V, q: float, K: int):
    P = np.asarray(P, np.float64); V = np.asarray(V, np.float64)
    return sum((1.0 - P - V) ** j for j in range(K)) * (P + V * (1.0 - q))


def pmp_same(P, V, q: float):
    return np.asarray(P, np.float64) + np.asarray(V, np.float64) * (1.0 - q)


def cost(L, G, N, PMP, cA: float, cU: float, cT: float):
    """Session cost J = c_A*PMP + L + c_U*G + c_T*(N-1)  (E061 definition)."""
    return cA * PMP + L + cU * G + cT * (N - 1.0)


def enumerate_m2(r: float, u: float, K: int, rho: float):
    """Exact oracle: sum over every outcome sequence of the M2 chain (test use only)."""
    a = 1.0 - r - u
    p = r + rho * (1.0 - r)
    su = u / (1.0 - r) if r < 1 else 0.0
    L = G = N = 0.0
    for seq in itertools.product("RUA", repeat=K):
        prob, prev_R = 1.0, None
        att, ended = 0, None
        for j, o in enumerate(seq):
            if j == 0:
                pr = {"R": r, "U": u, "A": a}[o]
            else:
                pr = {"R": p, "U": (1 - p) * su, "A": (1 - p) * (1 - su)}[o]
            prob *= pr
            att += 1
            if o != "R":
                ended = o
                break
        # only count each truncated path once: require the remaining symbols to be a fixed filler
        tail = seq[att:]
        if any(t != "A" for t in tail):
            continue
        L += prob * (ended is None)
        G += prob * (ended == "U")
        N += prob * att
    return L, G, N
