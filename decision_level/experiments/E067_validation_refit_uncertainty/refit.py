"""E067 validation-refitting uncertainty (post hoc; protocol docs/experiments/E067_validation_refit_uncertainty.md).

PLANNED, NOT FROZEN. This module only resamples VALIDATION inputs and refits the frozen E061 F1 rules
(`tcss_lib.core.fit_policies`, `thr_e023`). It never opens a score dump itself; callers pass arrays.

Replicate b: draw validation real SUBJECTS with replacement (a drawn subject brings all its validation captures,
paired across the degraded/restored conditions), and draw development-generator synthetic images with replacement;
refit every policy parameter on the replicate. The spread of the refitted parameters (and, after an authorised
test read, of the test decisions they induce) is the uncertainty the E060/E061 intervals condition away.
"""
from __future__ import annotations

import numpy as np

from experiments.tcss_lib import core as C

PARAMS = ("tau0", "tau1", "q_a", "tau_a", "tau_b", "tau_u")


def cluster_index(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Row indices of a cluster bootstrap draw over the unique values of `groups`."""
    uniq, inv = np.unique(groups, return_inverse=True)
    rows = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    pick = rng.integers(0, len(uniq), len(uniq))
    return np.concatenate([rows[k] for k in pick])


def refit_replicates(v0, v1, w0, w1, subj_real, B: int, seed: int) -> dict:
    """v0/v1: paired validation real scores (degraded/restored) with subject labels subj_real;
    w0/w1: paired development-generator synthetic scores. Returns {param: (B,) array} plus 'finite'."""
    v0, v1, w0, w1 = (np.asarray(a, np.float64) for a in (v0, v1, w0, w1))
    assert len(v0) == len(v1) == len(subj_real) and len(w0) == len(w1)
    ss = np.random.SeedSequence(seed).spawn(2)
    rs, rw = np.random.default_rng(ss[0]), np.random.default_rng(ss[1])
    out = {k: np.empty(B) for k in PARAMS}; out["finite"] = np.empty(B, bool)
    for b in range(B):
        i = cluster_index(np.asarray(subj_real), rs)
        j = rw.integers(0, len(w0), len(w0))
        f = C.fit_policies(v0[i], v1[i], w0[j], w1[j])
        for k in PARAMS:
            out[k][b] = f[k]
        out["finite"][b] = f["finite"]
    return out


def shared_replicates(keys: dict, subj_real, n_dev: int, B: int, seed: int) -> dict:
    """Protocol v3 replicate rule: ONE validation draw per replicate, shared by every key.

    keys: {key: (v0, v1, w0, w1)} with real rows aligned by val_patch_id across keys (same order as subj_real) and
    development rows aligned likewise (length n_dev). Returns {key: {param: (B,) array, 'finite': (B,) bool}}."""
    subj_real = np.asarray(subj_real)
    ss = np.random.SeedSequence(seed).spawn(2)
    rs, rw = np.random.default_rng(ss[0]), np.random.default_rng(ss[1])
    out = {k: {**{p: np.empty(B) for p in PARAMS}, "finite": np.empty(B, bool)} for k in keys}
    for b in range(B):
        i = cluster_index(subj_real, rs)                 # same subjects for every key
        j = rw.integers(0, n_dev, n_dev)                 # same dev images for every key
        for k, (v0, v1, w0, w1) in keys.items():
            assert len(v0) == len(subj_real) and len(w0) == n_dev
            f = C.fit_policies(np.asarray(v0)[i], np.asarray(v1)[i], np.asarray(w0)[j], np.asarray(w1)[j])
            for p in PARAMS:
                out[k][p][b] = f[p]
            out[k]["finite"][b] = f["finite"]
    return out


def summarise(rep: dict, frozen: dict, level: float = 0.95) -> dict:
    """Percentile spread of each refitted parameter and the share of replicates equal to the frozen value."""
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    s = {}
    for k in PARAMS:
        x = rep[k][np.isfinite(rep[k])]
        s[k] = {"frozen": frozen[k], "n_finite": int(x.size),
                "lo": float(np.quantile(x, lo)) if x.size else None,
                "hi": float(np.quantile(x, hi)) if x.size else None,
                "share_equal_frozen": float(np.mean(rep[k] == frozen[k]))}
    return s
