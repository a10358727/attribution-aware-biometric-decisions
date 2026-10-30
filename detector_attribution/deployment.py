"""Deployment diagnostic vector at a validation-locked threshold.

`thr_at_1pct`, `fast_auc` and the cluster bootstrap `Boot` are copied verbatim from the frozen
estimator of the research code. `deployment_vector` is written for this release: it restates, for one
(detector, seed, condition) and in-memory arrays, the computation that the frozen estimator performs
on per-image score files, so that it can be run on synthetic or user-supplied scores. The frozen
analysis itself is not released because it reads per-image score files that carry participant indices.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def thr_at_1pct(real_scores: np.ndarray) -> float:
    """Identical rule to diar.metrics.threshold_at_real_fpr (quantile, method='higher')."""
    return float(np.quantile(real_scores, 0.99, method="higher"))


def fast_auc(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney U AUC with average ranks for ties (scipy rankdata: vectorised C)."""
    ranks = rankdata(s, method="average")
    npos = float((y == 1).sum()); nneg = float((y == 0).sum())
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


class Boot:
    """Cluster bootstrap over real finger identities + within-generator synthetic resampling.

    Threshold-metric replicate means are computed exactly from draw counts, so no replicate
    is truncated; AUC replicates use explicit indices for the first n_auc draws only.
    """

    def __init__(self, rng, ident_real, real_idx, syn_by_gen, n_boot, n_auc):
        self.real_idx, self.n_boot, self.n_auc = real_idx, n_boot, n_auc
        uniq, inv = np.unique(ident_real, return_inverse=True)
        self.n_f = len(uniq)
        self.memb = [real_idx[inv == k] for k in range(self.n_f)]
        self.size_f = np.array([len(m) for m in self.memb], dtype=np.float64)
        self.C = np.empty((n_boot, self.n_f), dtype=np.int32)
        self.real_auc_idx = []
        for b in range(n_boot):
            d = rng.integers(0, self.n_f, size=self.n_f)
            self.C[b] = np.bincount(d, minlength=self.n_f)
            if b < n_auc:
                self.real_auc_idx.append(np.concatenate([self.memb[k] for k in d]))
        self.n_real_rep = self.C @ self.size_f                      # (n_boot,)
        self.syn_idx_all = np.concatenate([v for v in syn_by_gen.values()])
        self.SC = np.zeros((n_boot, len(self.syn_idx_all)), dtype=np.int32)
        self.syn_auc_idx = []
        off = 0
        picks = {}
        for g, v in syn_by_gen.items():
            n = len(v)
            picks[g] = rng.integers(0, n, size=(n_boot, n))
            for b in range(n_boot):
                self.SC[b, off:off + n] = np.bincount(picks[g][b], minlength=n)
            off += n
        for b in range(n_auc):
            self.syn_auc_idx.append(np.concatenate([syn_by_gen[g][picks[g][b]] for g in syn_by_gen]))
        self.n_syn_rep = float(len(self.syn_idx_all))

    def real_mean(self, flags: np.ndarray) -> np.ndarray:
        per_f = np.array([flags[m].sum() for m in self.memb], dtype=np.float64)
        return (self.C @ per_f) / self.n_real_rep

    def syn_mean(self, flags: np.ndarray) -> np.ndarray:
        return (self.SC @ flags[self.syn_idx_all].astype(np.float64)) / self.n_syn_rep

    def auc_reps(self, s: np.ndarray) -> np.ndarray:
        out = np.empty(self.n_auc)
        for b in range(self.n_auc):
            r, sy = self.real_auc_idx[b], self.syn_auc_idx[b]
            yb = np.concatenate([np.zeros(len(r)), np.ones(len(sy))])
            out[b] = fast_auc(yb, np.concatenate([s[r], s[sy]]))
        return out


def deployment_vector(val_real_anchor: np.ndarray, val_real_cond: np.ndarray,
                      y: np.ndarray, s_anchor: np.ndarray, s_cond: np.ndarray) -> dict:
    """The five quantities of the paper for one condition against its anchor.

    val_real_*: bona-fide VALIDATION scores (anchor; condition) -- they set tau_lock and tau_recal.
    y: test labels (0 bona fide, 1 synthetic); s_anchor / s_cond: test scores in the same order.
    tau_lock is set once on the anchor's validation bona-fide scores at 1 % FPR and reused unchanged.
    """
    thr_locked = thr_at_1pct(val_real_anchor)
    thr_recal = thr_at_1pct(val_real_cond)
    t = lambda s, thr: float((s[y == 1] > thr).mean())
    b = lambda s, thr: float((s[y == 0] > thr).mean())
    return dict(
        d_auc=fast_auc(y, s_cond) - fast_auc(y, s_anchor),
        d_tdr_locked=t(s_cond, thr_locked) - t(s_anchor, thr_locked),
        d_rejection=b(s_cond, thr_locked) - b(s_anchor, thr_locked),
        dtau=thr_recal - thr_locked,
        tdr_recal=t(s_cond, thr_recal),
        thr_locked=thr_locked, thr_recal=thr_recal)
