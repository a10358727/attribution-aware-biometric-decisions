"""Paired normalization-policy contrast: primitives and a reference estimator.

The primitives `thr_at_1pct`, `weighted_thr_at_1pct`, `weighted_auc`, `row_counts`, `pctl` and `pboot`
are copied verbatim from the frozen paired-bootstrap analysis of the research code.
`paired_contrast` is written for this release: a single-cell restatement of the estimand
DDT = dT_fixed - dT_legacy with a paired two-factor bootstrap (training seeds paired across the two
policies, crossed with bona-fide test clusters and within-generator synthetic draws shared by both
policies), for synthetic or user-supplied scores. It is not the frozen analysis, which reads
per-image score files that are not released and adds further secondary analyses.
"""
from __future__ import annotations

import numpy as np


def thr_at_1pct(real_scores: np.ndarray) -> float:
    """The frozen E023 rule (= diar.metrics.threshold_at_real_fpr): quantile 0.99, method 'higher'."""
    return float(np.quantile(real_scores, 0.99, method="higher"))


def weighted_thr_at_1pct(sorted_scores: np.ndarray, W: np.ndarray) -> np.ndarray:
    """thr_at_1pct of each row's multiset: sorted_scores (n,) ascending, W (r, n) multiplicities.
    np.quantile(x, 0.99, 'higher') is the element at 0-based rank ceil(0.99 (N - 1))."""
    cw = np.cumsum(W, axis=1)
    n_tot = cw[:, -1]
    k = np.ceil(0.99 * (n_tot - 1)).astype(np.int64)
    j = (cw < (k + 1)[:, None]).sum(axis=1)            # first position whose cumulative count > k
    return sorted_scores[j]


def weighted_auc(s_real: np.ndarray, s_syn: np.ndarray, Wr: np.ndarray, Ws: np.ndarray) -> np.ndarray:
    """P(syn > real) + 0.5 P(tie) under multiplicity weights, one value per row of Wr (r, n_real)
    and Ws (r, n_syn). With unit weights it is the average-rank Mann-Whitney AUC of E023."""
    order = np.argsort(s_real, kind="mergesort")
    sr = s_real[order]
    CR = np.zeros((Wr.shape[0], len(sr) + 1))
    np.cumsum(Wr[:, order], axis=1, out=CR[:, 1:])
    lo = np.searchsorted(sr, s_syn, side="left")
    hi = np.searchsorted(sr, s_syn, side="right")
    below = CR[:, lo]
    num = (Ws * (below + 0.5 * (CR[:, hi] - below))).sum(axis=1)
    return num / (Ws.sum(axis=1) * CR[:, -1])


def row_counts(rng: np.random.Generator, n_rows: int, n: int) -> np.ndarray:
    """n draws with replacement from n items, per row -> multiplicities (n_rows, n), float64."""
    d = rng.integers(0, n, size=(n_rows, n))
    out = np.zeros((n_rows, n))
    np.add.at(out, (np.repeat(np.arange(n_rows), n), d.ravel()), 1.0)
    return out


def pctl(v: np.ndarray, q) -> tuple[float, float]:
    return float(np.quantile(v, q[0])), float(np.quantile(v, q[1]))


def pboot(v: np.ndarray) -> float:
    """Descriptive two-sided bootstrap proportion; not a test statistic in the decision rule."""
    return float(min(1.0, 2 * min((v >= 0).mean(), (v <= 0).mean())))


def paired_contrast(arms: dict, clusters: np.ndarray, generators: np.ndarray, y: np.ndarray,
                    n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    """DDT and DDB for one (detector, protocol, operator) cell.

    arms: {"legacy": {"thr": (S,), "anchor": (S, n), "cond": (S, n)}, "fixed": {...}} -- per training
          seed, the locked threshold and the test scores of the anchor and the transformed condition.
          Seeds must be paired by index across the two arms.
    clusters: (n,) bona-fide cluster id (e.g. finger identity); ignored for synthetic rows.
    generators: (n,) generator id for synthetic rows; y: (n,) 0 bona fide / 1 synthetic.
    Thresholds stay at their full-sample values, as in the frozen estimator.
    """
    rng = np.random.default_rng(seed)
    S = next(iter(arms.values()))["anchor"].shape[0]
    real = np.flatnonzero(y == 0)
    ids, inv = np.unique(clusters[real], return_inverse=True)
    syn_groups = [np.flatnonzero((y == 1) & (generators == g)) for g in np.unique(generators[y == 1])]

    def deltas(arm, w_real, w_syn, seeds):
        out_t, out_b = [], []
        for s in seeds:
            fa = arm["anchor"][s] > arm["thr"][s]
            fc = arm["cond"][s] > arm["thr"][s]
            out_t.append(((fc[w_syn[0]] - fa[w_syn[0]].astype(float)) * w_syn[1]).sum() / w_syn[1].sum())
            out_b.append(((fc[real] - fa[real].astype(float)) * w_real).sum() / w_real.sum())
        return float(np.mean(out_t)), float(np.mean(out_b))

    full_real = np.ones(len(real))
    full_syn = (np.concatenate(syn_groups), np.ones(sum(len(g) for g in syn_groups)))
    point = {a: deltas(arms[a], full_real, full_syn, range(S)) for a in ("legacy", "fixed")}
    ddt = point["fixed"][0] - point["legacy"][0]
    ddb = point["fixed"][1] - point["legacy"][1]
    reps = np.empty((n_boot, 2))
    for b in range(n_boot):
        seeds = rng.integers(0, S, size=S)
        c = np.bincount(rng.integers(0, len(ids), size=len(ids)), minlength=len(ids))
        w_real = c[inv].astype(float)
        idx = np.concatenate([g[rng.integers(0, len(g), size=len(g))] for g in syn_groups])
        uniq, cnt = np.unique(idx, return_counts=True)
        w_syn = (uniq, cnt.astype(float))
        r = {a: deltas(arms[a], w_real, w_syn, seeds) for a in ("legacy", "fixed")}
        reps[b] = (r["fixed"][0] - r["legacy"][0], r["fixed"][1] - r["legacy"][1])
    q = (alpha / 2, 1 - alpha / 2)
    return dict(DDT=ddt, DDT_ci=pctl(reps[:, 0], q), DDB=ddb, DDB_ci=pctl(reps[:, 1], q), n_boot=n_boot)
