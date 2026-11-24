"""E071 gate G-S (synthetic inputs only; no image, dump or checkpoint is read)."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE.parents[3]))
import run_e071 as D   # noqa: E402


def _toy():
    ids = np.array(["a", "a", "a", "b", "b", "c"])
    pid = np.array([f"p{i}" for i in range(6)])
    return ids, pid


def test_pair_masks_exclude_own_image_and_split_fingers():
    ids, pid = _toy()
    g, i = D.pair_masks(ids, ids, pid, pid)
    assert not g[np.arange(6), np.arange(6)].any()            # own image never genuine
    assert g.sum() == 3 * 2 + 2 * 1                           # a: 3x2 ordered pairs, b: 2x1, c: none
    assert (g & i).sum() == 0 and (i == (ids[:, None] != ids[None, :])).all()


def test_tau_and_rates_against_bruteforce():
    rng = np.random.default_rng(1)
    ids, pid = _toy()
    S = rng.uniform(-1, 1, (6, 6))
    g, i = D.pair_masks(ids, ids, pid, pid)
    t = D.tau_from_pairs(S, i, 0.25)
    assert t == np.quantile(S[i], 0.75, method="higher")
    m, f = D.per_probe_rates(S, g, i, t)
    for r in range(6):
        gs = S[r][g[r]]; im = S[r][i[r]]
        if len(gs):
            assert np.isclose(m[r], (gs > t).mean())
        else:
            assert np.isnan(m[r])
        assert np.isclose(f[r], (im > t).mean())


def test_validation_fmr_close_to_nominal_on_large_sample():
    rng = np.random.default_rng(2)
    S = rng.normal(size=(400, 400)); imp = np.ones_like(S, bool); np.fill_diagonal(imp, False)
    t = D.tau_from_pairs(S, imp, 1e-2)
    assert abs((S[imp] > t).mean() - 1e-2) < 1e-3


def test_finger_max_and_mt_rates():
    ids, pid = _toy()
    S = np.arange(36, dtype=float).reshape(6, 6) / 36
    fingers = np.unique(ids); fidx = np.searchsorted(fingers, ids)
    excl = pid[:, None] == pid[None, :]
    FM = D.finger_max(S, ids, fingers, exclude=excl)
    assert np.isclose(FM[0, 0], S[0, 2])                       # own image (col 0) excluded, max over cols 1-2
    assert FM[5, 2] == -np.inf                                 # singleton finger c has no other template
    m, f = D.mt_rates(FM, fidx, 0.5)
    assert np.isnan(m[5])
    assert np.isclose(f[0], np.mean([FM[0, 1] > 0.5, FM[0, 2] > 0.5]))


def test_eer_hist_matches_direct_eer():
    rng = np.random.default_rng(3)
    gen = rng.normal(0.5, 0.15, 4000).clip(-1, 0.999); imp = rng.normal(0.0, 0.15, 40000).clip(-1, 0.999)
    S = np.concatenate([gen, imp])[None, :]
    Hg = D.row_hist(S, np.r_[np.ones(4000, bool), np.zeros(40000, bool)][None, :])
    Hi = D.row_hist(S, np.r_[np.zeros(4000, bool), np.ones(40000, bool)][None, :])
    e = D.eer_from_hist(Hg.astype(float), Hi.astype(float))[0]
    ts = np.linspace(-1, 1, 4001)
    fnr = np.array([(gen <= t).mean() for t in ts]); fpr = np.array([(imp > t).mean() for t in ts])
    k = np.argmin(np.abs(fnr - fpr))
    assert abs(e - (fnr[k] + fpr[k]) / 2) < 2e-3


def test_eer_hist_weighting_is_linear_in_probe_weights():
    rng = np.random.default_rng(4)
    S = rng.uniform(-1, 1, (5, 50)); gm = rng.uniform(size=(5, 50)) < 0.3; im = ~gm
    Hg, Hi = D.row_hist(S, gm), D.row_hist(S, im)
    assert np.allclose(Hg.sum(1), 1) and np.allclose(Hi.sum(1), 1)
    w = np.array([[2, 0, 1, 1, 0]], float)                     # duplicating probe 0 == weight 2
    S2 = np.concatenate([S[[0]], S[[0]], S[[2]], S[[3]]]); gm2 = np.concatenate([gm[[0]], gm[[0]], gm[[2]], gm[[3]]])
    a = D.eer_from_hist(w @ Hg, w @ Hi)
    b = D.eer_from_hist(D.row_hist(S2, gm2).sum(0, keepdims=True), D.row_hist(S2, ~gm2).sum(0, keepdims=True))
    assert np.allclose(a, b)


def test_synthetic_accept_uniform_claim():
    ids = np.array(["a", "a", "b"]); fingers = np.unique(ids)
    S = np.array([[0.9, 0.1, 0.9], [0.1, 0.1, 0.1]])
    a = D.synthetic_accept(S, ids, fingers, 0.5)
    assert np.allclose(a, [(0.5 + 1.0) / 2, 0.0])


def test_cascade_fail():
    rej = np.array([1.0, 0.0, 0.0]); m = np.array([1.0, 1.0, 0.25])
    assert np.allclose(D.cascade_fail(rej, m), [1.0, 0.0, 0.75])
