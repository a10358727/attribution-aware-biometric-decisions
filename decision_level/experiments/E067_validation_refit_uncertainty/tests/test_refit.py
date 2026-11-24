"""E067 gate G-S3 (synthetic inputs only; no score file is read)."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE.parents[3]))
import refit as R                                 # noqa: E402
from experiments.tcss_lib import core as C        # noqa: E402


def _data(seed=0, n_subj=30, per=8, n_syn=400):
    rng = np.random.default_rng(seed)
    subj = np.repeat(np.arange(n_subj), per)
    v0 = rng.normal(0, 1, subj.size) + rng.normal(0, .3, n_subj)[subj]
    v1 = v0 + rng.normal(.2, .3, subj.size)
    w0 = rng.normal(3, 1, n_syn); w1 = w0 + rng.normal(.1, .3, n_syn)
    return v0, v1, w0, w1, subj


def test_cluster_index_keeps_whole_subjects():
    g = np.repeat(np.arange(10), 3); rng = np.random.default_rng(1)
    idx = R.cluster_index(g, rng)
    _, c = np.unique(g[idx], return_counts=True)
    assert idx.size == g.size and np.all(c % 3 == 0)


def test_replicates_deterministic_and_shaped():
    v0, v1, w0, w1, s = _data()
    a = R.refit_replicates(v0, v1, w0, w1, s, B=20, seed=670067)
    b = R.refit_replicates(v0, v1, w0, w1, s, B=20, seed=670067)
    for k in R.PARAMS:
        assert a[k].shape == (20,) and np.array_equal(a[k], b[k], equal_nan=True)


def test_spread_covers_frozen_threshold():
    v0, v1, w0, w1, s = _data(2, n_subj=60)
    frozen = C.fit_policies(v0, v1, w0, w1)
    rep = R.refit_replicates(v0, v1, w0, w1, s, B=200, seed=3)
    sm = R.summarise(rep, frozen)
    assert sm["tau0"]["lo"] <= frozen["tau0"] <= sm["tau0"]["hi"]


def test_shared_draw_is_common_across_keys():
    v0, v1, w0, w1, s = _data(4)
    keys = {"a": (v0, v1, w0, w1), "b": (v0, v1, w0, w1), "c": (v0 + 0.5, v1 + 0.5, w0, w1)}
    out = R.shared_replicates(keys, s, len(w0), B=30, seed=5)
    for p in R.PARAMS:   # identical inputs -> identical refits only if the draw is shared
        assert np.array_equal(out["a"][p], out["b"][p], equal_nan=True)
    assert np.allclose(out["c"]["tau0"], out["a"]["tau0"] + 0.5)   # shift-equivariant under the same draw
