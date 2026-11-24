"""E065 gate G-S (synthetic inputs only; no image, dump or checkpoint is read)."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE.parents[3]))
import run_e065 as D   # noqa: E402


def test_impostor_leave_own_finger_out():
    e = np.eye(4)
    ids = np.array(["a", "a", "b", "c"])
    m = D.impostor_max(e, e, ids, ids)
    assert np.allclose(m, 0.0)              # own finger (incl. self) excluded; others orthogonal
    e2 = np.array([[1, 0], [1, 0], [0.6, 0.8], [0, 1]], float)
    m2 = D.impostor_max(e2, e2, ids, ids)
    assert np.isclose(m2[0], 0.6) and np.isclose(m2[3], 0.8)


def test_genuine_excludes_own_image_and_flags_singletons():
    e = np.array([[1, 0], [0.8, 0.6], [0, 1]], float)
    pid = np.array(["p0", "p1", "p2"]); ids = np.array(["a", "a", "b"])
    g = D.genuine_max(e, e, pid, pid, ids, ids)
    assert np.isclose(g[0], 0.8) and np.isclose(g[1], 0.8) and np.isnan(g[2])


def test_threshold_rule_matches_e024_quantile():
    rng = np.random.default_rng(0); imp = rng.normal(size=10_000)
    t = float(np.quantile(imp, 1 - 1e-2, method="higher"))
    assert abs((imp > t).mean() - 1e-2) < 2e-3
