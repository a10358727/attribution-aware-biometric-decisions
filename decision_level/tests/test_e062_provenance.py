"""[Public adaptation: only the sys.path setup differs from the research version.] E062 label-provenance helpers: SYNTHETIC tests only (no label source, image or DIAR file is read)."""
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "E062_quality_conditioned_eval"))
import ra_provenance_check as P  # noqa: E402


def test_tertile_votes_and_reversal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    V, cuts = P.tertile_votes(X)
    for j, s in enumerate(P.STATS):
        counts = np.bincount(V[:, j], minlength=3)
        assert counts.min() >= 99 and counts.max() <= 101
        hi = X[:, j] > cuts[j][1]
        assert (V[hi, j] == (0 if s in P.REVERSED else 2)).all()


def test_training_rule_three_of_four():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 4))
    r = P.reconstruct(X)
    cnt = np.stack([(r["V"] == k).sum(1) for k in range(3)], 1)
    assert (r["train"] == (cnt.max(1) >= 3)).all()
    assert (r["vote"][r["train"]] == cnt.argmax(1)[r["train"]]).all()
    assert (r["vote"][~r["train"]] == -1).all()
    assert ((r["support"] >= 1) & (r["support"] <= P.K)).all()


def test_fixed_cuts_reused():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(300, 4))
    _, cuts = P.tertile_votes(X)
    V2, cuts2 = P.tertile_votes(X + 10.0, cuts)
    assert cuts2 is cuts
    assert (V2[:, 0] == 2).all()          # every value above the old upper cut
