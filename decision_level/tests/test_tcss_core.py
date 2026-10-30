"""Gate G-S for E060/E061: closed-form checks of the shared core on synthetic data only."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # tcss_cpsi/
from experiments.tcss_lib import core as C  # noqa: E402


def test_thr_and_kmax_match_e023_rule():
    rng = np.random.default_rng(0)
    v = rng.normal(size=1775)
    t = C.thr_e023(v)
    assert t == np.sort(v)[math.ceil(0.99 * 1774)]
    assert C.k_max(1775) == 17 == int((v > t).sum())


def test_units_and_per_cred_mean():
    ids = np.array(["selfcollected_88:01_L_I", "selfcollected_88:01_L_I", "selfcollected_88:01_R_T",
                    "selfcollected_88:02_L_I"])
    u = C.make_units(ids)
    assert u.F == 3 and u.U == 2
    assert list(u.n_f()) == [2, 1, 1]
    p = C.per_cred_mean(np.array([1.0, 0.0, 1.0, 0.0]), u)
    assert np.allclose(p, [0.5, 1.0, 0.0])
    assert list(u.subj_names[u.subj_of_cred]) == ["01", "01", "02"]


def test_denial_attempts_closed_form():
    p = np.array([0.0, 0.5, 1.0])
    assert np.allclose(C.denial(p, 3), [0, 0.125, 1])
    assert np.allclose(C.attempts(p, 3), [1, 1.75, 3])
    W = np.array([[1.0, 1.0, 1.0], [2.0, 0.0, 1.0]])
    assert np.allclose(C.wmean(W, C.denial(p, 3)[:, None])[:, 0], [1.125 / 3, 1 / 3])


def test_d10_fractional_boundary_and_null_rows():
    rng = np.random.default_rng(1)
    p = rng.uniform(size=309)
    s = np.sort(p)[::-1]
    expected = (s[:30].sum() + 0.9 * s[30]) / s.sum()
    assert abs(C.d10(p, np.ones((1, 309)))[0] - expected) < 1e-12
    assert abs(C.d10_unit_rows(p[None, :])[0] - expected) < 1e-12
    # integer weights == duplicated credentials
    w = rng.integers(0, 3, size=309).astype(float)
    dup = np.repeat(p, w.astype(int))
    tot = len(dup); t = 0.1 * tot; sd = np.sort(dup)[::-1]; j = int(math.floor(t))
    exp_w = (sd[:j].sum() + (t - j) * sd[j]) / sd.sum()
    assert abs(C.d10(p, w[None, :])[0] - exp_w) < 1e-12
    assert np.isnan(C.d10(np.zeros(10), np.ones((1, 10)))[0])


def test_without_replacement():
    den, att, el = C.without_replacement(np.array([2, 3, 1]), np.array([4, 3, 2]), 3)
    assert list(el) == [True, True, False]
    assert abs(den[0] - 0.0) < 1e-12 and abs(den[1] - 1.0) < 1e-12
    assert abs(att[0] - (1 + 2 / 4 + 1 / 6)) < 1e-12


def test_session_and_pmp():
    r, u = np.array([0.2]), np.array([0.1])
    L, G, N = C.session_legit(r, u, 3)
    assert np.allclose([L[0], G[0], N[0]], [0.008, 0.1 * 1.24, 1.24])
    P, V, q = 0.3, 0.1, 0.9
    stay = 0.6
    assert abs(C.pmp_fresh(P, V, q, 3) - (1 + stay + stay ** 2) * (0.3 + 0.01)) < 1e-12
    assert abs(C.pmp_same(P, V, q) - 0.31) < 1e-12
    assert abs(C.cost_J(0.5, 0.1, 0.2, 1.5, 1, 0.2, 0.02) - (0.5 + 0.1 + 0.04 + 0.01)) < 1e-12


def test_fit_policies_constraints():
    rng = np.random.default_rng(2)
    v0 = rng.normal(size=1775); v1 = v0 + rng.normal(0.3, 0.5, size=1775)
    w0 = rng.normal(1.0, 1, size=1500); w1 = w0 + 0.5
    f = C.fit_policies(v0, v1, w0, w1)
    assert f["val_real_reject_pi0"] <= 0.01 + 1e-12 and f["val_real_reject_pi2"] <= 0.01 + 1e-12
    for row in f["q_a_table"]:
        assert row["val_real_reject"] * 1775 <= C.k_max(1775) + 1e-9
    assert int(round(f["val_real_stepup_pi4"] * 1775)) == math.ceil(0.02 * 1775)
    best = max(f["q_a_table"], key=lambda r: (r["val_dev_tdr"], r["q_a"]))
    assert f["q_a"] == best["q_a"]
    z, rest = C.decide("pi4", f, v0, v1)
    assert (z == C.REJ).sum() == (v0 > f["tau0"]).sum() and (z == C.STEP).sum() == math.ceil(0.02 * 1775)
    z3, rest3 = C.decide("pi3", f, v0, v1)
    assert rest3.sum() == (v0 > f["tau_a"]).sum() and (z3 == C.REJ).sum() <= C.k_max(1775)
    z1, _ = C.decide("pi1", f, v0, v1)
    assert (z1 == C.REJ).sum() == (v1 > f["tau0"]).sum()


def test_seed_average_nan_and_pctl():
    E = np.array([[1.0, np.nan, 3.0]]); m = np.array([[1.0, 5.0, 1.0]])
    assert C.seed_average(E, m)[0] == 2.0
    assert np.isnan(C.seed_average(np.array([[np.nan]]), np.array([[1.0]]))[0])
    lo, hi = C.pctl(np.arange(101.0), 0.95)
    assert abs(lo - 2.5) < 1e-12 and abs(hi - 97.5) < 1e-12


def test_access_guard_blocks_test_score(tmp_path):
    f = tmp_path / "x.npz"
    np.savez(f, test_score=np.zeros(3), val_score=np.ones(3))
    log = C.AccessLog("E060")
    assert C.read_members(f, ["val_score"], log)["val_score"].sum() == 3
    with pytest.raises(C.TestAccessDenied):
        C.read_members(f, ["test_score"], log)
    assert all(e["key"] != "test_score" for e in log.entries)


def test_authorisation_required(monkeypatch, tmp_path):
    monkeypatch.setattr(C, "ART", tmp_path)
    with pytest.raises(C.TestAccessDenied):
        C.check_authorisation("E999", [Path(C.__file__)])


def test_bootstrap_subject_clustering():
    rng = np.random.default_rng(3)
    Wsubj = C.row_counts(rng, 50, 4)
    assert np.allclose(Wsubj.sum(axis=1), 4)
    subj_of_cred = np.array([0, 0, 1, 2, 3, 3])
    W = Wsubj[:, subj_of_cred]
    assert (W[:, 0] == W[:, 1]).all() and (W[:, 4] == W[:, 5]).all()
