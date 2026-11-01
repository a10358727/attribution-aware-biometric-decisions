"""[Public adaptation: only the sys.path setup differs from the research version.] E062 v2 estimator: unit tests on SYNTHETIC data only (no DIAR score, label, manifest or image is read).

Passing these tests verifies code logic; it is not E062 evidence.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # decision_level/
from experiments.E062_quality_conditioned_eval import ra_analysis as E  # noqa: E402
from experiments.tcss_lib import core as C  # noqa: E402

DETS = E.HEADLINE


def synth(n_subj=30, creds=3, caps=4, base=0.05, restore=0.10, effects=None, seed=0, cls_fn=None, S=10,
          fix_equals_leg=False, zero_class=None):
    """Synthetic pool. effects[(arm, class)] adds to the restored rejection probability of that class."""
    rng = np.random.default_rng(seed)
    effects = effects or {}
    subj, cred, cls, ids = [], [], [], []
    for s in range(n_subj):
        for f in range(creds):
            for c in range(caps):
                subj.append(f"SUBJ{s:03d}")
                cred.append(f"SUBJ{s:03d}_F{f}")
                ids.append(f"SUBJ{s:03d}_F{f}_{c}.bmp")
                cls.append(cls_fn(s, f, c) if cls_fn else int(rng.integers(0, 3)))
    subj, cred, cls, ids = (np.array(x) for x in (subj, cred, cls, ids))
    n = len(subj)
    u = rng.normal(0, 0.02, n_subj)[np.array([int(x[4:]) for x in subj])]
    rej, el = {}, {}
    for det in DETS:
        for p in C.PROTOS:
            for arm in C.ARMS:
                el[(det, p, arm)] = True
                if fix_equals_leg and arm == "fix":
                    for cond in C.CONDS:
                        rej[(det, p, "fix", cond)] = rej[(det, p, "leg", cond)].copy()
                    continue
                p0 = np.clip(base + u, 0, 1)
                for cond in C.CONDS:
                    pr = p0.copy()
                    if cond != C.ANCHOR:
                        pr = pr + restore + np.array([effects.get((arm, g), 0.0) for g in cls])
                    pr = np.clip(pr, 0, 1)
                    if zero_class is not None:
                        pr = np.where(cls == zero_class, 0.0, pr)
                    rej[(det, p, arm, cond)] = (rng.random((S, n)) < pr).astype(np.float64)
    d = E.Data({p: subj for p in C.PROTOS}, {p: cred for p in C.PROTOS}, {p: cls for p in C.PROTOS}, rej, el,
               {p: {"vote": np.arange(n) % 3 == 0, "support": 5 + np.arange(n) % 5,
                    "severity": 1 + np.arange(n) % 4} for p in C.PROTOS},
               {p: ids for p in C.PROTOS})
    return d


def test_full_sample_closed_form():
    d = synth(seed=1)
    out = E.run(d, B=20, seed=3)
    for row in out["cells"][:4]:
        det, p, op, arm = row["det"], row["protocol"], row["op"], row["arm"]
        A = d.rej[(det, p, arm, C.ANCHOR)].mean(0)
        Rr = d.rej[(det, p, arm, C.OPS[op])].mean(0)
        for g, name in enumerate(E.CLASSES):
            m = d.cls[p] == g
            assert row[f"anc_{name}"] == pytest.approx(A[m].mean())
            assert row[f"dB_{name}"] == pytest.approx((Rr - A)[m].mean())
        assert row["theta_dry"] == pytest.approx(row["dB_RA-dry"] - row["dB_RA-middle"])


def test_aggregation_order_is_irrelevant_on_complete_data():
    d = synth(seed=2)
    R = d.rej[("Rn", "p1", "leg", C.ANCHOR)]
    m = np.array([[1, 0, 2, 1, 1, 0, 3, 1, 0, 1.0]])
    w = np.ones((1, R.shape[1]))
    g = d.cls["p1"] == 0
    a = E.group_mean(E.seed_avg(R, m), w, g)[0]
    b = float((m[0] * R[:, g].mean(1)).sum() / m.sum())
    assert a == pytest.approx(b)


def test_null_false_positive_rate():
    hits = total = 0
    for k in range(25):
        out = E.run(synth(n_subj=30, seed=100 + k), B=200, seed=k)
        for v in out["primary"].values():
            total += 1
            hits += v["reading"] in ("above 0", "below 0")
    assert hits / total <= 0.10


def test_planted_effects_and_interaction_recovered():
    eff = {("leg", 0): 0.15, ("leg", 2): -0.06}
    out = E.run(synth(n_subj=34, caps=6, effects=eff, seed=7), B=400, seed=1)
    P = out["primary"]
    assert P["theta_dry|leg"]["reading"] == "above 0"
    assert P["theta_wet|leg"]["reading"] == "below 0"
    assert P["theta_dry|leg"]["median"] == pytest.approx(0.15, abs=0.03)
    assert out["secondary"]["dtheta_dry|fix-leg"]["hi"] < 0       # effect only under legacy
    assert P["theta_dry|fix"]["reading"] == "includes 0"


def test_group_missing_is_undefined_not_imputed():
    d = synth(seed=3, cls_fn=lambda s, f, c: 1 + (c % 2))         # no RA-dry
    out = E.run(d, B=50, seed=0)
    assert out["groups"]["p1|RA-dry"]["captures"] == 0
    j = out["primary"]["theta_dry|leg"]
    assert j["median"] is None and j["reported"] is False and j["undefined_fraction"] == 1.0
    assert out["primary"]["theta_wet|leg"]["reported"] is True


def test_very_few_subjects_suppresses_interval():
    d = synth(n_subj=30, seed=4, cls_fn=lambda s, f, c: 0 if s < 2 else 1 + (c % 2))   # RA-dry in 2 subjects
    out = E.run(d, B=400, seed=0)
    assert out["groups"]["p1|RA-dry"]["subjects"] == 2
    j = out["primary"]["theta_dry|leg"]
    assert j["undefined_fraction"] > E.UNDEFINED_MAX and j["reported"] is False


def test_duplicate_ids_rejected():
    d = synth(seed=5)
    d.ids["p1"] = d.ids["p1"].copy()
    d.ids["p1"][1] = d.ids["p1"][0]
    with pytest.raises(ValueError, match="duplicated"):
        E.run(d, B=10)


def test_imbalanced_classes_closed_form():
    d = synth(seed=6, cls_fn=lambda s, f, c: 0 if (s % 10 == 0 and c == 0) else 1 + (s % 2))
    out = E.run(d, B=50, seed=0)
    assert out["groups"]["p1|RA-dry"]["captures"] == 9           # 3 subjects x 3 credentials x 1 capture
    row = out["cells"][0]
    m = d.cls["p1"] == 0
    A = d.rej[(row["det"], "p1", row["arm"], C.ANCHOR)].mean(0)
    assert row["anc_RA-dry"] == pytest.approx(A[m].mean())


def test_paired_draws_identical_arms_give_zero_interaction():
    out = E.run(synth(seed=8, fix_equals_leg=True), B=200, seed=0)
    for name in ("dry", "wet"):
        j = out["secondary"][f"dtheta_{name}|fix-leg"]
        assert j["median"] == 0.0 and j["lo"] == 0.0 and j["hi"] == 0.0


def test_floor_rule():
    out = E.run(synth(seed=9, zero_class=0), B=50, seed=0)
    j = out["primary"]["theta_dry|leg"]
    assert j["floor_uninformative"] is True and j["reading"] == "uninformative (floor)"
    assert out["primary"]["theta_wet|leg"]["floor_uninformative"] is False


def test_ineligible_cell_excluded():
    d = synth(seed=10)
    d.eligible[("Dt", "p2", "fix")] = False
    out = E.run(d, B=50, seed=0)
    assert out["primary"]["theta_dry|leg"]["n_cells"] == 12
    assert out["primary"]["theta_dry|fix"]["n_cells"] == 10       # Dt p2 bil + tv
    assert out["secondary"]["dtheta_dry|fix-leg"]["n_cells"] == 10


def test_determinism_and_seed_dependence():
    d = synth(seed=11)
    a, b, c = (E.run(d, B=100, seed=s) for s in (5, 5, 6))
    assert a["primary"] == b["primary"]
    assert a["primary"]["theta_dry|leg"]["lo"] != c["primary"]["theta_dry|leg"]["lo"]


def test_credential_weighting():
    # subject 0, credential F0 holds 4 RA-dry captures; every other RA-dry capture is alone in its credential
    def cls_fn(s, f, c):
        if s == 0 and f == 0:
            return 0
        return 0 if c == 0 else 1
    d = synth(n_subj=6, seed=12, cls_fn=cls_fn)
    p = "p1"
    w = E.capture_weights(d, p, np.ones((1, 6)), np.array([int(x[4:]) for x in d.subj[p]]), "credential",
                          np.ones(len(d.subj[p]), bool))[0]
    m = d.cls[p] == 0
    assert np.allclose(w[m & (d.cred[p] == "SUBJ000_F0")], 0.25)
    assert w[m].sum() == pytest.approx(1 + 17)                    # 18 credentials with RA-dry, one unit each


def test_run_all_prespecified_analyses():
    res = E.run_all(synth(seed=13), B=30, seed=0)
    assert set(res) == {"primary", "S-1", "S-2", "S-3a_sev12", "S-3b_sev34", "S-4", "S-5", "S-6_bil", "S-6_tv", "S-7",
                        "S-8w_dry", "S-8w_wet"}
    assert res["S-8w_dry"]["weighting"] == "credential"
    assert res["S-8w_dry"]["report_only"] == ["theta_dry|leg", "theta_dry|fix"]
    assert res["S-8w_wet"]["groups"]["p1|RA-dry"]["captures"] == 0
    assert res["S-4"]["primary"]["theta_dry|leg"]["n_cells"] == 8
    assert res["S-6_bil"]["primary"]["theta_dry|leg"]["n_cells"] == 6
    assert res["S-5"]["groups"]["p1|RA-dry"]["captures"] < res["primary"]["groups"]["p1|RA-dry"]["captures"]


def test_output_has_no_identifiers():
    text = json.dumps(E.run(synth(seed=14), B=20, seed=0))
    assert "SUBJ" not in text and ".bmp" not in text


def test_real_loader_refuses_without_authorisation(tmp_path):
    assert not (C.ART / "E062" / "TEST_READ_AUTHORISATION.json").exists()
    with pytest.raises(C.TestAccessDenied):
        E.load_real(tmp_path / "labels.csv", C.AccessLog("E062"))


def test_primary_level_is_amendment_1():
    assert (E.LEVEL_PRIMARY, E.LEVEL_SECONDARY) == (0.995, 0.975)                # D-26a; secondary unchanged
    assert (E.B_PRIMARY, E.SEED_DRAFT) == (10_000, 620_062)                     # D-24
    out = E.run(synth(seed=17), B=40, seed=0)
    assert {v["level"] for v in out["primary"].values()} == {0.995}
    assert out["secondary"]["dtheta_dry|fix-leg"]["level"] == 0.975


def test_unchosen_finger_masks_not_in_run_all():
    d = synth(seed=15)
    for p in C.PROTOS:
        d.attrs[p]["finger"] = np.where(np.char.endswith(d.cred[p].astype(str), "_F0"), "thumb", "non_thumb")
    mk = E.masks(d)
    assert {"S-8a", "S-8b", "S-8w_dry", "S-8w_wet"} <= set(mk)
    assert not mk["S-8a"]["p1"][np.char.endswith(d.cred["p1"].astype(str), "_F0")].any()
    ran = {k for k in E.run_all(d, B=10, seed=0) if k.startswith("S-8")}
    assert ran == {"S-8w_dry", "S-8w_wet"}                                      # D-23: S-8a/S-8b not chosen


def test_within_credential_closed_form():
    d = synth(seed=16)
    p = "p1"
    m = E.masks(d)["S-8w_dry"]
    out = E.run(d, B=20, seed=0, mask=m, weighting="credential")
    row = out["cells"][0]
    Rr = d.rej[(row["det"], p, row["arm"], C.OPS[row["op"]])].mean(0)
    A = d.rej[(row["det"], p, row["arm"], C.ANCHOR)].mean(0)
    dd = Rr - A
    cr = d.cred[p]
    diffs = [dd[(cr == c) & (d.cls[p] == 0)].mean() - dd[(cr == c) & (d.cls[p] == 1)].mean()
             for c in np.unique(cr[m[p]])]
    assert row["theta_dry"] == pytest.approx(np.mean(diffs))
    assert np.isnan(row["theta_wet"])                                           # RA-wet masked out
