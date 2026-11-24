"""E071 driver end-to-end on SYNTHETIC embeddings and pools (no image, dump, checkpoint or authorisation is read)."""
import json
import sys
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE.parents[3]))
import run_e071 as D                               # noqa: E402
from experiments.tcss_lib import core as C         # noqa: E402

DIM = 16
NOISE = {"clean": 0.3, "degraded": 0.6, "restored_bilateral": 0.8, "restored_tv": 0.9}


def _rows(split, n_subj, fingers, caps, singletons=()):
    rows = []
    for s in range(n_subj):
        for f in range(fingers):
            n = 1 if (s, f) in singletons else caps
            for c in range(n):
                rows.append({"patch_id": f"{split}_{s}_{f}_{c}", "source": "selfcollected_88", "label": "0",
                             "identity": f"selfcollected_88:{s}_F{f}", "split": split})
    return rows


def _emb(rows, cond, seed):
    out = []
    for r in rows:
        if r["label"] == "0":
            base = np.random.default_rng(zlib.crc32(r["identity"].encode())).normal(size=DIM)
        else:
            base = np.random.default_rng(zlib.crc32(r["source"].encode())).normal(size=DIM)
        e = base + NOISE[cond] * np.random.default_rng(zlib.crc32(f"{r['patch_id']}|{cond}|{seed}".encode())).normal(size=DIM) * 2
        out.append(e / np.linalg.norm(e))
    return np.array(out, np.float32)


def _setup(monkeypatch, tmp_path):
    val = _rows("val", 6, 3, 4, singletons={(0, 0)})
    test = _rows("test", 8, 3, 3, singletons={(1, 1), (2, 2)})
    gens = {"p1": ("g1", "g2"), "p2": ("g2", "g3")}
    syn = {p: [{"patch_id": f"syn_{g}_{i}", "source": g, "label": "1", "identity": "", "split": "test"}
               for g in gens[p] for i in range(25)] for p in gens}

    def manifest(p, split, label):
        if label == "1":
            return syn[p]
        return val if split == "val" else test

    cur = {"seed": None}
    monkeypatch.setattr(D, "manifest", manifest)
    monkeypatch.setattr(D.E65, "load_model", lambda mp: cur.update(seed=int(mp.name[-1])) or None)

    def embed(model, p, rows, cond, split, ilog, bs=512):
        assert all(r["split"] == split for r in rows)
        ilog.add(f"{split}|{p}", cond, len(rows))
        return _emb(rows, cond, cur["seed"])
    monkeypatch.setattr(D, "embed", embed)
    mats = []
    from diar.matcher_train import verification
    for s in (0, 1, 2):
        mp = tmp_path / "runs" / f"m_s{s}"; mp.mkdir(parents=True)
        (mp / "meta.json").write_text(json.dumps({"seed": s})); (mp / "best.pt").write_bytes(b"x")
        v = verification(_emb(test, "clean", s), [r["identity"] for r in test])
        (mp / "test_metrics.json").write_text(json.dumps({"eer": v["eer"], "n_gen": v["n_gen"]}))
        mats.append(mp)
    monkeypatch.setattr(D, "MATCHERS", mats)
    monkeypatch.setattr(D, "OUT", tmp_path / "E071")
    monkeypatch.setattr(D, "B", 40); monkeypatch.setattr(D, "CHUNK", 20)
    monkeypatch.setattr(D, "check_validation_auth", lambda: {"authorised_by": "TEST-FIXTURE"})
    monkeypatch.setattr(D, "check_test_auth", lambda: {"authorised_by": "TEST-FIXTURE"})
    monkeypatch.setattr(D, "preflight", lambda alog=None: {"pass": True, "fixture": True})
    # E065 validation thresholds as E065 would compute them from these validation embeddings (for gate G-D)
    vids = np.array([r["identity"] for r in val])
    tau65 = {}
    for s in (0, 1, 2):
        imp = D.E65.impostor_max(_emb(val, "degraded", s), _emb(val, "clean", s), vids, vids)
        for fmr in (0.01, 0.001):
            tau65[f"s{s}|{fmr}"] = float(np.quantile(imp, 1 - fmr, method="higher"))
    (tmp_path / "e065_tau_val.json").write_text(json.dumps(tau65))
    monkeypatch.setattr(D, "E065_TAU", tmp_path / "e065_tau_val.json")
    # detector side: fake pools whose real order is a permutation of the manifest order (tests the mapping)
    rng = np.random.default_rng(0)
    pools, fits = {}, {}
    for p in C.PROTOS:
        perm = rng.permutation(len(test))
        ids = np.array([test[i]["identity"] for i in perm]); pids = np.array([test[i]["patch_id"] for i in perm])
        pp = SimpleNamespace(units=C.make_units(ids), real_pid=pids,
                             syn_pid=np.array([r["patch_id"] for r in syn[p]]),
                             syn_gen=np.array([r["source"] for r in syn[p]]), real={}, syn={})
        for det in D.DETS:
            for arm in C.ARMS:
                for s in C.SEEDS:
                    v0 = rng.normal(0, 1, 300); w0 = rng.normal(3, 1, 300)
                    for cond in C.CONDS:
                        pp.real[(det, arm, s, cond)] = rng.normal(0.3 if cond != C.ANCHOR else 0, 1, len(pids))
                        pp.syn[(det, arm, s, cond)] = rng.normal(2.5, 1, len(pp.syn_pid))
                    for op, cond in C.OPS.items():
                        fits[f"{det}|{p}|{arm}|{s}|{op}"] = C.fit_policies(v0, v0 + 0.3, w0, w0 + 0.2)
        pools[p] = pp
    monkeypatch.setattr(D.POOL, "load_pool", lambda mode, dets, log, authorised=False: pools)
    fit = tmp_path / "fit.json"; fit.write_text(json.dumps({"fits": fits}, default=float))
    monkeypatch.setattr(D, "FIT", fit)
    return test


def _e065_refs(tmp_path, monkeypatch, summary_override=None):
    tau = {f"s{s}|{f}": 0.95 for s in (0, 1, 2) for f in (0.01, 0.001)}
    t = tmp_path / "e065_tau.json"; t.write_text(json.dumps(tau))
    summ = {f"dFNMR|{f}|{op}": {"fnmr_degraded": 0.5, "fnmr_restored": 0.6, "estimate": 0.1}
            for f in (0.01, 0.001) for op in ("bil", "tv")}
    summ["level|Ldet|pi0"] = {"leg": 0.0, "fix": 0.0}
    if summary_override:
        summ.update(summary_override)
    s = tmp_path / "e065_summary.json"; s.write_text(json.dumps(summ))
    monkeypatch.setattr(D, "E065_TAU", t); monkeypatch.setattr(D, "E065_SUMMARY", s)


def test_validation_then_test_end_to_end(monkeypatch, tmp_path):
    test = _setup(monkeypatch, tmp_path)
    assert D.validation_stage() == 0
    vdir = sorted((tmp_path / "E071").glob("validation_run_*"))[0]
    taus = json.loads((vdir / "tau_v.json").read_text())
    assert len(taus) == 3 * (2 + 2 + 2 + 1)
    acc = json.loads((vdir / "access_log.json").read_text())
    assert all(k.startswith("val|") for k in acc["images_read"])
    _e065_refs(tmp_path, monkeypatch)
    monkeypatch.setattr(D, "GX_TOL", np.inf); monkeypatch.setattr(D, "GC_TOL", np.inf)
    import time; time.sleep(1.1)
    monkeypatch.setattr(D, "EXPECTED_N", {"captures": len(test) - 2, "credentials": 8 * 3 - 2})
    assert D.test_stage() == 0
    tdir = sorted((tmp_path / "E071").glob("test_run_*"))[0]
    s = json.loads((tdir / "summary.json").read_text())
    for c in ("restored_bilateral", "restored_tv"):
        d = s["deltas"][f"dFNMR|anchor|0.01|{c}"]
        assert d["ci95"][0] <= d["estimate"] <= d["ci95"][1]
        assert s["V1_reading"][c].split(" (")[0].split(";")[0] in ("persists", "not resolved at this threshold", "reversed")
        assert "E065's estimate" in s["A1_reading"][c]
    lv = s["levels"]
    assert 0 <= lv["FNMR|anchor|0.01|degraded"]["estimate"] <= 1 and 0 <= lv["EER|none|0.0|degraded"]["estimate"] <= 0.5
    # pi4 rejects exactly where pi0 does, so its detector-only denial equals pi0's in every cell
    assert abs(s["cascade"]["delta|Ldet3|pi4|leg"]["median"]) < 1e-15
    assert abs(s["cascade"]["delta|Lcasc3|pi4|leg"]["median"]) < 1e-15
    # cascade can only add failures to the detector-only denial
    assert s["cascade"]["level|Lcasc1|pi0|leg"]["median"] >= s["cascade"]["level|Ldet1|pi0|leg"]["median"]
    prov = json.loads((tdir / "provenance.json").read_text())
    assert prov["launch"]["protocol_sha256"] and prov["B"] == 40


def test_gate_gx_stops_before_dumps(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    assert D.validation_stage() == 0
    _e065_refs(tmp_path, monkeypatch)
    called = {"n": 0}
    monkeypatch.setattr(D.POOL, "load_pool", lambda *a, **k: called.update(n=1))
    assert D.test_stage() == 2                    # E065 references do not match -> G-X fails
    assert called["n"] == 0                       # no detector dump was requested


def test_gate_gd_stops_validation_without_tau_file(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    bad = tmp_path / "bad_tau.json"; bad.write_text(json.dumps({f"s{s}|{f}": 0.123 for s in (0, 1, 2) for f in (0.01, 0.001)}))
    monkeypatch.setattr(D, "E065_TAU", bad)
    assert D.validation_stage() == 2
    vdir = sorted((tmp_path / "E071").glob("validation_run_*"))[0]
    assert not (vdir / "tau_v.json").exists() and (vdir / "gates.json").exists()


def test_fixed_template_session_against_bruteforce():
    rng = np.random.default_rng(5)
    ids = np.array(["selfcollected_88:0_A"] * 4 + ["selfcollected_88:0_B"] * 3 + ["selfcollected_88:1_C"])
    units = C.make_units(ids)
    n = len(ids); keep = np.array([1.0] * 7 + [0.0])            # singleton C has no pair
    ip = np.arange(n)
    acc = {(0, "degraded"): rng.uniform(size=(n, n)) < 0.6, (0, "restored_tv"): rng.uniform(size=(n, n)) < 0.4}
    T, pm, Pidx, valid = D.fixed_template_tensors(units, keep, ip, acc, [0])
    rej = (rng.uniform(size=n) < 0.3).astype(float); restored = rng.uniform(size=n) < 0.5
    l1, l3 = D.fixed_template_session(rej, restored, T[(0, "restored_tv")], T[(0, "degraded")], pm, Pidx, valid)
    for f in range(units.F):
        caps = [i for i in range(n) if units.cred_of_capture[i] == f and keep[i] > 0]
        if len(caps) < 2:
            assert l1[f] == 0 and l3[f] == 0; continue
        rb = []
        for b in caps:
            fa = [rej[a] + (1 - rej[a]) * (1 - (acc[(0, "restored_tv")] if restored[a] else acc[(0, "degraded")])[a, b])
                  for a in caps if a != b]
            rb.append(np.mean(fa))
        assert np.isclose(l1[f], np.mean(rb)) and np.isclose(l3[f], np.mean(np.array(rb) ** 3))
    # per-attempt value equals the probe-side average (pairs are balanced within a credential)
    caps = [i for i in range(n) if units.cred_of_capture[i] == 0]
    m = [np.mean([(acc[(0, "restored_tv")] if restored[a] else acc[(0, "degraded")])[a, b] for b in caps if b != a]) for a in caps]
    assert np.isclose(l1[0], np.mean([D.cascade_fail(rej[a], m[i]) for i, a in enumerate(caps)]))


def test_gx_gc_pass_at_real_tolerances_against_frozen_e065(monkeypatch, tmp_path):
    """Cross-implementation: frozen E065 test_stage and E071 test_stage on the same fixture; G-X and G-C at the real
    tolerances (1e-9, 1e-12). Adapted from the pre-freeze review."""
    import time
    test = _setup(monkeypatch, tmp_path)
    E = D.E65
    cur = {"seed": None}
    monkeypatch.setattr(E, "manifest_rows", lambda split: list(test) if split == "test" else None)
    monkeypatch.setattr(E, "load_model", lambda mp: cur.update(seed=int(mp.name[-1])) or None)
    monkeypatch.setattr(E, "embed", lambda model, rows, cond, split, ilog, bs=512: (ilog.add(split, cond, len(rows)), _emb(rows, cond, cur["seed"]))[1])
    def embed71(model, p, rows, cond, split, ilog, bs=512):
        ilog.add(f"{split}|{p}", cond, len(rows)); return _emb(rows, cond, cur["seed"])
    monkeypatch.setattr(D, "embed", embed71)
    for k, v in (("MATCHERS", D.MATCHERS), ("OUT", tmp_path / "E065"), ("B", 40), ("CHUNK", 20), ("FIT", D.FIT)):
        monkeypatch.setattr(E, k, v)
    monkeypatch.setattr(E, "check_test_auth", lambda: {"a": 1})
    vd = tmp_path / "E065" / "validation_run_x"; vd.mkdir(parents=True)
    (vd / "tau_m.json").write_text((tmp_path / "e065_tau_val.json").read_text())
    assert E.test_stage() == 0
    tdir = sorted((tmp_path / "E065").glob("test_run_*"))[0]
    monkeypatch.setattr(D, "E065_SUMMARY", tdir / "summary.json")
    assert D.validation_stage() == 0
    time.sleep(1.1)
    monkeypatch.setattr(D, "EXPECTED_N", {"captures": len(test) - 2, "credentials": 22})
    assert D.test_stage() == 0
    g = json.loads((sorted((tmp_path / "E071").glob("test_run_*"))[0] / "gates.json").read_text())
    assert g["G_X"]["pass"] and g["G_C"]["pass"]
    assert max(v["abs_dev"] for v in g["G_X"]["values"].values()) <= 1e-9
