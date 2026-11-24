"""E067 driver end-to-end on SYNTHETIC pools (no score file, manifest or authorisation is read)."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE.parents[3]))
import run_e067 as D                               # noqa: E402
from experiments.tcss_lib import core as C         # noqa: E402


def _fake(monkeypatch, tmp_path, corrupt=False):
    rng = np.random.default_rng(0)
    n_subj, per, n_dev = 20, 6, 150
    vpid = np.array([f"v{i}" for i in range(n_subj * per)])
    ident = {f"v{i}": f"selfcollected_88:{i // per}_Left_Index" for i in range(n_subj * per)}
    pools, frozen = {}, {}
    for p in C.PROTOS:
        pp = SimpleNamespace(val_real={}, val_dev={})
        for det in D.DETS:
            for arm in C.ARMS:
                for s in C.SEEDS:
                    for cond in C.CONDS:
                        pp.val_real[(det, arm, s, cond)] = rng.normal(0, 1, vpid.size)
                        pp.val_dev[(det, arm, s, cond)] = rng.normal(3, 1, n_dev)
                    for op, cond in C.OPS.items():
                        k = f"{det}|{p}|{arm}|{s}|{op}"
                        frozen[k] = C.fit_policies(pp.val_real[(det, arm, s, C.ANCHOR)], pp.val_real[(det, arm, s, cond)],
                                                   pp.val_dev[(det, arm, s, C.ANCHOR)], pp.val_dev[(det, arm, s, cond)])
        pools[p] = pp
    if corrupt:
        frozen["Rn|p1|leg|0|bil"]["tau0"] += 1.0
    fit = tmp_path / "validation_fit.json"; fit.write_text(json.dumps({"experiment": "E061", "fits": frozen}, default=float))  # real F2 layout
    monkeypatch.setattr(D, "FIT", fit)
    monkeypatch.setattr(D, "OUT", tmp_path / "E067")
    monkeypatch.setattr(D, "B_V", 20)
    monkeypatch.setattr(D, "check_validation_authorisation", lambda: {"authorised_by": "TEST-FIXTURE"})
    monkeypatch.setattr(D.POOL, "load_pool", lambda mode, dets, log: (_ for _ in ()).throw(AssertionError) if mode != "meta" else pools)
    monkeypatch.setattr(D.POOL, "val_identity_map", lambda protos: ident)
    def rm(path, keys, log, authorised=False):
        assert "test_score" not in keys
        return {"val_patch_id": vpid, "val_y": np.zeros(vpid.size, int)}
    monkeypatch.setattr(D.C, "read_members", rm)


def test_stage_v_synthetic_end_to_end(monkeypatch, tmp_path):
    _fake(monkeypatch, tmp_path)
    assert D.stage_v() == 0
    run = next((tmp_path / "E067").iterdir())
    g = json.loads((run / "gates.json").read_text())
    assert g["G_V0"]["pass"]
    sp = json.loads((run / "param_spread.json").read_text())
    assert len(sp) == 240 and "tau0" in sp["Rn|p1|leg|0|bil"]


def test_stage_v_stops_on_g_v0_failure(monkeypatch, tmp_path):
    _fake(monkeypatch, tmp_path, corrupt=True)
    assert D.stage_v() == 2
    run = next((tmp_path / "E067").iterdir())
    assert not (run / "param_spread.json").exists()
