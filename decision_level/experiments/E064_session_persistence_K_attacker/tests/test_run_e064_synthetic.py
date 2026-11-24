"""E064 driver end-to-end on SYNTHETIC pools (no score file, manifest or authorisation is read).

G0 and G-E063 references are produced by the frozen E061 and E063 code on the same synthetic pools, so the gates are
checked against independent implementations, not against E064's own numbers."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE.parents[3]))
import run_e064 as D                               # noqa: E402
import run_e063 as E63                             # noqa: E402
from experiments.tcss_lib import core as C         # noqa: E402
from experiments.tcss_lib.pool import SharedUnits, ones_draw   # noqa: E402


def _pools(seed=0):
    rng = np.random.default_rng(seed)
    ids = []
    for s in range(10):
        for f in range(3):
            ids += [f"selfcollected_88:{s}_F{f}"] * int(rng.integers(1, 7))     # n_f in 1..6: E_K differ by K
    ids = np.array(ids)
    gens = {"p1": ("ga", "gb"), "p2": ("gb", "gc")}
    pools = {}
    for p in C.PROTOS:
        sp = np.array([f"{g}_{i}" for g in gens[p] for i in range(40)]); sg = np.array([g for g in gens[p] for _ in range(40)])
        pp = SimpleNamespace(units=C.make_units(ids), real_pid=np.array([f"r{i}" for i in range(len(ids))]),
                             syn_pid=sp, syn_gen=sg, real={}, syn={}, val_real={}, val_dev={})
        for det in D.C.HEADLINE:
            for arm in C.ARMS:
                for s in C.SEEDS:
                    for cond in C.CONDS:
                        sh = 0.0 if cond == C.ANCHOR else 0.4
                        pp.real[(det, arm, s, cond)] = rng.normal(sh, 1, len(ids))
                        pp.syn[(det, arm, s, cond)] = rng.normal(1.8 + sh, 1, len(sp))
                        pp.val_real[(det, arm, s, cond)] = rng.normal(sh, 1, 400)
                        pp.val_dev[(det, arm, s, cond)] = rng.normal(2.5 + sh, 1, 200)
        pools[p] = pp
    return pools


def _frozen_refs(pools, fits, dets):
    shared = SharedUnits(pools)
    gens_by_p = {p: {g: np.flatnonzero(pp.syn_gen == g) for g in sorted(set(map(str, pp.syn_gen)))} for p, pp in pools.items()}
    V = D.E61.per_key_vectors(pools, fits, dets)
    ones_m = {(d, p): np.ones((1, 10)) for d in dets for p in C.PROTOS}
    out61 = D.E61.components(pools, shared, V, dets, ones_draw(shared), ones_m, gens_by_p)
    cs = C.C_STAR
    fr = {}
    for (det, p, arm, op, pol, c), v in out61.items():
        fr.setdefault((det, p, op, arm, pol), {})[c] = float(v[0])
    for k in fr:
        det, p, op, arm, pol = k
        fr[k]["J_cstar"] = float(D.E61.J(out61, (det, p, arm, op, pol), cs["c_A"], cs["c_U"], cs["c_T"], cs["q"])[0])
    X, elig = E63.per_cred_models(pools, V, dets)
    out63 = E63.components(pools, shared, V, X, elig, dets, ones_draw(shared), ones_m)
    fr63 = {}
    for (det, p, arm, op, pol, mo, q), v in out63.items():
        if mo in ("M0e", "M1") and q in "LGN":
            fr63[(det, p, op, arm, pol, mo, q)] = float(v[0])
    return fr, fr63


def _run(monkeypatch, tmp_path, frozen=True, corrupt=False):
    dets = C.HEADLINE
    pools = _pools()
    fits = D.E61.fit_all(pools, dets); el = D.E61.eligibility(fits, pools, dets)
    refs = _frozen_refs(pools, fits, dets) if frozen else None
    if corrupt:
        k = next(iter(refs[0])); refs[0][k]["L"] += 1e-6
    monkeypatch.setattr(D, "CHUNK", 20)
    logs = []
    res = D.run_analysis(pools, fits, el, dets, 40, tmp_path / "out", logs.append, frozen=refs)
    return res, tmp_path / "out"


def test_end_to_end_gates_pass_against_e061_e063_code(monkeypatch, tmp_path):
    res, out = _run(monkeypatch, tmp_path)
    g = res["gates"]
    assert g["G0"]["pass"] and g["G0"]["max_abs_dev"] <= 1e-12, g["G0"]
    assert g["G_E063"]["pass"] and g["G_E063"]["max_abs_dev"] <= 1e-12, g["G_E063"]
    assert g["G_STRUCT_full"]["pass"], g["G_STRUCT_full"]
    assert g["G_STRUCT_replicates"]["pass"], g["G_STRUCT_replicates"]
    import csv
    rows = list(csv.DictReader(open(out / "grid_readings.csv")))
    m2 = [r for r in rows if r["model"] == "M2"]
    assert len(m2) == 75 * 9
    base = [r for r in m2 if float(r["rho"]) == 0.0 and r["K"] == "3" and r["attacker"] == "fresh"]
    assert all(r["label_vs_baseline"] in ("unchanged", "still indistinguishable") for r in base)
    # K = 1: every rho gives the same legitimate side, so R-H1 medians agree across rho for the same attacker
    k1 = {r["rho"]: float(r["median"]) for r in m2 if r["K"] == "1" and r["attacker"] == "fresh" and r["reading"] == "R-H1"}
    assert max(k1.values()) - min(k1.values()) < 1e-6
    dec = json.loads((out / "decomposition.json").read_text())
    s, w, t = (dec[f"K3|dJ_pi1_leg|{n}"]["median"] for n in ("selection", "without_replacement", "total_M1_vs_M0"))
    assert np.isfinite([s, w, t]).all()
    rk = json.loads((out / "rankings.json").read_text())["E064"]
    assert sum(rk["M2|rho=0.0|K=3|fresh|leg"]["argmin_cstar"].values()) == 12
    assert (out / "levels.csv").exists() and (out / "rho_star.json").exists()


def test_g0_failure_stops_before_bootstrap(monkeypatch, tmp_path):
    res, out = _run(monkeypatch, tmp_path, corrupt=True)
    assert res.get("stopped_at_gate") and not res["gates"]["G0"]["pass"]
    assert not (out / "grid_readings.csv").exists()


def test_pilot_path_without_frozen_refs(monkeypatch, tmp_path):
    res, out = _run(monkeypatch, tmp_path, frozen=False)
    assert "G0" not in res["gates"] and res["gates"]["G_STRUCT_full"]["pass"]
