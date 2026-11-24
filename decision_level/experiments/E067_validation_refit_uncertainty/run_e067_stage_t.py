#!/usr/bin/env python3
"""E067 Stage T driver: joint bootstrap of validation refitting and test sampling (protocol v3, frozen 3e1389f).

    python experiments/E067_validation_refit_uncertainty/run_e067_stage_t.py --check-auth
    python experiments/E067_validation_refit_uncertainty/run_e067_stage_t.py --pilot   # validation pool as stand-in test; NOT evidence
    python experiments/E067_validation_refit_uncertainty/run_e067_stage_t.py --test    # needs TEST_READ_AUTHORISATION.json

Each replicate b draws (1) validation subjects (shared across every key) and development images, refits the frozen
E061 F1 rules for all 240 keys, and (2) test subjects, training seeds and synthetic images exactly as E061 does.
The E061 statistics are recomputed with the refitted parameters ("refit on") and, on the SAME test draw, with the
frozen F2 parameters ("refit off", the paired comparator). E061 code is imported unchanged.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(ROOT / "experiments" / "E061_policy_simulation"))
from experiments.tcss_lib import core as C                       # noqa: E402
from experiments.tcss_lib import pool as POOL                    # noqa: E402
from experiments.tcss_lib.pool import SharedUnits, ones_draw     # noqa: E402
import run_e061 as E61                                           # noqa: E402  (frozen, unchanged)
import refit as R                                                # noqa: E402

EXP = "E067"
OUT = C.ART / EXP
PROTOCOL = ROOT / "docs" / "experiments" / "E067_validation_refit_uncertainty.md"
FIT = C.ART / "E061" / "validation_fit.json"
FIT_SHA = "c3d01b7d84d8858bfd091e9bcdd8d8db4b24597fc4ccf8adeaf3554e9dae22e4"
E061_JOINT = C.ART / "E061" / "joint_summary.json"
E060_OUTCOME = C.ART / "E060" / "OUTCOME.json"
E060_ELIG = C.ART / "E060" / "eligibility.json"
SOURCES_T = [HERE, HERE.parent / "refit.py", ROOT / "experiments" / "tcss_lib" / "core.py",
             ROOT / "experiments" / "tcss_lib" / "pool.py", ROOT / "experiments" / "E061_policy_simulation" / "run_e061.py"]
DETS = C.HEADLINE
B_T, B_PILOT, SEED_T = 2000, 50, 670068
L95, L9875, L975 = 0.95, 1 - 0.05 / 4, 0.975
TOL = 1e-12


def check_test_authorisation() -> dict:
    auth = C.check_authorisation(EXP, SOURCES_T)
    miss = [k for k in ("authorised_by", "date", "scope", "protocol_sha256", "validation_fit_sha256",
                        "e059_post_test_reads_decision", "freeze_commit") if not auth.get(k)]
    if miss:
        raise C.TestAccessDenied(f"{EXP} Stage T authorisation lacks {miss}")
    if auth["protocol_sha256"] != C.sha256(PROTOCOL):
        raise C.TestAccessDenied("protocol differs from the authorised version")
    if auth["validation_fit_sha256"] != FIT_SHA or C.sha256(FIT) != FIT_SHA:
        raise C.TestAccessDenied("validation_fit.json is not the frozen E061 F2 file")
    return auth


def val_rows(p, alog):
    """Validation real patch ids (dump row order) and their subjects; dev row count."""
    z = C.read_members(C.dump_path(DETS[0], p, "leg", 0, C.ANCHOR), ("val_patch_id", "val_y", "val_source"), alog)
    vpid = z["val_patch_id"][z["val_y"] == 0]
    ident = POOL.val_identity_map((p,))
    subj = np.array([C.subject_of(ident[str(x)]) for x in vpid])
    n_dev = int(((z["val_y"] == 1) & (z["val_source"] == C.DEV_GENERATOR)).sum())
    return subj, n_dev


def refit_all(pools, subj_draw_rows, dev_draw):
    fits = {}
    for p, pp in pools.items():
        i, j = subj_draw_rows[p], dev_draw[p]
        for det in DETS:
            for arm in C.ARMS:
                for s in C.SEEDS:
                    for op, cond in C.OPS.items():
                        fits[f"{det}|{p}|{arm}|{s}|{op}"] = C.fit_policies(
                            pp.val_real[(det, arm, s, C.ANCHOR)][i], pp.val_real[(det, arm, s, cond)][i],
                            pp.val_dev[(det, arm, s, C.ANCHOR)][j], pp.val_dev[(det, arm, s, cond)][j])
    return fits


def stats(comp, el61, el60):
    """Statistics of one replicate (comp values are arrays of length 1 or floats)."""
    cs = C.C_STAR
    g = lambda k: float(np.asarray(comp[k]).ravel()[0])
    def J(det, p, arm, op, pol):
        k = (det, p, arm, op, pol)
        return C.cost_J(g(k + (f"PMP_q{cs['q']}",)), g(k + ("L",)), g(k + ("G",)), g(k + ("N",)), cs["c_A"], cs["c_U"], cs["c_T"])
    out = {}
    cells = E61.cells(DETS)
    for pol in C.POLICIES[1:]:
        for arm in ("leg", "fix", "dd"):
            v = []
            for (det, p, op) in cells:
                ok = el61[f"{det}|{p}|leg"] and el61[f"{det}|{p}|fix"] if arm == "dd" else el61[f"{det}|{p}|{arm}"]
                if not ok:
                    continue
                d = lambda a: J(det, p, a, op, pol) - J(det, p, a, op, "pi0")
                v.append(d("fix") - d("leg") if arm == "dd" else d(arm))
            out[f"dJ|{pol}|{arm}"] = float(np.median(v))
    v = [g((det, p, "leg", op, "pi1", "L")) - g((det, p, "leg", op, "pi0", "L"))
         for (det, p, op) in cells if el60[f"{det}|{p}|leg"]]
    out["E060_dL3|leg"] = float(np.median(v))
    return out


def label_resolved(lo, hi, lo0, hi0, positive_expected):
    """unchanged / narrowed / reversed of a resolved comparator reading."""
    if positive_expected:
        res0, res = lo0 > 0, lo > 0
        rev = hi < 0
    else:
        res0, res = hi0 < 0, hi < 0
        rev = lo > 0
    if not res0:
        return "comparator_not_resolved"
    return "reversed" if rev else ("unchanged" if res else "narrowed")


def label_null(lo, hi):
    return "newly_resolved_better" if hi < 0 else "newly_resolved_worse" if lo > 0 else "still_indistinguishable"


def run(mode: str) -> int:
    auth = check_test_authorisation() if mode == "test" else {"pilot_not_evidence": True}
    stamp = time.strftime("%Y%m%d_%H%M%S")
    outdir = OUT / (f"test_run_{stamp}" if mode == "test" else f"pilot_stage_t_{stamp}")
    outdir.mkdir(parents=True, exist_ok=False)
    alog = C.AccessLog(EXP, authorisation=auth)
    logf = open(outdir / "run.log", "w")
    def log(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()
    B = B_T if mode == "test" else B_PILOT
    pools = POOL.load_pool("test" if mode == "test" else "pilot", DETS, alog, authorised=(mode == "test"))
    F2 = json.loads(FIT.read_text()); frozen, el61 = F2["fits"], F2["eligibility"]
    el60 = json.loads(E060_ELIG.read_text())
    vinfo = {p: val_rows(p, alog) for p in pools}
    shared = SharedUnits(pools)
    gens = {p: {g: np.flatnonzero(pp.syn_gen == g) for g in sorted(set(map(str, pp.syn_gen)))} for p, pp in pools.items()}
    V0 = E61.per_key_vectors(pools, frozen, DETS)
    ones_m = {(d, p): np.ones((1, 10)) for d in DETS for p in C.PROTOS}
    full0 = stats(E61.components(pools, shared, V0, DETS, ones_draw(shared), ones_m, gens), el61, el60)
    gates = {}
    # G-V0 (validation): full-sample refit reproduces F2 exactly
    allv = {p: np.arange(len(vinfo[p][0])) for p in pools}; alld = {p: np.arange(vinfo[p][1]) for p in pools}
    rf = refit_all(pools, allv, alld)
    mism = [k for k in rf for prm in R.PARAMS if not (rf[k][prm] == frozen[k][prm] or
            (np.isinf(rf[k][prm]) and np.isinf(frozen[k][prm]) and np.sign(rf[k][prm]) == np.sign(frozen[k][prm])))]
    gates["G_V0"] = {"pass": not mism, "n_mismatch": len(mism), "first": mism[:5]}
    if mode == "test":
        j61 = json.loads(E061_JOINT.read_text()); o60 = json.loads(E060_OUTCOME.read_text())
        d61 = {k: abs(full0[k] - j61[k]["median"]) for k in full0 if k.startswith("dJ|")}
        gates["G0"] = {"pass": max(d61.values()) <= TOL, "max_abs_dev": max(d61.values())}
        dev60 = abs(full0["E060_dL3|leg"] - o60["H1"]["median"])
        gates["G0_E060"] = {"pass": dev60 <= TOL, "abs_dev": dev60, "frozen": o60["H1"]["median"], "recomputed": full0["E060_dL3|leg"]}
    (outdir / "gates.json").write_text(json.dumps(gates, indent=1, default=float))
    failed = [k for k, v in gates.items() if not v["pass"]]
    if failed:
        alog.dump(outdir / "access_log.json")
        log(f"GATE FAILED {failed}: stopping before any bootstrap (see gates.json)"); return 2
    log(f"gates passed {list(gates)}; bootstrap B={B}")
    ss = np.random.SeedSequence(SEED_T).spawn(5)
    rng_seed, rng_subj, rng_syn, rng_vs, rng_vd = (np.random.default_rng(x) for x in ss)
    keys = list(full0)
    on = {k: np.empty(B) for k in keys}; off = {k: np.empty(B) for k in keys}
    subj_names = sorted(set(vinfo["p1"][0]) | set(vinfo["p2"][0]))
    t0 = time.time()
    for b in range(B):
        pick = rng_vs.integers(0, len(subj_names), len(subj_names))          # one subject draw for every key/protocol
        rows = {p: np.concatenate([np.flatnonzero(vinfo[p][0] == subj_names[k]) for k in pick]) for p in pools}
        devs = {}
        for p in pools:
            devs[p] = rng_vd.integers(0, vinfo[p][1], vinfo[p][1])
        fits_b = refit_all(pools, rows, devs)
        Vb = E61.per_key_vectors(pools, fits_b, DETS)
        m_seed = {(d, p): C.row_counts(rng_seed, 1, 10) for d in DETS for p in C.PROTOS}
        dr = shared.draw(rng_subj, rng_syn, 1)
        s_on = stats(E61.components(pools, shared, Vb, DETS, dr, m_seed, gens), el61, el60)
        s_off = stats(E61.components(pools, shared, V0, DETS, dr, m_seed, gens), el61, el60)
        for k in keys:
            on[k][b], off[k][b] = s_on[k], s_off[k]
        if (b + 1) % 50 == 0:
            log(f"  replicate {b + 1}/{B} ({time.time() - t0:.0f}s)")
    # readings
    j61 = json.loads(E061_JOINT.read_text()) if E061_JOINT.exists() else {}
    rows_out, readings = [], {}
    def iv(x, lvl): return C.pctl(x, lvl)
    for k in keys:
        lvl = L975 if k.startswith("E060") else L9875
        (lo, hi), (lo0, hi0) = iv(on[k], lvl), iv(off[k], lvl)
        mc = float(np.std([np.quantile(np.random.default_rng(i).choice(on[k], B), (1 - lvl) / 2) for i in range(200)]))
        rows_out.append(dict(statistic=k, full_sample=full0[k], level=lvl, refit_on_lo=lo, refit_on_hi=hi,
                             refit_off_lo=lo0, refit_off_hi=hi0, width_ratio=(hi - lo) / (hi0 - lo0) if hi0 > lo0 else "",
                             mc_sd_lower_bound=mc,
                             e061_frozen_decision_ci=json.dumps(j61.get(k, {}).get("ci_decision", ""))))
    g = {r["statistic"]: r for r in rows_out}
    readings["R-H1"] = label_resolved(g["dJ|pi1|leg"]["refit_on_lo"], g["dJ|pi1|leg"]["refit_on_hi"], g["dJ|pi1|leg"]["refit_off_lo"], g["dJ|pi1|leg"]["refit_off_hi"], True)
    readings["R-H2"] = label_resolved(g["dJ|pi1|dd"]["refit_on_lo"], g["dJ|pi1|dd"]["refit_on_hi"], g["dJ|pi1|dd"]["refit_off_lo"], g["dJ|pi1|dd"]["refit_off_hi"], False)
    for arm in C.ARMS:
        r = g[f"dJ|pi4|{arm}"]
        readings[f"R-H4|{arm}"] = label_resolved(r["refit_on_lo"], r["refit_on_hi"], r["refit_off_lo"], r["refit_off_hi"], False)
        for pol in ("pi2", "pi3"):
            r = g[f"dJ|{pol}|{arm}"]
            readings[f"R-H3|{pol}|{arm}"] = label_null(r["refit_on_lo"], r["refit_on_hi"])
            readings[f"R-H3|{pol}|{arm}|comparator"] = label_null(r["refit_off_lo"], r["refit_off_hi"])
    r = g["E060_dL3|leg"]
    readings["E060-H1"] = label_resolved(r["refit_on_lo"], r["refit_on_hi"], r["refit_off_lo"], r["refit_off_hi"], True)
    with open(outdir / "intervals.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0])); w.writeheader(); w.writerows(rows_out)
    (outdir / "readings.json").write_text(json.dumps({"mode": mode, "B": B, "readings": readings,
                                                      "pilot_not_evidence": mode != "test"}, indent=1))
    np.savez_compressed(outdir / "replicates.npz", **{f"on__{k}": v for k, v in on.items()}, **{f"off__{k}": v for k, v in off.items()})
    alog.dump(outdir / "access_log.json")
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.count("\n")
    (outdir / "provenance.json").write_text(json.dumps({"experiment": EXP, "stage": "T", "mode": mode, "git": git,
        "dirty_files": dirty, "B": B, "seed": SEED_T, "python": sys.version, "numpy": np.__version__,
        "sources": {C.rel(s): C.sha256(s) for s in SOURCES_T}, "protocol_sha256": C.sha256(PROTOCOL),
        "validation_fit_sha256": FIT_SHA, "command": " ".join(sys.argv), "seconds": time.time() - t0}, indent=1))
    log(f"done -> {outdir}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check-auth", action="store_true"); g.add_argument("--pilot", action="store_true")
    g.add_argument("--test", action="store_true")
    a = ap.parse_args()
    try:
        if a.check_auth:
            check_test_authorisation(); print("authorisation: OK"); return 0
        return run("pilot" if a.pilot else "test")
    except C.TestAccessDenied as e:
        print(f"authorisation: REFUSED -- {e}"); return 3


if __name__ == "__main__":
    sys.exit(main())
