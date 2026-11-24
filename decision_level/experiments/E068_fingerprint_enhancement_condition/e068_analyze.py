#!/usr/bin/env python3
"""E068 analysis of the single scoring read (CPU).

    python e068_analyze.py --selftest
    python e068_analyze.py --run artifacts/results/E068/score_run_<ts>

R1 (primary): E049's analysis functions imported unchanged (Gate 0 against the frozen anchors, full-sample cells,
paired bootstrap, cells(), decide()), with COND = restored_fpenh and the bootstrap seed 680068. Primary clusters:
the 34 test SUBJECTS (each subject's fingers drawn together); secondary: E049's finger clusters (E036.Units).
Primary readout S_fix over the 8 primary cells (N_fixed arm only); secondary E049 classes A-E/MIXED.
R2: E061 headline detectors' bona-fide rejection change at the F2 tau0 on the test reals.
R3: ArcFace-88 V1 FNMR change at E071's tau_v (FMR 1e-2 primary).
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e068_common as E  # noqa: E402

ROOT = E.ROOT
SEED = 680068
B = 10_000
CHUNK = 1_000
E071_SUMMARY = ROOT / "artifacts" / "results" / "E071" / "test_run_20261123_005124" / "summary.json"


def load_e049():
    spec = importlib.util.spec_from_file_location("e049_analyze", ROOT / "experiments" / "E049_learned_restorer_sensitivity" / "analyze.py")
    m = importlib.util.module_from_spec(spec); sys.modules["e049_analyze"] = m; spec.loader.exec_module(m)
    m.COND = E.COND                 # the only condition change
    m.E036.SEED = SEED              # E049's private E036 instance: bootstrap seed of this experiment
    return m


def subject_of(identity: str) -> str:
    sys.path.insert(0, str(ROOT))
    from experiments.tcss_lib.core import subject_of as so
    return so(identity)


def subject_units(E49, G, expect: int = 34):
    """E036.Units with the real-finger draw replaced by a subject draw (a subject's fingers share its count)."""
    U = E49.E036.Units(G)
    subj = np.array([subject_of(f) for f in U.fingers])
    names, s_of_f = np.unique(subj, return_inverse=True)
    base_draw = U.draw

    def draw(rng, n_rows):
        d = base_draw(rng, n_rows)                        # keeps E036's per-generator synthetic image draws
        d["real"] = E49.E036.row_counts(rng, n_rows, len(names))[:, s_of_f]
        return d
    assert len(names) == expect, len(names)
    U.draw = draw; U.subjects = names
    return U


def s_fix(rows) -> dict:
    prim = [r for r in rows if r["role"] == "primary"]
    cells = [f"{r['instrument']}|{r['protocol']}" for r in prim if r["dT_fix_est"] or r["dB_fix_est"]]
    n = len(cells)
    reading = ("the enhancer shifts locked behaviour even without per-image normalization" if n >= 6 else
               "no shift beyond what per-image normalization carries" if n <= 1 else "mixed")
    return {"S_fix": n, "of": len(prim), "cells": cells, "reading": reading,
            "rule": "S_fix = #primary cells whose N_fixed dT_lock or dB_lock 95% interval excludes 0; >=6 shift, <=1 none, else mixed"}


# ------------------------------------------------------------------ R2 / R3 helpers (pure)
def cluster_index(identities):
    names = np.array([subject_of(str(x)) for x in identities])
    u, idx = np.unique(names, return_inverse=True)
    return u, idx


def weighted_rate(Wcap, x):
    """Wcap (R, n) capture weights; x (n,) or (n, k) -> weighted means."""
    return (Wcap @ x) / Wcap.sum(1, keepdims=x.ndim > 1)


def r2(run: Path, log) -> list[dict]:
    files = sorted((run / "headline").glob(f"*__{E.ANCHOR}.npz"))
    D = {}
    for f in files:
        with np.load(f, allow_pickle=False) as a, np.load(str(f).replace(f"__{E.ANCHOR}.npz", f"__{E.COND}.npz"), allow_pickle=False) as b:
            assert (a["test_patch_id"] == b["test_patch_id"]).all()
            t = float(a["tau0"])
            key = (str(a["family"]), str(a["protocol"]), str(a["arm"]))
            D.setdefault(key, {})[int(a["seed"])] = ((a["score"] > t).astype(float), (b["score"] > t).astype(float), a["test_identity"])
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 2]))
    ident = next(iter(D.values()))[0][2]
    subj, sidx = cluster_index(ident)
    assert len(subj) == 34 and len(ident) == 1653, (len(subj), len(ident))
    for v in D.values():
        for s in v.values():
            assert (s[2] == ident).all()
    out = []
    reps = {k: [] for k in D}
    for b0 in range(0, B, CHUNK):
        n = min(CHUNK, B - b0)
        cnt = row_counts(rng, n, len(subj)); Wcap = cnt[:, sidx]
        for k, v in D.items():
            m = row_counts(rng, n, 10)
            dB = np.stack([weighted_rate(Wcap, v[s][1] - v[s][0]) for s in range(10)], 1)
            reps[k].append((m * dB).sum(1) / 10)
    for k, v in D.items():
        bd = np.mean([v[s][0].mean() for s in range(10)]); be = np.mean([v[s][1].mean() for s in range(10)])
        r = np.concatenate(reps[k]); lo, hi = np.quantile(r, (0.025, 0.975))
        out.append(dict(family=k[0], protocol=k[1].upper(), arm=k[2], B_degraded=bd, B_fpenh=be, dB=be - bd,
                        dB_lo=float(lo), dB_hi=float(hi), resolved=bool(lo > 0 or hi < 0), n_reals=len(ident),
                        n_subjects=len(subj)))
    log(f"R2: {len(out)} cells")
    return out


def r3(run: Path, log) -> dict:
    z = np.load(run / "matcher_arcface88.npz", allow_pickle=False)
    hp = z["has_pair"]; seeds = [int(s) for s in z["seeds"]]
    subj, sidx = cluster_index(z["test_identity"])
    assert len(subj) == 34 and int(hp.sum()) == 1621, (len(subj), int(hp.sum()))
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 3]))
    ref = json.loads(E071_SUMMARY.read_text())
    out = {}
    for fmr in (0.01, 0.001):
        fn = {c: np.stack([np.nan_to_num(1 - z[f"m__s{ms}__{fmr}__{c}"], nan=0.0) for ms in seeds], 1)
              for c in (E.ANCHOR, E.COND)}
        full = {c: float(np.mean(fn[c][hp].mean(0))) for c in fn}
        reps = []
        for b0 in range(0, B, CHUNK):
            n = min(CHUNK, B - b0)
            W = row_counts(rng, n, len(subj))[:, sidx] * hp[None, :]
            reps.append(((W @ fn[E.COND]) / W.sum(1, keepdims=True) - (W @ fn[E.ANCHOR]) / W.sum(1, keepdims=True)).mean(1))
        r = np.concatenate(reps); lo, hi = np.quantile(r, (0.025, 0.975))
        out[str(fmr)] = {"FNMR_degraded": full[E.ANCHOR], "FNMR_fpenh": full[E.COND], "dFNMR": full[E.COND] - full[E.ANCHOR],
                         "ci95": [float(lo), float(hi)], "resolved": bool(lo > 0 or hi < 0),
                         "e071_dFNMR_bilateral": ref["deltas"][f"dFNMR|anchor|{fmr}|restored_bilateral"]["estimate"],
                         "e071_dFNMR_tv": ref["deltas"][f"dFNMR|anchor|{fmr}|restored_tv"]["estimate"],
                         "captures": int(hp.sum()), "subjects": len(subj), "matcher_seeds": seeds}
    log(f"R3: {out['0.01']['dFNMR']:+.3f} {out['0.01']['ci95']}")
    return out


def row_counts(rng, n_rows, n):
    d = rng.integers(0, n, size=(n_rows, n)); out = np.zeros((n_rows, n))
    np.add.at(out, (np.repeat(np.arange(n_rows), n), d.ravel()), 1.0)
    return out


def effect(builds: list[Path]) -> tuple[dict, dict]:
    stats = []
    for b in builds:
        with open(b / "effect_per_image.csv") as fh:
            for r in csv.DictReader(fh):
                stats.append({k: (float(v) if k not in ("protocol", "split", "label") else v) for k, v in r.items()})
    eff = E.summarise_effect(stats)
    reals = [s for s in stats if s["label"] == "0"]
    rho = np.array([s["rho"] for s in reals])
    dose = {sp: {"n": int(sum(s["split"] == sp for s in reals)),
                 "median_rho": float(np.median([s["rho"] for s in reals if s["split"] == sp])),
                 "q25_q75_rho": [float(x) for x in np.quantile([s["rho"] for s in reals if s["split"] == sp], (0.25, 0.75))]}
            for sp in ("val", "test")}
    dose["all_reals_median_rho"] = float(np.median(rho))
    return eff, dose


def analyze(run: Path) -> int:
    launch = E.git_state()
    summ = json.loads((run / "score_summary.json").read_text())
    builds = [ROOT / json.loads((E.OUT / "TEST_READ_AUTHORISATION.json").read_text())[k] for k in ("stage_q_build", "test_build")]
    builds = [b.parent for b in builds]
    out = run / f"analysis_{time.strftime('%Y%m%d_%H%M%S')}"; out.mkdir(exist_ok=False)
    logf = open(out / "analyze.log", "w")

    def log(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()

    E49 = load_e049(); E49.OUT = run
    res = {"run": E.rel(run), "launch": launch}
    eff, dose = effect(builds); res["effect"] = eff; res["dose_rho_reals"] = dose
    # ---------------- R1
    S, meta = E49.load()
    g0 = E49.gate0(S, meta); (out / "gate0.json").write_text(json.dumps(g0, indent=1))
    log(f"Gate 0: {g0['checked']} anchor dumps, {len(g0['failures'])} failures, max |diff| {g0['max_abs_diff']:.2e}")
    res["gate0"] = {k: g0[k] for k in ("checked", "max_abs_diff", "pass")} | {"n_failures": len(g0["failures"]), "n_a1": len(g0["a1_passed"])}
    if not g0["pass"]:
        res["R1"] = "Gate 0 FAILED: reproduction failure; no restored_fpenh score of R1 is interpreted"
        log(res["R1"])
    else:
        G = {p: E49.groups(meta, p) for p in E49.PROTOS}
        fs = E49.full_sample(S, G)
        for label, U in (("subject", subject_units(E49, G)), ("finger", E49.E036.Units(G))):
            boot = E49.bootstrap(S, G, U, log)
            rows = E49.cells(fs, boot)
            with open(out / f"cells_{label}.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
            dec = E49.decide(rows, eff); dec["S_fix"] = s_fix(rows)
            res[f"R1_{label}"] = dec
            log(f"R1 [{label}] S_fix {dec['S_fix']['S_fix']}/8 ({dec['S_fix']['reading']}); class {dec['outcome']} "
                f"(S {dec['n_S']}, I {dec['n_I']}, A {dec['n_A']})")
    # ---------------- R2
    if summ.get("G_R68", {}).get("pass"):
        rows2 = r2(run, log)
        with open(out / "headline_dB.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows2[0])); w.writeheader(); w.writerows(rows2)
        res["R2"] = {"n_cells": len(rows2), "resolved_pos": sum(r["resolved"] and r["dB_lo"] > 0 for r in rows2),
                     "resolved_neg": sum(r["resolved"] and r["dB_hi"] < 0 for r in rows2)}
    else:
        res["R2"] = "G-R68 failed: no headline readout"
    # ---------------- R3
    if summ.get("G_X68", {}).get("pass"):
        res["R3"] = r3(run, log)
    else:
        res["R3"] = "G-X68 failed: no matcher readout"
    (out / "class.json").write_text(json.dumps(res, indent=1, default=str))
    log(f"analysis -> {out}")
    return 0


def selftest() -> int:
    ok = True
    mk = lambda k: [dict(role="primary", instrument=f"x{i}", protocol="P1", dT_fix_est=i < k, dB_fix_est=False) for i in range(8)]
    ok &= s_fix(mk(6))["S_fix"] == 6 and s_fix(mk(6))["reading"].startswith("the enhancer")
    ok &= s_fix(mk(1))["reading"].startswith("no shift") and s_fix(mk(3))["reading"] == "mixed"
    rng = np.random.default_rng(0); c = row_counts(rng, 5, 7)
    ok &= bool((c.sum(1) == 7).all())
    W = np.ones((1, 4)); ok &= bool(np.isclose(weighted_rate(W, np.array([0., 1, 1, 0]))[0], 0.5))
    ok &= subject_of("selfcollected_88:56_Left_Middle") == "56"
    print(f"selftest: {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true"); ap.add_argument("--run")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    return analyze(ROOT / a.run)


if __name__ == "__main__":
    sys.exit(main())
