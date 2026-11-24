#!/usr/bin/env python3
"""E049 step 3 -- Gate 0 (anchor reproduction), the deployment vector and the paired arm contrast at 1 % FDR
(E036 estimator), the restorer-effect check and the frozen interpretation classes.

    python3 experiments/E049_learned_restorer_sensitivity/analyze.py --selftest
    python3 experiments/E049_learned_restorer_sensitivity/analyze.py
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "results"
OUT = ART / "E049"
_spec = importlib.util.spec_from_file_location("run_e036", ROOT / "experiments/E036_interaction_statistics/run_e036.py")
E036 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(E036)

PRIMARY = ("d3", "resnet18_imagenet", "resnet50_imagenet", "efficientnet_b0_imagenet")
CONTRAST = ("npr",)
INSTS = PRIMARY + CONTRAST
NAME = {"d3": "D3", "resnet18_imagenet": "ResNet-18 (ImageNet)", "resnet50_imagenet": "ResNet-50 (ImageNet)",
        "efficientnet_b0_imagenet": "EfficientNet-B0 (ImageNet)", "npr": "NPR"}
ARMS, PROTOS, SEEDS = ("leg", "fix"), ("p1", "p2"), tuple(range(10))
ANCHOR, COND = "degraded", "restored_restormer"
TOL_GATE0 = 1e-4


def load():
    S = {}
    meta = {}
    for inst in INSTS:
        for arm in ARMS:
            for p in PROTOS:
                for s in SEEDS:
                    for c in (ANCHOR, COND):
                        with np.load(OUT / "scores" / f"{inst}_{arm}_{p}__s{s}__{c}.npz", allow_pickle=False) as z:
                            S[(inst, arm, p, s, c)] = (z["test_score"].astype(np.float64), z["val_score"].astype(np.float64))
                            if p not in meta:
                                meta[p] = {k: z[k] for k in ("test_patch_id", "test_source", "test_identity", "test_y", "val_patch_id", "val_y")}
                            else:
                                for k in ("test_patch_id", "val_patch_id"):
                                    assert (z[k] == meta[p][k]).all()
    return S, meta


def frozen_anchor(inst, arm, p, s):
    if inst == "d3":
        with np.load(ART / "E038" / "scores" / f"d3_{arm}_{p}__s{s}__degraded.npz") as z:
            return z["test_patch_id"], z["test_score"][0], z["val_patch_id"], z["val_score"][0]
    if inst == "npr":
        root, rn, _ = E036.SOURCES[("Np", p, arm)]
    else:
        root = f"E035/scores_{'legacy' if arm == 'leg' else 'fixed'}_tracka_{p}"
        rn = f"e035_{inst}_t{p}_{'minmax' if arm == 'leg' else 'div255'}"
    for f in sorted((ART / root).glob("*.npz")):
        with np.load(f, allow_pickle=False) as z:
            if str(z["run_name"]) == rn and int(z["seed"]) == s and str(z["condition"]) == ANCHOR:
                return z["test_patch_id"], z["test_score"], z["val_patch_id"], z["val_score"]
    raise FileNotFoundError((inst, arm, p, s))


def gate0(S, meta) -> dict:
    fails, worst, n, a1 = [], 0.0, 0, []
    for inst in INSTS:
        for arm in ARMS:
            for p in PROTOS:
                for s in SEEDS:
                    tp, ts, vp, vs = frozen_anchor(inst, arm, p, s)
                    ok_ids = (tp == meta[p]["test_patch_id"]).all() and (vp == meta[p]["val_patch_id"]).all()
                    d = max(float(np.abs(ts - S[(inst, arm, p, s, ANCHOR)][0]).max()), float(np.abs(vs - S[(inst, arm, p, s, ANCHOR)][1]).max()))
                    worst = max(worst, d); n += 1
                    if not ok_ids or d > TOL_GATE0:
                        f = dict(inst=inst, arm=arm, p=p, seed=s, ids_equal=bool(ok_ids), max_abs_diff=d)
                        # Amendment A1 (PI, 2026-10-08, before any restored score was read): contrast instrument only --
                        # a dump beyond 1e-4 passes on the E033 determinism criterion. Primary instruments unchanged.
                        if ok_ids and inst in CONTRAST:
                            f.update(a1_check(ts, vs, *S[(inst, arm, p, s, ANCHOR)], meta[p], p))
                            if f["a1_pass"]:
                                a1.append(f); continue
                        fails.append(f)
    return dict(checked=n, failures=fails, max_abs_diff=worst, tolerance=TOL_GATE0, a1_passed=a1,
                amendment="A1: contrast dumps beyond 1e-4 judged by Pearson >= 0.9999, |dAUC| <= 0.001, |dT_lock| <= 0.002, "
                          "|dB_lock| <= 0.002 vs the frozen dump (E033 criterion); primary instruments 1e-4 unchanged",
                **{"pass": not fails})


def a1_check(ts_f, vs_f, ts_n, vs_n, m, p) -> dict:
    y = m["test_y"]; src = m["test_source"]
    keep = (y == 0) | np.isin(src, sorted(E036.UNSEEN[p]))
    vR = m["val_y"] == 0
    r = float(np.corrcoef(np.r_[ts_f, vs_f], np.r_[ts_n, vs_n])[0, 1])

    def anchor(ts, vs):
        tau = E036.thr_at_1pct(vs[vR]); t, yy = ts[keep], y[keep]
        auc = float(E036.weighted_auc(t[yy == 0], t[yy == 1], np.ones((1, (yy == 0).sum())), np.ones((1, (yy == 1).sum())))[0])
        return auc, float((t[yy == 1] > tau).mean()), float((t[yy == 0] > tau).mean())
    a, b = anchor(ts_f, vs_f), anchor(ts_n, vs_n)
    dA, dT, dB = abs(a[0] - b[0]), abs(a[1] - b[1]), abs(a[2] - b[2])
    return dict(a1_pearson=r, a1_dAUC=dA, a1_dT_lock=dT, a1_dB_lock=dB,
                a1_pass=bool(r >= 0.9999 and dA <= 0.001 and dT <= 0.002 and dB <= 0.002))


def groups(meta, p):
    y, src = meta[p]["test_y"], meta[p]["test_source"]
    m = (y == 0) | np.isin(src, sorted(E036.UNSEEN[p]))
    Rr, Ss = np.flatnonzero(m & (y == 0)), np.flatnonzero(m & (y == 1))
    vR = np.flatnonzero(meta[p]["val_y"] == 0)
    ns = SimpleNamespace(R=Rr, S=Ss, vR=vR, syn_gen=src[Ss], real_ident=meta[p]["test_identity"][Rr],
                         real_pid=meta[p]["test_patch_id"][Rr], syn_pid=meta[p]["test_patch_id"][Ss], val_pid=meta[p]["val_patch_id"][vR])
    return ns


def full_sample(S, G):
    out = {}
    for inst in INSTS:
        for arm in ARMS:
            for p in PROTOS:
                g = G[p]; o_r, o_s = np.ones((1, len(g.R))), np.ones((1, len(g.S)))
                for s in SEEDS:
                    ta, va = S[(inst, arm, p, s, ANCHOR)]; tc, vc = S[(inst, arm, p, s, COND)]
                    tau = E036.thr_at_1pct(va[g.vR]); tau_c = E036.thr_at_1pct(vc[g.vR])
                    A = ((ta[g.S] > tau).mean(), (ta[g.R] > tau).mean(), float(E036.weighted_auc(ta[g.R], ta[g.S], o_r, o_s)[0]))
                    C = ((tc[g.S] > tau).mean(), (tc[g.R] > tau).mean(), float(E036.weighted_auc(tc[g.R], tc[g.S], o_r, o_s)[0]))
                    out[(inst, arm, p, s)] = dict(tau=tau, T_anchor=A[0], B_anchor=A[1], AUC_anchor=A[2], dT=C[0] - A[0], dB=C[1] - A[1],
                                                  dAUC=C[2] - A[2], dtau=tau_c - tau, T_recal=float((tc[g.S] > tau_c).mean()))
    return out


def bootstrap(S, G, U, log):
    ss = np.random.SeedSequence(E036.SEED)
    rng_seed, rng_clu, _ = (np.random.default_rng(c) for c in ss.spawn(3))
    R = {(i, p): {k: [] for k in ("DDT", "DDB", "dT_leg", "dT_fix", "dB_leg", "dB_fix", "dA_leg", "dA_fix")} for i in INSTS for p in PROTOS}
    F = {}
    for p in PROTOS:
        g = G[p]
        for inst in INSTS:
            for arm in ARMS:
                for s in SEEDS:
                    ta, va = S[(inst, arm, p, s, ANCHOR)]; tc, _ = S[(inst, arm, p, s, COND)]
                    tau = E036.thr_at_1pct(va[g.vR])
                    F[(inst, arm, p, s)] = ((tc[g.S] > tau).astype(float) - (ta[g.S] > tau), (tc[g.R] > tau).astype(float) - (ta[g.R] > tau))
    done = 0
    while done < E036.B_PRIMARY:
        n = min(E036.CHUNK, E036.B_PRIMARY - done)
        M = {(i, p): E036.row_counts(rng_seed, n, len(SEEDS)) for i in INSTS for p in PROTOS}
        D = U.draw(rng_clu, n)
        n_auc = max(0, min(n, E036.B_AUC - done))
        for p in PROTOS:
            Wr, Ws = U.weights(D, p); g = G[p]
            for inst in INSTS:
                Mp = M[(inst, p)]; r = R[(inst, p)]
                dT = {a: np.stack([(Ws @ F[(inst, a, p, s)][0]) / Ws.sum(1) for s in SEEDS], 1) for a in ARMS}
                dB = {a: np.stack([(Wr @ F[(inst, a, p, s)][1]) / Wr.sum(1) for s in SEEDS], 1) for a in ARMS}
                for a in ARMS:
                    r[f"dT_{a}"].append((Mp * dT[a]).sum(1) / 10); r[f"dB_{a}"].append((Mp * dB[a]).sum(1) / 10)
                r["DDT"].append(((Mp * dT["fix"]).sum(1) - (Mp * dT["leg"]).sum(1)) / 10)
                r["DDB"].append(((Mp * dB["fix"]).sum(1) - (Mp * dB["leg"]).sum(1)) / 10)
                if n_auc:
                    for a in ARMS:
                        d = []
                        for s in SEEDS:
                            ta, _ = S[(inst, a, p, s, ANCHOR)]; tc, _ = S[(inst, a, p, s, COND)]
                            d.append(E036.weighted_auc(tc[g.R], tc[g.S], Wr[:n_auc], Ws[:n_auc]) - E036.weighted_auc(ta[g.R], ta[g.S], Wr[:n_auc], Ws[:n_auc]))
                        r[f"dA_{a}"].append((Mp[:n_auc] * np.stack(d, 1)).sum(1) / 10)
        done += n
        log(f"  replicates {done}/{E036.B_PRIMARY}")
    return {c: {k: np.concatenate(v) for k, v in d.items()} for c, d in R.items()}


def ci(v):
    lo, hi = np.quantile(v, E036.Q)
    return float(lo), float(hi)


def cells(fs, boot):
    rows = []
    for inst in INSTS:
        for p in PROTOS:
            b = boot[(inst, p)]
            m = lambda a, k: float(np.mean([fs[(inst, a, p, s)][k] for s in SEEDS]))
            r = dict(instrument=NAME[inst], inst=inst, role="primary" if inst in PRIMARY else "contrast", protocol=p.upper())
            for a in ARMS:
                for k in ("dAUC", "dT", "dB", "dtau", "T_recal", "T_anchor", "B_anchor", "AUC_anchor"):
                    r[f"{k}_{a}"] = m(a, k)
                for k, bk in (("dT", "dT"), ("dB", "dB"), ("dAUC", "dA")):
                    lo, hi = ci(b[f"{bk}_{a}"]); r[f"{k}_{a}_lo"], r[f"{k}_{a}_hi"] = lo, hi
                    r[f"{k}_{a}_est"] = bool(lo > 0 or hi < 0)
            r["DDT"], r["DDB"] = r["dT_fix"] - r["dT_leg"], r["dB_fix"] - r["dB_leg"]
            for q in ("DDT", "DDB"):
                lo, hi = ci(b[q]); r[f"{q}_lo"], r[f"{q}_hi"] = lo, hi; r[f"{q}_est"] = bool(lo > 0 or hi < 0)
            r["S"] = bool(r["dT_leg_est"] or r["dT_fix_est"] or r["dB_leg_est"] or r["dB_fix_est"])
            r["I"] = bool(r["DDT_est"] or r["DDB_est"])
            r["A"] = bool(r["dAUC_leg_est"] or r["dAUC_fix_est"])
            rows.append(r)
    return rows


def decide(rows, effect):
    prim = [r for r in rows if r["role"] == "primary"]
    nS, nI, nA = sum(r["S"] for r in prim), sum(r["I"] for r in prim), sum(r["A"] for r in prim)
    if effect["identity_like"] or (nS <= 1 and nA <= 1):
        c = "D"
    elif nS >= 6 and nI >= 6:
        c = "A"
    elif nS >= 6:
        c = "B"
    elif nA >= 6:
        c = "C"
    else:
        c = "MIXED"
    return dict(outcome=c, n_cells=len(prim), n_S=nS, n_I=nI, n_A=nA,
                S_cells=[f"{r['instrument']}|{r['protocol']}" for r in prim if r["S"]],
                I_cells=[f"{r['instrument']}|{r['protocol']}" for r in prim if r["I"]],
                A_cells=[f"{r['instrument']}|{r['protocol']}" for r in prim if r["A"]],
                identity_like=effect["identity_like"],
                rule="S: dT_lock or dB_lock interval excl. 0 in >=1 arm; I: DDT or DDB interval excl. 0; A: dAUC excl. 0 in >=1 arm. "
                     "D: identity-like or (nS<=1 and nA<=1); A: nS>=6 and nI>=6; B: nS>=6; C: nA>=6; else MIXED (preregistered, E049)")


def selftest() -> int:
    ok = E036.selftest() == 0
    eff0 = dict(identity_like=False)
    mk = lambda S, I, A: [dict(role="primary", instrument=f"x{k}", protocol="P1", S=k < S, I=k < I, A=k < A) for k in range(8)]
    for (S_, I_, A_, idl), want in (((8, 7, 8, False), "A"), ((6, 5, 0, False), "B"), ((2, 0, 7, False), "C"), ((1, 0, 1, False), "D"),
                                    ((8, 8, 8, True), "D"), ((4, 2, 3, False), "MIXED")):
        ok &= decide(mk(S_, I_, A_), dict(identity_like=idl))["outcome"] == want
    print(f"selftest: {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true"); a = ap.parse_args()
    if a.selftest:
        return selftest()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "experiments/E049_learned_restorer_sensitivity/analyze.py",
                            "experiments/E036_interaction_statistics/run_e036.py"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if dirty:
        print("ABORT: analysis script differs from the committed (frozen) version"); return 3
    (ROOT / "artifacts/logs/E049").mkdir(parents=True, exist_ok=True)
    logf = open(ROOT / f"artifacts/logs/E049/analyze_{time.strftime('%Y%m%d_%H%M%S')}.log", "w")

    def log(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()

    gate = json.loads((OUT / "restorer_gate.json").read_text())
    S, meta = load()
    g0 = gate0(S, meta)
    (OUT / "gate0.json").write_text(json.dumps(g0, indent=2))
    log(f"Gate 0: {g0['checked']} anchor dumps, {len(g0['failures'])} failures, max |diff| {g0['max_abs_diff']:.2e}")
    if not g0["pass"]:
        log("Gate 0 FAILED -- E049 stops before any restored score is read"); return 2
    G = {p: groups(meta, p) for p in PROTOS}
    U = E036.Units(G)
    fs = full_sample(S, G)
    boot = bootstrap(S, G, U, log)
    rows = cells(fs, boot)
    with open(OUT / "cells.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    dec = decide(rows, gate["effect"])
    dec.update(restorer=gate["restorer_used"], effect=gate["effect"], gate0=dict(checked=g0["checked"], max_abs_diff=g0["max_abs_diff"]),
               git=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    (OUT / "OUTCOME.json").write_text(json.dumps(dec, indent=2))
    log(f"OUTCOME {dec['outcome']}: S {dec['n_S']}/8, I {dec['n_I']}/8, A {dec['n_A']}/8; identity-like {dec['identity_like']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
