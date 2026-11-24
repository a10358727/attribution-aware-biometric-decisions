#!/usr/bin/env python3
"""E064 -- RQ3 session sensitivity II: persistence (M2, rho), attempt budget K = 1..5, attacker variants, and the
population-matched decomposition of retry-model vs population-selection effects (post hoc; simulation).

Protocol: docs/experiments/E064_session_persistence_K_attacker.md (v4). Frozen E061 F2 parameters (SHA-256 pinned),
E061 per-capture decisions (imported, unchanged), E063 session models (imported, unchanged), E064 M2 closed forms.

    python experiments/E064_session_persistence_K_attacker/run_e064.py --check-auth
    python experiments/E064_session_persistence_K_attacker/run_e064.py --pilot   # L2; validation-only; NOT evidence
    python experiments/E064_session_persistence_K_attacker/run_e064.py --test    # L3; one test read

PMP is pre-matcher progression, never attack success.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(ROOT / "experiments" / "E061_policy_simulation"))
sys.path.insert(0, str(ROOT / "experiments" / "E063_session_robustness"))
from experiments.tcss_lib import core as C                                   # noqa: E402
from experiments.tcss_lib.pool import SharedUnits, load_pool, ones_draw     # noqa: E402
import run_e061 as E61                                                       # noqa: E402  (frozen, unchanged)
import session_models as SM                                                  # noqa: E402  (E063, frozen)
import session_m2 as M2                                                      # noqa: E402

EXP = "E064"
OUT = C.ART / EXP
LOGDIR = ROOT / "artifacts" / "logs" / EXP
PROTOCOL = ROOT / "docs" / "experiments" / "E064_session_persistence_K_attacker.md"
SOURCES_RUN = [HERE, HERE.parent / "session_m2.py", ROOT / "experiments/E063_session_robustness/session_models.py",
               ROOT / "experiments/tcss_lib/core.py", ROOT / "experiments/tcss_lib/pool.py",
               ROOT / "experiments/E061_policy_simulation/run_e061.py"]
FIT = C.ART / "E061" / "validation_fit.json"
FIT_SHA = "c3d01b7d84d8858bfd091e9bcdd8d8db4b24597fc4ccf8adeaf3554e9dae22e4"
E061_SESSION = C.ART / "E061" / "session.csv"
E063_RUN = C.ART / "E063" / "test_run_20261122_195531"
B_PRIMARY, B_PILOT, CHUNK, SEED = 10_000, 200, 500, 640064
LEVEL = 1 - 0.05 / 4                     # 98.75 %, the E061 decision level (readings only)
RHOS = M2.RHO_GRID
KS = M2.K_GRID
ATTACKERS = ("fresh", "same", "worstgen")
Q_LIST = C.COST_GRID["q"]
CS = C.C_STAR
TOL = 1e-12
BASE = (0.0, 3, "fresh")                 # Monte Carlo baseline configuration (E061 design, new seed)
# E061 frozen outcomes per reading: +1 / -1 = resolved direction, 0 = indistinguishable. Checked against
# E061 OUTCOME.json at run time (check_e061_dir); the fixed-arm pi1 reading is descriptive in E061 (joint_summary).
E061_DIR = {"R-H1": +1, "fixed_pi1": +1, "R-H2": -1, "R-H4|leg": -1, "R-H4|fix": -1,
            "R-H3|pi2|leg": 0, "R-H3|pi2|fix": 0, "R-H3|pi3|leg": 0, "R-H3|pi3|fix": 0}
E061_OUTCOME = C.ART / "E061" / "OUTCOME.json"


def check_e061_dir() -> dict:
    o = json.loads(E061_OUTCOME.read_text())
    ok = (o["H1_supported"] is True and o["H2_supported"] is True and o["H4_supported"] == {"leg": True, "fix": True}
          and all(v == "indistinguishable" for v in o["H3_class"].values()))
    return {"pass": bool(ok), "outcome": o["outcome"]}


# ------------------------------------------------------------------------------ column layout
def legit_cols():
    """Column names of the per-credential legitimate matrix: M2 on all credentials, M2 and M1 on E_K."""
    cols = [("M2", r, k, q) for k in KS for r in RHOS for q in "LGN"]
    cols += [("M2E", r, k, q) for k in KS for r in RHOS for q in "LGN"]
    cols += [("M1E", None, k, q) for k in KS for q in "LGN"]
    return cols


def syn_cols(gens):
    cols = [("PMP", k, q) for k in KS for q in Q_LIST] + [("SAME", None, q) for q in Q_LIST]
    cols += [("GEN", k, g) for k in KS for g in gens]
    return cols


def per_cred_inputs(pools, V):
    """Per key (det, p, arm, seed, op, pol): X_all (F, 75) M2 on all credentials and X_K (F, 18) [M2(rho) at K, M1 at K]
    for the E_K-weighted means. Captures are never resampled within a credential, so these are fixed."""
    X, elig = {}, {}
    for p, pp in pools.items():
        nf = pp.units.n_f()
        elig[p] = {k: (nf >= k) for k in KS}
        for key, v in V.items():
            if key[1] != p:
                continue
            R = SM.counts(v["rej_cap"], pp.units.cred_of_capture, pp.units.F)
            U = SM.counts(v["stp_cap"], pp.units.cred_of_capture, pp.units.F)
            m2 = {(r, k): M2.m2(v["rf"], v["uf"], k, r) for k in KS for r in RHOS}
            xa = np.stack([m2[(r, k)][i] for k in KS for r in RHOS for i in range(3)], 1)
            xk = {}
            for k in KS:
                l1, g1, n1, e1 = SM.m1(R, U, nf, k)
                assert (e1 == elig[p][k]).all()
                xk[k] = np.stack([m2[(r, k)][i] for r in RHOS for i in range(3)]
                                 + [np.nan_to_num(l1), np.nan_to_num(g1), np.nan_to_num(n1)], 1)
            X[key] = (xa, xk)
    return X, elig


def components(pools, shared, V, X, elig, dets, d, m_seed, gens_by_p):
    """Seed-averaged component matrix (R, n_comp) per (det, p, arm, op, pol); float64."""
    R = d["subj"].shape[0]
    out = {}
    nl = len(legit_cols())
    for p, pp in pools.items():
        W = shared.cred_weights(d, p, pp.units)
        WK = {k: W * elig[p][k][None, :] for k in KS}
        Ws = shared.syn_weights(d, p, len(pp.syn_pid)); wsum = Ws.sum(1)
        gens = gens_by_p[p]; ns = len(syn_cols(list(gens)))
        for det in dets:
            m = m_seed[(det, p)]
            for arm in C.ARMS:
                for op in C.OPS:
                    for pol in C.POLICIES:
                        E = np.empty((R, nl + ns, 10))
                        for j, s in enumerate(C.SEEDS):
                            key = (det, p, arm, s, op, pol)
                            xa, xk = X[key]
                            E[:, :75, j] = C.wmean(W, xa)
                            for ki, k in enumerate(KS):
                                Mk = C.wmean(WK[k], xk[k])                       # (R, 18)
                                E[:, 75 + ki * 15: 75 + (ki + 1) * 15, j] = Mk[:, :15]
                                E[:, 150 + ki * 3: 150 + (ki + 1) * 3, j] = Mk[:, 15:]
                            v = V[key]
                            Pv = (Ws @ v["acc_syn"]) / wsum; Vv = (Ws @ v["stp_syn"]) / wsum
                            c = nl
                            for k in KS:
                                for q in Q_LIST:
                                    E[:, c, j] = M2.pmp_fresh(Pv, Vv, q, k); c += 1
                            for q in Q_LIST:
                                E[:, c, j] = M2.pmp_same(Pv, Vv, q); c += 1
                            pg = {}
                            for g, idx in gens.items():
                                Wg = Ws[:, idx]; sg = Wg.sum(1)
                                pg[g] = ((Wg @ v["acc_syn"][idx]) / sg, (Wg @ v["stp_syn"][idx]) / sg)
                            for k in KS:
                                for g in gens:
                                    E[:, c, j] = M2.pmp_fresh(pg[g][0], pg[g][1], CS["q"], k); c += 1
                        assert not np.isnan(E).any()
                        out[(det, p, arm, op, pol)] = (E * m[:, None, :]).sum(2) / m.sum(1)[:, None]
    return out


class Index:
    """Column lookup per protocol (generator sets differ by protocol)."""

    def __init__(self, gens_by_p):
        self.L = {c: i for i, c in enumerate(legit_cols())}
        nl = len(self.L)
        self.S = {p: {c: nl + i for i, c in enumerate(syn_cols(list(g)))} for p, g in gens_by_p.items()}
        self.gens = {p: list(g) for p, g in gens_by_p.items()}

    def legit(self, model, rho, k, q):
        return self.L[(model, rho if model != "M1E" else None, k, q)]


# ------------------------------------------------------------------------------ analysis
def run_analysis(pools, fits, el, dets, B, outdir: Path, log, frozen=None) -> dict:
    shared = SharedUnits(pools)
    ss = np.random.SeedSequence(SEED).spawn(3)
    rng_seed, rng_subj, rng_syn = (np.random.default_rng(x) for x in ss)
    gens_by_p = {p: {g: np.flatnonzero(pp.syn_gen == g) for g in sorted(set(map(str, pp.syn_gen)))}
                 for p, pp in pools.items()}
    IX = Index(gens_by_p)
    V = E61.per_key_vectors(pools, fits, dets)
    X, elig = per_cred_inputs(pools, V)
    cells = [(d_, p, op) for d_ in dets for p in C.PROTOS for op in C.OPS]
    ones_m = {(d_, p): np.ones((1, 10)) for d_ in dets for p in C.PROTOS}
    full = components(pools, shared, V, X, elig, dets, ones_draw(shared), ones_m, gens_by_p)

    # ---------------------------------------------------------------- accessors (work on full or replicate dicts)
    def leg(src, cell, arm, pol, model, rho, k, q):
        det, p, op = cell
        return src[(det, p, arm, op, pol)][:, IX.legit(model, rho, k, q)].astype(np.float64)

    def worst_idx(src, cell, k):
        """Per row: generator index maximising pi0's fresh PMP (q*) averaged over the two arms (one choice per cell)."""
        det, p, op = cell
        cols = [IX.S[p][("GEN", k, g)] for g in IX.gens[p]]
        A = np.mean([src[(det, p, arm, op, "pi0")][:, cols] for arm in C.ARMS], axis=0)
        return np.argmax(A, axis=1), cols

    def pmp(src, cell, arm, pol, k, attacker, q=CS["q"]):
        det, p, op = cell
        M = src[(det, p, arm, op, pol)].astype(np.float64)
        if attacker == "fresh":
            return M[:, IX.S[p][("PMP", k, q)]]
        if attacker == "same":
            return M[:, IX.S[p][("SAME", None, q)]]
        wi, cols = worst_idx(src, cell, k)
        return M[:, cols][np.arange(M.shape[0]), wi]

    def J(src, cell, arm, pol, model, rho, k, attacker, cA=CS["c_A"], cU=CS["c_U"], cT=CS["c_T"], q=CS["q"]):
        g = lambda qq: leg(src, cell, arm, pol, model, rho, k, qq)
        return C.cost_J(pmp(src, cell, arm, pol, k, attacker, q), g("L"), g("G"), g("N"), cA, cU, cT)

    def elg(cell, arm):
        det, p, _ = cell
        return el[f"{det}|{p}|{arm}"] if arm != "dd" else el[f"{det}|{p}|leg"] and el[f"{det}|{p}|fix"]

    # ---------------------------------------------------------------- gates on the full sample
    gates = {}
    f1 = lambda a: float(a[0])
    if frozen is not None:
        fr, fr63 = frozen
        dev, n = 0.0, 0
        for cell in cells:
            det, p, op = cell
            for arm in C.ARMS:
                for pol in C.POLICIES:
                    ref = fr[(det, p, op, arm, pol)]
                    mine = {"L": leg(full, cell, arm, pol, "M2", 0.0, 3, "L"), "G": leg(full, cell, arm, pol, "M2", 0.0, 3, "G"),
                            "N": leg(full, cell, arm, pol, "M2", 0.0, 3, "N"), "PMP_q0.9": pmp(full, cell, arm, pol, 3, "fresh"),
                            "PMPsame_q0.9": pmp(full, cell, arm, pol, 3, "same"),
                            "J_cstar": J(full, cell, arm, pol, "M2", 0.0, 3, "fresh"),
                            "L1": leg(full, cell, arm, pol, "M2", 0.0, 1, "L"), "L5": leg(full, cell, arm, pol, "M2", 0.0, 5, "L")}
                    for g in IX.gens[p]:
                        mine[f"PMPgen_{g}"] = full[(det, p, arm, op, pol)][:, IX.S[p][("GEN", 3, g)]]
                    for k_, v_ in mine.items():
                        dev = max(dev, abs(f1(v_) - ref[k_])); n += 1
        gates["G0"] = {"max_abs_dev": dev, "n_values": n, "pass": dev <= TOL}
        dev63, n63 = 0.0, 0
        for cell in cells:
            det, p, op = cell
            for arm in C.ARMS:
                for pol in C.POLICIES:
                    for q in "LGN":
                        for mo, model in (("M0e", "M2E"), ("M1", "M1E")):
                            v_ = f1(leg(full, cell, arm, pol, model, 0.0, 3, q))
                            dev63 = max(dev63, abs(v_ - fr63[(det, p, op, arm, pol, mo, q)])); n63 += 1
        gates["G_E063"] = {"max_abs_dev": dev63, "n_values": n63, "pass": dev63 <= TOL}
        gates["G_E061_DIR"] = check_e061_dir()
    gates["G_STRUCT_full"] = structural_checks(full, cells, leg, pmp, J, IX)
    outdir.mkdir(parents=True, exist_ok=False)
    (outdir / "gates.json").write_text(json.dumps(gates, indent=1, default=bool))
    if not all(g["pass"] for g in gates.values()):
        log(f"GATE FAILED: {json.dumps(gates, default=bool)} -> stop; no bootstrap")
        return {"stopped_at_gate": True, "gates": gates}

    # ---------------------------------------------------------------- bootstrap
    reps = {}
    t0 = time.time()
    for b0 in range(0, B, CHUNK):
        n = min(CHUNK, B - b0)
        m_seed = {(d_, p): C.row_counts(rng_seed, n, 10) for d_ in dets for p in C.PROTOS}
        dr = shared.draw(rng_subj, rng_syn, n)
        for k_, v_ in components(pools, shared, V, X, elig, dets, dr, m_seed, gens_by_p).items():
            reps.setdefault(k_, []).append(v_.astype(np.float32))
        log(f"  bootstrap {b0 + n}/{B} ({time.time() - t0:.0f}s)")
    reps = {k_: np.concatenate(v_) for k_, v_ in reps.items()}             # float32 (protocol v4: memory)
    gates["G_STRUCT_replicates"] = structural_checks(reps, cells, leg, pmp, J, IX, replicate=True)
    (outdir / "gates.json").write_text(json.dumps(gates, indent=1, default=bool))
    if not gates["G_STRUCT_replicates"]["pass"]:
        log("G-STRUCT failed on replicates -> stop; no result written")
        return {"stopped_at_gate": True, "gates": gates}

    # ---------------------------------------------------------------- statistics
    def cellstat(fun, arm):
        cl = [c for c in cells if elg(c, arm)]
        if arm == "dd":
            fv = [f1(fun(full, c, "fix")) - f1(fun(full, c, "leg")) for c in cl]
            rv = np.stack([fun(reps, c, "fix").astype(np.float64) - fun(reps, c, "leg") for c in cl], 1)
        else:
            fv = [f1(fun(full, c, arm)) for c in cl]
            rv = np.stack([fun(reps, c, arm).astype(np.float64) for c in cl], 1)
        med = np.median(rv, 1)
        return {"n_cells": len(cl), "median": float(np.median(fv)), "ci95": C.pctl(med, 0.95),
                "ci9875": C.pctl(med, LEVEL)}

    def dJ(pol, model, rho, k, att):
        return lambda src, c, a: J(src, c, a, pol, model, rho, k, att) - J(src, c, a, "pi0", model, rho, k, att)

    def readings_for(model, rho, k, att):
        st = {"R-H1": cellstat(dJ("pi1", model, rho, k, att), "leg"),
              "fixed_pi1": cellstat(dJ("pi1", model, rho, k, att), "fix"),
              "R-H2": cellstat(dJ("pi1", model, rho, k, att), "dd"),
              "R-H4|leg": cellstat(dJ("pi4", model, rho, k, att), "leg"),
              "R-H4|fix": cellstat(dJ("pi4", model, rho, k, att), "fix")}
        for pol in ("pi2", "pi3"):
            for arm in C.ARMS:
                st[f"R-H3|{pol}|{arm}"] = cellstat(dJ(pol, model, rho, k, att), arm)
        return st

    def sign(st):
        lo, hi = st["ci9875"]
        return +1 if lo > 0 else -1 if hi < 0 else 0

    def label(ref_dir, s):
        if ref_dir == 0:
            return {0: "still indistinguishable", -1: "newly resolved better", +1: "newly resolved worse"}[s]
        return "unchanged" if s == ref_dir else "narrowed" if s == 0 else "reversed"

    grid = {}
    for rho in RHOS:
        for k in KS:
            for att in ATTACKERS:
                grid[(rho, k, att)] = readings_for("M2", rho, k, att)
        log(f"  readings rho={rho} done")
    base_dir = {r: sign(st) for r, st in grid[BASE].items()}
    rows = []
    for (rho, k, att), rd in grid.items():
        for r, st in rd.items():
            s = sign(st)
            rows.append(dict(model="M2", population="all", rho=rho, K=k, attacker=att, reading=r, n_cells=st["n_cells"],
                             median=st["median"], lo95=st["ci95"][0], hi95=st["ci95"][1], lo9875=st["ci9875"][0],
                             hi9875=st["ci9875"][1], label_vs_E061=label(E061_DIR[r], s),
                             label_vs_baseline=label(base_dir[r], s)))
    # descriptive: M0|E_K and M1|E_K readings (E063 extended to K = 1..5), fresh attacker
    for k in KS:
        for model, mname in (("M2E", "M0|E_K"), ("M1E", "M1|E_K")):
            for r, st in readings_for(model, 0.0, k, "fresh").items():
                s = sign(st)
                rows.append(dict(model=mname, population=f"E_{k}", rho=0.0 if model == "M2E" else "", K=k, attacker="fresh",
                                 reading=r, n_cells=st["n_cells"], median=st["median"], lo95=st["ci95"][0],
                                 hi95=st["ci95"][1], lo9875=st["ci9875"][0], hi9875=st["ci9875"][1],
                                 label_vs_E061=label(E061_DIR[r], s), label_vs_baseline=label(base_dir[r], s)))
    # rho* (K = 3, fresh) for R-H1, R-H2 and pi2
    rho_star = {}
    for r in ("R-H1", "R-H2", "R-H3|pi2|leg", "R-H3|pi2|fix"):
        dep = [rho for rho in RHOS if sign(grid[(rho, 3, "fresh")][r]) != base_dir[r]]
        rho_star[r] = min(dep) if dep else "none"
    k_range = {}
    for att in ATTACKERS:
        for r in ("R-H1", "R-H2", "R-H4|leg", "R-H4|fix"):
            k_range[f"{r}|{att}"] = [k for k in KS if sign(grid[(0.0, k, att)][r]) == base_dir[r]]
    # decomposition (fresh, c*)
    dec = {}
    for k in KS:
        for qn, pol, arm in (("dJ_pi1_leg", "pi1", "leg"), ("ddJ_pi1", "pi1", "dd"), ("dJ_pi4_leg", "pi4", "leg"),
                             ("dJ_pi4_fix", "pi4", "fix")):
            f0 = dJ(pol, "M2", 0.0, k, "fresh"); f0e = dJ(pol, "M2E", 0.0, k, "fresh"); f1e = dJ(pol, "M1E", 0.0, k, "fresh")
            dec[f"K{k}|{qn}|selection"] = cellstat(lambda s_, c, a: f0e(s_, c, a) - f0(s_, c, a), arm)
            dec[f"K{k}|{qn}|without_replacement"] = cellstat(lambda s_, c, a: f1e(s_, c, a) - f0e(s_, c, a), arm)
            dec[f"K{k}|{qn}|total_M1_vs_M0"] = cellstat(lambda s_, c, a: f1e(s_, c, a) - f0(s_, c, a), arm)
            for rho in RHOS[1:]:
                fr_ = dJ(pol, "M2", rho, k, "fresh"); fre = dJ(pol, "M2E", rho, k, "fresh")
                dec[f"K{k}|{qn}|persistence_rho{rho}|all"] = cellstat(lambda s_, c, a: fr_(s_, c, a) - f0(s_, c, a), arm)
                dec[f"K{k}|{qn}|persistence_rho{rho}|E_K"] = cellstat(lambda s_, c, a: fre(s_, c, a) - f0e(s_, c, a), arm)
    log("  decomposition done")
    # levels (median over cells, 95 %)
    lev = []
    for rho in RHOS:
        for k in KS:
            for pol in C.POLICIES:
                for arm in C.ARMS:
                    for q in "LGN":
                        st = cellstat(lambda s_, c, a: leg(s_, c, a, pol, "M2", rho, k, q), arm)
                        lev.append(dict(rho=rho, K=k, attacker="", policy=pol, arm=arm, quantity=q, median=st["median"],
                                        lo95=st["ci95"][0], hi95=st["ci95"][1]))
                    for att in ATTACKERS:
                        st = cellstat(lambda s_, c, a: J(s_, c, a, pol, "M2", rho, k, att), arm)
                        lev.append(dict(rho=rho, K=k, attacker=att, policy=pol, arm=arm, quantity="J_cstar",
                                        median=st["median"], lo95=st["ci95"][0], hi95=st["ci95"][1]))
                        if rho == 0.0:
                            st = cellstat(lambda s_, c, a: pmp(s_, c, a, pol, k, att), arm)
                            lev.append(dict(rho="any", K=k, attacker=att, policy=pol, arm=arm, quantity="PMP_q0.9",
                                            median=st["median"], lo95=st["ci95"][0], hi95=st["ci95"][1]))
    # rankings (full sample; ties to pi0 as in E061)
    cgrid = list(itertools.product(C.COST_GRID["c_A"], C.COST_GRID["c_U"], C.COST_GRID["c_T"], C.COST_GRID["q"]))
    rank = {}
    configs = [("M2", rho, k, att) for rho in RHOS for k in KS for att in ATTACKERS] + \
              [(m, 0.0, k, "fresh") for m in ("M2E", "M1E") for k in KS]
    for (model, rho, k, att) in configs:
        for arm in C.ARMS:
            cnt = {pol: 0 for pol in C.POLICIES}; gs = {pol: 0 for pol in C.POLICIES}
            for cell in cells:
                jf = [f1(J(full, cell, arm, pol, model, rho, k, att)) for pol in C.POLICIES]
                cnt[C.POLICIES[int(np.argmin(jf))]] += 1
                if att == "fresh":
                    for (cA, cU, cT, q) in cgrid:
                        jg = [f1(J(full, cell, arm, pol, model, rho, k, att, cA, cU, cT, q)) for pol in C.POLICIES]
                        gs[C.POLICIES[int(np.argmin(jg))]] += 1
            name = {"M2": f"M2|rho={rho}", "M2E": "M0|E_K", "M1E": "M1|E_K"}[model]
            rank[f"{name}|K={k}|{att}|{arm}"] = {"argmin_cstar": cnt,
                                                 "grid_share": ({p_: v / (len(cells) * len(cgrid)) for p_, v in gs.items()}
                                                                if att == "fresh" else None)}
    ref_rank = None   # E063's M0 entries are E061's own design (E063 G0 reproduced E061 exactly)
    if (E063_RUN / "RESULTS.json").exists() and frozen is not None:
        r63 = json.loads((E063_RUN / "RESULTS.json").read_text())
        ref_rank = {"E063_argmin_cstar_counts": r63["argmin_cstar_counts"], "E063_grid_argmin_share": r63["grid_argmin_share"]}
    # ---------------------------------------------------------------- write
    with open(outdir / "grid_readings.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(outdir / "levels.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(lev[0])); w.writeheader(); w.writerows(lev)
    (outdir / "rho_star.json").write_text(json.dumps({"rho_star_K3_fresh": rho_star, "K_range_rho0": k_range,
                                                      "baseline": list(BASE), "baseline_signs": base_dir}, indent=1))
    (outdir / "decomposition.json").write_text(json.dumps(dec, indent=1))
    (outdir / "rankings.json").write_text(json.dumps({"E064": rank, "reference": ref_rank}, indent=1))
    res = {"gates": gates, "B": B, "seed": SEED, "rho_star": rho_star, "K_range": k_range,
           "n_configurations": len(grid), "n_readings": len(grid) * 9}
    (outdir / "RESULTS.json").write_text(json.dumps(res, indent=1, default=bool))
    return res


def structural_checks(src, cells, leg, pmp, J, IX, replicate=False) -> dict:
    """G-STRUCT: invariances that hold by mathematics (protocol v4 table). Per-row checks (all populations/models for
    pi4 dL = dN = 0; M2: dG(pi4-pi0) >= 0, non-increasing in rho, non-decreasing in K; dPMP_fresh(pi4-pi0) <= 0 and
    non-increasing in K; dJ(pi4-pi0) non-increasing in rho for every attacker and non-decreasing in K for the same
    attacker) run on the full sample AND every replicate. K = 1 and rho = 1 identities run on the full sample."""
    tol = 1e-5 if replicate else TOL                       # replicates are stored in float32
    bad = {}

    def flag(name, ok):
        if not ok:
            bad[name] = bad.get(name, 0) + 1
    for cell in cells:
        for arm in C.ARMS:
            for k in KS:
                for model, rhos in (("M2", RHOS), ("M2E", RHOS), ("M1E", (None,))):
                    for rho in rhos:
                        for q in "LN":
                            d = leg(src, cell, arm, "pi4", model, rho, k, q) - leg(src, cell, arm, "pi0", model, rho, k, q)
                            flag(f"d{q}_pi4_nonzero", np.max(np.abs(d)) <= tol)
                for rho in RHOS:                                       # dG(pi4-pi0) >= 0 (M2)
                    dG = leg(src, cell, arm, "pi4", "M2", rho, k, "G") - leg(src, cell, arm, "pi0", "M2", rho, k, "G")
                    flag("dG_pi4_negative", np.all(dG >= -tol))
                    if rho > 0:
                        prevG = leg(src, cell, arm, "pi4", "M2", RHOS[RHOS.index(rho) - 1], k, "G") - \
                                leg(src, cell, arm, "pi0", "M2", RHOS[RHOS.index(rho) - 1], k, "G")
                        flag("dG_pi4_increases_in_rho", np.all(dG <= prevG + tol))
                    if k > 1:
                        prevK = leg(src, cell, arm, "pi4", "M2", rho, k - 1, "G") - leg(src, cell, arm, "pi0", "M2", rho, k - 1, "G")
                        flag("dG_pi4_decreases_in_K", np.all(dG >= prevK - tol))
                        dJs = J(src, cell, arm, "pi4", "M2", rho, k, "same") - J(src, cell, arm, "pi0", "M2", rho, k, "same")
                        dJs0 = J(src, cell, arm, "pi4", "M2", rho, k - 1, "same") - J(src, cell, arm, "pi0", "M2", rho, k - 1, "same")
                        flag("dJ_pi4_same_decreases_in_K", np.all(dJs >= dJs0 - tol))
                dP = pmp(src, cell, arm, "pi4", k, "fresh") - pmp(src, cell, arm, "pi0", k, "fresh")
                flag("dPMP_pi4_fresh_positive", np.all(dP <= tol))
                if k > 1:
                    dP0 = pmp(src, cell, arm, "pi4", k - 1, "fresh") - pmp(src, cell, arm, "pi0", k - 1, "fresh")
                    flag("dPMP_pi4_fresh_increases_in_K", np.all(dP <= dP0 + tol))
                for att in ATTACKERS:
                    prev = None
                    for rho in RHOS:
                        d = J(src, cell, arm, "pi4", "M2", rho, k, att) - J(src, cell, arm, "pi0", "M2", rho, k, att)
                        if prev is not None:
                            flag("dJ_pi4_increases_in_rho", np.all(d <= prev + tol))
                        prev = d
            if replicate:
                continue
            for pol in C.POLICIES:
                for q in "LGN":
                    v = [leg(src, cell, arm, pol, "M2", rho, 1, q) for rho in RHOS] + [leg(src, cell, arm, pol, "M1E", None, 1, q)]
                    flag("K1_models_differ", max(np.max(np.abs(x - v[0])) for x in v) <= tol)
                    flag("K1_population_differs", np.max(np.abs(leg(src, cell, arm, pol, "M2E", 0.0, 1, q)
                                                                 - leg(src, cell, arm, pol, "M2", 0.0, 1, q))) <= tol)
                flag("K1_fresh_ne_same", np.max(np.abs(pmp(src, cell, arm, pol, 1, "fresh") - pmp(src, cell, arm, pol, 1, "same"))) <= tol)
                lr = [leg(src, cell, arm, pol, "M2", 1.0, k, "L") for k in KS]
                flag("rho1_L_depends_on_K", max(np.max(np.abs(x - lr[0])) for x in lr) <= tol)
    return {"pass": not bad, "violations": bad, "replicate_level": replicate}


def load_frozen():
    fr = {}
    for r in csv.DictReader(open(E061_SESSION)):
        fr.setdefault((r["det"], r["protocol"], r["op"], r["arm"], r["policy"]), {})[r["quantity"]] = float(r["estimate"])
    fr63 = {}
    for r in csv.DictReader(open(E063_RUN / "session_models.csv")):
        if r["model"] in ("M0e", "M1") and r["quantity"] in ("L", "G", "N"):
            fr63[(r["det"], r["protocol"], r["op"], r["arm"], r["policy"], r["model"], r["quantity"])] = float(r["estimate"])
    return fr, fr63


# ------------------------------------------------------------------------------ authorisation
def check_test_authorisation() -> dict:
    auth = C.check_authorisation(EXP, SOURCES_RUN)
    missing = [k for k in ("authorised_by", "date", "scope", "protocol_sha256", "validation_fit_sha256",
                           "e059_post_test_reads_decision", "freeze_commit") if not auth.get(k)]
    if missing:
        raise C.TestAccessDenied(f"E064 authorisation lacks {missing}")
    if not str(auth["scope"]).startswith("e064_test"):
        raise C.TestAccessDenied("E064 authorisation scope must start with 'e064_test'")
    if auth["protocol_sha256"] != C.sha256(PROTOCOL):
        raise C.TestAccessDenied("protocol differs from the authorised version")
    if auth["validation_fit_sha256"] != FIT_SHA or C.sha256(FIT) != FIT_SHA:
        raise C.TestAccessDenied("validation_fit.json is not the frozen E061 F2 file")
    return auth


def check_pilot_authorisation() -> dict:
    f = OUT / "VALIDATION_AUTHORISATION.json"
    if not f.exists():
        raise C.TestAccessDenied(f"E064: no {C.rel(f)}")
    a = json.loads(f.read_text())
    if not str(a.get("scope", "")).startswith("e064_pilot") or not a.get("authorised_by"):
        raise C.TestAccessDenied("E064 pilot authorisation lacks scope 'e064_pilot' or authorised_by")
    if a.get("protocol_sha256") != C.sha256(PROTOCOL):
        raise C.TestAccessDenied("protocol differs from the authorised version")
    for s in SOURCES_RUN:
        if a.get("script_sha256", {}).get(C.rel(s)) != C.sha256(s):
            raise C.TestAccessDenied(f"E064: {C.rel(s)} differs from the authorised version")
    return a


def launch_state() -> dict:
    run = lambda *a: subprocess.run(a, cwd=ROOT, capture_output=True, text=True).stdout
    return {"git": run("git", "rev-parse", "HEAD").strip(),
            "dirty_tracked": run("git", "status", "--porcelain", "--untracked-files=no").splitlines(),
            "protocol_sha256": C.sha256(PROTOCOL), "script_sha256": {C.rel(s): C.sha256(s) for s in SOURCES_RUN},
            "python": sys.version.split()[0], "numpy": np.__version__, "host": platform.node(),
            "launch_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "command": " ".join(sys.argv), "gpu": "none (CPU only)"}


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pilot", action="store_true"); g.add_argument("--test", action="store_true")
    g.add_argument("--check-auth", action="store_true")
    a = ap.parse_args()
    dets = C.HEADLINE
    if a.check_auth:
        for name, f in (("pilot", check_pilot_authorisation), ("test", check_test_authorisation)):
            try:
                f(); print(f"{name} authorisation: OK")
            except C.TestAccessDenied as e:
                print(f"{name} authorisation: REFUSED -- {e}")
        return 0
    prov = launch_state()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    mode = "pilot" if a.pilot else "test"
    try:
        auth = check_pilot_authorisation() if a.pilot else check_test_authorisation()
    except C.TestAccessDenied as e:
        print(f"REFUSED before any data read: {e}"); return 3
    LOGDIR.mkdir(parents=True, exist_ok=True)
    logf = open(LOGDIR / f"{mode}_{stamp}.log", "w")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()

    alog = C.AccessLog(EXP, authorisation=auth)
    if a.pilot:
        pools = load_pool("pilot", dets, alog)
        assert C.TEST_SCORE not in {e["key"] for e in alog.entries}
        fits = E61.fit_all(pools, dets); el = E61.eligibility(fits, pools, dets)
        outdir = OUT / f"pilot_validation_{stamp}"
        res = run_analysis(pools, fits, el, dets, B_PILOT, outdir, log, frozen=None)
        (outdir / "README.txt").write_text("VALIDATION-ONLY PILOT (code exercise; policies fitted and evaluated on validation;"
                                           " circular). B=200. NOT EVIDENCE; never cited. No test score was read.\n")
    else:
        rec = json.loads(FIT.read_text()); fits, el = rec["fits"], rec["eligibility"]
        pools = load_pool("test", dets, alog, authorised=True)
        outdir = OUT / f"test_run_{stamp}"
        res = run_analysis(pools, fits, el, dets, B_PRIMARY, outdir, log, frozen=load_frozen())
        prov["e061_session_csv_sha256"] = C.sha256(E061_SESSION)
        prov["e063_session_models_sha256"] = C.sha256(E063_RUN / "session_models.csv")
    alog.dump(outdir / "access_log.json")
    prov.update({"auth": auth, "B": B_PILOT if a.pilot else B_PRIMARY, "seed": SEED,
                 "end_time": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    (outdir / "provenance.json").write_text(json.dumps(prov, indent=1))
    log(f"{mode} finished -> {outdir}" if not res.get("stopped_at_gate") else f"{mode} stopped at a gate -> {outdir}")
    return 0 if not res.get("stopped_at_gate") else 2


if __name__ == "__main__":
    sys.exit(main())
