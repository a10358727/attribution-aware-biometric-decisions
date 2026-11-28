#!/usr/bin/env python3
"""E073 -- distribution-aware access-policy evaluation (Task C; post hoc, exploratory).

Frozen plan: docs/C5/C5_Research_Protocol.md §4-§6; freeze record docs/C5/C5_FREEZE.json.
Reuses the frozen E061 F2 parameters (artifacts/results/E061/validation_fit.json) unchanged.

    python experiments/E073_distribution_policy/run_e073.py --pilot
    python experiments/E073_distribution_policy/run_e073.py --test     # needs E073/TEST_READ_AUTHORISATION.json
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
from experiments.tcss_lib import core as C                                   # noqa: E402
from experiments.tcss_lib.pool import SharedUnits, load_pool, ones_draw    # noqa: E402
from experiments.c5_lib import stats as S                                    # noqa: E402
sys.path.insert(0, str(C.ROOT / "experiments" / "E063_session_robustness"))
import session_models as SM                                                  # noqa: E402

EXP = "E073"
SOURCES_RUN = [HERE, C.ROOT / "experiments/c5_lib/stats.py", C.ROOT / "experiments/tcss_lib/core.py",
               C.ROOT / "experiments/tcss_lib/pool.py", C.ROOT / "experiments/E063_session_robustness/session_models.py"]
DETS = C.HEADLINE
B = 2000
SEED = 730701
K = C.C_STAR["K"]
CS = C.C_STAR
PHIS = (0.0, 0.02, 0.05, 0.1, 0.2)
LAMBDAS = (0.0, 0.5, 1.0, 2.0, 5.0)
DELTAS = (0.0, 0.01)
LVL_P2 = 1 - 0.05 / 8
POL = C.POLICIES


def log(msg, fh=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fh:
        fh.write(line + "\n"); fh.flush()


def git_state():
    run = lambda *a: subprocess.run(a, cwd=C.ROOT, capture_output=True, text=True).stdout.strip()
    return {"sha": run("git", "rev-parse", "HEAD"), "dirty": run("git", "status", "--porcelain", "--untracked-files=no").splitlines()}


def cells():
    return [(d, p, op) for d in DETS for p in C.PROTOS for op in C.OPS]


def per_cred_counts(z, units):
    R = np.bincount(units.cred_of_capture, weights=(z == C.REJ).astype(float), minlength=units.F)
    U = np.bincount(units.cred_of_capture, weights=(z == C.STEP).astype(float), minlength=units.F)
    return R, U


def session(Rc, Uc, n, model, phi_fail=0.0):
    """Per-credential (L, G, N, valid) under M0 (all) or M1 (n >= K)."""
    if model == "M0":
        L, G, N = S.session_phi(Rc / n, Uc / n, K, phi_fail)
        return L, G, N, np.ones(len(n), bool)
    L, G, N, v = SM.m1(Rc, Uc, n, K)
    L = np.where(v, L + phi_fail * G, np.nan)
    return L, G, N, v


def vectors(pools, fits):
    """Per (det, p, arm, seed, op, policy): credential counts and synthetic outcome indicators."""
    Vv = {}
    for p, pp in pools.items():
        for det in DETS:
            for arm in C.ARMS:
                for s in C.SEEDS:
                    for op, cond in C.OPS.items():
                        prm = fits[f"{det}|{p}|{arm}|{s}|{op}"]
                        r0, r1 = pp.real[(det, arm, s, C.ANCHOR)], pp.real[(det, arm, s, cond)]
                        y0, y1 = pp.syn[(det, arm, s, C.ANCHOR)], pp.syn[(det, arm, s, cond)]
                        for pol in POL:
                            zr, _ = C.decide(pol, prm, r0, r1)
                            zs, _ = C.decide(pol, prm, y0, y1)
                            Rc, Uc = per_cred_counts(zr, pp.units)
                            Vv[(det, p, arm, s, op, pol)] = (Rc, Uc, (zs == C.ACC).astype(float), (zs == C.STEP).astype(float))
    return Vv


def endpoints(pools, shared, Vv, d, m_seed, model, cU, cT, phi_fail=0.0, pmp_kind="fresh", q=CS["q"], pols=POL):
    """Seed-averaged endpoints per (det, p, arm, op, policy). Returns dict -> (R,) arrays, plus
    subject burden per seed for worse-off computations (computed inside)."""
    R_ = d["subj"].shape[0]
    out = {}
    for p, pp in pools.items():
        u = pp.units; n = u.n_f().astype(float)
        Wc = shared.cred_weights(d, p, u); Wsub = shared.subj_weights(d, p)
        Ws = shared.syn_weights(d, p, len(pp.syn_pid))
        for det in DETS:
            m = m_seed[(det, p)]
            for arm in C.ARMS:
                for op in C.OPS:
                    per = {}
                    for j, s in enumerate(C.SEEDS):
                        bsub0 = None
                        for pol in pols if "pi0" in pols else ("pi0",) + tuple(pols):
                            Rc, Uc, acc, stp = Vv[(det, p, arm, s, op, pol)]
                            ph = phi_fail if pol == "pi4" else 0.0
                            L, G, N, valid = session(Rc, Uc, n, model, ph)
                            Lz, Gz, Nz = (np.where(valid, x, 0.0) for x in (L, G, N))
                            Wv = Wc * valid[None, :]
                            mL, mG, mN = (S.wmean_rows(x, Wv) for x in (Lz, Gz, Nz))
                            Pv = (Ws @ acc) / Ws.sum(1); Vs = (Ws @ stp) / Ws.sum(1)
                            pmp = C.pmp_fresh(Pv, Vs, q, K) if pmp_kind == "fresh" else C.pmp_same(Pv, Vs, q)
                            bf = Lz + cU * Gz + cT * (Nz - 1.0)
                            A = S.subject_matrix(u.subj_of_cred, u.U, valid)
                            has = A.sum(1) > 0
                            bsub = A @ bf
                            Wsv = Wsub * has[None, :]
                            e = {"L": mL, "G": mG, "N": mN, "PMP": pmp,
                                 "meanb_cred": mL + cU * mG + cT * (mN - 1.0),
                                 "meanb_subj": S.wmean_rows(bsub, Wsv),
                                 "CVaR80": S.wtail_mean(bsub[has], Wsv[:, has], 0.8),
                                 "CVaR90": S.wtail_mean(bsub[has], Wsv[:, has], 0.9),
                                 "n_subj_used": np.full(R_, float(has.sum()))}
                            if pol == "pi0":
                                bsub0 = bsub
                            else:
                                dlt = bsub - bsub0
                                e["worse_gt0"] = S.wshare((dlt > 1e-12) & has, Wsv)
                                e["worse_gt001"] = S.wshare((dlt > 0.01) & has, Wsv)
                                for qq in (10, 50, 90):
                                    e[f"dQ{qq}"] = S.wquantile(dlt[has], Wsv[:, has], qq / 100)
                            if pol not in pols:
                                continue
                            for nm, v in e.items():
                                per.setdefault((pol, nm), np.empty((R_, 10)))[:, j] = v
                    for (pol, nm), E in per.items():
                        out[(det, p, arm, op, pol, nm)] = C.seed_average(E, m)
    return out


def J0(e, key, cA):
    g = lambda nm: e[key + (nm,)]
    return cA * g("PMP") + g("meanb_cred")


def evaluations(e, det, p, arm, op, cA=CS["c_A"]):
    """Per-policy objective arrays (R,) for E-0, E-1, E-7(lambda) and constraint quantities."""
    k = lambda pol: (det, p, arm, op, pol)
    ev = {"E0": {pol: J0(e, k(pol), cA) for pol in POL},
          "E1": {pol: cA * e[k(pol) + ("PMP",)] + e[k(pol) + ("meanb_subj",)] for pol in POL}}
    for lam in LAMBDAS:
        ev[f"E7_l{lam}"] = {pol: cA * e[k(pol) + ("PMP",)] + e[k(pol) + ("meanb_subj",)] + lam * e[k(pol) + ("CVaR80",)]
                            for pol in POL}
    cv = {pol: e[k(pol) + ("CVaR80",)] for pol in POL}
    for dl in DELTAS:
        feas = {pol: cv[pol] <= cv["pi0"] + dl + 1e-12 for pol in POL}
        ev[f"E6_d{dl}"] = {pol: np.where(feas[pol], ev["E0"][pol], np.inf) for pol in POL}
    return ev


def argmin_pol(obj):
    M = np.stack([obj[pol] for pol in POL], 1)
    return np.argmin(M, 1)          # ties -> first (pi0 first, as E061 counts ties with pi0)


def pareto(points):
    """Indices of non-dominated policies (minimise all coordinates)."""
    nd = []
    for i, a in enumerate(points):
        dom = any(all(b[j] <= a[j] for j in range(len(a))) and any(b[j] < a[j] for j in range(len(a)))
                  for jj, b in enumerate(points) if jj != i)
        if not dom:
            nd.append(i)
    return nd


def gate_reproduce(e_full, fh):
    rows = list(csv.DictReader(open(C.ART / "E061" / "session.csv")))
    name = {"L": "L", "G": "G", "N": "N", "PMP_q0.9": "PMP"}
    md, n = 0.0, 0
    for r in rows:
        key = (r["det"], r["protocol"], r["arm"], r["op"], r["policy"])
        if r["quantity"] in name:
            got = float(e_full[key + (name[r["quantity"]],)][0])
        elif r["quantity"] == "J_cstar":
            got = float(J0(e_full, key, CS["c_A"])[0])
        else:
            continue
        md = max(md, abs(got - float(r["estimate"]))); n += 1
    ok = md < 1e-9
    log(f"  G-R E061 reproduction: n={n} max|diff|={md:.3g} -> {'PASS' if ok else 'FAIL'}", fh)
    return {"n": n, "max_abs_diff": md, "pass": ok}


def write_csv(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pilot", action="store_true"); g.add_argument("--test", action="store_true")
    ap.add_argument("--B", type=int, default=None)
    a = ap.parse_args()
    pilot = a.pilot
    out = C.ART / EXP / ("pilot_validation" if pilot else "")
    out.mkdir(parents=True, exist_ok=True)
    logdir = C.ROOT / "artifacts" / "logs" / EXP; logdir.mkdir(parents=True, exist_ok=True)
    fh = open(logdir / f"{'pilot' if pilot else 'test'}_{time.strftime('%Y%m%d_%H%M%S')}.log", "w")
    B_ = a.B or (200 if pilot else B)
    acc = C.AccessLog(EXP)
    if not pilot:
        acc.authorisation = C.check_authorisation(EXP, SOURCES_RUN)
    log(f"E073 {'PILOT (validation stand-in; not evidence)' if pilot else 'TEST (authorised)'}; B={B_}", fh)
    t0 = time.time()
    fits = json.load(open(C.ART / "E061" / "validation_fit.json"))["fits"]
    pools = load_pool("pilot" if pilot else "test", DETS, acc, authorised=not pilot)
    shared = SharedUnits(pools)
    Vv = vectors(pools, fits)
    log("  decision vectors built", fh)
    ss = np.random.SeedSequence(SEED).spawn(3)
    rng_seed, rng_subj, rng_syn = (np.random.default_rng(x) for x in ss)
    d = shared.draw(rng_subj, rng_syn, B_)
    full = ones_draw(shared)
    d = {k: np.vstack([full[k], v]) for k, v in d.items()}                 # row 0 = full sample
    m_seed = {(dd, p): np.vstack([np.ones((1, 10)), C.row_counts(rng_seed, B_, 10)]) for dd in DETS for p in C.PROTOS}

    gates = {}
    e = endpoints(pools, shared, Vv, d, m_seed, "M0", CS["c_U"], CS["c_T"])
    log(f"  M0 endpoints ({time.time() - t0:.0f}s)", fh)
    if not pilot:
        gates["G-R_E061"] = gate_reproduce(e, fh)
        if not gates["G-R_E061"]["pass"]:
            json.dump(gates, open(out / "gates.json", "w"), indent=1); acc.dump(out / "access_log.json")
            raise SystemExit("G-R failed: stop (no adjustment)")

    # ---------------- per-cell tables and primary analyses
    pol_rows, min_rows, pareto_rows = [], [], []
    agree = {}
    for (det, p, op) in cells():
        for arm in C.ARMS:
            for pol in POL:
                for nm in ("PMP", "L", "G", "N", "meanb_cred", "meanb_subj", "CVaR80", "CVaR90", "worse_gt0",
                           "worse_gt001", "dQ10", "dQ50", "dQ90", "n_subj_used"):
                    key = (det, p, arm, op, pol, nm)
                    if key not in e:
                        continue
                    v = e[key]; lo, hi = C.pctl(v[1:], 0.95)
                    pol_rows.append(dict(det=det, protocol=p, op=op, arm=arm, policy=pol, quantity=nm,
                                         estimate=float(v[0]), lo95=lo, hi95=hi))
            ev = evaluations(e, det, p, arm, op)
            base = argmin_pol(ev["E0"])
            for nm, obj in ev.items():
                am = argmin_pol(obj)
                agree.setdefault((nm, arm), []).append(float(np.mean(am[1:] == base[1:])))
                min_rows.append(dict(det=det, protocol=p, op=op, arm=arm, evaluation=nm, minimizer=POL[am[0]],
                                     e0_minimizer=POL[base[0]], changed=bool(am[0] != base[0]),
                                     boot_share_minimizer_eq_full=float(np.mean(am[1:] == am[0])),
                                     boot_share_agree_with_E0=float(np.mean(am[1:] == base[1:]))))
            pts = [(float(e[(det, p, arm, op, pol, "PMP")][0]), float(e[(det, p, arm, op, pol, "meanb_subj")][0]),
                    float(e[(det, p, arm, op, pol, "CVaR80")][0])) for pol in POL]
            nd = pareto(pts)
            Rr = len(e[(det, p, arm, op, "pi0", "PMP")])
            freq = np.zeros(len(POL))
            for r in range(1, Rr):
                ptr = [(e[(det, p, arm, op, pol, "PMP")][r], e[(det, p, arm, op, pol, "meanb_subj")][r],
                        e[(det, p, arm, op, pol, "CVaR80")][r]) for pol in POL]
                for i in pareto(ptr):
                    freq[i] += 1
            for i, pol in enumerate(POL):
                pareto_rows.append(dict(det=det, protocol=p, op=op, arm=arm, policy=pol, nondominated_full=i in nd,
                                        boot_share_nondominated=freq[i] / (Rr - 1)))
    write_csv(out / "policy_endpoints.csv", pol_rows)
    write_csv(out / "minimizers_cstar.csv", min_rows)
    write_csv(out / "pareto3.csv", pareto_rows)
    cp1 = {}
    for arm in C.ARMS:
        for nm in sorted({r["evaluation"] for r in min_rows}):
            rs = [r for r in min_rows if r["arm"] == arm and r["evaluation"] == nm]
            cp1[f"{nm}|{arm}"] = {"n_changed": sum(r["changed"] for r in rs), "n_cells": len(rs),
                                  "minimizers": {pp: sum(r["minimizer"] == pp for r in rs) for pp in POL},
                                  "median_boot_agree_with_E0": float(np.median(agree[(nm, arm)]))}
    def cls(n):
        return "changes selection" if n >= 4 else "no change" if n == 0 else "marginal"
    cp1_class = {arm: {"E6_d0.0": cls(cp1[f"E6_d0.0|{arm}"]["n_changed"]),
                       "E7_l1.0": cls(cp1[f"E7_l1.0|{arm}"]["n_changed"]),
                       "E1": cp1[f"E1|{arm}"]["n_changed"]} for arm in C.ARMS}

    cp2 = {}
    for pol in ("pi1", "pi4"):
        for arm in C.ARMS:
            for nm in ("meanb_subj", "CVaR80", "worse_gt0", "worse_gt001", "PMP", "dQ90"):
                vals = []
                for (det, p, op) in cells():
                    v = e[(det, p, arm, op, pol, nm)]
                    if nm in ("meanb_subj", "CVaR80", "PMP"):
                        v = v - e[(det, p, arm, op, "pi0", nm)]
                    vals.append(v)
                M = np.median(np.stack(vals, 1), 1)
                lev = LVL_P2 if nm in ("meanb_subj", "CVaR80") else 0.95
                cp2[f"{pol}|{arm}|{nm}"] = {"median": float(M[0]), "ci95": C.pctl(M[1:], 0.95),
                                           "ci_decision": C.pctl(M[1:], lev), "decision_level": lev}
    # diagnostic: spread across policies relative to interval width
    diag = {}
    for arm in C.ARMS:
        for nm in ("meanb_subj", "CVaR80", "PMP"):
            spreads, widths = [], []
            for (det, p, op) in cells():
                vals = [float(e[(det, p, arm, op, pol, nm)][0]) for pol in POL]
                spreads.append(max(vals) - min(vals))
                widths.append(np.median([np.subtract(*reversed(C.pctl(e[(det, p, arm, op, pol, nm)][1:], 0.95))) for pol in POL]))
            diag[f"{nm}|{arm}"] = {"median_spread_across_policies": float(np.median(spreads)),
                                   "median_ci95_width": float(np.median(widths))}
    log(f"  primary analyses done ({time.time() - t0:.0f}s)", fh)

    # ---------------- C-S1 retry model M1
    e1 = endpoints(pools, shared, Vv, d, m_seed, "M1", CS["c_U"], CS["c_T"])
    s1_rows, s1_min = [], []
    for (det, p, op) in cells():
        for arm in C.ARMS:
            for pol in POL:
                for nm in ("meanb_cred", "meanb_subj", "CVaR80", "worse_gt0", "n_subj_used"):
                    key = (det, p, arm, op, pol, nm)
                    if key in e1:
                        v = e1[key]; lo, hi = C.pctl(v[1:], 0.95)
                        s1_rows.append(dict(det=det, protocol=p, op=op, arm=arm, policy=pol, quantity=nm,
                                            estimate=float(v[0]), lo95=lo, hi95=hi))
            ev = evaluations(e1, det, p, arm, op)
            for nm in ("E0", "E1", "E6_d0.0", "E7_l1.0"):
                s1_min.append(dict(det=det, protocol=p, op=op, arm=arm, evaluation=nm, minimizer=POL[argmin_pol(ev[nm])[0]],
                                   m0_minimizer=[r["minimizer"] for r in min_rows if (r["det"], r["protocol"], r["op"], r["arm"], r["evaluation"]) == (det, p, op, arm, nm)][0]))
    write_csv(out / "cs1_m1_endpoints.csv", s1_rows); write_csv(out / "cs1_m1_minimizers.csv", s1_min)
    log(f"  C-S1 done ({time.time() - t0:.0f}s)", fh)

    # ---------------- C-S3 hypothetical secondary-check failure (pi4 only, plus pi0 reference)
    s3_rows = []
    flips = {}
    for ph in PHIS:
        e3 = endpoints(pools, shared, Vv, d, m_seed, "M0", CS["c_U"], CS["c_T"], phi_fail=ph, pols=("pi0", "pi4"))
        for arm in C.ARMS:
            vals = {nm: [] for nm in ("dJ", "dmeanb", "dCVaR80", "worse_gt0")}
            nflip = 0
            for (det, p, op) in cells():
                k4, k0 = (det, p, arm, op, "pi4"), (det, p, arm, op, "pi0")
                dJ = J0(e3, k4, CS["c_A"]) - J0(e3, k0, CS["c_A"])
                vals["dJ"].append(dJ)
                vals["dmeanb"].append(e3[k4 + ("meanb_subj",)] - e3[k0 + ("meanb_subj",)])
                vals["dCVaR80"].append(e3[k4 + ("CVaR80",)] - e3[k0 + ("CVaR80",)])
                vals["worse_gt0"].append(e3[k4 + ("worse_gt0",)])
                base_dJ = float(J0(e, k4, CS["c_A"])[0] - J0(e, k0, CS["c_A"])[0])
                nflip += int(np.sign(dJ[0]) != np.sign(base_dJ))
            flips[(ph, arm)] = nflip
            for nm, vs in vals.items():
                M = np.median(np.stack(vs, 1), 1)
                s3_rows.append(dict(phi=ph, arm=arm, quantity=nm, median=float(M[0]), lo95=C.pctl(M[1:], 0.95)[0],
                                    hi95=C.pctl(M[1:], 0.95)[1], n_cells_sign_flip_dJ=nflip))
    write_csv(out / "cs3_secondary_failure.csv", s3_rows)
    e074_go_i = any(flips[(ph, arm)] >= 4 for ph in PHIS if ph <= 0.1 for arm in C.ARMS)
    log(f"  C-S3 done ({time.time() - t0:.0f}s)", fh)

    # ---------------- C-S2 cost grid (full sample) and C-S4 same-image attacker
    d1 = {k: v[:1] for k, v in d.items()}; m1s = {k: v[:1] for k, v in m_seed.items()}
    grid_rows = []
    G = C.COST_GRID
    for cU, cT in itertools.product(G["c_U"], G["c_T"]):
        for q in G["q"]:
            eg = endpoints(pools, shared, Vv, d1, m1s, "M0", cU, cT, q=q)
            for cA in G["c_A"]:
                for (det, p, op) in cells():
                    for arm in C.ARMS:
                        ev = evaluations(eg, det, p, arm, op, cA=cA)
                        rec = dict(c_A=cA, c_U=cU, c_T=cT, q=q, det=det, protocol=p, op=op, arm=arm)
                        for nm in ("E0", "E1", "E6_d0.0", "E7_l1.0"):
                            rec[nm] = POL[argmin_pol(ev[nm])[0]]
                        grid_rows.append(rec)
    write_csv(out / "cs2_cost_grid_minimizers.csv", grid_rows)
    es = endpoints(pools, shared, Vv, d1, m1s, "M0", CS["c_U"], CS["c_T"], pmp_kind="same")
    s4 = []
    for (det, p, op) in cells():
        for arm in C.ARMS:
            ev = evaluations(es, det, p, arm, op)
            s4.append(dict(det=det, protocol=p, op=op, arm=arm, **{nm: POL[argmin_pol(ev[nm])[0]] for nm in ("E0", "E1", "E7_l1.0")}))
    write_csv(out / "cs4_same_image_minimizers.csv", s4)
    log(f"  C-S2, C-S4 done ({time.time() - t0:.0f}s)", fh)

    grid_summary = {}
    for nm in ("E0", "E1", "E6_d0.0", "E7_l1.0"):
        grid_summary[nm] = {arm: {pol: float(np.mean([r[nm] == pol for r in grid_rows if r["arm"] == arm])) for pol in POL}
                            for arm in C.ARMS}
    grid_summary["share_points_E7_differs_from_E0"] = {arm: float(np.mean([r["E7_l1.0"] != r["E0"] for r in grid_rows if r["arm"] == arm])) for arm in C.ARMS}
    grid_summary["share_points_E6_differs_from_E0"] = {arm: float(np.mean([r["E6_d0.0"] != r["E0"] for r in grid_rows if r["arm"] == arm])) for arm in C.ARMS}

    outcome = {"pilot_not_evidence": pilot, "C_P1": cp1_class, "C_P1_detail": cp1,
               "C_P2": cp2, "diagnostic_spread": diag, "C_S2_grid": grid_summary,
               "C_S3_sign_flips": {f"phi={k[0]}|{k[1]}": v for k, v in flips.items()},
               "E074_gate_condition_i": bool(e074_go_i)}
    json.dump(outcome, open(out / "OUTCOME.json", "w"), indent=1)
    json.dump(gates, open(out / "gates.json", "w"), indent=1)
    acc.dump(out / "access_log.json")
    prov = {"experiment": EXP, "mode": "pilot" if pilot else "test", "git": git_state(), "python": sys.version.split()[0],
            "numpy": np.__version__, "host": platform.node(), "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "B": B_, "seed": SEED, "runtime_s": time.time() - t0,
            "script_sha256": {C.rel(s): C.sha256(s) for s in SOURCES_RUN},
            "pool_checks": {p: pp.checks for p, pp in pools.items()}}
    json.dump(prov, open(out / "provenance.json", "w"), indent=1)
    log(f"E073 done in {time.time() - t0:.0f}s", fh)


if __name__ == "__main__":
    main()
