#!/usr/bin/env python3
"""E061 -- access-decision policy simulation (post-hoc secondary analysis; simulation).

Frozen analysis plan: docs/experiments/E061_policy_simulation.md (F1, commit 3ff5b85).

    python experiments/E061_policy_simulation/run_e061.py --fit      # F2: validation_fit.json (validation only)
    python experiments/E061_policy_simulation/run_e061.py --pilot    # validation-only pilot (not evidence)
    python experiments/E061_policy_simulation/run_e061.py --test     # needs TEST_READ_AUTHORISATION.json
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
sys.path.insert(0, str(HERE.parents[2]))
from experiments.tcss_lib import core as C                                   # noqa: E402
from experiments.tcss_lib.pool import SharedUnits, load_pool, ones_draw    # noqa: E402

EXP = "E061"
OUT = C.ART / EXP
LOGDIR = C.ROOT / "artifacts" / "logs" / EXP
B_PRIMARY = 10_000
B_PILOT = 200
CHUNK = 500
SEED = 610601
FIT = OUT / "validation_fit.json"
SOURCES_RUN = [HERE, C.ROOT / "experiments/tcss_lib/core.py", C.ROOT / "experiments/tcss_lib/pool.py"]
LEVEL_PRIMARY = 1 - 0.05 / 4          # 98.75 %
Q_LIST = C.COST_GRID["q"]             # (0.5, 0.9, 1.0); 0.9 is c*
K = C.C_STAR["K"]
sys.path.insert(0, str(C.ROOT / "experiments" / "E036_interaction_statistics"))


def git_state() -> dict:
    def run(*a):
        return subprocess.run(a, cwd=C.ROOT, capture_output=True, text=True).stdout.strip()
    return {"sha": run("git", "rev-parse", "HEAD"), "dirty": run("git", "status", "--porcelain", "--untracked-files=no").splitlines()}


def provenance(extra: dict) -> dict:
    return {"experiment": EXP, "git": git_state(), "python": sys.version.split()[0], "numpy": np.__version__,
            "host": platform.node(), "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "script_sha256": {C.rel(s): C.sha256(s) for s in SOURCES_RUN}, **extra}


def cells(dets):
    return [(d, p, op) for d in dets for p in C.PROTOS for op in C.OPS]


# ------------------------------------------------------------------------------ F2 fitting
def fit_all(pools, dets) -> dict:
    fits = {}
    for p, pp in pools.items():
        for det in dets:
            for arm in C.ARMS:
                for s in C.SEEDS:
                    for op, cond in C.OPS.items():
                        f = C.fit_policies(pp.val_real[(det, arm, s, C.ANCHOR)], pp.val_real[(det, arm, s, cond)],
                                           pp.val_dev[(det, arm, s, C.ANCHOR)], pp.val_dev[(det, arm, s, cond)])
                        fits[f"{det}|{p}|{arm}|{s}|{op}"] = f
    return fits


def eligibility(fits, pools, dets) -> dict:
    """E060 §4 rule on tau0 per arm-cell (all seeds), plus finite pi parameters."""
    el = {}
    for p, pp in pools.items():
        for det in dets:
            for arm in C.ARMS:
                ok = True
                for s in C.SEEDS:
                    for op in C.OPS:
                        f = fits[f"{det}|{p}|{arm}|{s}|{op}"]
                        ok &= C.eligible_tau(pp.val_real[(det, arm, s, C.ANCHOR)], f["tau0"]) and f["finite"]
                el[f"{det}|{p}|{arm}"] = bool(ok)
    return el


# ------------------------------------------------------------------------------ analysis
def per_key_vectors(pools, fits, dets):
    """Outcome indicators per (det, p, arm, seed, op, policy) on the pool, from frozen parameters."""
    V = {}
    for p, pp in pools.items():
        u = pp.units
        for det in dets:
            for arm in C.ARMS:
                for s in C.SEEDS:
                    for op, cond in C.OPS.items():
                        prm = fits[f"{det}|{p}|{arm}|{s}|{op}"]
                        r0, r1 = pp.real[(det, arm, s, C.ANCHOR)], pp.real[(det, arm, s, cond)]
                        y0, y1 = pp.syn[(det, arm, s, C.ANCHOR)], pp.syn[(det, arm, s, cond)]
                        for pol in C.POLICIES:
                            zr, rr = C.decide(pol, prm, r0, r1)
                            zs, rs = C.decide(pol, prm, y0, y1)
                            rej = (zr == C.REJ).astype(float); stp = (zr == C.STEP).astype(float)
                            rf = C.per_cred_mean(rej, u); uf = C.per_cred_mean(stp, u)
                            V[(det, p, arm, s, op, pol)] = dict(
                                rej_cap=rej, stp_cap=stp, rf=rf, uf=uf,
                                acc_syn=(zs == C.ACC).astype(float), stp_syn=(zs == C.STEP).astype(float),
                                restored_real=float(rr.mean()), restored_syn=float(rs.mean()))
    return V


COMP = ["L", "G", "N", "B", "U", "P", "V"] + [f"PMP_q{q}" for q in Q_LIST] + [f"PMPsame_q{q}" for q in Q_LIST] \
       + ["L1", "L5", "N1", "N5", "G1", "G5"]


def components(pools, shared, V, dets, d, m_seed, gens_by_p):
    """Seed-averaged components per (det, p, arm, op, policy) under one block of draws."""
    R = d["subj"].shape[0]
    out = {}
    for p, pp in pools.items():
        u = pp.units
        W = shared.cred_weights(d, p, u)
        Wcap = W[:, u.cred_of_capture]
        Ws = shared.syn_weights(d, p, len(pp.syn_pid))
        gens = gens_by_p[p]
        for det in dets:
            m = m_seed[(det, p)]
            for arm in C.ARMS:
                for op in C.OPS:
                    for pol in C.POLICIES:
                        per = {c: np.empty((R, 10)) for c in COMP}
                        pergen = {g: np.empty((R, 10)) for g in gens}
                        for j, s in enumerate(C.SEEDS):
                            v = V[(det, p, arm, s, op, pol)]
                            rf, uf = v["rf"], v["uf"]
                            X = []
                            for k in (K, 1, 5):
                                Lk, Gk, Nk = C.session_legit(rf, uf, k)
                                X += [Lk, Gk, Nk]
                            M = C.wmean(W, np.stack(X, axis=1))
                            per["L"][:, j], per["G"][:, j], per["N"][:, j] = M[:, 0], M[:, 1], M[:, 2]
                            per["L1"][:, j], per["G1"][:, j], per["N1"][:, j] = M[:, 3], M[:, 4], M[:, 5]
                            per["L5"][:, j], per["G5"][:, j], per["N5"][:, j] = M[:, 6], M[:, 7], M[:, 8]
                            per["B"][:, j] = (Wcap @ v["rej_cap"]) / Wcap.sum(axis=1)
                            per["U"][:, j] = (Wcap @ v["stp_cap"]) / Wcap.sum(axis=1)
                            Pv = (Ws @ v["acc_syn"]) / Ws.sum(axis=1); Vv = (Ws @ v["stp_syn"]) / Ws.sum(axis=1)
                            per["P"][:, j], per["V"][:, j] = Pv, Vv
                            for q in Q_LIST:
                                per[f"PMP_q{q}"][:, j] = C.pmp_fresh(Pv, Vv, q, K)
                                per[f"PMPsame_q{q}"][:, j] = C.pmp_same(Pv, Vv, q)
                            for g, idx in gens.items():
                                Wg = Ws[:, idx]
                                Pg = (Wg @ v["acc_syn"][idx]) / Wg.sum(axis=1)
                                Vg = (Wg @ v["stp_syn"][idx]) / Wg.sum(axis=1)
                                pergen[g][:, j] = C.pmp_fresh(Pg, Vg, C.C_STAR["q"], K)
                        for c in COMP:
                            out[(det, p, arm, op, pol, c)] = C.seed_average(per[c], m)
                        for g in gens:
                            out[(det, p, arm, op, pol, f"PMPgen_{g}")] = C.seed_average(pergen[g], m)
    return out


def J(comp, key, c_A, c_U, c_T, q):
    det, p, arm, op, pol = key
    g = lambda c: comp[(det, p, arm, op, pol, c)]
    return C.cost_J(g(f"PMP_q{q}"), g("L"), g("G"), g("N"), c_A, c_U, c_T)


def run_analysis(pools, fits, el, dets, B, outdir: Path, log, pilot: bool) -> dict:
    shared = SharedUnits(pools)
    ss = np.random.SeedSequence(SEED).spawn(4)
    rng_seed, rng_subj, rng_syn, _ = (np.random.default_rng(x) for x in ss)
    gens_by_p = {p: {g: np.flatnonzero(pp.syn_gen == g) for g in sorted(set(map(str, pp.syn_gen)))}
                 for p, pp in pools.items()}
    V = per_key_vectors(pools, fits, dets)
    full = components(pools, shared, V, dets, ones_draw(shared), {(d, p): np.ones((1, 10)) for d in dets for p in C.PROTOS}, gens_by_p)
    full = {k: float(v[0]) for k, v in full.items()}
    reps = {}
    t0 = time.time()
    for b0 in range(0, B, CHUNK):
        n = min(CHUNK, B - b0)
        m_seed = {(d, p): C.row_counts(rng_seed, n, 10) for d in dets for p in C.PROTOS}
        dr = shared.draw(rng_subj, rng_syn, n)
        r = components(pools, shared, V, dets, dr, m_seed, gens_by_p)
        for k, v in r.items():
            reps.setdefault(k, []).append(v)
        log(f"  bootstrap {b0 + n}/{B} ({time.time() - t0:.0f}s)")
    reps = {k: np.concatenate(v) for k, v in reps.items()}
    cs = C.C_STAR

    def jstar(src, det, p, arm, op, pol):
        return J(src, (det, p, arm, op, pol), cs["c_A"], cs["c_U"], cs["c_T"], cs["q"])

    def elig(det, p, arm):
        return el[f"{det}|{p}|{arm}"] if arm != "dd" else el[f"{det}|{p}|leg"] and el[f"{det}|{p}|fix"]

    def joint(pol, arm, cell_dets, level):
        cl = [c for c in cells(cell_dets) if elig(c[0], c[1], arm)]
        def dJ(src, c, a):
            return jstar(src, *c[:2], a, c[2], pol) - jstar(src, *c[:2], a, c[2], "pi0")
        if arm == "dd":
            fv = [dJ(full, c, "fix") - dJ(full, c, "leg") for c in cl]
            rv = np.stack([dJ(reps, c, "fix") - dJ(reps, c, "leg") for c in cl], axis=1)
        else:
            fv = [dJ(full, c, arm) for c in cl]
            rv = np.stack([dJ(reps, c, arm) for c in cl], axis=1)
        med = np.median(rv, axis=1)
        return {"n_cells": len(cl), "median": float(np.median(fv)), "ci95": C.pctl(med, 0.95),
                "ci_decision": C.pctl(med, level), "decision_level": level}

    summary = {}
    for pol in C.POLICIES[1:]:
        for arm in ("leg", "fix", "dd"):
            summary[f"dJ|{pol}|{arm}"] = joint(pol, arm, dets, LEVEL_PRIMARY)
            summary[f"dJ|{pol}|{arm}|noD3"] = joint(pol, arm, [d for d in dets if d != "Dt"], LEVEL_PRIMARY)
    n_inel = {arm: sum(not elig(d, p, arm) for (d, p, op) in cells(dets)) for arm in C.ARMS}
    uninterpretable = any(v > 3 for v in n_inel.values())
    H1 = summary["dJ|pi1|leg"]["ci_decision"][0] > 0
    H2 = summary["dJ|pi1|dd"]["ci_decision"][1] < 0
    H4 = {arm: summary[f"dJ|pi4|{arm}"]["ci_decision"][1] < 0 for arm in C.ARMS}
    H3 = {}
    for pol in ("pi2", "pi3"):
        for arm in C.ARMS:
            lo, hi = summary[f"dJ|{pol}|{arm}"]["ci_decision"]
            H3[f"{pol}|{arm}"] = "better" if hi < 0 else "worse" if lo > 0 else "indistinguishable"
    outcome = "UNINTERPRETABLE" if uninterpretable else ("P-A" if H1 and H2 else "P-B" if H1 else "P-C")
    decision = {"pilot_not_evidence": pilot, "outcome": outcome, "H1_supported": bool(H1), "H2_supported": bool(H2),
                "H4_supported": {k: bool(v) for k, v in H4.items()}, "H3_class": H3, "ineligible_cells": n_inel,
                "c_star": cs, "decision_level": LEVEL_PRIMARY}

    # per arm-cell tables
    single, session, pareto_s, tol_rows = [], [], [], []
    for (det, p, op) in cells(dets):
        for arm in C.ARMS:
            for pol in C.POLICIES:
                key = (det, p, arm, op, pol)
                for c in ("B", "U", "P", "V"):
                    lo, hi = C.pctl(reps[key + (c,)], 0.95)
                    single.append(dict(det=det, protocol=p, op=op, arm=arm, policy=pol, quantity=c,
                                       estimate=full[key + (c,)], lo95=lo, hi95=hi, eligible=elig(det, p, arm)))
                rr = float(np.mean([V[(det, p, arm, s, op, pol)]["restored_real"] for s in C.SEEDS]))
                rsy = float(np.mean([V[(det, p, arm, s, op, pol)]["restored_syn"] for s in C.SEEDS]))
                single.append(dict(det=det, protocol=p, op=op, arm=arm, policy=pol, quantity="R_real",
                                   estimate=rr, lo95="", hi95="", eligible=elig(det, p, arm)))
                single.append(dict(det=det, protocol=p, op=op, arm=arm, policy=pol, quantity="R_syn",
                                   estimate=rsy, lo95="", hi95="", eligible=elig(det, p, arm)))
                for c in ["L", "G", "N", f"PMP_q{cs['q']}", f"PMPsame_q{cs['q']}", "L1", "L5"] + \
                         [f"PMPgen_{g}" for g in gens_by_p[p]]:
                    lo, hi = C.pctl(reps[key + (c,)], 0.95)
                    session.append(dict(det=det, protocol=p, op=op, arm=arm, policy=pol, quantity=c,
                                        estimate=full[key + (c,)], lo95=lo, hi95=hi, eligible=elig(det, p, arm)))
                gen_keys = [key + (f"PMPgen_{g}",) for g in gens_by_p[p]]
                worst = np.max(np.stack([reps[k] for k in gen_keys], axis=1), axis=1)
                lo, hi = C.pctl(worst, 0.95)
                session.append(dict(det=det, protocol=p, op=op, arm=arm, policy=pol, quantity="PMP_worst_generator",
                                    estimate=max(full[k] for k in gen_keys), lo95=lo, hi95=hi, eligible=elig(det, p, arm)))
                jf = jstar(full, *key[:2], arm, op, pol); jr = jstar(reps, *key[:2], arm, op, pol)
                lo, hi = C.pctl(jr, 0.95)
                session.append(dict(det=det, protocol=p, op=op, arm=arm, policy=pol, quantity="J_cstar",
                                    estimate=jf, lo95=lo, hi95=hi, eligible=elig(det, p, arm)))
                if pol == "pi0":
                    continue
                k0 = (det, p, arm, op, "pi0")
                pm = f"PMP_q{cs['q']}"
                dP = reps[key + (pm,)] - reps[k0 + (pm,)]; dL = reps[key + ("L",)] - reps[k0 + ("L",)]
                (pl, ph), (ll, lh) = C.pctl(dP, 0.95), C.pctl(dL, 0.95)
                base = dict(det=det, protocol=p, op=op, arm=arm, policy=pol,
                            dPMP=full[key + (pm,)] - full[k0 + (pm,)], dPMP_lo95=pl, dPMP_hi95=ph,
                            dL=full[key + ("L",)] - full[k0 + ("L",)], dL_lo95=ll, dL_hi95=lh, eligible=elig(det, p, arm))
                strict = ph < 0 and lh < 0
                tol = (not strict) and ((ph < 0 and lh < C.TOL["L"]) or (lh < 0 and ph < C.TOL["pmp"]))
                pareto_s.append({**base, "strict_pareto_improvement": strict})
                tol_rows.append({**base, "tolerance_bounded_tradeoff": tol,
                                 "tol_pmp": C.TOL["pmp"], "tol_L": C.TOL["L"]})

    # cost grid (S1, tier 1): minimiser per cell (full sample) and between-arm disagreement
    grid = list(itertools.product(C.COST_GRID["c_A"], C.COST_GRID["c_U"], C.COST_GRID["c_T"], C.COST_GRID["q"]))
    grid_rows, disagree = [], {}
    for (det, p, op) in cells(dets):
        dis_rep = np.zeros(B)
        dis_full = 0
        for (cA, cU, cT, q) in grid:
            amin = {}
            amin_rep = {}
            for arm in C.ARMS:
                jf = [J(full, (det, p, arm, op, pol), cA, cU, cT, q) for pol in C.POLICIES]
                amin[arm] = C.POLICIES[int(np.argmin(jf))]
                jr = np.stack([J(reps, (det, p, arm, op, pol), cA, cU, cT, q) for pol in C.POLICIES], axis=1)
                amin_rep[arm] = np.argmin(jr, axis=1)
                grid_rows.append(dict(det=det, protocol=p, op=op, arm=arm, c_A=cA, c_U=cU, c_T=cT, q=q,
                                      argmin=amin[arm], **{f"J_{pol}": v for pol, v in zip(C.POLICIES, jf)}))
            dis_full += amin["leg"] != amin["fix"]
            dis_rep += amin_rep["leg"] != amin_rep["fix"]
        lo, hi = C.pctl(dis_rep / len(grid), 0.95)
        disagree[f"{det}|{p}|{op}"] = {"fraction_full": dis_full / len(grid), "ci95": (lo, hi)}

    outdir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("single_attempt.csv", single), ("session.csv", session), ("pareto_strict.csv", pareto_s),
                       ("tradeoff_tolerance.csv", tol_rows), ("cost_grid.csv", grid_rows)):
        with open(outdir / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary["cost_grid_arm_disagreement"] = disagree
    (outdir / "joint_summary.json").write_text(json.dumps(summary, indent=1))
    (outdir / "OUTCOME.json").write_text(json.dumps(decision, indent=1))
    return decision


# ------------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fit", action="store_true")
    g.add_argument("--pilot", action="store_true")
    g.add_argument("--test", action="store_true")
    a = ap.parse_args()
    dets = C.HEADLINE
    OUT.mkdir(parents=True, exist_ok=True); LOGDIR.mkdir(parents=True, exist_ok=True)
    mode = "fit" if a.fit else "pilot" if a.pilot else "test"
    logf = open(LOGDIR / f"{mode}_{time.strftime('%Y%m%d_%H%M%S')}.log", "w")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line); logf.write(line + "\n"); logf.flush()

    alog = C.AccessLog(EXP)
    if a.fit:
        if FIT.exists():
            log(f"{FIT} exists; F2 file is never overwritten"); return 2
        pools = load_pool("meta", dets, alog)
        fits = fit_all(pools, dets); el = eligibility(fits, pools, dets)
        rec = {"experiment": EXP, "stage": "F2 validation-only policy parameters",
               "plan": "docs/experiments/E061_policy_simulation.md (F1 3ff5b85)",
               "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "git": git_state(),
               "script_sha256": {C.rel(s): C.sha256(s) for s in SOURCES_RUN},
               "eligibility": el, "fits": fits}
        FIT.write_text(json.dumps(rec, indent=1))
        alog.dump(OUT / "access_log_fit.json")
        assert "test_score" not in {e["key"] for e in alog.entries}
        log(f"F2 written: {FIT.name} sha256={C.sha256(FIT)}  eligible={sum(el.values())}/{len(el)}")
        return 0
    if a.pilot:
        pools = load_pool("pilot", dets, alog)
        fits = fit_all(pools, dets); el = eligibility(fits, pools, dets)
        dec = run_analysis(pools, {k: v for k, v in fits.items()}, el, dets, B_PILOT, OUT / "pilot_validation", log, pilot=True)
        (OUT / "pilot_validation" / "README.txt").write_text(
            "VALIDATION-ONLY PILOT (code exercise). Policies fitted and evaluated on the validation pool\n"
            "(circular; attacker pool = all validation synthetic sources incl. the development generator). B=200.\n"
            "NOT EVIDENCE; never cited.\n")
        alog.dump(OUT / "pilot_validation" / "access_log.json")
        assert "test_score" not in {e["key"] for e in alog.entries}
        log(f"pilot done; outcome field (meaningless, pilot): {dec['outcome']}")
        return 0
    # --test
    auth = C.check_authorisation(EXP, SOURCES_RUN)
    if auth.get("validation_fit_sha256") != C.sha256(FIT):
        raise C.TestAccessDenied("validation_fit.json differs from the F2-frozen file")
    alog.authorisation = auth
    rec = json.loads(FIT.read_text()); fits, el = rec["fits"], rec["eligibility"]
    import run_e036
    P = {p: run_e036.Proto(p) for p in C.PROTOS}
    for p in C.PROTOS:
        for f in P[p].files:
            alog.add(f, "test_score+val_score (run_e036.Proto, G0)")
    fs = run_e036.full_sample(P); g0 = run_e036.gate0(fs)
    (OUT / "gate_G0_e036.json").write_text(json.dumps(g0, indent=1, default=str))
    if not g0["pass"]:
        log("G0 FAILED -> stop"); return 2
    pools = load_pool("test", dets, alog, authorised=True)
    # G0-P: pi0/pi1/pi2 single-attempt T and B on the sigmoid (D3) cells equal the E036 code path
    devs = []
    for p in C.PROTOS:
        Pp = P[p]
        for arm in C.ARMS:
            for s in C.SEEDS:
                for op, cond in C.OPS.items():
                    prm = fits[f"Dt|{p}|{arm}|{s}|{op}"]
                    tau = fs[("Dt", p, arm, s)]["tau"]
                    devs.append(abs(prm["tau0"] - tau))
                    for c_ in (C.ANCHOR, cond):
                        sc = Pp.test[("Dt", arm, s, c_)]
                        mine_r = pools[p].real[("Dt", arm, s, c_)] > prm["tau0"]
                        mine_s = pools[p].syn[("Dt", arm, s, c_)] > prm["tau0"]
                        devs.append(abs(mine_r.mean() - (sc[Pp.R] > tau).mean()))
                        devs.append(abs(mine_s.mean() - (sc[Pp.S] > tau).mean()))
    g0p = {"max_abs_deviation": float(max(devs)), "pass": float(max(devs)) <= 1e-12}
    (OUT / "gate_G0P.json").write_text(json.dumps(g0p, indent=1))
    if not g0p["pass"]:
        log("G0-P FAILED -> stop"); return 2
    dec = run_analysis(pools, fits, el, dets, B_PRIMARY, OUT, log, pilot=False)
    alog.dump(OUT / "access_log.json")
    (OUT / "provenance.json").write_text(json.dumps(provenance({"mode": "test", "auth": auth}), indent=1))
    log(f"E061 outcome: {dec['outcome']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
