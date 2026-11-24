#!/usr/bin/env python3
"""E063 -- RQ3 session robustness: M0 vs M0|elig vs M1 (post hoc secondary analysis; simulation).

Frozen protocol: docs/experiments/E063_session_robustness.md. Re-uses the frozen E061 policy parameters
(validation_fit.json, SHA-256 pinned), the E061 per-capture decision code (imported, unchanged) and the E060/E061
bootstrap units. Nothing is refitted.

    python experiments/E063_session_robustness/run_e063.py --pilot   # validation-only code exercise; NOT evidence
    python experiments/E063_session_robustness/run_e063.py --test    # needs artifacts/results/E063/TEST_READ_AUTHORISATION.json

The --test path refuses to read any test score unless the PI authorisation file exists, pins the SHA-256 of every
source in SOURCES_RUN and of the protocol, names the frozen validation_fit.json and records a PI decision on the
open E059 post-test integrity reads.
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
from experiments.tcss_lib import core as C                                   # noqa: E402
from experiments.tcss_lib import pool as POOL                                # noqa: E402
from experiments.tcss_lib.pool import SharedUnits, load_pool, ones_draw     # noqa: E402
import run_e061 as E61                                                       # noqa: E402  (frozen, unchanged)
import session_models as SM                                                  # noqa: E402

EXP = "E063"
OUT = C.ART / EXP
LOGDIR = ROOT / "artifacts" / "logs" / EXP
PROTOCOL = ROOT / "docs" / "experiments" / "E063_session_robustness.md"
SOURCES_RUN = [HERE, HERE.parent / "session_models.py", ROOT / "experiments/tcss_lib/core.py",
               ROOT / "experiments/tcss_lib/pool.py", ROOT / "experiments/E061_policy_simulation/run_e061.py"]
FIT = C.ART / "E061" / "validation_fit.json"
FIT_SHA = "c3d01b7d84d8858bfd091e9bcdd8d8db4b24597fc4ccf8adeaf3554e9dae22e4"
E061_SESSION = C.ART / "E061" / "session.csv"
E061_SESSION_SHA = None            # filled from git-committed file at run time and recorded (no manifest exists)
# Track A manifests (needed only by the validation pilot for validation identities). The repo's data/ symlink
# points to a path that no longer exists on this host; the same files live here and are pinned by SHA-256.
MANIFEST_FALLBACK = {"p1": Path("PRIVATE_TRACK_A_ROOT/p1/manifest.csv"),   # [public copy: local path redacted]
                     "p2": Path("PRIVATE_TRACK_A_ROOT/p2/manifest.csv")}
MANIFEST_SHA = {"p1": "474cac5f7b1720e0b9be313173a4c1afc53e0749ecf390fd029c2e5bf0e298d3",
                "p2": "5601f5f31c36682a8ebd667c2ffd825212cdf1c169e8cdf080785fb8ce8f6ee6"}
B_PRIMARY = 10_000
B_PILOT = 200
CHUNK = 500
SEED = 630063
K = 3
LEVEL_DECISION = 1 - 0.05 / 4          # 98.75 %, the E061 decision level (readings only; no hypothesis is re-decided)
MODELS = ("M0", "M0e", "M1")
Q_LIST = C.COST_GRID["q"]
EXPECTED_M1_EXCLUDED = {"credentials": 60, "captures": 88}      # E060 post-freeze annotation, K = 3, per protocol
G0_TOL = 1e-12


def git_state() -> dict:
    def run(*a):
        return subprocess.run(a, cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"sha": run("git", "rev-parse", "HEAD"),
            "dirty_tracked": run("git", "status", "--porcelain", "--untracked-files=no").splitlines()}


def ensure_manifests():
    for p, f in C.MANIFEST.items():
        if not f.exists():
            alt = MANIFEST_FALLBACK[p]
            if C.sha256(alt) != MANIFEST_SHA[p]:
                raise RuntimeError(f"manifest {alt} hash mismatch")
            C.MANIFEST[p] = alt; POOL.MANIFEST[p] = alt
        elif C.sha256(f) != MANIFEST_SHA[p]:
            raise RuntimeError(f"manifest {f} hash mismatch")


# ------------------------------------------------------------------------------ per-credential model inputs
def per_cred_models(pools, V, dets):
    """Fixed per-credential session values for every (det, p, arm, seed, op, pol) under M0 and M1.
    Captures are never resampled within a credential, so these do not change across bootstrap replicates."""
    X = {}
    elig = {}
    for p, pp in pools.items():
        u = pp.units; nf = u.n_f()
        el = nf >= K; elig[p] = el
        for key, v in V.items():
            if key[1] != p:
                continue
            l0, g0, n0 = SM.m0(v["rf"], v["uf"], K)
            R = SM.counts(v["rej_cap"], u.cred_of_capture, u.F)
            U = SM.counts(v["stp_cap"], u.cred_of_capture, u.F)
            l1, g1, n1, e1 = SM.m1(R, U, nf, K)
            assert (e1 == el).all()
            X[key] = {"M0": np.stack([l0, g0, n0], 1), "M1": np.stack([np.nan_to_num(l1), np.nan_to_num(g1),
                                                                     np.nan_to_num(n1)], 1)}
    return X, elig


COMP_SYN = [f"PMP_q{q}" for q in Q_LIST] + ["PMPsame_q0.9"]


def components(pools, shared, V, X, elig, dets, d, m_seed):
    """Seed-averaged components per (det, p, arm, op, pol, model, quantity) under one block of draws."""
    R = d["subj"].shape[0]
    out = {}
    for p, pp in pools.items():
        u = pp.units
        W = shared.cred_weights(d, p, u)
        We = W * elig[p][None, :]
        Ws = shared.syn_weights(d, p, len(pp.syn_pid))
        for det in dets:
            m = m_seed[(det, p)]
            for arm in C.ARMS:
                for op in C.OPS:
                    for pol in C.POLICIES:
                        per = {(mo, q): np.empty((R, 10)) for mo in MODELS for q in ("L", "G", "N")}
                        per.update({("syn", c): np.empty((R, 10)) for c in COMP_SYN})
                        for j, s in enumerate(C.SEEDS):
                            key = (det, p, arm, s, op, pol)
                            x = X[key]
                            for mo, Wm, arr in (("M0", W, x["M0"]), ("M0e", We, x["M0"]), ("M1", We, x["M1"])):
                                M = C.wmean(Wm, arr)
                                per[(mo, "L")][:, j], per[(mo, "G")][:, j], per[(mo, "N")][:, j] = M[:, 0], M[:, 1], M[:, 2]
                            v = V[key]
                            Pv = (Ws @ v["acc_syn"]) / Ws.sum(axis=1); Vv = (Ws @ v["stp_syn"]) / Ws.sum(axis=1)
                            for q in Q_LIST:
                                per[("syn", f"PMP_q{q}")][:, j] = C.pmp_fresh(Pv, Vv, q, K)
                            per[("syn", "PMPsame_q0.9")][:, j] = C.pmp_same(Pv, Vv, 0.9)
                        for (a, b), E in per.items():
                            out[(det, p, arm, op, pol, a, b)] = C.seed_average(E, m)
    return out


def J(comp, det, p, arm, op, pol, model, c_A, c_U, c_T, q, attacker="fresh"):
    g = lambda mo, c: comp[(det, p, arm, op, pol, mo, c)]
    pm = g("syn", f"PMP_q{q}") if attacker == "fresh" else g("syn", "PMPsame_q0.9")
    return C.cost_J(pm, g(model, "L"), g(model, "G"), g(model, "N"), c_A, c_U, c_T)


# ------------------------------------------------------------------------------ analysis
def run_analysis(pools, fits, el, dets, B, outdir: Path, log, frozen_session=None) -> dict:
    shared = SharedUnits(pools)
    ss = np.random.SeedSequence(SEED).spawn(3)
    rng_seed, rng_subj, rng_syn = (np.random.default_rng(x) for x in ss)
    V = E61.per_key_vectors(pools, fits, dets)
    X, elig = per_cred_models(pools, V, dets)
    elig_counts = {p: {"credentials": int(pp.units.F), "eligible_K3": int(elig[p].sum()),
                       "excluded_K3": int((~elig[p]).sum()),
                       "excluded_K3_captures": int(pp.units.n_f()[~elig[p]].sum())} for p, pp in pools.items()}
    full = components(pools, shared, V, X, elig, dets, ones_draw(shared),
                      {(d_, p): np.ones((1, 10)) for d_ in dets for p in C.PROTOS})
    full = {k: float(v[0]) for k, v in full.items()}
    cs = C.C_STAR
    cells = [(d_, p, op) for d_ in dets for p in C.PROTOS for op in C.OPS]

    def jstar(src, det, p, arm, op, pol, mo, attacker="fresh"):
        return J(src, det, p, arm, op, pol, mo, cs["c_A"], cs["c_U"], cs["c_T"], cs["q"], attacker)

    gates = {"elig_counts": elig_counts}
    if frozen_session is not None:
        # G-M1: eligibility equals the E060 post-freeze annotation
        gm1 = all(c["excluded_K3"] == EXPECTED_M1_EXCLUDED["credentials"]
                  and c["excluded_K3_captures"] == EXPECTED_M1_EXCLUDED["captures"] for c in elig_counts.values())
        # G0: M0 full-sample values reproduce the frozen E061 session.csv
        dev = 0.0; n = 0
        for (det, p, op) in cells:
            for arm in C.ARMS:
                for pol in C.POLICIES:
                    fr = frozen_session[(det, p, op, arm, pol)]
                    mine = {"L": full[(det, p, arm, op, pol, "M0", "L")], "G": full[(det, p, arm, op, pol, "M0", "G")],
                            "N": full[(det, p, arm, op, pol, "M0", "N")],
                            "PMP_q0.9": full[(det, p, arm, op, pol, "syn", "PMP_q0.9")],
                            "PMPsame_q0.9": full[(det, p, arm, op, pol, "syn", "PMPsame_q0.9")],
                            "J_cstar": jstar(full, det, p, arm, op, pol, "M0")}
                    for k, v in mine.items():
                        dev = max(dev, abs(v - fr[k])); n += 1
        gates.update({"G_M1_eligibility_pass": bool(gm1), "G0_max_abs_dev": dev, "G0_n_values": n,
                      "G0_pass": dev <= G0_TOL})
        outdir.mkdir(parents=True, exist_ok=False)
        (outdir / "gates.json").write_text(json.dumps(gates, indent=1))
        if not (gm1 and dev <= G0_TOL):
            log(f"GATE FAILED: {gates}  -> stop; no bootstrap, no result written")
            return {"stopped_at_gate": True, "gates": gates}
    else:
        outdir.mkdir(parents=True, exist_ok=False)
        (outdir / "gates.json").write_text(json.dumps(gates, indent=1))

    reps = {}
    t0 = time.time()
    for b0 in range(0, B, CHUNK):
        n = min(CHUNK, B - b0)
        m_seed = {(d_, p): C.row_counts(rng_seed, n, 10) for d_ in dets for p in C.PROTOS}
        dr = shared.draw(rng_subj, rng_syn, n)
        r = components(pools, shared, V, X, elig, dets, dr, m_seed)
        for k, v in r.items():
            reps.setdefault(k, []).append(v)
        log(f"  bootstrap {b0 + n}/{B} ({time.time() - t0:.0f}s)")
    reps = {k: np.concatenate(v) for k, v in reps.items()}

    def elg(det, p, arm):
        return el[f"{det}|{p}|{arm}"] if arm != "dd" else el[f"{det}|{p}|leg"] and el[f"{det}|{p}|fix"]

    def cellstat(fun, arm):
        """Median over eligible cells of a per-cell statistic fun(src, cell, arm) (arm 'dd' = fix - leg)."""
        cl = [c for c in cells if elg(c[0], c[1], arm)]
        if arm == "dd":
            fv = [fun(full, c, "fix") - fun(full, c, "leg") for c in cl]
            rv = np.stack([fun(reps, c, "fix") - fun(reps, c, "leg") for c in cl], axis=1)
        else:
            fv = [fun(full, c, arm) for c in cl]
            rv = np.stack([fun(reps, c, arm) for c in cl], axis=1)
        med = np.median(rv, axis=1)
        return {"n_cells": len(cl), "median": float(np.median(fv)), "ci95": C.pctl(med, 0.95),
                "ci9875": C.pctl(med, LEVEL_DECISION),
                "per_cell_ci95_excl0_pos": int(sum(C.pctl(rv[:, i], 0.95)[0] > 0 for i in range(rv.shape[1]))),
                "per_cell_ci95_excl0_neg": int(sum(C.pctl(rv[:, i], 0.95)[1] < 0 for i in range(rv.shape[1])))}

    def dq(pol, mo, qty, attacker="fresh"):
        if qty == "J":
            return lambda src, c, a: (jstar(src, c[0], c[1], a, c[2], pol, mo, attacker)
                                      - jstar(src, c[0], c[1], a, c[2], "pi0", mo, attacker))
        return lambda src, c, a: src[(c[0], c[1], a, c[2], pol, mo, qty)] - src[(c[0], c[1], a, c[2], "pi0", mo, qty)]

    def eff(pol, qty, m_hi, m_lo, attacker="fresh"):
        f_hi, f_lo = dq(pol, m_hi, qty, attacker), dq(pol, m_lo, qty, attacker)
        return lambda src, c, a: f_hi(src, c, a) - f_lo(src, c, a)

    def lvl(pol, mo, qty):
        if qty == "J":
            return lambda src, c, a: jstar(src, c[0], c[1], a, c[2], pol, mo)
        return lambda src, c, a: src[(c[0], c[1], a, c[2], pol, mo, qty)]

    summary = {"contrasts": {}, "effects": {}, "levels": {}, "same_image_attacker": {}}
    for mo in MODELS:
        for pol in C.POLICIES[1:]:
            for qty in ("J", "L", "G", "N"):
                for arm in ("leg", "fix", "dd"):
                    summary["contrasts"][f"d{qty}|{pol}|{arm}|{mo}"] = cellstat(dq(pol, mo, qty), arm)
            for arm in ("leg", "fix", "dd"):
                summary["same_image_attacker"][f"dJ|{pol}|{arm}|{mo}"] = cellstat(dq(pol, mo, "J", "same"), arm)
        for pol in C.POLICIES:
            for qty in ("J", "L", "G", "N"):
                for arm in ("leg", "fix"):
                    summary["levels"][f"{qty}|{pol}|{arm}|{mo}"] = cellstat(lvl(pol, mo, qty), arm)
    for pol in C.POLICIES[1:]:
        for qty in ("J", "L", "G", "N"):
            for name, hi, lo in (("selection", "M0e", "M0"), ("model", "M1", "M0e"), ("total", "M1", "M0")):
                for arm in ("leg", "fix", "dd"):
                    summary["effects"][f"{name}|d{qty}|{pol}|{arm}"] = cellstat(eff(pol, qty, hi, lo), arm)
    for pol in C.POLICIES:
        for arm in ("leg", "fix"):
            summary["levels"][f"PMP_q0.9|{pol}|{arm}"] = cellstat(
                lambda src, c, a, pol=pol: src[(c[0], c[1], a, c[2], pol, "syn", "PMP_q0.9")], arm)

    # pre-specified readings of the E061 hypotheses under each model (descriptive; the E061 outcome is not re-decided)
    readings = {}
    for mo in MODELS:
        ct = summary["contrasts"]
        h3 = {}
        for pol in ("pi2", "pi3"):
            for arm in C.ARMS:
                lo, hi = ct[f"dJ|{pol}|{arm}|{mo}"]["ci9875"]
                h3[f"{pol}|{arm}"] = "better" if hi < 0 else "worse" if lo > 0 else "indistinguishable"
        readings[mo] = {"H1_dJ_pi1_leg_lower9875_gt0": ct[f"dJ|pi1|leg|{mo}"]["ci9875"][0] > 0,
                        "H2_ddJ_pi1_upper9875_lt0": ct[f"dJ|pi1|dd|{mo}"]["ci9875"][1] < 0,
                        "H4_leg_upper9875_lt0": ct[f"dJ|pi4|leg|{mo}"]["ci9875"][1] < 0,
                        "H4_fix_upper9875_lt0": ct[f"dJ|pi4|fix|{mo}"]["ci9875"][1] < 0,
                        "fixed_arm_dJ_pi1_lower9875_gt0": ct[f"dJ|pi1|fix|{mo}"]["ci9875"][0] > 0,
                        "H3_class": h3}

    # rankings: argmin at c* per cell, and cost-grid shares (full sample; ties to pi0 as in E061)
    grid = list(itertools.product(C.COST_GRID["c_A"], C.COST_GRID["c_U"], C.COST_GRID["c_T"], C.COST_GRID["q"]))
    rank, grid_share, pair = {}, {}, {}
    for mo in MODELS:
        for arm in C.ARMS:
            cnt = {pol: 0 for pol in C.POLICIES}; gs = {pol: 0 for pol in C.POLICIES}; pw = {}
            for (det, p, op) in cells:
                jf = [jstar(full, det, p, arm, op, pol, mo) for pol in C.POLICIES]
                cnt[C.POLICIES[int(np.argmin(jf))]] += 1
                for (cA, cU, cT, q) in grid:
                    jg = [J(full, det, p, arm, op, pol, mo, cA, cU, cT, q) for pol in C.POLICIES]
                    gs[C.POLICIES[int(np.argmin(jg))]] += 1
                    k = f"c_A={cA}"
                    pw.setdefault(k, [0, 0, 0])
                    pw[k][0] += 1; pw[k][1] += jg[1] < jg[0]; pw[k][2] += jg[4] < jg[0]
            rank[f"{arm}|{mo}"] = cnt
            grid_share[f"{arm}|{mo}"] = {k: v / (len(cells) * len(grid)) for k, v in gs.items()}
            pair[f"{arm}|{mo}"] = {k: {"cell_points": v[0], "pi1_lt_pi0": v[1], "pi4_lt_pi0": v[2]} for k, v in pw.items()}

    # per-cell table and Pareto/tolerance readings per model
    rows, par = [], []
    pm = "PMP_q0.9"
    for (det, p, op) in cells:
        for arm in C.ARMS:
            for pol in C.POLICIES:
                for mo in MODELS:
                    for qty in ("L", "G", "N"):
                        k = (det, p, arm, op, pol, mo, qty)
                        lo, hi = C.pctl(reps[k], 0.95)
                        rows.append(dict(det=det, protocol=p, op=op, arm=arm, policy=pol, model=mo, quantity=qty,
                                         estimate=full[k], lo95=lo, hi95=hi, eligible=elg(det, p, arm)))
                    jf = jstar(full, det, p, arm, op, pol, mo); lo, hi = C.pctl(jstar(reps, det, p, arm, op, pol, mo), 0.95)
                    rows.append(dict(det=det, protocol=p, op=op, arm=arm, policy=pol, model=mo, quantity="J_cstar",
                                     estimate=jf, lo95=lo, hi95=hi, eligible=elg(det, p, arm)))
                    if pol == "pi0":
                        continue
                    k1, k0 = (det, p, arm, op, pol), (det, p, arm, op, "pi0")
                    dP = reps[k1 + ("syn", pm)] - reps[k0 + ("syn", pm)]
                    dL = reps[k1 + (mo, "L")] - reps[k0 + (mo, "L")]
                    (pl, ph), (ll, lh) = C.pctl(dP, 0.95), C.pctl(dL, 0.95)
                    strict = ph < 0 and lh < 0
                    tol = (not strict) and ((ph < 0 and lh < C.TOL["L"]) or (lh < 0 and ph < C.TOL["pmp"]))
                    par.append(dict(det=det, protocol=p, op=op, arm=arm, policy=pol, model=mo,
                                    dPMP=full[k1 + ("syn", pm)] - full[k0 + ("syn", pm)], dPMP_lo95=pl, dPMP_hi95=ph,
                                    dL=full[k1 + (mo, "L")] - full[k0 + (mo, "L")], dL_lo95=ll, dL_hi95=lh,
                                    strict_pareto=strict, tolerance_tradeoff=tol))
        for qn in ("PMP_q0.5", "PMP_q0.9", "PMP_q1.0", "PMPsame_q0.9"):
            for arm in C.ARMS:
                for pol in C.POLICIES:
                    k = (det, p, arm, op, pol, "syn", qn)
                    lo, hi = C.pctl(reps[k], 0.95)
                    rows.append(dict(det=det, protocol=p, op=op, arm=arm, policy=pol, model="attacker(all models)",
                                     quantity=qn, estimate=full[k], lo95=lo, hi95=hi, eligible=elg(det, p, arm)))

    for name, data in (("session_models.csv", rows), ("pareto_tolerance_models.csv", par)):
        with open(outdir / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    (outdir / "joint_summary.json").write_text(json.dumps(summary, indent=1))
    res = {"readings": readings, "argmin_cstar_counts": rank, "grid_argmin_share": grid_share,
           "grid_pairwise_vs_pi0": pair, "strict_pareto_count": {mo: sum(r["strict_pareto"] for r in par if r["model"] == mo)
                                                              for mo in MODELS},
           "tolerance_count": {f"{mo}|{pol}|{arm}": sum(r["tolerance_tradeoff"] for r in par
                                                        if r["model"] == mo and r["policy"] == pol and r["arm"] == arm)
                               for mo in MODELS for pol in C.POLICIES[1:] for arm in C.ARMS},
           "gates": gates, "B": B, "seed": SEED, "K": K, "decision_level_for_readings": LEVEL_DECISION}
    (outdir / "RESULTS.json").write_text(json.dumps(res, indent=1, default=bool))
    return res


def load_frozen_session():
    fr = {}
    for r in csv.DictReader(open(E061_SESSION)):
        fr.setdefault((r["det"], r["protocol"], r["op"], r["arm"], r["policy"]), {})[r["quantity"]] = float(r["estimate"])
    return fr


def check_authorisation_e063() -> dict:
    auth = C.check_authorisation(EXP, SOURCES_RUN)          # pins every analysis source by SHA-256
    missing = [k for k in ("authorised_by", "date", "scope", "protocol_sha256", "validation_fit_sha256",
                           "e059_post_test_reads_decision", "freeze_commit") if not auth.get(k)]
    if missing:
        raise C.TestAccessDenied(f"E063 authorisation lacks {missing}")
    if auth["protocol_sha256"] != C.sha256(PROTOCOL):
        raise C.TestAccessDenied("protocol differs from the authorised version")
    if auth["validation_fit_sha256"] != FIT_SHA or C.sha256(FIT) != FIT_SHA:
        raise C.TestAccessDenied("validation_fit.json is not the frozen E061 F2 file")
    return auth


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pilot", action="store_true")
    g.add_argument("--test", action="store_true")
    g.add_argument("--check-auth", action="store_true", help="only evaluate the authorisation guard; reads no data")
    a = ap.parse_args()
    dets = C.HEADLINE
    stamp = time.strftime("%Y%m%d_%H%M%S")
    if a.check_auth:
        try:
            check_authorisation_e063(); print("authorisation: OK"); return 0
        except C.TestAccessDenied as e:
            print(f"authorisation: REFUSED -- {e}"); return 3
    LOGDIR.mkdir(parents=True, exist_ok=True)
    mode = "pilot" if a.pilot else "test"
    logf = open(LOGDIR / f"{mode}_{stamp}.log", "w")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line); logf.write(line + "\n"); logf.flush()

    alog = C.AccessLog(EXP)
    prov = {"experiment": EXP, "mode": mode, "git": git_state(), "python": sys.version.split()[0],
            "numpy": np.__version__, "host": platform.node(), "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "script_sha256": {C.rel(s): C.sha256(s) for s in SOURCES_RUN}, "protocol_sha256": C.sha256(PROTOCOL),
            "gpu": "none (CPU only)"}
    if a.pilot:
        ensure_manifests()
        pools = load_pool("pilot", dets, alog)
        assert C.TEST_SCORE not in {e["key"] for e in alog.entries}
        fits = E61.fit_all(pools, dets); el = E61.eligibility(fits, pools, dets)
        outdir = OUT / f"pilot_validation_{stamp}"
        run_analysis(pools, fits, el, dets, B_PILOT, outdir, log, frozen_session=None)
        (outdir / "README.txt").write_text(
            "VALIDATION-ONLY PILOT (code exercise). Policies fitted AND evaluated on the validation pool (circular;\n"
            "attacker pool = all validation synthetic sources incl. the development generator). B=200.\n"
            "NOT EVIDENCE; never cited. No test score was read (access log).\n")
        alog.dump(outdir / "access_log.json")
        (outdir / "provenance.json").write_text(json.dumps(prov, indent=1))
        log(f"pilot done -> {outdir}")
        return 0
    # --test
    try:
        auth = check_authorisation_e063()
    except C.TestAccessDenied as e:
        log(f"REFUSED before any data read: {e}")
        return 3
    alog.authorisation = auth
    rec = json.loads(FIT.read_text()); fits, el = rec["fits"], rec["eligibility"]
    pools = load_pool("test", dets, alog, authorised=True)
    outdir = OUT / f"test_run_{stamp}"
    res = run_analysis(pools, fits, el, dets, B_PRIMARY, outdir, log, frozen_session=load_frozen_session())
    alog.dump(outdir / "access_log.json")
    prov["auth"] = auth
    prov["e061_session_csv_sha256"] = C.sha256(E061_SESSION)
    (outdir / "provenance.json").write_text(json.dumps(prov, indent=1))
    log("E063 test run finished" if not res.get("stopped_at_gate") else "E063 stopped at a gate")
    return 0 if not res.get("stopped_at_gate") else 2


if __name__ == "__main__":
    sys.exit(main())
