#!/usr/bin/env python3
"""E060 -- credential- and subject-level legitimate-user burden (post-hoc secondary analysis).

Frozen analysis plan: docs/experiments/E060_credential_burden.md (F1, commit 3ff5b85).

    python experiments/E060_credential_burden/run_e060.py --pretest   # G-M, G-tau, eligibility (no test score)
    python experiments/E060_credential_burden/run_e060.py --pilot     # validation-only pilot (not evidence)
    python experiments/E060_credential_burden/run_e060.py --test      # needs TEST_READ_AUTHORISATION.json
"""
from __future__ import annotations

import argparse
import csv
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

EXP = "E060"
OUT = C.ART / EXP
LOGDIR = C.ROOT / "artifacts" / "logs" / EXP
B_PRIMARY = 10_000
B_PILOT = 200
CHUNK = 1_000
SEED = 600601
K_LIST = (1, 2, 3, 5)
CELL_OPS = tuple(C.OPS)                     # ("bil", "tv")
SOURCES_RUN = [HERE, C.ROOT / "experiments/tcss_lib/core.py", C.ROOT / "experiments/tcss_lib/pool.py"]

# sigmoid reference sources = run_e036.SOURCES (G-tau / G0)
sys.path.insert(0, str(C.ROOT / "experiments" / "E036_interaction_statistics"))


def e036():
    import run_e036
    return run_e036


def git_state() -> dict:
    def run(*a):
        return subprocess.run(a, cwd=C.ROOT, capture_output=True, text=True).stdout.strip()
    return {"sha": run("git", "rev-parse", "HEAD"), "dirty": run("git", "status", "--porcelain", "--untracked-files=no").splitlines()}


def provenance(extra: dict) -> dict:
    return {"experiment": EXP, "git": git_state(), "python": sys.version.split()[0], "numpy": np.__version__,
            "host": platform.node(), "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "script_sha256": {str(s.relative_to(C.ROOT)): C.sha256(s) for s in SOURCES_RUN}, **extra}


# ------------------------------------------------------------------------------ thresholds
def thresholds(pools: dict, dets) -> dict:
    """tau per (det, p, arm, seed) on the primary scale from validation real degraded scores."""
    tau = {}
    for p, pp in pools.items():
        for det in dets:
            for arm in C.ARMS:
                for s in C.SEEDS:
                    tau[(det, p, arm, s)] = C.thr_e023(pp.val_real[(det, arm, s, C.ANCHOR)])
    return tau


def eligibility(pools: dict, tau: dict, dets) -> dict:
    """F1 §4: arm-cell (det, p, arm) eligible iff every seed has finite tau and val rejection >= 0.5 %."""
    el = {}
    for p, pp in pools.items():
        for det in dets:
            for arm in C.ARMS:
                el[(det, p, arm)] = all(C.eligible_tau(pp.val_real[(det, arm, s, C.ANCHOR)], tau[(det, p, arm, s)])
                                        for s in C.SEEDS)
    return el


def cells(dets):
    return [(d, p, op) for d in dets for p in C.PROTOS for op in CELL_OPS]


# ------------------------------------------------------------------------------ analysis
ENDPOINTS = ["B_cred", "C50"] + [f"L{k}" for k in K_LIST] + [f"A{k}" for k in K_LIST] + \
            ["B_subj", "C50_subj", "L3_subj", "L5_subj", "A3_subj", "L3_M1", "A3_M1", "L5_M1"]


def per_seed_values(pp, det, arm, s, cond, tau, W, Wsubj):
    """All endpoints for one dump under credential weights W (R, F) and subject weights Wsubj (R, U)."""
    u = pp.units
    rej = (pp.real[(det, arm, s, cond)] > tau).astype(np.float64)
    pf = C.per_cred_mean(rej, u)
    cols = {"B_cred": pf, "C50": (pf >= 0.5).astype(np.float64)}
    for k in K_LIST:
        cols[f"L{k}"] = C.denial(pf, k)
        cols[f"A{k}"] = C.attempts(pf, k)
    X = np.stack(list(cols.values()), axis=1)
    out = dict(zip(cols, C.wmean(W, X).T))
    # subject-level (S4): per-subject mean over its credentials, then subject weights
    Msub = np.zeros((u.U, u.F)); Msub[u.subj_of_cred, np.arange(u.F)] = 1.0
    Msub /= Msub.sum(axis=1, keepdims=True)
    Xs = Msub @ np.stack([pf, (pf >= 0.5).astype(float), pf ** 3, pf ** 5, C.attempts(pf, 3)], axis=1)
    for name, v in zip(("B_subj", "C50_subj", "L3_subj", "L5_subj", "A3_subj"), C.wmean(Wsubj, Xs).T):
        out[name] = v
    # M1 without replacement (S1), credentials with n_f >= K
    nf = u.n_f(); rc = np.bincount(u.cred_of_capture, weights=rej, minlength=u.F)
    for k, names in ((3, ("L3_M1", "A3_M1")), (5, ("L5_M1", None))):
        den, att, el = C.without_replacement(rc, nf, k)
        We = W * el[None, :]
        out[names[0]] = C.wmean(We, den[:, None])[:, 0]
        if names[1]:
            out[names[1]] = C.wmean(We, att[:, None])[:, 0]
    out["D10"] = C.d10(pf, W)
    out["_B_cap"] = rej.mean()
    out["_pf"] = pf
    return out


def run_analysis(pools, tau, el, dets, B, outdir: Path, log, pilot: bool) -> dict:
    shared = SharedUnits(pools)
    ss = np.random.SeedSequence(SEED).spawn(4)
    rng_seed, rng_subj, _, rng_null = (np.random.default_rng(x) for x in ss)
    names = ENDPOINTS + ["D10"]
    keys = [(det, p, arm, cond) for det in dets for p in C.PROTOS for arm in C.ARMS for cond in C.CONDS]

    def block(d, m_seed):
        R = d["subj"].shape[0]
        res = {}
        for (det, p, arm, cond) in keys:
            pp = pools[p]
            W = shared.cred_weights(d, p, pp.units); Ws = shared.subj_weights(d, p)
            per = {n: np.empty((R, len(C.SEEDS))) for n in names}
            for j, s in enumerate(C.SEEDS):
                v = per_seed_values(pp, det, arm, s, cond, tau[(det, p, arm, s)], W, Ws)
                for n in names:
                    per[n][:, j] = v[n]
            m = m_seed[(det, p)]
            for n in names:
                res[(det, p, arm, cond, n)] = C.seed_average(per[n], m)
        return res

    # full sample
    full = block(ones_draw(shared, with_syn=False), {(d, p): np.ones((1, 10)) for d in dets for p in C.PROTOS})
    full = {k: float(v[0]) for k, v in full.items()}
    # bootstrap
    reps = {}
    t0 = time.time()
    for b0 in range(0, B, CHUNK):
        n = min(CHUNK, B - b0)
        m_seed = {(d, p): C.row_counts(rng_seed, n, 10) for d in dets for p in C.PROTOS}
        d = shared.draw(rng_subj, None, n)
        r = block(d, m_seed)
        for k, v in r.items():
            reps.setdefault(k, []).append(v)
        log(f"  bootstrap {b0 + n}/{B} ({time.time() - t0:.0f}s)")
    reps = {k: np.concatenate(v) for k, v in reps.items()}

    # contrasts
    rows_obs, rows_mod = [], []
    obs_names = {"B_cred", "C50", "D10", "B_subj", "C50_subj"}
    for (det, p, op) in cells(dets):
        cond = C.OPS[op]
        for n in names:
            for arm in C.ARMS:
                for kind, fv, rv in (("level_anchor", full[(det, p, arm, C.ANCHOR, n)], reps[(det, p, arm, C.ANCHOR, n)]),
                                     ("level_restored", full[(det, p, arm, cond, n)], reps[(det, p, arm, cond, n)]),
                                     ("delta", full[(det, p, arm, cond, n)] - full[(det, p, arm, C.ANCHOR, n)],
                                      reps[(det, p, arm, cond, n)] - reps[(det, p, arm, C.ANCHOR, n)])):
                    lo, hi = C.pctl(rv, 0.95)
                    row = dict(det=det, protocol=p, op=op, arm=arm, endpoint=n, kind=kind, estimate=fv, lo95=lo, hi95=hi,
                               eligible=el[(det, p, arm)])
                    (rows_obs if n in obs_names else rows_mod).append(row)
            dd_f = (full[(det, p, "fix", cond, n)] - full[(det, p, "fix", C.ANCHOR, n)]) - \
                   (full[(det, p, "leg", cond, n)] - full[(det, p, "leg", C.ANCHOR, n)])
            dd_r = (reps[(det, p, "fix", cond, n)] - reps[(det, p, "fix", C.ANCHOR, n)]) - \
                   (reps[(det, p, "leg", cond, n)] - reps[(det, p, "leg", C.ANCHOR, n)])
            lo, hi = C.pctl(dd_r, 0.95)
            (rows_obs if n in obs_names else rows_mod).append(
                dict(det=det, protocol=p, op=op, arm="fix-leg", endpoint=n, kind="ddelta", estimate=dd_f, lo95=lo, hi95=hi,
                     eligible=el[(det, p, "leg")] and el[(det, p, "fix")]))

    def joint(n, arm, cell_dets):
        cs = [c for c in cells(cell_dets) if (el[(c[0], c[1], arm)] if arm != "dd" else el[(c[0], c[1], "leg")] and el[(c[0], c[1], "fix")])]
        def delta(src, c, a):
            d_, p_, op_ = c
            return src[(d_, p_, a, C.OPS[op_], n)] - src[(d_, p_, a, C.ANCHOR, n)]
        if arm == "dd":
            fv = [delta(full, c, "fix") - delta(full, c, "leg") for c in cs]
            rv = np.stack([delta(reps, c, "fix") - delta(reps, c, "leg") for c in cs], axis=1)
        else:
            fv = [delta(full, c, arm) for c in cs]
            rv = np.stack([delta(reps, c, arm) for c in cs], axis=1)
        med_r = np.nanmedian(rv, axis=1)
        return {"n_cells": len(cs), "median": float(np.nanmedian(fv)),
                "ci95": C.pctl(med_r, 0.95), "ci975": C.pctl(med_r, 0.975)}

    summary = {}
    for n in names:
        for arm in ("leg", "fix", "dd"):
            summary[f"{n}|{arm}"] = joint(n, arm, dets)
            summary[f"{n}|{arm}|noD3"] = joint(n, arm, [d for d in dets if d != "Dt"])
    # eligibility counts and decision
    n_inel = {arm: sum(not el[(d, p, arm)] for (d, p, op) in cells(dets)) for arm in C.ARMS}
    uninterpretable = any(v > 3 for v in n_inel.values())
    h1 = summary["L3|leg"]["ci975"][0] > 0
    h2 = summary["L3|dd"]["ci975"][1] < 0
    outcome = "UNINTERPRETABLE" if uninterpretable else ("U-A" if h1 and h2 else ("U-B" if h1 else "U-C"))
    decision = {"pilot_not_evidence": pilot, "H1_supported": bool(h1), "H2_supported": bool(h2),
                "outcome": outcome, "ineligible_cells": n_inel, "H1": summary["L3|leg"], "H2": summary["L3|dd"],
                "rule": "H1: lower 97.5% bound of median dL3 (legacy) > 0; H2: upper 97.5% bound of median ddL3 < 0"}

    # D10 homogeneity null (full sample; descriptive)
    null_rows = []
    for (det, p, arm, cond) in keys:
        pp = pools[p]; nf = pp.units.n_f()
        draws = []
        for s in C.SEEDS:
            b = float(((pp.real[(det, arm, s, cond)] > tau[(det, p, arm, s)])).mean())
            draws.append(C.d10_null(nf, b, rng_null, 2000 if not pilot else 200))
        dn = np.nanmean(np.stack(draws), axis=0)
        null_rows.append(dict(det=det, protocol=p, arm=arm, condition=cond, D10=full[(det, p, arm, cond, "D10")],
                              null_mean=float(np.nanmean(dn)), null_q975=float(np.nanquantile(dn, 0.975))))

    outdir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("cells_observed.csv", rows_obs), ("cells_modelled.csv", rows_mod), ("d10_null.csv", null_rows)):
        with open(outdir / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (outdir / "joint_summary.json").write_text(json.dumps(summary, indent=1))
    (outdir / "OUTCOME.json").write_text(json.dumps(decision, indent=1))
    # B_cap point estimates only (context; E052 boundary: no interval for its ddelta)
    bcap = [dict(det=det, protocol=p, arm=arm, condition=cond, B_cap=full_bcap(pools, det, p, arm, cond, tau))
            for (det, p, arm, cond) in keys]
    with open(outdir / "b_cap_point_estimates.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(bcap[0])); w.writeheader(); w.writerows(bcap)
    return decision


def full_bcap(pools, det, p, arm, cond, tau):
    return float(np.mean([(pools[p].real[(det, arm, s, cond)] > tau[(det, p, arm, s)]).mean() for s in C.SEEDS]))


# ------------------------------------------------------------------------------ gates
def gate_m(pools) -> dict:
    """G-M: metadata identity (test metadata only; no score) against E052 Stage 0 and the manifest."""
    out = {"per_protocol": {p: pp.checks for p, pp in pools.items()}}
    ok = all(pp.checks["n_real"] == 1653 and pp.checks["n_cred"] == 309 and pp.checks["n_subj"] == 34
             and pp.checks["n_val_real"] == 1775 for pp in pools.values())
    same = (pools["p1"].real_pid == pools["p2"].real_pid).all() and \
           (pools["p1"].units.cred_names == pools["p2"].units.cred_names).all()
    # manifest identity of every real test patch
    man = {}
    for p in C.PROTOS:
        with open(C.MANIFEST[p]) as fh:
            for r in csv.DictReader(fh):
                man[(p, r["patch_id"])] = (r["identity"], r["split"])
    mism = 0
    for p, pp in pools.items():
        names = pp.units.cred_names[pp.units.cred_of_capture]
        for pid, ident in zip(pp.real_pid, names):
            mi = man.get((p, str(pid)))
            if mi is None or mi[0] != ident or mi[1] != "test":
                mism += 1
    vident = {}
    for p in C.PROTOS:
        with open(C.MANIFEST[p]) as fh:
            for r in csv.DictReader(fh):
                if r["split"] == "val" and r["label"] == "0":
                    vident[r["patch_id"]] = r["identity"]
    vsub = {C.subject_of(v) for v in vident.values()}
    out.update({"counts_ok": bool(ok), "p1_equals_p2_real": bool(same), "manifest_mismatches": mism,
                "val_subjects_in_manifest": len(vsub)})
    out["pass"] = bool(ok and same and mism == 0 and len(vsub) == 20)
    return out


def gate_tau(pools_primary, tau_primary, dets) -> dict:
    """G-tau: sigmoid tau vs published thr_locked (6e-7); logit tau vs E051 tau_logit (exact)."""
    r036 = e036()
    pub = {}
    for (det, p, arm), (_, rn, csvp) in r036.SOURCES.items():
        if det not in dets:
            continue
        with open(C.ART / csvp) as fh:
            for r in csv.DictReader(fh):
                if r["run_name"] == rn and r["group"] == "unseen" and r["condition"] == "restored_bilateral":
                    pub[(det, p, arm, int(r["seed"]))] = float(r["thr_locked"])
    e051 = {}
    with open(C.ART / "E051" / "validation_logit_thresholds.csv") as fh:
        for r in csv.DictReader(fh):
            if r["condition"] == "degraded":
                e051[(r["run_id"].rsplit("_s", 1)[0], int(r["seed"]))] = float(r["tau_logit"])
    fails, checked = [], 0
    log = C.AccessLog(EXP)
    for (det, p, arm), (root, rn, _) in r036.SOURCES.items():
        if det not in dets:
            continue
        for s in C.SEEDS:
            f = C.ART / root / f"{rn}__s{s}__degraded.npz"
            z = C.read_members(f, ("val_y", "val_score"), log)
            t_sig = C.thr_e023(z["val_score"][z["val_y"] == 0])
            checked += 1
            if abs(t_sig - pub[(det, p, arm, s)]) > 6e-7:
                fails.append(dict(det=det, p=p, arm=arm, seed=s, check="sigmoid", mine=t_sig, published=pub[(det, p, arm, s)]))
            scale = C.SRC[(det, p, arm)][2]
            tp = tau_primary[(det, p, arm, s)]
            if scale == "sigmoid":
                if abs(tp - t_sig) > 0:
                    fails.append(dict(det=det, p=p, arm=arm, seed=s, check="primary=sigmoid", mine=tp, ref=t_sig))
            else:
                ref = e051.get((C.SRC[(det, p, arm)][1], s))
                checked += 1
                if ref is None or abs(tp - ref) > 1e-6 * max(1.0, abs(ref)):
                    fails.append(dict(det=det, p=p, arm=arm, seed=s, check="logit_vs_E051", mine=tp, ref=ref))
    return {"checked": checked, "failures": fails, "pass": not fails}


# ------------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pretest", action="store_true")
    g.add_argument("--pilot", action="store_true")
    g.add_argument("--test", action="store_true")
    a = ap.parse_args()
    dets = C.HEADLINE
    OUT.mkdir(parents=True, exist_ok=True); LOGDIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    mode = "pretest" if a.pretest else "pilot" if a.pilot else "test"
    logf = open(LOGDIR / f"{mode}_{stamp}.log", "w")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line); logf.write(line + "\n"); logf.flush()

    alog = C.AccessLog(EXP)
    if a.pretest:
        pools = load_pool("meta", dets, alog)
        tau = thresholds(pools, dets); el = eligibility(pools, tau, dets)
        gm = gate_m(pools); gt = gate_tau(pools, tau, dets)
        (OUT / "gate_M.json").write_text(json.dumps(gm, indent=1))
        (OUT / "gate_tau.json").write_text(json.dumps(gt, indent=1))
        (OUT / "eligibility.json").write_text(json.dumps({"|".join(k): v for k, v in el.items()}, indent=1))
        with open(OUT / "validation_thresholds.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["det", "protocol", "arm", "seed", "scale", "tau_lock", "val_real_reject"])
            for (det, p, arm, s), t in sorted(tau.items()):
                w.writerow([det, p, arm, s, C.SRC[(det, p, arm)][2], repr(t),
                            float(np.mean(pools[p].val_real[(det, arm, s, C.ANCHOR)] > t))])
        alog.dump(OUT / "access_log_pretest.json")
        (OUT / "provenance_pretest.json").write_text(json.dumps(provenance({"mode": "pretest"}), indent=1))
        log(f"G-M pass={gm['pass']}  G-tau pass={gt['pass']} ({gt['checked']} checks)  eligible={sum(el.values())}/{len(el)}")
        assert "test_score" not in {e["key"] for e in alog.entries}
        return 0 if gm["pass"] and gt["pass"] else 2
    if a.pilot:
        pools = load_pool("pilot", dets, alog)
        tau = thresholds(pools, dets); el = eligibility(pools, tau, dets)
        dec = run_analysis(pools, tau, el, dets, B_PILOT, OUT / "pilot_validation", log, pilot=True)
        (OUT / "pilot_validation" / "README.txt").write_text(
            "VALIDATION-ONLY PILOT (code exercise). Validation real pool used as a stand-in test set with the\n"
            "threshold from the same pool (circular). B=200. NOT EVIDENCE; never cited.\n")
        alog.dump(OUT / "pilot_validation" / "access_log.json")
        assert "test_score" not in {e["key"] for e in alog.entries}
        log(f"pilot done; outcome field (meaningless, pilot): {dec['outcome']}")
        return 0
    # --test
    auth = C.check_authorisation(EXP, SOURCES_RUN)
    alog.authorisation = auth
    log("authorisation OK; G0 (E036 Gate 0 on the frozen sigmoid dumps)")
    r036 = e036()
    P = {p: r036.Proto(p) for p in C.PROTOS}
    for p in C.PROTOS:
        for f in P[p].files:
            alog.add(f, "test_score+val_score (run_e036.Proto, G0)")
    fs = r036.full_sample(P); g0 = r036.gate0(fs)
    (OUT / "gate_G0_e036.json").write_text(json.dumps(g0, indent=1, default=str))
    if not g0["pass"]:
        log("G0 FAILED -> stop"); return 2
    pools = load_pool("test", dets, alog, authorised=True)
    tau = thresholds(pools, dets); el = eligibility(pools, tau, dets)
    # G0-L: decisions on the primary scale vs the E036 sigmoid decisions
    dis = []
    for p in C.PROTOS:
        Pp = P[p]
        for det in dets:
            for arm in C.ARMS:
                for s in C.SEEDS:
                    for cond in C.CONDS:
                        sig = Pp.test[(det, arm, s, cond)][Pp.R] > fs[(det, p, arm, s)]["tau"]
                        mine = pools[p].real[(det, arm, s, cond)] > tau[(det, p, arm, s)]
                        n = int((sig != mine).sum())
                        if n:
                            dis.append(dict(det=det, p=p, arm=arm, seed=s, cond=cond, n_disagree=n))
    (OUT / "gate_G0L.json").write_text(json.dumps({"disagreements": dis, "n": len(dis)}, indent=1))
    log(f"G0 pass; G0-L disagreements listed: {len(dis)}")
    dec = run_analysis(pools, tau, el, dets, B_PRIMARY, OUT, log, pilot=False)
    alog.dump(OUT / "access_log.json")
    (OUT / "provenance.json").write_text(json.dumps(provenance({"mode": "test", "auth": auth}), indent=1))
    log(f"E060 outcome: {dec['outcome']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
