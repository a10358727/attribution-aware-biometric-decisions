#!/usr/bin/env python3
"""E072 -- burden validity (Task A) and individual/distributional burden (Task B). Post hoc, exploratory.

Frozen plan: docs/C5/C5_Research_Protocol.md (§1-§3, §5-§6); freeze record docs/C5/C5_FREEZE.json.

    python experiments/E072_burden_validity/run_e072.py --pilot   # validation pool stand-in (not evidence)
    python experiments/E072_burden_validity/run_e072.py --test    # needs artifacts/results/E072/TEST_READ_AUTHORISATION.json
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
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
from experiments.tcss_lib import core as C                     # noqa: E402
from experiments.tcss_lib.pool import load_pool                # noqa: E402
from experiments.c5_lib import stats as S                      # noqa: E402

EXP = "E072"
SOURCES_RUN = [HERE, C.ROOT / "experiments/c5_lib/stats.py", C.ROOT / "experiments/tcss_lib/core.py",
               C.ROOT / "experiments/tcss_lib/pool.py"]
DETS = C.HEADLINE
B = 2000
D_NULL = 1000
SEED = 720701
KS = (3, 5)
LVL_A2 = 0.975
LVL_B = 1 - 0.05 / 3
TYPES = ("blur", "contrast", "dry", "noise", "smear", "wet")


def log(msg, fh=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fh:
        fh.write(line + "\n"); fh.flush()


def git_state():
    run = lambda *a: subprocess.run(a, cwd=C.ROOT, capture_output=True, text=True).stdout.strip()
    return {"sha": run("git", "rev-parse", "HEAD"), "dirty": run("git", "status", "--porcelain", "--untracked-files=no").splitlines()}


def strata_for(pids):
    """deg type index, severity, stratum code (type*4 + sev-1) from the manifest (identical in P1/P2)."""
    meta = {}
    with open(C.MANIFEST["p1"]) as fh:
        for r in csv.DictReader(fh):
            meta[r["patch_id"]] = (r["deg_type"], int(r["deg_severity"]))
    t = np.array([TYPES.index(meta[str(x)][0]) for x in pids]); s = np.array([meta[str(x)][1] for x in pids])
    return t, s, t * 4 + (s - 1)


def cells():
    return [(d, p, op) for d in DETS for p in C.PROTOS for op in C.OPS]


# ------------------------------------------------------------------ decisions
def decisions(pools):
    """y[(det,p,arm,s,cond)] = bool rejections of real rows at tau_s (E023 rule on validation anchor)."""
    Y, tau = {}, {}
    for p, pp in pools.items():
        for det in DETS:
            for arm in C.ARMS:
                for s in C.SEEDS:
                    t = C.thr_e023(pp.val_real[(det, arm, s, C.ANCHOR)])
                    tau[(det, p, arm, s)] = t
                    for cond in C.CONDS:
                        Y[(det, p, arm, s, cond)] = pp.real[(det, arm, s, cond)] > t
    return Y, tau


# ------------------------------------------------------------------ A1 (full sample, parametric nulls)
def a1_job(args):
    """One (det, p, arm, cond) with all seeds: observed and null statistics for every model."""
    key, ys, anchors, g, tidx, cred, F, subj_cap, U, keep3, seedseq = args
    rng = np.random.default_rng(seedseq)
    restored = anchors is not None
    models = {"N0": (np.zeros_like(g), 1), "N1": (g, 24), "S-A1a_N1type": (tidx, 6)}
    if restored:
        models["N2"] = (g * 2, 48)               # placeholder; anchor added per seed below
        models["S-A1a_N2type"] = (tidx * 2, 12)
    out = {}
    n_f = np.bincount(cred, minlength=F).astype(float)
    for mname in models:
        T_obs = {"phi": [], "d10": [], "phi_n3": [], "phi_subj": []}
        T_null = {k: np.zeros((len(ys), D_NULL)) for k in T_obs}
        for j, y in enumerate(ys):
            gg, G = models[mname]
            if mname in ("N2", "S-A1a_N2type"):
                gg = gg + anchors[j].astype(int)
            y8 = y.astype(np.uint8)
            T_obs["phi"].append(float(S.phi_conditioned(y8, gg, G, cred, F)))
            T_obs["phi_n3"].append(float(S.phi_conditioned(y8, gg, G, cred, F, keep=keep3)))
            T_obs["phi_subj"].append(float(S.phi_conditioned(y8, gg, G, subj_cap, U)))
            pf = np.bincount(cred, weights=y.astype(float), minlength=F) / n_f
            T_obs["d10"].append(float(C.d10_unit_rows(pf[None, :])[0]))
            pi = S.stratum_rates(y8, gg, G)
            Ysim = S.simulate_null(pi, D_NULL, rng)
            T_null["phi"][j] = S.phi_conditioned(Ysim, gg, G, cred, F)
            T_null["phi_n3"][j] = S.phi_conditioned(Ysim, gg, G, cred, F, keep=keep3)
            T_null["phi_subj"][j] = S.phi_conditioned(Ysim, gg, G, subj_cap, U)
            T_null["d10"][j] = C.d10_unit_rows(S.unit_sums(Ysim, cred, F) / n_f[None, :])
        res = {}
        for st in T_obs:
            T = float(np.nanmean(T_obs[st])); Tn = np.nanmean(T_null[st], axis=0)
            res[st] = {"T": T, "null_mean": float(np.nanmean(Tn)), "null_q975": float(np.nanquantile(Tn, 0.975)),
                       "exceeds": bool(T > np.nanquantile(Tn, 0.975)),
                       "ratio": float(T / np.nanmean(Tn)) if np.nanmean(Tn) > 0 else math.nan,
                       "n_seed_nan": int(np.isnan(T_obs[st]).sum())}
        out[mname] = res
    return key, out


def run_a1(pools, Y, strata, workers, fh):
    jobs = []
    ss = np.random.SeedSequence([SEED, 1])
    children = iter(ss.spawn(200))
    for p, pp in pools.items():
        u = pp.units
        tidx, sev, g = strata[p]
        F, U = u.F, u.U
        cred = u.cred_of_capture; subj_cap = u.subj_of_cred[cred]
        keep3 = u.n_f() >= 3
        for det in DETS:
            for arm in C.ARMS:
                ys = [Y[(det, p, arm, s, C.ANCHOR)] for s in C.SEEDS]
                jobs.append(((det, p, arm, "anchor"), ys, None, g, tidx, cred, F, subj_cap, U, keep3, next(children)))
                for op, cond in C.OPS.items():
                    ys = [Y[(det, p, arm, s, cond)] for s in C.SEEDS]
                    jobs.append(((det, p, arm, op), ys, [Y[(det, p, arm, s, C.ANCHOR)] for s in C.SEEDS],
                                 g, tidx, cred, F, subj_cap, U, keep3, next(children)))
    res = {}
    with ProcessPoolExecutor(workers) as ex:
        for i, (k, v) in enumerate(ex.map(a1_job, jobs)):
            res[k] = v
            if (i + 1) % 6 == 0:
                log(f"  A1 {i + 1}/{len(jobs)}", fh)
    return res


# ------------------------------------------------------------------ A3 (GLMM)
def a3_job(args):
    key, y, X, cred, sub_of_cred, U, rate = args
    out = {"rate": rate}
    try:
        full = S.glmm_fit(X, y, cred, sub_of_cred, U)
        no_c = S.glmm_fit(X, y, cred, sub_of_cred, U, use_c=False)
        no_s = S.glmm_fit(X, y, cred, sub_of_cred, U, use_s=False)
        lr_c = 2 * (no_c["negll"] - full["negll"]); lr_s = 2 * (no_s["negll"] - full["negll"])
        ss, sc = full["sig_s"], full["sig_c"]
        flag = (not full["ok"]) or ss < 0.01 or sc < 0.01 or ss > 10 or sc > 10 or rate < 0.02
        tot = ss ** 2 + sc ** 2 + math.pi ** 2 / 3
        out.update(ok=full["ok"], sig_s=ss, sig_c=sc, share_s=ss ** 2 / tot, share_c=sc ** 2 / tot,
                   lr_c=lr_c, p_c=S.lrt_boundary_p(lr_c), lr_s=lr_s, p_s=S.lrt_boundary_p(lr_s), flagged=bool(flag))
    except Exception as e:                                   # reported, never retried with other settings
        out.update(ok=False, error=repr(e), flagged=True)
    return key, out


def run_a3(pools, Y, strata, workers, fh):
    jobs = []
    for p, pp in pools.items():
        u = pp.units; tidx, sev, g = strata[p]
        tnames = np.array([TYPES[i] for i in tidx])
        for det in DETS:
            for arm in C.ARMS:
                for s in C.SEEDS:
                    ya = Y[(det, p, arm, s, C.ANCHOR)]
                    X0 = S.design(tnames, sev)
                    jobs.append(((det, p, arm, "anchor", s, "R1"), ya.astype(float), X0, u.cred_of_capture,
                                 u.subj_of_cred, u.U, float(ya.mean())))
                    for op, cond in C.OPS.items():
                        yr = Y[(det, p, arm, s, cond)]
                        jobs.append(((det, p, arm, op, s, "R1"), yr.astype(float), X0, u.cred_of_capture,
                                     u.subj_of_cred, u.U, float(yr.mean())))
                        jobs.append(((det, p, arm, op, s, "R2"), yr.astype(float), S.design(tnames, sev, ya),
                                     u.cred_of_capture, u.subj_of_cred, u.U, float(yr.mean())))
    res = {}
    with ProcessPoolExecutor(workers) as ex:
        for i, (k, v) in enumerate(ex.map(a3_job, jobs, chunksize=4)):
            res[k] = v
            if (i + 1) % 100 == 0:
                log(f"  A3 {i + 1}/{len(jobs)}", fh)
    return res


def summarise_a3(res):
    rows = []
    groups = {}
    for (det, p, arm, cond, s, var), v in res.items():
        kind = "anchor" if cond == "anchor" else "restored"
        groups.setdefault((arm, kind, var), []).append(v)
    for (arm, kind, var), vs in sorted(groups.items()):
        ok = [v for v in vs if not v.get("flagged", True)]
        n, nf = len(vs), len(vs) - len(ok)
        ident = nf <= n / 2
        r = {"arm": arm, "condition": kind, "variant": var, "n_fits": n, "n_flagged": nf, "identifiable": ident}
        if ident:
            for q in ("sig_s", "sig_c", "share_s", "share_c"):
                r[q + "_median"] = float(np.median([v[q] for v in ok]))
            r["share_p_c_lt_0.05"] = float(np.mean([v["p_c"] < 0.05 for v in ok]))
            r["share_p_s_lt_0.05"] = float(np.mean([v["p_s"] < 0.05 for v in ok]))
        rows.append(r)
    return rows


# ------------------------------------------------------------------ A2 + B (bootstrap)
def endpoints_seed(k, n, units, Wc, Ws, A, valid3):
    """All A2/B endpoints for one (cell, arm, cond, seed). Wc (R,F) credential weights, Ws (R,U) subject weights.
    Returns dict name -> (R,) and subject vector b (U,) for paired deltas."""
    p = k / n
    out = {}
    for K in KS:
        out[f"L{K}_E1"] = S.wmean_rows(p ** K, Wc)
        vK = n >= K
        out[f"L{K}_E2"] = S.wmean_rows((p ** K)[vK], Wc[:, vK])
        out[f"L{K}_E3"] = S.wmean_rows(S.unbiased_pK(k, n, K)[vK], Wc[:, vK])
    a, b, m, rho = S.bb_moments(k, n, Wc)
    for K in KS:
        post = S.eb_pK(k[None, :], n[None, :], a[:, None], b[:, None], K)      # (R, F)
        out[f"L{K}_E4"] = (Wc * post).sum(1) / Wc.sum(1)
    out["B_cred"] = S.wmean_rows(p, Wc)
    out["C50"] = S.wmean_rows((p >= 0.5).astype(float), Wc)
    out["cred_CVaR80"] = S.wtail_mean(p, Wc, 0.8)
    # subject level
    bsub = A @ p                                           # credential-averaged single-attempt rate
    out["subj_mean"] = S.wmean_rows(bsub, Ws)
    for q in (0.25, 0.5, 0.75, 0.9):
        out[f"subj_Q{int(q * 100)}"] = S.wquantile(bsub, Ws, q)
    out["subj_CVaR80"] = S.wtail_mean(bsub, Ws, 0.8)
    out["subj_CVaR90"] = S.wtail_mean(bsub, Ws, 0.9)
    out["subj_top20_share"] = S.wtop_share(bsub, Ws, 0.2)
    L3s = A @ (p ** 3)
    out["subj_L3_E1_mean"] = S.wmean_rows(L3s, Ws); out["subj_L3_E1_CVaR80"] = S.wtail_mean(L3s, Ws, 0.8)
    # S-B1..3 (aggregation sensitivity)
    U = units.U
    k_s = np.bincount(units.subj_of_cred, weights=k, minlength=U); n_s = np.bincount(units.subj_of_cred, weights=n, minlength=U)
    pooled = k_s / n_s
    mn = np.full(U, np.inf); mx = np.full(U, -np.inf)
    np.minimum.at(mn, units.subj_of_cred, p); np.maximum.at(mx, units.subj_of_cred, p)
    for nm, v in (("SB1_pooled", pooled), ("SB2_best", mn), ("SB3_worst", mx)):
        out[f"{nm}_mean"] = S.wmean_rows(v, Ws); out[f"{nm}_CVaR80"] = S.wtail_mean(v, Ws, 0.8)
    return out, bsub, (a, b)


def run_bootstrap(pools, Y, fh, B_):
    ss = np.random.SeedSequence([SEED, 2]).spawn(2)
    rng_seed, rng_subj = (np.random.default_rng(x) for x in ss)
    subjects = sorted(set().union(*[set(pp.units.subj_names.tolist()) for pp in pools.values()]))
    pos = {n: i for i, n in enumerate(subjects)}
    Wsubj_all = np.vstack([np.ones((1, len(subjects))), C.row_counts(rng_subj, B_, len(subjects))])  # row 0 = full sample
    mseed = {(d, p): np.vstack([np.ones((1, 10)), C.row_counts(rng_seed, B_, 10)]) for d in DETS for p in C.PROTOS}
    R = B_ + 1
    res, subj_vecs, ebprior = {}, {}, {}
    for p, pp in pools.items():
        u = pp.units
        Ws = Wsubj_all[:, [pos[n] for n in u.subj_names]]
        Wc = Ws[:, u.subj_of_cred]
        A = S.subject_matrix(u.subj_of_cred, u.U)
        n = u.n_f().astype(float)
        for det in DETS:
            m = mseed[(det, p)]
            for arm in C.ARMS:
                for cond in C.CONDS:
                    per = {}
                    bvecs = []
                    for j, s in enumerate(C.SEEDS):
                        k = np.bincount(u.cred_of_capture, weights=Y[(det, p, arm, s, cond)].astype(float), minlength=u.F)
                        e, bsub, ab = endpoints_seed(k, n, u, Wc, Ws, A, n >= 3)
                        for name, v in e.items():
                            per.setdefault(name, np.empty((R, 10)))[:, j] = v
                        bvecs.append(bsub)
                        if j == 0:
                            ebprior[(det, p, arm, cond)] = (float(ab[0][0]), float(ab[1][0]))
                    for name, E in per.items():
                        res[(det, p, arm, cond, name)] = C.seed_average(E, m)
                    subj_vecs[(det, p, arm, cond)] = np.stack(bvecs)        # (10, U)
            log(f"  bootstrap endpoints {det} {p} done", fh)
        for det in DETS:
            m = mseed[(det, p)]
            for arm in C.ARMS:
                for op, cond in C.OPS.items():
                    d = subj_vecs[(det, p, arm, cond)] - subj_vecs[(det, p, arm, C.ANCHOR)]      # (10, U)
                    per = {"worse_gt0": np.empty((R, 10)), "worse_gt005": np.empty((R, 10)),
                           "dQ10": np.empty((R, 10)), "dQ50": np.empty((R, 10)), "dQ90": np.empty((R, 10))}
                    for j in range(10):
                        per["worse_gt0"][:, j] = S.wshare(d[j] > 0, Ws)
                        per["worse_gt005"][:, j] = S.wshare(d[j] > 0.05, Ws)
                        for q in (10, 50, 90):
                            per[f"dQ{q}"][:, j] = S.wquantile(d[j], Ws, q / 100)
                    for name, E in per.items():
                        res[(det, p, arm, op, "paired_" + name)] = C.seed_average(E, m)
    return res, subj_vecs, ebprior


def summarise_boot(res, level_map):
    """Per-cell levels, deltas, ddeltas and across-cell medians with intervals. Row 0 = full sample."""
    names = sorted({k[4] for k in res if not k[4].startswith("paired_")})
    rows, joint = [], {}

    def lvl(det, p, arm, cond, nm):
        return res[(det, p, arm, cond, nm)]

    for nm in names:
        for arm in C.ARMS:
            for kind in ("anchor", "restored", "delta"):
                vals = []
                for (det, p, op) in cells():
                    if kind == "anchor":
                        v = lvl(det, p, arm, C.ANCHOR, nm)
                    elif kind == "restored":
                        v = lvl(det, p, arm, C.OPS[op], nm)
                    else:
                        v = lvl(det, p, arm, C.OPS[op], nm) - lvl(det, p, arm, C.ANCHOR, nm)
                    vals.append(v)
                    lo, hi = C.pctl(v[1:], 0.95)
                    rows.append(dict(det=det, protocol=p, op=op, arm=arm, endpoint=nm, kind=kind,
                                     estimate=float(v[0]), lo95=lo, hi95=hi))
                M = np.nanmedian(np.stack(vals, 1), axis=1)
                lev = level_map.get((nm, arm, kind), 0.95)
                joint[f"{nm}|{arm}|{kind}"] = {"median": float(M[0]), "ci95": C.pctl(M[1:], 0.95),
                                              "ci_decision": C.pctl(M[1:], lev), "decision_level": lev}
        vals = []
        for (det, p, op) in cells():
            dl = lvl(det, p, "leg", C.OPS[op], nm) - lvl(det, p, "leg", C.ANCHOR, nm)
            df = lvl(det, p, "fix", C.OPS[op], nm) - lvl(det, p, "fix", C.ANCHOR, nm)
            v = df - dl
            vals.append(v)
            lo, hi = C.pctl(v[1:], 0.95)
            rows.append(dict(det=det, protocol=p, op=op, arm="dd", endpoint=nm, kind="ddelta",
                             estimate=float(v[0]), lo95=lo, hi95=hi))
        M = np.nanmedian(np.stack(vals, 1), axis=1)
        lev = level_map.get((nm, "dd", "ddelta"), 0.95)
        joint[f"{nm}|dd|ddelta"] = {"median": float(M[0]), "ci95": C.pctl(M[1:], 0.95),
                                   "ci_decision": C.pctl(M[1:], lev), "decision_level": lev}
    pnames = sorted({k[4] for k in res if k[4].startswith("paired_")})
    for nm in pnames:
        for arm in C.ARMS:
            vals = []
            for (det, p, op) in cells():
                v = res[(det, p, arm, op, nm)]
                vals.append(v)
                lo, hi = C.pctl(v[1:], 0.95)
                rows.append(dict(det=det, protocol=p, op=op, arm=arm, endpoint=nm, kind="paired",
                                 estimate=float(v[0]), lo95=lo, hi95=hi))
            M = np.nanmedian(np.stack(vals, 1), axis=1)
            lev = level_map.get((nm, arm, "paired"), 0.95)
            joint[f"{nm}|{arm}|paired"] = {"median": float(M[0]), "ci95": C.pctl(M[1:], 0.95),
                                          "ci_decision": C.pctl(M[1:], lev), "decision_level": lev}
    return rows, joint


# ------------------------------------------------------------------ A2 bias simulation, n_f strata, LOO, variance
def bias_sim(pools, Y, ebprior, fh):
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 3]))
    rows = []
    for p, pp in pools.items():
        u = pp.units; n = u.n_f()
        for det in DETS:
            for arm in C.ARMS:
                for cond in C.CONDS:
                    a, b = ebprior[(det, p, arm, cond)]
                    acc = {}
                    for _ in range(500):
                        pt = rng.beta(a, b, u.F); k = rng.binomial(n, pt); ph = k / n
                        for nm, f in (("L3", lambda x: np.mean(x ** 3)),
                                      ("C25", lambda x: np.mean(x >= 0.25)), ("C50", lambda x: np.mean(x >= 0.5)),
                                      ("C75", lambda x: np.mean(x >= 0.75)),
                                      ("D10", lambda x: C.d10_unit_rows(x[None, :])[0]),
                                      ("CVaR80", lambda x: S.wtail_mean(x, np.ones((1, len(x))), 0.8)[0]),
                                      ("CVaR90", lambda x: S.wtail_mean(x, np.ones((1, len(x))), 0.9)[0])):
                            acc.setdefault(nm, []).append((f(ph), f(pt)))
                    for nm, v in acc.items():
                        v = np.array(v, float)
                        tr = np.nanmean(v[:, 1]); pl = np.nanmean(v[:, 0])
                        rows.append(dict(det=det, protocol=p, arm=arm, condition=cond, endpoint=nm, prior_a=a, prior_b=b,
                                         truth_mean=tr, plugin_mean=pl, abs_bias=pl - tr,
                                         rel_bias=(pl - tr) / tr if tr > 0 else math.nan))
        log(f"  bias simulation {p} done", fh)
    return rows


def descriptive_full(pools, Y, ebprior):
    """n_f-stratum C25/C50/C75 (plug-in and E4), LOO ranges of subject CVaR, variance components. Full sample."""
    strat_rows, loo_rows, var_rows = [], [], []
    for p, pp in pools.items():
        u = pp.units; n = u.n_f().astype(float)
        A = S.subject_matrix(u.subj_of_cred, u.U)
        band = np.where(n <= 2, "1-2", np.where(n <= 4, "3-4", ">=5"))
        for det in DETS:
            for arm in C.ARMS:
                for cond in C.CONDS:
                    acc = {}
                    lo80, hi80, lo90, hi90, bvar, wvar = [], [], [], [], [], []
                    for s in C.SEEDS:
                        k = np.bincount(u.cred_of_capture, weights=Y[(det, p, arm, s, cond)].astype(float), minlength=u.F)
                        ph = k / n
                        a, b, _, _ = S.bb_moments(k, n)
                        pe = S.eb_mean(k, n, a, b)
                        for bd in ("1-2", "3-4", ">=5"):
                            msk = band == bd
                            for c in (0.25, 0.5, 0.75):
                                acc.setdefault((bd, c, "plugin"), []).append(np.mean(ph[msk] >= c))
                                acc.setdefault((bd, c, "E4"), []).append(np.mean(pe[msk] >= c))
                        bsub = A @ ph
                        r80 = S.loo_range(bsub, lambda x, W: S.wtail_mean(x, W, 0.8))
                        r90 = S.loo_range(bsub, lambda x, W: S.wtail_mean(x, W, 0.9))
                        lo80.append(r80[0]); hi80.append(r80[1]); lo90.append(r90[0]); hi90.append(r90[1])
                        bvar.append(np.var(bsub))
                        wv = [np.var(ph[u.subj_of_cred == i]) for i in range(u.U)]
                        wvar.append(np.mean(wv))
                    for (bd, c, est), v in acc.items():
                        strat_rows.append(dict(det=det, protocol=p, arm=arm, condition=cond, nf_band=bd,
                                               n_cred=int((band == bd).sum()), threshold=c, estimator=est,
                                               share=float(np.mean(v))))
                    loo_rows.append(dict(det=det, protocol=p, arm=arm, condition=cond,
                                         CVaR80_loo_min=float(np.mean(lo80)), CVaR80_loo_max=float(np.mean(hi80)),
                                         CVaR90_loo_min=float(np.mean(lo90)), CVaR90_loo_max=float(np.mean(hi90))))
                    bv, wv = float(np.mean(bvar)), float(np.mean(wvar))
                    var_rows.append(dict(det=det, protocol=p, arm=arm, condition=cond, between_subject_var=bv,
                                         mean_within_subject_var=wv,
                                         between_share=bv / (bv + wv) if bv + wv > 0 else math.nan))
    return strat_rows, loo_rows, var_rows


# ------------------------------------------------------------------ G-R
def gate_reproduce(res, fh):
    obs = list(csv.DictReader(open(C.ART / "E060" / "cells_observed.csv")))
    mod = list(csv.DictReader(open(C.ART / "E060" / "cells_modelled.csv")))
    want = {}
    for r in obs + mod:
        if r["endpoint"] in ("B_cred", "C50", "L3") and r["kind"] in ("level_anchor", "level_restored"):
            want[(r["det"], r["protocol"], r["op"], r["arm"], r["endpoint"], r["kind"])] = float(r["estimate"])
    mine = {"B_cred": "B_cred", "C50": "C50", "L3": "L3_E1"}
    maxdiff = 0.0
    for (det, p, op, arm, ep, kind), v in want.items():
        cond = C.ANCHOR if kind == "level_anchor" else C.OPS[op]
        got = float(res[(det, p, arm, cond, mine[ep])][0])
        maxdiff = max(maxdiff, abs(got - v))
    ok = maxdiff < 1e-9
    log(f"  G-R E060 reproduction: n={len(want)} max|diff|={maxdiff:.3g} -> {'PASS' if ok else 'FAIL'}", fh)
    return {"n": len(want), "max_abs_diff": maxdiff, "pass": ok}


def gate_reproduce_d10(a1, fh):
    obs = list(csv.DictReader(open(C.ART / "E060" / "cells_observed.csv")))
    maxdiff, n = 0.0, 0
    for r in obs:
        if r["endpoint"] == "D10" and r["kind"] in ("level_anchor", "level_restored"):
            key = (r["det"], r["protocol"], r["arm"], "anchor" if r["kind"] == "level_anchor" else r["op"])
            got = a1[key]["N0"]["d10"]["T"]
            maxdiff = max(maxdiff, abs(got - float(r["estimate"]))); n += 1
    ok = maxdiff < 1e-9
    log(f"  G-R D10 reproduction: n={n} max|diff|={maxdiff:.3g} -> {'PASS' if ok else 'FAIL'}", fh)
    return {"n": n, "max_abs_diff": maxdiff, "pass": ok}


# ------------------------------------------------------------------ main
def classify(n_exceed):
    return "persists" if n_exceed >= 9 else "explained" if n_exceed <= 3 else "mixed"


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
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
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--B", type=int, default=None)
    a = ap.parse_args()
    pilot = a.pilot
    out = C.ART / EXP / ("pilot_validation" if pilot else "")
    out.mkdir(parents=True, exist_ok=True)
    logdir = C.ROOT / "artifacts" / "logs" / EXP; logdir.mkdir(parents=True, exist_ok=True)
    fh = open(logdir / f"{'pilot' if pilot else 'test'}_{time.strftime('%Y%m%d_%H%M%S')}.log", "w")
    B_ = a.B or (200 if pilot else B)
    acc = C.AccessLog(EXP)
    auth = None
    if not pilot:
        auth = C.check_authorisation(EXP, SOURCES_RUN)
        acc.authorisation = auth
    log(f"E072 {'PILOT (validation stand-in; not evidence)' if pilot else 'TEST (authorised)'}; B={B_}", fh)
    t0 = time.time()
    pools = load_pool("pilot" if pilot else "test", DETS, acc, authorised=not pilot)
    strata = {p: strata_for(pp.real_pid) for p, pp in pools.items()}
    checks = {p: pp.checks for p, pp in pools.items()}
    log(f"  pools loaded {json.dumps({p: {k: v for k, v in c.items() if k in ('n_real', 'n_cred', 'n_subj')} for p, c in checks.items()})}", fh)
    Y, tau = decisions(pools)

    res, subj_vecs, ebprior = run_bootstrap(pools, Y, fh, B_)
    gates = {}
    if not pilot:
        gates["G-R_E060"] = gate_reproduce(res, fh)
        if not gates["G-R_E060"]["pass"]:
            json.dump(gates, open(out / "gates.json", "w"), indent=1); acc.dump(out / "access_log.json")
            raise SystemExit("G-R failed: stop (no adjustment)")
    level_map = {("L3_E3", "leg", "delta"): LVL_A2, ("L3_E4", "leg", "delta"): LVL_A2,
                 ("subj_CVaR80", "leg", "delta"): LVL_B, ("subj_CVaR80", "dd", "ddelta"): LVL_B,
                 ("paired_worse_gt0", "leg", "paired"): LVL_B}
    rows, joint = summarise_boot(res, level_map)
    write_csv(out / "cells_boot.csv", rows)

    log("  A1 conditioned nulls", fh)
    a1 = run_a1(pools, Y, strata, a.workers, fh)
    if not pilot:
        gates["G-R_D10"] = gate_reproduce_d10(a1, fh)
    a1_rows, a1_class = [], {}
    for (det, p, arm, cond), mm in sorted(a1.items()):
        for mname, sts in mm.items():
            for st, v in sts.items():
                a1_rows.append(dict(det=det, protocol=p, arm=arm, condition=cond, model=mname, statistic=st, **v))
    for arm in C.ARMS:
        for mname in ("N0", "N1", "N2", "S-A1a_N1type", "S-A1a_N2type"):
            for st in ("phi", "d10", "phi_n3", "phi_subj"):
                for kind in ("anchor", "restored"):
                    keys = [(d, p, arm, "anchor") for (d, p, op) in cells()] if kind == "anchor" else \
                           [(d, p, arm, op) for (d, p, op) in cells()]
                    vs = [a1[k][mname][st] for k in keys if mname in a1[k]]
                    if not vs:
                        continue
                    nx = sum(v["exceeds"] for v in vs)
                    a1_class[f"{mname}|{st}|{arm}|{kind}"] = {
                        "n_cells": len(vs), "n_exceed": nx, "class": classify(nx) if len(vs) == 12 else None,
                        "median_ratio": float(np.nanmedian([v["ratio"] for v in vs]))}
    write_csv(out / "a1_nulls.csv", a1_rows)

    log("  A2 bias simulation", fh)
    write_csv(out / "a2_bias_sim.csv", bias_sim(pools, Y, ebprior, fh))
    strat_rows, loo_rows, var_rows = descriptive_full(pools, Y, ebprior)
    write_csv(out / "a2_nf_strata.csv", strat_rows)
    write_csv(out / "b_tail_loo.csv", loo_rows)
    write_csv(out / "b_variance_components.csv", var_rows)

    log("  A3 GLMM", fh)
    a3 = run_a3(pools, Y, strata, a.workers, fh)
    write_csv(out / "a3_glmm_fits.csv", [dict(det=k[0], protocol=k[1], arm=k[2], condition=k[3], seed=k[4], variant=k[5],
                                               **{kk: vv for kk, vv in v.items()}) for k, v in sorted(a3.items(), key=str)])
    a3_sum = summarise_a3(a3)
    write_csv(out / "a3_glmm_summary.csv", a3_sum)

    # subject vectors (aggregate per subject index; no identifiers beyond the corpus's numeric subject index)
    np.savez_compressed(out / "subject_burden_vectors.npz",
                        **{"|".join(k): v for k, v in subj_vecs.items()})

    a2 = {nm: joint[f"{nm}|leg|delta"] for nm in ("L3_E1", "L3_E2", "L3_E3", "L3_E4")}
    decisions_ = {
        "pilot_not_evidence": pilot,
        "A1_primary_N2_restored": {arm: a1_class.get(f"N2|phi|{arm}|restored") for arm in C.ARMS},
        "A2_P1_E3_survives": joint["L3_E3|leg|delta"]["ci_decision"][0] > 0,
        "A2_P2_E4_survives": joint["L3_E4|leg|delta"]["ci_decision"][0] > 0,
        "B_P1_CVaR80_increase_leg": joint["subj_CVaR80|leg|delta"]["ci_decision"][0] > 0,
        "B_P2_CVaR80_reduced_fix": joint["subj_CVaR80|dd|ddelta"]["ci_decision"][1] < 0,
        "B_P3_worse_share_leg": joint["paired_worse_gt0|leg|paired"],
        "CVaR90_unstable": None,
    }
    # CVaR90 stability rule (§3): interval width > point estimate in most cells
    wide = [r for r in rows if r["endpoint"] == "subj_CVaR90" and r["kind"] in ("anchor", "restored")]
    decisions_["CVaR90_unstable"] = bool(np.mean([(r["hi95"] - r["lo95"]) > r["estimate"] for r in wide]) > 0.5)
    json.dump({"joint": joint, "a1_class": a1_class, "a2_leg_delta": a2}, open(out / "joint_summary.json", "w"), indent=1)
    json.dump(decisions_, open(out / "OUTCOME.json", "w"), indent=1, default=str)
    json.dump(gates, open(out / "gates.json", "w"), indent=1)
    acc.dump(out / "access_log.json")
    prov = {"experiment": EXP, "mode": "pilot" if pilot else "test", "git": git_state(), "python": sys.version.split()[0],
            "numpy": np.__version__, "host": platform.node(), "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "B": B_, "D_null": D_NULL, "seed": SEED, "runtime_s": time.time() - t0,
            "script_sha256": {C.rel(s): C.sha256(s) for s in SOURCES_RUN}, "pool_checks": checks}
    json.dump(prov, open(out / "provenance.json", "w"), indent=1)
    log(f"E072 done in {time.time() - t0:.0f}s", fh)


if __name__ == "__main__":
    main()
