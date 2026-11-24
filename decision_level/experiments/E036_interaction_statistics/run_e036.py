#!/usr/bin/env python3
"""E036 -- direct interaction statistics for the E027 verdict (paired two-factor bootstrap).

Preregistration: docs/experiments/E036_direct_interaction_statistics.md, frozen (F1) by the commit
that adds this file. Analysis only: it reads the frozen per-image score dumps behind Table 1 of the
manuscript. No model is loaded, no image is read, nothing is trained.

    python3 experiments/E036_interaction_statistics/run_e036.py --selftest          # synthetic data only
    python3 experiments/E036_interaction_statistics/run_e036.py --check-thresholds  # validation pool only
    python3 experiments/E036_interaction_statistics/run_e036.py                     # the frozen analysis

Estimands, per (detector d, protocol p, operator o); each arm a is measured against its own anchor
and its own locked threshold with the frozen E023 estimator on the unseen group:
    dT_a = mean_s [ T_{a,s}(o) - T_{a,s}(anchor) ]        (dB_a likewise, bona-fide rejection)
    DDT  = dT_fixed - dT_legacy                            DDB = dB_fixed - dB_legacy
Primary interval: 95 % percentile interval of the paired two-factor bootstrap -- training seeds
(resampled per detector and protocol, paired by seed ID across the two arms and shared by both
operators) crossed with test clusters (real finger identities; synthetic images within generator),
one draw per replicate shared by both arms, both conditions and every detector, and by both
protocols wherever their test images coincide. Thresholds are held at their full-sample values,
as in the frozen E023 estimator; a secondary analysis also resamples validation fingers.

Steps, in this order; each is fixed by the preregistration:
  1 Gate 0     full-sample per-seed dT, dB, dAUC and tau_lock reproduce the published result files.
               Any failure stops the run before step 2 (exit code 2), with no substitution.
  2 primary    DDT, DDB per cell with 95 % intervals (B_PRIMARY replicates).
  3 secondary  Bonferroni-width intervals; per-seed paired intervals (all-seed analogue);
               validation-finger resampling of tau_lock; DDAUC; NPR cells; across-cell summaries.
  4 D3         descriptive exclusion summaries (declared post hoc in the preregistration).
  5 decision   the frozen rule -> OUTCOME.json.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "results"
OUT = ART / "E036"
LOGDIR = ROOT / "artifacts" / "logs" / "E036"

ANCHOR = "degraded"
OPS = {"bil": "restored_bilateral", "tv": "restored_tv"}
CONDS = (ANCHOR, OPS["bil"], OPS["tv"])
UNSEEN = {"p1": {"dcgan", "progan", "stylegan2ada", "stylegan3"},      # identical to E023 / E027
          "p2": {"adm", "dcgan", "stylegan2ada", "stylegan3"}}
PROTOS = ("p1", "p2")
SEEDS = tuple(range(10))
HEADLINE = ("Rn", "Bs", "Dt")
SUPPLEMENTAL = ("Np",)
DETS = HEADLINE + SUPPLEMENTAL
ARMS = ("leg", "fix")
DET_NAME = {"Rn": "ResNet-18", "Bs": "Benjamin et al.", "Dt": "D3", "Np": "NPR"}

B_PRIMARY = 10_000   # primary: paired two-factor bootstrap
B_SEED = 2_000       # secondary: per-seed paired cluster bootstrap (first B_SEED cluster draws)
B_VAL = 2_000        # secondary: validation-finger resampling of tau_lock (first B_VAL replicates)
B_AUC = 2_000        # secondary: DDAUC (first B_AUC replicates)
CHUNK = 1_000
SEED = 361001        # bootstrap RNG seed, fixed in the preregistration
ALPHA = 0.05
N_HEADLINE_CELLS = 12
Q = (ALPHA / 2, 1 - ALPHA / 2)
Q_BONF = (ALPHA / N_HEADLINE_CELLS / 2, 1 - ALPHA / N_HEADLINE_CELLS / 2)
TOL_D = 6e-6         # Gate 0: published d_* are rounded to 5 decimals
TOL_THR = 6e-7       # Gate 0: published thr_locked is rounded to 6 decimals

# (detector, protocol, arm) -> (dump root, run_name, published result CSV).  Same mapping as the
# manuscript's number generator (paper/figs/mk_numbers.py SOURCES), which produces Table 1.
SOURCES = {
    ("Rn", "p1", "leg"): ("E009/scores_tracka_p1", "bl_resnet18_e0_tp1", "E023/results_tracka_p1_restored_indep.csv"),
    ("Rn", "p2", "leg"): ("E009/scores_tracka_p2", "bl_resnet18_e0_tp2", "E023/results_tracka_p2_restored_indep.csv"),
    ("Bs", "p1", "leg"): ("E025/scores_tracka_p1_screened", "bl_bcesupcon_tp1", "E025/results_tracka_p1_restored_indep_E025detectors.csv"),
    ("Bs", "p2", "leg"): ("E025/scores_tracka_p2_screened", "bl_bcesupcon_tp2", "E025/results_tracka_p2_restored_indep_E025detectors.csv"),
    ("Dt", "p1", "leg"): ("E026/scores_tracka_p1_screened", "bl_d3_84_tp1", "E026/results_tracka_p1_restored_indep_E026detectors.csv"),
    ("Dt", "p2", "leg"): ("E027/scores_legacy_tracka_p2", "e027_d3_84_tp2_minmax", "E027/results_legacy_tracka_p2.csv"),
    ("Np", "p1", "leg"): ("E003/scores_tracka_p1", "bl_npr_tp1", "E003/results_tracka_p1_restored_indep_E003detectors.csv"),
    ("Np", "p2", "leg"): ("E027/scores_legacy_tracka_p2", "e027_npr_tp2_minmax", "E027/results_legacy_tracka_p2.csv"),
    ("Rn", "p1", "fix"): ("E027/scores_fixed_tracka_p1", "e027_resnet18_e0_tp1_div255", "E027/results_fixed_tracka_p1.csv"),
    ("Rn", "p2", "fix"): ("E027/scores_fixed_tracka_p2", "e027_resnet18_e0_tp2_div255", "E027/results_fixed_tracka_p2.csv"),
    ("Bs", "p1", "fix"): ("E027/scores_fixed_tracka_p1", "e027_bcesupcon_tp1_div255", "E027/results_fixed_tracka_p1.csv"),
    ("Bs", "p2", "fix"): ("E027/scores_fixed_tracka_p2", "e027_bcesupcon_tp2_div255", "E027/results_fixed_tracka_p2.csv"),
    ("Dt", "p1", "fix"): ("E027/scores_fixed_tracka_p1", "e027_d3_84_tp1_div255", "E027/results_fixed_tracka_p1.csv"),
    ("Dt", "p2", "fix"): ("E027/scores_fixed_tracka_p2", "e027_d3_84_tp2_div255", "E027/results_fixed_tracka_p2.csv"),
    ("Np", "p1", "fix"): ("E027/scores_fixed_tracka_p1", "e027_npr_tp1_div255", "E027/results_fixed_tracka_p1.csv"),
    ("Np", "p2", "fix"): ("E027/scores_fixed_tracka_p2", "e027_npr_tp2_div255", "E027/results_fixed_tracka_p2.csv"),
}
MANIFEST = {p: ROOT / "data" / "track_a" / p / "manifest.csv" for p in PROTOS}


# ------------------------------------------------------------------------------ primitives
def thr_at_1pct(real_scores: np.ndarray) -> float:
    """The frozen E023 rule (= diar.metrics.threshold_at_real_fpr): quantile 0.99, method 'higher'."""
    return float(np.quantile(real_scores, 0.99, method="higher"))


def weighted_thr_at_1pct(sorted_scores: np.ndarray, W: np.ndarray) -> np.ndarray:
    """thr_at_1pct of each row's multiset: sorted_scores (n,) ascending, W (r, n) multiplicities.
    np.quantile(x, 0.99, 'higher') is the element at 0-based rank ceil(0.99 (N - 1))."""
    cw = np.cumsum(W, axis=1)
    n_tot = cw[:, -1]
    k = np.ceil(0.99 * (n_tot - 1)).astype(np.int64)
    j = (cw < (k + 1)[:, None]).sum(axis=1)            # first position whose cumulative count > k
    return sorted_scores[j]


def weighted_auc(s_real: np.ndarray, s_syn: np.ndarray, Wr: np.ndarray, Ws: np.ndarray) -> np.ndarray:
    """P(syn > real) + 0.5 P(tie) under multiplicity weights, one value per row of Wr (r, n_real)
    and Ws (r, n_syn). With unit weights it is the average-rank Mann-Whitney AUC of E023."""
    order = np.argsort(s_real, kind="mergesort")
    sr = s_real[order]
    CR = np.zeros((Wr.shape[0], len(sr) + 1))
    np.cumsum(Wr[:, order], axis=1, out=CR[:, 1:])
    lo = np.searchsorted(sr, s_syn, side="left")
    hi = np.searchsorted(sr, s_syn, side="right")
    below = CR[:, lo]
    num = (Ws * (below + 0.5 * (CR[:, hi] - below))).sum(axis=1)
    return num / (Ws.sum(axis=1) * CR[:, -1])


def row_counts(rng: np.random.Generator, n_rows: int, n: int) -> np.ndarray:
    """n draws with replacement from n items, per row -> multiplicities (n_rows, n), float64."""
    d = rng.integers(0, n, size=(n_rows, n))
    out = np.zeros((n_rows, n))
    np.add.at(out, (np.repeat(np.arange(n_rows), n), d.ravel()), 1.0)
    return out


def pctl(v: np.ndarray, q) -> tuple[float, float]:
    return float(np.quantile(v, q[0])), float(np.quantile(v, q[1]))


def pboot(v: np.ndarray) -> float:
    """Descriptive two-sided bootstrap proportion; not a test statistic in the decision rule."""
    return float(min(1.0, 2 * min((v >= 0).mean(), (v <= 0).mean())))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


# ------------------------------------------------------------------------------ data
class Proto:
    """Every dump of one protocol, aligned on one test and one validation index."""

    def __init__(self, p: str, need_test: bool = True):
        self.p = p
        self.test, self.val, self.files = {}, {}, []
        meta = None
        for (det, proto, arm), (root, rn, _) in SOURCES.items():
            if proto != p:
                continue
            index = {}
            for f in sorted((ART / root).glob("*.npz")):
                with np.load(f, allow_pickle=False) as z:
                    if str(z["run_name"]) == rn:
                        index[(int(z["seed"]), str(z["condition"]))] = f
            for s in SEEDS:
                for cond in (CONDS if need_test else (ANCHOR,)):
                    f = index[(s, cond)]
                    self.files.append(f)
                    with np.load(f, allow_pickle=False) as z:
                        if meta is None:
                            meta = {k: z[k] for k in ("test_y", "test_source", "test_identity",
                                                      "test_patch_id", "val_y", "val_patch_id")}
                        else:
                            for k in ("test_patch_id", "test_y", "val_patch_id", "val_y"):
                                assert (z[k] == meta[k]).all(), f"{k} differs in {f}"
                        if need_test:
                            self.test[(det, arm, s, cond)] = z["test_score"].astype(np.float64)
                        if cond == ANCHOR:
                            self.val[(det, arm, s)] = z["val_score"].astype(np.float64)
        self.meta = meta
        y, src = meta["test_y"], meta["test_source"]
        m = (y == 0) | np.isin(src, sorted(UNSEEN[p]))                 # the E023 'unseen' group
        self.R = np.flatnonzero(m & (y == 0))
        self.S = np.flatnonzero(m & (y == 1))
        self.syn_gen = src[self.S]
        self.real_ident = meta["test_identity"][self.R]
        self.real_pid = meta["test_patch_id"][self.R]
        self.syn_pid = meta["test_patch_id"][self.S]
        self.vR = np.flatnonzero(meta["val_y"] == 0)
        self.val_pid = meta["val_patch_id"][self.vR]


class Units:
    """Resampling units shared across protocols: real test fingers, synthetic test images by
    generator, validation fingers. A unit present in both protocols gets one draw per replicate."""

    def __init__(self, P: dict[str, Proto]):
        self.fingers = sorted(set().union(*[set(P[p].real_ident.tolist()) for p in PROTOS]))
        fpos = {f: i for i, f in enumerate(self.fingers)}
        self.real_col = {p: np.array([fpos[f] for f in P[p].real_ident]) for p in PROTOS}
        gens = sorted(set().union(*UNSEEN.values()))
        self.gen_pids = {}
        for g in gens:
            sets = [set(P[p].syn_pid[P[p].syn_gen == g].tolist()) for p in PROTOS if g in UNSEEN[p]]
            assert all(s == sets[0] for s in sets), f"generator {g}: test images differ between protocols"
            self.gen_pids[g] = sorted(sets[0])
        pos = {g: {pid: i for i, pid in enumerate(v)} for g, v in self.gen_pids.items()}
        self.syn_map = {}   # p -> [(generator, positions in the protocol's S order, unit columns)]
        for p in PROTOS:
            gen = np.array([str(g) for g in P[p].syn_gen])
            self.syn_map[p] = []
            for g in sorted(set(gen.tolist())):
                idx = np.flatnonzero(gen == g)
                col = np.array([pos[g][str(pid)] for pid in P[p].syn_pid[idx]])
                self.syn_map[p].append((g, idx, col))
        # validation fingers, from the Track A manifests (the dumps carry no validation identity)
        ident = {}
        for p in PROTOS:
            with open(MANIFEST[p]) as fh:
                for r in csv.DictReader(fh):
                    if r["split"] == "val":
                        ident[r["patch_id"]] = r["identity"]
        self.val_ident = {p: np.array([ident[str(x)] for x in P[p].val_pid]) for p in PROTOS}
        self.vfingers = sorted(set().union(*[set(v.tolist()) for v in self.val_ident.values()]))
        vpos = {f: i for i, f in enumerate(self.vfingers)}
        self.val_col = {p: np.array([vpos[f] for f in self.val_ident[p]]) for p in PROTOS}

    def draw(self, rng: np.random.Generator, n_rows: int) -> dict:
        """One block of test-cluster replicates: finger counts and per-generator image counts."""
        d = {"real": row_counts(rng, n_rows, len(self.fingers))}
        for g in sorted(self.gen_pids):
            d[g] = row_counts(rng, n_rows, len(self.gen_pids[g]))
        return d

    def weights(self, d: dict, p: str) -> tuple[np.ndarray, np.ndarray]:
        Wr = d["real"][:, self.real_col[p]]
        Ws = np.empty((Wr.shape[0], sum(len(idx) for _, idx, _ in self.syn_map[p])))
        for g, idx, col in self.syn_map[p]:
            Ws[:, idx] = d[g][:, col]
        return Wr, Ws


# ------------------------------------------------------------------------------ analysis
def full_sample(P: dict[str, Proto]) -> dict:
    """Unweighted estimates per (det, p, arm, seed): tau, and dT, dB, dAUC per operator."""
    out = {}
    for p in PROTOS:
        Pp = P[p]
        ones_r = np.ones((1, len(Pp.R))); ones_s = np.ones((1, len(Pp.S)))
        for det in DETS:
            for arm in ARMS:
                for s in SEEDS:
                    tau = thr_at_1pct(Pp.val[(det, arm, s)][Pp.vR])
                    r = {"tau": tau}
                    base = {}
                    for cond in CONDS:
                        sc = Pp.test[(det, arm, s, cond)]
                        f = sc > tau
                        base[cond] = (f[Pp.S].mean(), f[Pp.R].mean(),
                                      float(weighted_auc(sc[Pp.R], sc[Pp.S], ones_r, ones_s)[0]))
                    for op, cond in OPS.items():
                        r[op] = {k: base[cond][i] - base[ANCHOR][i] for i, k in enumerate(("dT", "dB", "dAUC"))}
                    out[(det, p, arm, s)] = r
    return out


def gate0(fs: dict) -> dict:
    fails, maxdev = [], {"d_tdr_locked": 0.0, "d_rejection": 0.0, "d_auc": 0.0, "thr_locked": 0.0}
    n = 0
    for (det, p, arm), (_, rn, csvp) in SOURCES.items():
        with open(ART / csvp) as fh:
            rows = [r for r in csv.DictReader(fh) if r["run_name"] == rn and r["group"] == "unseen"]
        pub = {(int(r["seed"]), r["condition"]): r for r in rows}
        for s in SEEDS:
            got = fs[(det, p, arm, s)]
            for op, cond in OPS.items():
                r = pub[(s, cond)]
                for col, mine, tol in (("d_tdr_locked", got[op]["dT"], TOL_D),
                                       ("d_rejection", got[op]["dB"], TOL_D),
                                       ("d_auc", got[op]["dAUC"], TOL_D),
                                       ("thr_locked", got["tau"], TOL_THR)):
                    dev = abs(mine - float(r[col]))
                    maxdev[col] = max(maxdev[col], dev)
                    n += 1
                    if dev > tol:
                        fails.append(dict(det=det, p=p, arm=arm, seed=s, op=op, col=col,
                                          mine=mine, published=float(r[col])))
    return {"checked": n, "failures": fails, "max_abs_deviation": maxdev, "pass": not fails}


def bootstrap(P: dict[str, Proto], U: Units, log) -> dict:
    """Replicate arrays per (det, p, op): DDT, DDB (B_PRIMARY); per-seed DDT, DDB (B_SEED x 10);
    DDT/DDB with validation-resampled thresholds (B_VAL); DDAUC (B_AUC)."""
    ss = np.random.SeedSequence(SEED)
    rng_seed, rng_clu, rng_val = (np.random.default_rng(c) for c in ss.spawn(3))
    cells = [(det, p, op) for det in DETS for p in PROTOS for op in OPS]
    R = {c: {"DDT": [], "DDB": [], "seedDDT": [], "seedDDB": [], "valDDT": [], "valDDB": [], "DDAUC": []}
         for c in cells}
    # full-sample thresholds and flags, arranged as columns
    keys = {p: [(det, arm, s, cond) for det in DETS for arm in ARMS for s in SEEDS for cond in CONDS]
            for p in PROTOS}
    kpos = {p: {k: i for i, k in enumerate(keys[p])} for p in PROTOS}
    Fs, Fr = {}, {}
    for p in PROTOS:
        Pp = P[p]
        tau = {(det, arm, s): thr_at_1pct(Pp.val[(det, arm, s)][Pp.vR]) for det in DETS for arm in ARMS for s in SEEDS}
        F = np.stack([Pp.test[k] > tau[k[:3]] for k in keys[p]], axis=1).astype(np.float64)
        Fs[p], Fr[p] = F[Pp.S], F[Pp.R]
    done = 0
    while done < B_PRIMARY:
        n = min(CHUNK, B_PRIMARY - done)
        t0 = time.time()
        M = {(det, p): row_counts(rng_seed, n, len(SEEDS)) for det in DETS for p in PROTOS}
        D = U.draw(rng_clu, n)
        n_val = max(0, min(n, B_VAL - done))
        n_auc = max(0, min(n, B_AUC - done))
        n_seed = max(0, min(n, B_SEED - done))
        V = row_counts(rng_val, n_val, len(U.vfingers)) if n_val else None
        for p in PROTOS:
            Pp = P[p]
            Wr, Ws = U.weights(D, p)
            T = (Ws @ Fs[p]) / Ws.sum(axis=1, keepdims=True)
            Bm = (Wr @ Fr[p]) / Wr.sum(axis=1, keepdims=True)
            for det in DETS:
                Mdp = M[(det, p)]
                for op, cond in OPS.items():
                    dTs = {arm: np.stack([T[:, kpos[p][(det, arm, s, cond)]] - T[:, kpos[p][(det, arm, s, ANCHOR)]]
                                          for s in SEEDS], axis=1) for arm in ARMS}
                    dBs = {arm: np.stack([Bm[:, kpos[p][(det, arm, s, cond)]] - Bm[:, kpos[p][(det, arm, s, ANCHOR)]]
                                          for s in SEEDS], axis=1) for arm in ARMS}
                    c = R[(det, p, op)]
                    c["DDT"].append(((Mdp * dTs["fix"]).sum(1) - (Mdp * dTs["leg"]).sum(1)) / len(SEEDS))
                    c["DDB"].append(((Mdp * dBs["fix"]).sum(1) - (Mdp * dBs["leg"]).sum(1)) / len(SEEDS))
                    if n_seed:
                        c["seedDDT"].append((dTs["fix"] - dTs["leg"])[:n_seed])
                        c["seedDDB"].append((dBs["fix"] - dBs["leg"])[:n_seed])
            if n_val:   # validation fingers resampled too: tau_lock re-estimated per replicate
                Wv = V[:, U.val_col[p]]
                Wr_v, Ws_v = Wr[:n_val], Ws[:n_val]
                for det in DETS:
                    Mdp = M[(det, p)][:n_val]
                    dT_v = {op: {arm: np.zeros((n_val, len(SEEDS))) for arm in ARMS} for op in OPS}
                    dB_v = {op: {arm: np.zeros((n_val, len(SEEDS))) for arm in ARMS} for op in OPS}
                    for arm in ARMS:
                        for s in SEEDS:
                            vs = Pp.val[(det, arm, s)][Pp.vR]
                            order = np.argsort(vs, kind="mergesort")
                            tau_b = weighted_thr_at_1pct(vs[order], Wv[:, order])
                            rates = {}
                            for cond in CONDS:
                                sc = Pp.test[(det, arm, s, cond)]
                                fs_ = (sc[Pp.S][None, :] > tau_b[:, None])
                                fr_ = (sc[Pp.R][None, :] > tau_b[:, None])
                                rates[cond] = ((Ws_v * fs_).sum(1) / Ws_v.sum(1), (Wr_v * fr_).sum(1) / Wr_v.sum(1))
                            for op, cond in OPS.items():
                                dT_v[op][arm][:, s] = rates[cond][0] - rates[ANCHOR][0]
                                dB_v[op][arm][:, s] = rates[cond][1] - rates[ANCHOR][1]
                    for op in OPS:
                        c = R[(det, p, op)]
                        c["valDDT"].append(((Mdp * dT_v[op]["fix"]).sum(1) - (Mdp * dT_v[op]["leg"]).sum(1)) / len(SEEDS))
                        c["valDDB"].append(((Mdp * dB_v[op]["fix"]).sum(1) - (Mdp * dB_v[op]["leg"]).sum(1)) / len(SEEDS))
            if n_auc:
                Wr_a, Ws_a = Wr[:n_auc], Ws[:n_auc]
                for det in DETS:
                    Mdp = M[(det, p)][:n_auc]
                    auc = {}
                    for arm in ARMS:
                        for s in SEEDS:
                            for cond in CONDS:
                                sc = Pp.test[(det, arm, s, cond)]
                                auc[(arm, s, cond)] = weighted_auc(sc[Pp.R], sc[Pp.S], Wr_a, Ws_a)
                    for op, cond in OPS.items():
                        d = {arm: np.stack([auc[(arm, s, cond)] - auc[(arm, s, ANCHOR)] for s in SEEDS], axis=1)
                             for arm in ARMS}
                        R[(det, p, op)]["DDAUC"].append(((Mdp * d["fix"]).sum(1) - (Mdp * d["leg"]).sum(1)) / len(SEEDS))
        done += n
        log(f"  replicates {done:>6}/{B_PRIMARY}  (block {time.time() - t0:.1f}s)")
    return {c: {k: (np.concatenate(v, axis=0) if v else None) for k, v in R[c].items()} for c in cells}


def summarise(fs: dict, boot: dict) -> tuple[list[dict], dict]:
    rows = []
    for det in DETS:
        for p in PROTOS:
            for op in OPS:
                b = boot[(det, p, op)]
                mean = lambda arm, q: st.fmean(fs[(det, p, arm, s)][op][q] for s in SEEDS)
                dT = {arm: mean(arm, "dT") for arm in ARMS}
                dB = {arm: mean(arm, "dB") for arm in ARMS}
                dA = {arm: mean(arm, "dAUC") for arm in ARMS}
                r = dict(detector=DET_NAME[det], det=det, protocol=p.upper(), operator=op,
                         role="headline" if det in HEADLINE else "supplemental",
                         dT_leg=dT["leg"], dT_fix=dT["fix"], DDT=dT["fix"] - dT["leg"],
                         dB_leg=dB["leg"], dB_fix=dB["fix"], DDB=dB["fix"] - dB["leg"],
                         dAUC_leg=dA["leg"], dAUC_fix=dA["fix"], DDAUC=dA["fix"] - dA["leg"])
                for q in ("DDT", "DDB"):
                    r[f"{q}_lo"], r[f"{q}_hi"] = pctl(b[q], Q)
                    r[f"{q}_lo_bonf"], r[f"{q}_hi_bonf"] = pctl(b[q], Q_BONF)
                    r[f"{q}_pboot"] = pboot(b[q])
                    r[f"{q}_supports"] = r[f"{q}_hi"] < 0          # interval entirely below zero
                    r[f"{q}_opposite"] = r[f"{q}_lo"] > 0
                    r[f"{q}_supports_bonf"] = r[f"{q}_hi_bonf"] < 0
                    r[f"{q}_val_lo"], r[f"{q}_val_hi"] = pctl(b[f"val{q}"], Q)
                    seedrep = b[f"seed{q}"]
                    hi = np.quantile(seedrep, Q[1], axis=0)
                    lo = np.quantile(seedrep, Q[0], axis=0)
                    r[f"{q}_seeds_below0"] = int((hi < 0).sum())
                    r[f"{q}_seeds_above0"] = int((lo > 0).sum())
                r["DDAUC_lo"], r["DDAUC_hi"] = pctl(b["DDAUC"], Q)
                ret = abs(dT["fix"]) / abs(dT["leg"]) if dT["leg"] else float("nan")
                r.update(retention=ret, flip=bool(dT["leg"] * dT["fix"] < 0),
                         cost_only=bool(dB["fix"] > 0 and dT["fix"] <= 0.01),
                         ret60_sign=bool(dT["leg"] * dT["fix"] >= 0 and ret >= 0.60),
                         n_boot=B_PRIMARY)
                rows.append(r)

    def block(sel):
        cells = [(r["det"], r["protocol"].lower(), r["operator"]) for r in sel]
        out = {"n_cells": len(sel)}
        for q in ("DDT", "DDB"):
            med_b = np.median(np.stack([boot[c][q] for c in cells], axis=1), axis=1)
            out[q] = dict(n_supports=sum(r[f"{q}_supports"] for r in sel),
                          n_opposite=sum(r[f"{q}_opposite"] for r in sel),
                          n_supports_bonf=sum(r[f"{q}_supports_bonf"] for r in sel),
                          n_val_supports=sum(r[f"{q}_val_hi"] < 0 for r in sel),
                          n_cells_all_seeds_below0=sum(r[f"{q}_seeds_below0"] == len(SEEDS) for r in sel),
                          n_point_negative=sum(r[q] < 0 for r in sel),
                          median_point=float(np.median([r[q] for r in sel])),
                          median_ci=pctl(med_b, Q), p_median_below0=float((med_b < 0).mean()))
        out["DDAUC"] = dict(n_point_positive=sum(r["DDAUC"] > 0 for r in sel),
                            n_ci_above0=sum(r["DDAUC_lo"] > 0 for r in sel),
                            median_point=float(np.median([r["DDAUC"] for r in sel])))
        bil = [r["retention"] for r in sel if r["operator"] == "bil"]
        tv = [r["retention"] for r in sel if r["operator"] == "tv"]
        kept = [r["retention"] for r in sel if not r["flip"]]
        out["descriptive"] = dict(retention_median_bil=float(np.median(bil)), retention_median_tv=float(np.median(tv)),
                                  retention_range_sign_kept=[min(kept), max(kept)] if kept else None,
                                  flips=sum(r["flip"] for r in sel), cost_only=sum(r["cost_only"] for r in sel),
                                  ret60_sign=sum(r["ret60_sign"] for r in sel))
        return out

    head = [r for r in rows if r["role"] == "headline"]
    summary = {"headline": block(head),
               "headline_excluding_D3": block([r for r in head if r["det"] != "Dt"]),
               "D3_only": block([r for r in head if r["det"] == "Dt"]),
               "NPR_supplemental": block([r for r in rows if r["det"] == "Np"])}
    return rows, summary


def decide(summary: dict) -> dict:
    nT = summary["headline"]["DDT"]["n_supports"]
    nB = summary["headline"]["DDB"]["n_supports"]
    if nT >= 9 and nB >= 9:
        outcome = "I-A"
    elif nT <= 4:
        outcome = "I-C"
    else:
        outcome = "I-B"
    return {"outcome": outcome, "n_DDT_supports": nT, "n_DDB_supports": nB, "n_cells": N_HEADLINE_CELLS,
            "rule": "support = primary 95% interval entirely below zero; I-A: nDDT>=9 and nDDB>=9; "
                    "I-C: nDDT<=4; I-B: otherwise (preregistered, E036)"}


# ------------------------------------------------------------------------------ self-test
def selftest() -> int:
    from scipy.stats import rankdata
    rng = np.random.default_rng(0)
    ok = True
    for _ in range(50):                                   # weighted AUC == average-rank AUC
        a = np.round(rng.normal(size=60), 1); b = np.round(rng.normal(0.5, 1, size=40), 1)
        y = np.r_[np.zeros(60), np.ones(40)]; s = np.r_[a, b]
        rk = rankdata(s, method="average")
        ref = (rk[y == 1].sum() - 40 * 41 / 2) / (40 * 60)
        got = weighted_auc(a, b, np.ones((1, 60)), np.ones((1, 40)))[0]
        ok &= abs(ref - got) < 1e-12
        w_r = rng.integers(0, 4, size=60).astype(float); w_s = rng.integers(0, 4, size=40).astype(float)
        a2, b2 = np.repeat(a, w_r.astype(int)), np.repeat(b, w_s.astype(int))
        y2 = np.r_[np.zeros(len(a2)), np.ones(len(b2))]; rk2 = rankdata(np.r_[a2, b2], method="average")
        ref2 = (rk2[y2 == 1].sum() - len(b2) * (len(b2) + 1) / 2) / (len(a2) * len(b2))
        ok &= abs(ref2 - weighted_auc(a, b, w_r[None], w_s[None])[0]) < 1e-12
    print(f"  weighted AUC matches the average-rank AUC (unit and integer weights): {ok}")
    ok_q = True
    for _ in range(200):                                  # weighted 'higher' quantile == np.quantile
        v = np.sort(np.round(rng.normal(size=rng.integers(5, 300)), 2))
        w = rng.integers(0, 4, size=len(v)).astype(float); w[rng.integers(0, len(v))] += 1
        ref = thr_at_1pct(np.repeat(v, w.astype(int)))
        ok_q &= ref == weighted_thr_at_1pct(v, w[None])[0]
    print(f"  weighted thr_at_1pct matches np.quantile on the expanded sample: {ok_q}")
    ok_c = True                                           # row_counts: multinomial counts, right total
    C = row_counts(rng, 500, 37)
    ok_c &= bool((C.sum(1) == 37).all()) and abs(C.mean() - 1.0) < 0.02
    print(f"  row_counts gives n draws per row with mean multiplicity 1: {ok_c}")
    # the paired two-factor interval on synthetic two-arm data: coverage under a zero interaction,
    # and detection under a true one (same seeds x test-images layout as the real analysis)
    def sim(shift_fix: float) -> tuple[float, float]:
        nf, per, ns = 40, 5, 150
        seed_eff = {arm: rng.normal(0, 0.02, 10) for arm in ARMS}
        fing = np.repeat(np.arange(nf), per); fe = rng.normal(0, 0.8, nf)[fing]
        base_r = rng.normal(0, 1, nf * per) + fe
        base_s = rng.normal(1.0, 1, ns)
        dT = {}
        for arm in ARMS:
            vals = []
            for s in range(10):
                shift = (0.3 if arm == "leg" else shift_fix) + seed_eff[arm][s]
                tau = np.quantile(base_r + rng.normal(0, 0.1, base_r.size), 0.95)
                a = (base_s + rng.normal(0, 0.1, ns) > tau).astype(float)
                r_ = (base_s + shift + rng.normal(0, 0.1, ns) > tau).astype(float)
                vals.append((a, r_))
            dT[arm] = vals
        B = 400
        M = row_counts(rng, B, 10); Ws = row_counts(rng, B, ns)
        est = {arm: np.stack([(Ws @ (dT[arm][s][1] - dT[arm][s][0])) / ns for s in range(10)], 1) for arm in ARMS}
        rep = ((M * est["fix"]).sum(1) - (M * est["leg"]).sum(1)) / 10
        return tuple(np.quantile(rep, [0.025, 0.975]))
    n_sim = 200
    cover = sum(lo <= 0 <= hi for lo, hi in (sim(0.3) for _ in range(n_sim))) / n_sim
    detect = sum(hi < 0 for lo, hi in (sim(0.1) for _ in range(n_sim))) / n_sim
    ok_cov, ok_pow = cover >= 0.90, detect >= 0.80
    print(f"  null coverage of the paired two-factor 95% interval (synthetic): {cover:.3f} (>= 0.90): {ok_cov}")
    print(f"  detection of a true interaction, interval below zero (synthetic): {detect:.3f} (>= 0.80): {ok_pow}")
    allok = bool(ok and ok_q and ok_c and ok_cov and ok_pow)
    print(f"selftest: {'OK' if allok else 'FAILED'}")
    return 0 if allok else 1


def check_thresholds() -> int:
    """Validation pool only: tau_lock recomputed from the anchor dumps vs the published thr_locked."""
    worst, n, bad = 0.0, 0, []
    for p in PROTOS:
        Pp = Proto(p, need_test=False)
        for (det, proto, arm), (_, rn, csvp) in SOURCES.items():
            if proto != p:
                continue
            with open(ART / csvp) as fh:
                rows = [r for r in csv.DictReader(fh) if r["run_name"] == rn and r["group"] == "unseen"
                        and r["condition"] == ANCHOR]
            pub = {int(r["seed"]): float(r["thr_locked"]) for r in rows}
            for s in SEEDS:
                dev = abs(thr_at_1pct(Pp.val[(det, arm, s)][Pp.vR]) - pub[s])
                worst, n = max(worst, dev), n + 1
                if dev > TOL_THR:
                    bad.append((det, p, arm, s, dev))
    print(f"thresholds: {n} checked, max |dev| {worst:.2e}, failures {len(bad)}")
    return 0 if not bad else 1


# ------------------------------------------------------------------------------ main
def git_state() -> dict:
    def run(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"sha": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "script_dirty": bool(run("status", "--porcelain", "--", str(Path(__file__).relative_to(ROOT))))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--check-thresholds", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.check_thresholds:
        return check_thresholds()

    OUT.mkdir(parents=True, exist_ok=True); LOGDIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    logf = open(LOGDIR / f"run_{stamp}.log", "w")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    t_start = time.time()
    gs = git_state()
    prov = {"experiment": "E036", "started": time.strftime("%Y-%m-%d %H:%M:%S"), "git": gs,
            "script_sha256": sha256(Path(__file__)), "python": sys.version.split()[0],
            "numpy": np.__version__, "host": platform.node(), "cpu_only": True,
            "B_PRIMARY": B_PRIMARY, "B_SEED": B_SEED, "B_VAL": B_VAL, "B_AUC": B_AUC, "SEED": SEED,
            "argv": sys.argv}
    log(f"E036 start  git {gs['sha'][:7]} ({gs['branch']}), script dirty: {gs['script_dirty']}")
    if gs["script_dirty"]:
        log("ABORT: the analysis script differs from the committed (frozen) version")
        return 3
    P = {p: Proto(p) for p in PROTOS}
    prov["inputs"] = {str(f.relative_to(ROOT)): sha256(f) for p in PROTOS for f in P[p].files}
    for p in PROTOS:
        log(f"  {p}: {len(P[p].files)} dumps, unseen group {len(P[p].R)} real / {len(P[p].S)} synthetic")
    U = Units(P)
    log(f"  units: {len(U.fingers)} real test fingers, generators "
        + ", ".join(f"{g} {len(v)}" for g, v in U.gen_pids.items()) + f", {len(U.vfingers)} validation fingers")

    fs = full_sample(P)
    g0 = gate0(fs)
    (OUT / "gate0.json").write_text(json.dumps(g0, indent=2))
    log(f"Gate 0: {g0['checked']} values checked, {len(g0['failures'])} failures, max |dev| {g0['max_abs_deviation']}")
    if not g0["pass"]:
        log("Gate 0 FAILED -- E036 stops here (preregistered); nothing further is computed")
        prov["finished"] = time.strftime("%Y-%m-%d %H:%M:%S"); prov["stopped_at"] = "gate0"
        (OUT / "provenance.json").write_text(json.dumps(prov, indent=2))
        return 2

    boot = bootstrap(P, U, log)
    rows, summary = summarise(fs, boot)
    outcome = decide(summary)
    outcome.update(git=gs["sha"], script_sha256=prov["script_sha256"])
    with open(OUT / "interaction_cells.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    (OUT / "interaction_summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "OUTCOME.json").write_text(json.dumps(outcome, indent=2))
    np.savez_compressed(OUT / "bootstrap_replicates.npz",
                        **{f"{d}_{p}_{op}_{k}": v for (d, p, op), b in boot.items() for k, v in b.items()
                           if v is not None})
    prov["finished"] = time.strftime("%Y-%m-%d %H:%M:%S"); prov["seconds"] = round(time.time() - t_start, 1)
    (OUT / "provenance.json").write_text(json.dumps(prov, indent=2))
    log(f"OUTCOME {outcome['outcome']}: DDT supports {outcome['n_DDT_supports']}/12, "
        f"DDB supports {outcome['n_DDB_supports']}/12  ({prov['seconds']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
