"""E062 v2 estimator: RA-class-stratified restoration-induced bona-fide rejection (draft, not frozen).

Specification: docs/experiments/E062_quality_conditioned_eval/E062_PRE_DATA_PROTOCOL_v2.md §6–§8 (estimands from v1 §5,
inference from v1 §7). Conventions follow the frozen E060 analysis (experiments/E060_credential_burden/run_e060.py):
validation-locked tau per (detector, protocol, arm, seed); rejection = score > tau; the twelve headline cells
{Rn, Bs, Dt} x {p1, p2} x {bil, tv}; E060 arm-cell eligibility; seed multiplicities drawn per (detector, protocol)
and shared across arms, operators and conditions; subject multiplicities shared across protocols; the median over
eligible cells recomputed in every replicate.

Estimand (capture-weighted primary): for capture i, d_i = seed-average of [r_i(restored) - r_i(anchor)];
dB_{a,c}(g) = mean of d_i over captures of RA class g; theta^g_{a,c} = dB_{a,c}(g) - dB_{a,c}(RA-middle) for
g in {RA-dry, RA-wet}; summary = median over eligible cells. Primary family: theta^dry and theta^wet in each arm,
two-sided 99.5 % percentile intervals (Amendment 1, D-26a: simulation-calibrated from the 98.75 % of protocol v2,
whose percentile intervals were liberal on the real design). Secondary (97.5 %): dtheta^g = theta^g_fix -
theta^g_leg; wet - dry. Sensitivity S-1...S-7 (v2 §8) and S-8w within-credential (Amendment 1, D-23).
Amendment 1 choices take effect only when the PI signs it and it is committed in the freeze commit.

This module never reads data by itself. `load_real` refuses to run without
artifacts/results/E062/TEST_READ_AUTHORISATION.json pinning this file (tcss_lib guard). In this round only
synthetic data were used.
"""
from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
from experiments.tcss_lib import core as C          # noqa: E402

EXP = "E062"
CLASSES = ("RA-dry", "RA-middle", "RA-wet")         # class index 0, 1, 2 (q3_v2_all codes)
REF = 1                                             # RA-middle
CONTRASTS = {"dry": 0, "wet": 2}
LEVEL_PRIMARY, LEVEL_SECONDARY, LEVEL_DISPLAY = 0.995, 0.975, 0.95   # primary: Amendment 1 D-26a (v2: 0.9875)
UNDEFINED_MAX = 0.05                                # v1 §7 rule 2
B_PRIMARY, CHUNK = 10_000, 1_000
SEED_DRAFT = 620_062                                # Amendment 1 D-24 (with B_PRIMARY); fixed in the freeze commit
OPS = tuple(C.OPS)                                  # ("bil", "tv")
HEADLINE = C.HEADLINE                               # ("Rn", "Bs", "Dt")


@dataclass
class Data:
    """Bona-fide test captures per protocol and their rejection indicators (0/1), seeds on axis 0."""
    subj: dict                       # p -> (n,) subject name per capture
    cred: dict                       # p -> (n,) credential name per capture
    cls: dict                        # p -> (n,) RA class index 0/1/2
    rej: dict                        # (det, p, arm, cond) -> (S, n) float, cond in C.CONDS
    eligible: dict                   # (det, p, arm) -> bool (E060 rule)
    attrs: dict = field(default_factory=dict)   # p -> {"vote": bool (n,), "support": int (n,), "severity": int (n,)}
    ids: dict = field(default_factory=dict)     # p -> (n,) capture identifiers (source filename); must be unique

    def validate(self):
        for p in self.subj:
            n = len(self.subj[p])
            if p in self.ids:
                if len(self.ids[p]) != n:
                    raise ValueError(f"{p}: identifier list does not match the captures")
                check_unique_ids(self.ids[p])
            if not (len(self.cred[p]) == len(self.cls[p]) == n):
                raise ValueError(f"{p}: capture arrays differ in length")
            if not set(np.unique(self.cls[p]).tolist()) <= {0, 1, 2}:
                raise ValueError(f"{p}: RA class outside 0/1/2")
        for k, r in self.rej.items():
            if r.shape[1] != len(self.subj[k[1]]):
                raise ValueError(f"{k}: rejection array does not match the capture list")
            if not np.isin(r, (0.0, 1.0)).all():
                raise ValueError(f"{k}: rejection indicators must be 0/1")


def check_unique_ids(ids) -> None:
    """Duplicate capture identifiers are a join failure (v2 §5), never silently averaged."""
    ids = np.asarray(ids)
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"{len(ids) - len(np.unique(ids))} duplicated capture identifiers")


# ------------------------------------------------------------------------------ draws
class Draws:
    """Bootstrap draws: subject multiplicities shared across protocols, seed multiplicities per (det, p)."""

    def __init__(self, data: Data, dets, B: int, seed: int):
        self.subjects = sorted(set().union(*[set(map(str, v)) for v in data.subj.values()]))
        pos = {s: i for i, s in enumerate(self.subjects)}
        self.col = {p: np.array([pos[str(s)] for s in v]) for p, v in data.subj.items()}
        self.dets, self.B, self.seed = tuple(dets), B, seed
        self.S = next(iter(data.rej.values())).shape[0]

    def chunks(self):
        ss = np.random.SeedSequence(self.seed).spawn(2)
        rng_seed, rng_subj = (np.random.default_rng(x) for x in ss)
        for b0 in range(0, self.B, CHUNK):
            n = min(CHUNK, self.B - b0)
            m = {(d, p): C.row_counts(rng_seed, n, self.S) for d in self.dets for p in C.PROTOS}
            yield m, C.row_counts(rng_subj, n, len(self.subjects))

    def full(self):
        return {(d, p): np.ones((1, self.S)) for d in self.dets for p in C.PROTOS}, np.ones((1, len(self.subjects)))


# ------------------------------------------------------------------------------ estimator core
def seed_avg(R: np.ndarray, m: np.ndarray) -> np.ndarray:
    """R (S, n) indicators, m (k, S) seed multiplicities -> (k, n) seed-averaged per capture."""
    return (m @ R) / m.sum(axis=1, keepdims=True)


def group_mean(A: np.ndarray, w: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """A (k, n), w (k, n) capture weights, mask (n,) -> (k,) weighted mean over masked captures; NaN if empty."""
    ww = w * mask[None, :]
    den = ww.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (ww * A).sum(axis=1) / den
    out[den == 0] = np.nan
    return out


def capture_weights(data: Data, p: str, subj_mult: np.ndarray, col: np.ndarray, weighting: str, keep: np.ndarray):
    """Per-capture weights for one protocol: subject multiplicity x (1 / captures of the credential within its
    RA class) for credential weighting (S-1), else 1."""
    w = subj_mult[:, col]
    if weighting == "credential":
        key = np.char.add(np.asarray(data.cred[p], dtype=str), np.char.add("|", data.cls[p].astype(str)))
        key = np.where(keep, key, "")
        _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        w = w / cnt[inv][None, :]
    elif weighting != "capture":
        raise ValueError(weighting)
    return w


def block(data: Data, dets, m_seed, subj_mult, col, mask, weighting):
    """Group levels for every (det, p, arm, cond, g) -> (k,) arrays."""
    out = {}
    for p in C.PROTOS:
        keep = mask[p]
        w = capture_weights(data, p, subj_mult, col[p], weighting, keep)
        gm = {g: keep & (data.cls[p] == g) for g in range(3)}
        for det in dets:
            m = m_seed[(det, p)]
            for arm in C.ARMS:
                for cond in C.CONDS:
                    A = seed_avg(data.rej[(det, p, arm, cond)], m)
                    for g in range(3):
                        out[(det, p, arm, cond, g)] = group_mean(A, w, gm[g])
    return out


def cell_list(dets, ops=OPS):
    return [(d, p, op) for d in dets for p in C.PROTOS for op in ops]


def derive(L, cells):
    """Per cell and arm: dB(g), theta^dry, theta^wet; per cell: dtheta, wet - dry."""
    out = {}
    for (d, p, op) in cells:
        cond = C.OPS[op]
        for arm in C.ARMS:
            for g in range(3):
                out[("anc", d, p, op, arm, g)] = L[(d, p, arm, C.ANCHOR, g)]
                out[("res", d, p, op, arm, g)] = L[(d, p, arm, cond, g)]
                out[("dB", d, p, op, arm, g)] = L[(d, p, arm, cond, g)] - L[(d, p, arm, C.ANCHOR, g)]
            for name, g in CONTRASTS.items():
                out[("theta", name, d, p, op, arm)] = out[("dB", d, p, op, arm, g)] - out[("dB", d, p, op, arm, REF)]
            out[("wet_minus_dry", d, p, op, arm)] = out[("dB", d, p, op, arm, 2)] - out[("dB", d, p, op, arm, 0)]
        for name in CONTRASTS:
            out[("dtheta", name, d, p, op)] = out[("theta", name, d, p, op, "fix")] - out[("theta", name, d, p, op, "leg")]
    return out


def strict_median(stack: np.ndarray) -> np.ndarray:
    """Median over cells (axis 1); undefined (NaN) in a replicate if any eligible cell is undefined."""
    med = np.median(stack, axis=1)
    med[np.isnan(stack).any(axis=1)] = np.nan
    return med


def interval(rv: np.ndarray, level: float):
    rv = np.asarray(rv, np.float64)
    und = float(np.isnan(rv).mean()) if len(rv) else 1.0
    if und > UNDEFINED_MAX:
        return {"lo": None, "hi": None, "undefined_fraction": und, "reported": False}
    lo, hi = C.pctl(rv, level)
    return {"lo": lo, "hi": hi, "undefined_fraction": und, "reported": True}


def reading(iv):
    if not iv["reported"]:
        return "not reported (undefined replicates > 5 %)"
    return "above 0" if iv["lo"] > 0 else ("below 0" if iv["hi"] < 0 else "includes 0")


def run(data: Data, dets=HEADLINE, B: int = B_PRIMARY, seed: int = SEED_DRAFT, mask=None, ops=OPS,
        weighting: str = "capture", label: str = "primary") -> dict:
    """One analysis (primary or a pre-specified sensitivity). Returns an aggregate dict (no identifiers)."""
    data.validate()
    mask = mask or {p: np.ones(len(data.subj[p]), bool) for p in data.subj}
    dr = Draws(data, dets, B, seed)
    cells = cell_list(dets, ops)
    m_full, s_full = dr.full()
    full = derive(block(data, dets, m_full, s_full, dr.col, mask, weighting), cells)
    full = {k: float(v[0]) for k, v in full.items()}
    reps = {}
    for m_seed, subj_mult in dr.chunks():
        r = derive(block(data, dets, m_seed, subj_mult, dr.col, mask, weighting), cells)
        for k, v in r.items():
            reps.setdefault(k, []).append(v)
    reps = {k: np.concatenate(v) for k, v in reps.items()}

    def elig(c, arm):
        d, p, _ = c
        return data.eligible[(d, p, "leg")] and data.eligible[(d, p, "fix")] if arm == "both" else data.eligible[(d, p, arm)]

    def joint(keyf, arm, level):
        cs = [c for c in cells if elig(c, arm)]
        if not cs:
            return {"n_cells": 0}
        fv = np.array([full[keyf(c)] for c in cs])
        rv = strict_median(np.stack([reps[keyf(c)] for c in cs], axis=1))
        med = float(np.median(fv)) if not np.isnan(fv).any() else None
        iv = interval(rv, level)
        return {"n_cells": len(cs), "median": med, "level": level, **iv, "reading": reading(iv)}

    # floor rule (v1 §7 rule 3), per arm and class, on the full sample
    floor = {}
    for arm in C.ARMS:
        cs = [c for c in cells if elig(c, arm)]
        for g in range(3):
            z = sum(1 for c in cs if full[("anc", *c, arm, g)] == 0 and full[("res", *c, arm, g)] == 0)
            floor[(arm, g)] = z > len(cs) / 2

    out = {"label": label, "weighting": weighting, "dets": list(dets), "ops": list(ops), "B": B, "seed": seed,
           "groups": {}, "primary": {}, "secondary": {}, "cells": []}
    for p in data.subj:
        for g in range(3):
            sel = mask[p] & (data.cls[p] == g)
            out["groups"][f"{p}|{CLASSES[g]}"] = {"captures": int(sel.sum()),
                                                  "subjects": int(len(np.unique(data.subj[p][sel]))),
                                                  "credentials": int(len(np.unique(data.cred[p][sel])))}
    for arm in C.ARMS:
        for g in range(3):
            for q in ("anc", "res", "dB"):
                out["secondary"][f"{q}|{arm}|{CLASSES[g]}"] = joint(lambda c, q=q, g=g, arm=arm: (q, *c, arm, g),
                                                                   arm, LEVEL_DISPLAY)
        for name, g in CONTRASTS.items():
            j = joint(lambda c, name=name, arm=arm: ("theta", name, *c, arm), arm, LEVEL_PRIMARY)
            j["floor_uninformative"] = bool(floor[(arm, g)] or floor[(arm, REF)])
            if j["floor_uninformative"]:
                j["reading"] = "uninformative (floor)"
            out["primary"][f"theta_{name}|{arm}"] = j
        out["secondary"][f"wet_minus_dry|{arm}"] = joint(lambda c, arm=arm: ("wet_minus_dry", *c, arm), arm,
                                                         LEVEL_SECONDARY)
    for name in CONTRASTS:
        out["secondary"][f"dtheta_{name}|fix-leg"] = joint(lambda c, name=name: ("dtheta", name, *c), "both",
                                                           LEVEL_SECONDARY)
    for c in cells:
        for arm in C.ARMS:
            row = {"det": c[0], "protocol": c[1], "op": c[2], "arm": arm, "eligible": elig(c, arm)}
            for g in range(3):
                for q in ("anc", "res", "dB"):
                    row[f"{q}_{CLASSES[g]}"] = full[(q, *c, arm, g)]
            for name in CONTRASTS:
                k = ("theta", name, *c, arm)
                lo, hi = C.pctl(reps[k], LEVEL_DISPLAY)
                row.update({f"theta_{name}": full[k], f"theta_{name}_lo95": lo, f"theta_{name}_hi95": hi})
            out["cells"].append(row)
    return out


# ------------------------------------------------------------------------------ pre-specified analyses
def masks(data: Data):
    """Capture masks of v2 §8: S-2 subjects with all three classes, S-3 severity strata, S-5 vote-labelled,
    S-7 KNN support >= 7/9. Computed from labels and metadata only."""
    out = {}
    for p in data.subj:
        subj = np.asarray(data.subj[p]).astype(str)
        three = {s for s in np.unique(subj) if len(np.unique(data.cls[p][subj == s])) == 3}
        a = data.attrs.get(p, {})
        out.setdefault("S-2", {})[p] = np.isin(subj, list(three))
        if "severity" in a:
            out.setdefault("S-3a_sev12", {})[p] = np.isin(a["severity"], (1, 2))
            out.setdefault("S-3b_sev34", {})[p] = np.isin(a["severity"], (3, 4))
        if "vote" in a:
            out.setdefault("S-5", {})[p] = np.asarray(a["vote"], bool)
        if "support" in a:
            out.setdefault("S-7", {})[p] = np.asarray(a["support"]) >= 7
        if "finger" in a:                 # options considered for D-23 and not chosen: not run by run_all
            out.setdefault("S-8a", {})[p] = np.asarray(a["finger"]) != "thumb"
            out.setdefault("S-8b", {})[p] = np.asarray(a["finger"]) == "non_thumb"
        # S-8w (Amendment 1, D-23): within-credential masks -- captures of credentials (subject x finger) that carry
        # both the contrast class and RA-middle; run with weighting="credential" and read only that contrast
        cred = np.asarray(data.cred[p]).astype(str)
        for name, g in CONTRASTS.items():
            both = {c for c in np.unique(cred) if {g, REF} <= set(data.cls[p][cred == c].tolist())}
            out.setdefault(f"S-8w_{name}", {})[p] = np.isin(cred, list(both)) & np.isin(data.cls[p], (g, REF))
    return out


def run_all(data: Data, B: int = B_PRIMARY, seed: int = SEED_DRAFT) -> dict:
    """Primary + S-1…S-7 + S-8w with the same seed (identical bootstrap draws across analyses)."""
    res = {"primary": run(data, B=B, seed=seed)}
    res["S-1"] = run(data, B=B, seed=seed, weighting="credential", label="S-1")
    mk = masks(data)
    for name in ("S-2", "S-3a_sev12", "S-3b_sev34", "S-5", "S-7"):
        if name in mk:
            res[name] = run(data, B=B, seed=seed, mask=mk[name], label=name)
    res["S-4"] = run(data, dets=("Rn", "Bs"), B=B, seed=seed, label="S-4")
    for op in OPS:
        res[f"S-6_{op}"] = run(data, B=B, seed=seed, ops=(op,), label=f"S-6_{op}")
    for name in CONTRASTS:               # S-8w: only theta_<name>|leg and theta_<name>|fix are estimands
        r = run(data, B=B, seed=seed, mask=mk[f"S-8w_{name}"], weighting="credential", label=f"S-8w_{name}")
        r["report_only"] = [f"theta_{name}|{arm}" for arm in C.ARMS]
        res[f"S-8w_{name}"] = r
    return res


# ------------------------------------------------------------------------------ real-data loader (guarded)
def sources() -> list:
    return [HERE, HERE.parent / "ra_audit.py", C.ROOT / "experiments/tcss_lib/core.py",
            C.ROOT / "experiments/tcss_lib/pool.py", C.ROOT / "experiments/E060_credential_burden/run_e060.py"]


def load_real(labels_csv: Path, log: C.AccessLog) -> Data:
    """Frozen E060 decision inputs joined to RA labels. Refuses without the E062 test-read authorisation.
    NOT executed before the freeze and the PI's written authorisation (v2 §10 G-F, G-A)."""
    auth = C.check_authorisation(EXP, sources())          # raises TestAccessDenied otherwise
    log.authorisation = auth
    from experiments.tcss_lib.pool import load_pool
    from experiments.E060_credential_burden import run_e060 as E060
    from experiments.E062_quality_conditioned_eval import ra_audit as RA
    if RA.sha256_file(labels_csv) != auth.get("labels_csv_sha256"):
        raise C.TestAccessDenied("label CSV differs from the authorised version")
    labels, dups, bad = RA.read_labels(labels_csv)
    if dups or bad:
        raise ValueError("label file has duplicates or invalid codes")
    pools = load_pool("test", HEADLINE, log, authorised=True)
    tau = E060.thresholds(pools, HEADLINE)
    el = E060.eligibility(pools, tau, HEADLINE)
    data = Data({}, {}, {}, {}, el, {}, {})
    for p, pp in pools.items():
        meta = {}
        with open(C.MANIFEST[p]) as fh:
            for r in csv.DictReader(fh):
                if r["split"] == "test" and r["label"] == "0":
                    meta[r["patch_id"]] = (Path(r["src_file"]).name, int(r["deg_severity"]))
        files = [meta[str(pid)][0] for pid in pp.real_pid]
        data.ids[p] = np.array(files)
        data.subj[p] = pp.units.subj_names[pp.units.subj_of_cred[pp.units.cred_of_capture]]
        data.cred[p] = pp.units.cred_names[pp.units.cred_of_capture]
        data.cls[p] = np.array([labels[f][0] for f in files])
        data.attrs[p] = {"vote": np.array([labels[f][1] for f in files]),
                         "support": np.array([labels[f][2] for f in files]),
                         "severity": np.array([meta[str(pid)][1] for pid in pp.real_pid])}
        for det in HEADLINE:
            for arm in C.ARMS:
                for cond in C.CONDS:
                    data.rej[(det, p, arm, cond)] = np.stack(
                        [(pp.real[(det, arm, s, cond)] > tau[(det, p, arm, s)]).astype(np.float64) for s in C.SEEDS])
    return data
