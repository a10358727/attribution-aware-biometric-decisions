#!/usr/bin/env python3
"""E071 -- matcher verification threshold: does E065's genuine-matching penalty survive a pairwise verification
operating point? (post hoc; protocol docs/experiments/E071_matcher_verification_threshold.md)

ArcFace-88 (E018, 3 seeds), CPU, deterministic. E065's frozen helpers are imported unchanged.

    python experiments/E071_matcher_verification/run_e071.py --check-auth
    python experiments/E071_matcher_verification/run_e071.py --validation   # L2: thresholds from validation pairs
    python experiments/E071_matcher_verification/run_e071.py --test         # L3: G-E, G-X, G-N, G-C, V1-V6

Vocabulary (binding): genuine verification acceptance and zero-effort impostor acceptance are measured. Pre-matcher
progression is E061's. Synthetic zero-effort acceptance (V6) is NOT attack success and is never reported as such.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(ROOT / "experiments" / "E065_matcher_cascade_genuine"))
from experiments.tcss_lib import core as C                      # noqa: E402
from experiments.tcss_lib import pool as POOL                   # noqa: E402
from experiments.tcss_lib.pool import SharedUnits, ones_draw    # noqa: E402
import run_e065 as E65                                          # noqa: E402  (frozen at 666be4d, unchanged)

EXP = "E071"
OUT = C.ART / EXP
PROTOCOL = ROOT / "docs" / "experiments" / "E071_matcher_verification_threshold.md"
FIT = C.ART / "E061" / "validation_fit.json"
FIT_SHA = "c3d01b7d84d8858bfd091e9bcdd8d8db4b24597fc4ccf8adeaf3554e9dae22e4"
E065_TAU = C.ART / "E065" / "validation_run_20261122_200457" / "tau_m.json"
E065_SUMMARY = C.ART / "E065" / "test_run_20261122_200825" / "summary.json"
DATA = ROOT / "data" / "track_a"
MATCHERS = [ROOT / "artifacts" / "runs" / "E018" / f"matcher_arcface88_tracka_s{s}" for s in (0, 1, 2)]
SOURCES = [HERE, ROOT / "experiments" / "E065_matcher_cascade_genuine" / "run_e065.py",
           ROOT / "experiments" / "tcss_lib" / "core.py", ROOT / "experiments" / "tcss_lib" / "pool.py",
           ROOT / "diar" / "models" / "matcher88.py", ROOT / "diar" / "matcher_train.py"]
FMRS = (1e-2, 1e-3)
PRIMARY_FMR = 1e-2
PROBE_CONDS = ("degraded", "restored_bilateral", "restored_tv")
OPCOND = {"bil": "restored_bilateral", "tv": "restored_tv"}
B, CHUNK, SEED = 10_000, 500, 710071
K = 3
Q = 0.9
NB = 2000                                   # EER histogram bins on [-1, 1], width 0.001
DETS = C.HEADLINE
EXPECTED_N = {"captures": 1621, "credentials": 277}
GX_TOL, GC_TOL = 1e-9, 1e-12


# ------------------------------------------------------------------------------ pure scoring functions (G-S)
def pair_masks(probe_ids, gal_ids, probe_pid, gal_pid):
    """Genuine: same finger, different source image. Impostor: different finger."""
    same = probe_ids[:, None] == gal_ids[None, :]
    return same & (probe_pid[:, None] != gal_pid[None, :]), ~same


def tau_from_pairs(S, imp_mask, fmr):
    return float(np.quantile(S[imp_mask], 1.0 - fmr, method="higher"))


def per_probe_rates(S, gen_mask, imp_mask, tau):
    """m(i): fraction of probe i's genuine pairs above tau (NaN if none); f(i): fraction of impostor pairs above tau."""
    acc = S > tau
    ng = gen_mask.sum(1); ni = imp_mask.sum(1)
    m = np.where(ng > 0, (acc & gen_mask).sum(1) / np.maximum(ng, 1), np.nan)
    f = np.where(ni > 0, (acc & imp_mask).sum(1) / np.maximum(ni, 1), np.nan)
    return m, f


def finger_max(S, gal_ids, fingers, exclude=None):
    """(n_probe, n_gallery) -> (n_probe, n_finger) max cosine within each finger's templates; `exclude` is a boolean
    (n_probe, n_gallery) mask of templates to ignore (the probe's own source image). Fingers with no usable template
    get -inf."""
    S = np.where(exclude, -np.inf, S) if exclude is not None else S
    out = np.full((S.shape[0], len(fingers)), -np.inf)
    for k, fg in enumerate(fingers):
        cols = np.flatnonzero(gal_ids == fg)
        out[:, k] = S[:, cols].max(1)
    return out


def mt_rates(FM, probe_fidx, tau):
    """Multi-template sensitivity: genuine = own finger's max (NaN if -inf), impostor = every other finger's max."""
    n = FM.shape[0]; own = FM[np.arange(n), probe_fidx]
    m = np.where(np.isfinite(own), (own > tau).astype(float), np.nan)
    other = np.ones_like(FM, bool); other[np.arange(n), probe_fidx] = False
    f = ((FM > tau) & other).sum(1) / other.sum(1)
    return m, f


def mt_tau(FM, probe_fidx, fmr):
    other = np.ones_like(FM, bool); other[np.arange(FM.shape[0]), probe_fidx] = False
    return float(np.quantile(FM[other & np.isfinite(FM)], 1.0 - fmr, method="higher"))


def row_hist(S, mask):
    """Per-row normalised histogram of the masked scores (rows with no entry are all-zero), float32 (n, NB)."""
    b = np.clip(((S + 1.0) / 2.0 * NB).astype(np.int64), 0, NB - 1)
    n = S.shape[0]
    H = np.zeros(n * NB, np.float64)
    rows = np.repeat(np.arange(n), S.shape[1]).reshape(S.shape)
    np.add.at(H, (rows * NB + b)[mask], 1.0)
    H = H.reshape(n, NB); s = H.sum(1, keepdims=True)
    return (H / np.where(s > 0, s, 1.0)).astype(np.float32)


def eer_from_hist(Hg, Hi):
    """Hg, Hi (R, NB) genuine / impostor mass per bin -> EER per row. Threshold between bins k-1 and k:
    FNR_k = genuine mass below k, FPR_k = impostor mass at or above k (accept iff score bin >= k)."""
    Hg = Hg / Hg.sum(1, keepdims=True); Hi = Hi / Hi.sum(1, keepdims=True)
    z = np.zeros((Hg.shape[0], 1))
    fnr = np.concatenate([z, np.cumsum(Hg, 1)], 1)
    fpr = 1.0 - np.concatenate([z, np.cumsum(Hi, 1)], 1)
    k = np.argmin(np.abs(fnr - fpr), 1); r = np.arange(len(k))
    return (fnr[r, k] + fpr[r, k]) / 2.0


def synthetic_accept(S_syn, gal_ids, fingers, tau):
    """Zero-effort random-claim acceptance per synthetic image: mean over fingers (uniform claim) of the fraction of
    the finger's templates above tau (one template drawn uniformly)."""
    A = (S_syn > tau).astype(np.float64)
    out = np.zeros(S_syn.shape[0])
    for fg in fingers:
        cols = np.flatnonzero(gal_ids == fg)
        out += A[:, cols].mean(1)
    return out / len(fingers)


def cascade_fail(rej, m):
    """Per-attempt legitimate failure: detector REJECT, or (not REJECT and matcher rejects). m = acceptance prob."""
    return rej + (1.0 - rej) * (1.0 - m)


def fixed_template_tensors(units, keep, ip, accm, seeds):
    """Padded per-credential structures for the fixed-enrolment session (protocol V5).
    Pidx (F, nmax): pool positions of each credential's eligible captures (-1 pad); T[(ms, cond)] (F, nmax, nmax):
    acceptance of probe a (condition cond) against the clean template of capture b; pmask: valid pairs a != b."""
    cred = units.cred_of_capture; F = units.F
    members = [np.flatnonzero((cred == f) & (keep > 0)) for f in range(F)]
    nmax = max(1, max(len(m) for m in members))
    Pidx = np.full((F, nmax), -1)
    for f, m in enumerate(members):
        Pidx[f, :len(m)] = m
    valid = Pidx >= 0
    rows = np.asarray(ip)[np.clip(Pidx, 0, None)]
    pmask = valid[:, :, None] & valid[:, None, :] & ~np.eye(nmax, dtype=bool)[None]
    T = {k: v[rows[:, :, None], rows[:, None, :]].astype(np.float64) for k, v in accm.items()}
    return T, pmask, Pidx, valid


def fixed_template_session(rej, restored, T_rest, T_deg, pmask, Pidx, valid, K=K):
    """Per credential: the claimant is enrolled with ONE clean template b (uniform over the credential's captures);
    each attempt draws a probe a != b with replacement (M0). r_f(b) = mean_a fail(a, b); session denial
    L_K = mean_b r_f(b)^K, per-attempt L_1 = mean_b r_f(b). Credentials without a pair get 0 (masked by weight)."""
    R = np.where(valid, np.asarray(rej, float)[np.clip(Pidx, 0, None)], 0.0)
    Rs = np.where(valid, np.asarray(restored, bool)[np.clip(Pidx, 0, None)], False)
    A = np.where(Rs[:, :, None], T_rest, T_deg)
    fail = R[:, :, None] + (1.0 - R[:, :, None]) * (1.0 - A)
    den = pmask.sum(1)                                            # (F, nmax): probes per template
    r = np.where(den > 0, (fail * pmask).sum(1) / np.maximum(den, 1), 0.0)
    tm = den > 0; nt = np.maximum(tm.sum(1), 1)
    return (r * tm).sum(1) / nt, ((r ** K) * tm).sum(1) / nt


# ------------------------------------------------------------------------------ authorisation
def _check(fname, scope_prefix, extra=()):
    f = OUT / fname
    if not f.exists():
        raise C.TestAccessDenied(f"{EXP}: no {C.rel(f)}")
    a = json.loads(f.read_text())
    miss = [k for k in ("authorised_by", "date", "scope", "freeze_commit", "protocol_sha256", "script_sha256")
            + tuple(extra) if not a.get(k)]
    if miss:
        raise C.TestAccessDenied(f"{EXP}: {fname} lacks {miss}")
    if not str(a["scope"]).startswith(scope_prefix):
        raise C.TestAccessDenied(f"{EXP}: {fname} scope must start with {scope_prefix!r}")
    if a["protocol_sha256"] != C.sha256(PROTOCOL):
        raise C.TestAccessDenied(f"{EXP}: protocol differs from the authorised version")
    for s in SOURCES:
        if a["script_sha256"].get(C.rel(s)) != C.sha256(s):
            raise C.TestAccessDenied(f"{EXP}: {C.rel(s)} differs from the authorised version")
    return a


def check_validation_auth():
    return _check("VALIDATION_AUTHORISATION.json", "e071_validation")


def check_test_auth():
    a = _check("TEST_READ_AUTHORISATION.json", "e071_test", ("e059_post_test_reads_decision", "tau_v_sha256"))
    if C.sha256(FIT) != FIT_SHA:
        raise C.TestAccessDenied("validation_fit.json is not the frozen E061 F2 file")
    tau = latest_tau()
    if tau is None or C.sha256(tau) != a["tau_v_sha256"]:
        raise C.TestAccessDenied("tau_v.json differs from the authorised validation output")
    return a


def latest_tau():
    runs = sorted(OUT.glob("validation_run_*/tau_v.json"))
    return runs[-1] if runs else None


# ------------------------------------------------------------------------------ image helpers
def manifest(p, split, label):
    return [r for r in csv.DictReader(open(DATA / p / "manifest.csv")) if r["split"] == split and r["label"] == label]


def load_imgs(p, rows, cond, split, ilog):
    assert all(r["split"] == split for r in rows)
    imgs = [cv2.imread(str(DATA / p / cond / r["source"] / f"{r['patch_id']}.png"), 0) for r in rows]
    assert all(g is not None and g.shape == (88, 88) for g in imgs)
    ilog.add(f"{split}|{p}", cond, len(rows))
    return np.stack(imgs).astype(np.float32) / 255.0                      # matcher_train convention


@torch.no_grad()
def embed(model, p, rows, cond, split, ilog, bs=512):
    x = load_imgs(p, rows, cond, split, ilog)
    return np.concatenate([model(torch.from_numpy(x[i:i + bs]).unsqueeze(1)).float().numpy()
                           for i in range(0, len(x), bs)])


def seed_of(mp):
    return int(json.loads((mp / "meta.json").read_text())["seed"])


def launch_state():
    """Git, dirty diff, protocol and source hashes captured at LAUNCH (E067 audit D7: end-of-run capture misrecorded)."""
    run = lambda *a: subprocess.run(a, cwd=ROOT, capture_output=True, text=True).stdout
    return {"git": run("git", "rev-parse", "HEAD").strip(),
            "dirty_tracked": run("git", "status", "--porcelain", "--untracked-files=no").splitlines(),
            "protocol_sha256": C.sha256(PROTOCOL), "sources": {C.rel(s): C.sha256(s) for s in SOURCES},
            "launch_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "command": " ".join(sys.argv)}


def write_prov(outdir, stage, launch):
    (outdir / "provenance.json").write_text(json.dumps({
        "experiment": EXP, "stage": stage, "launch": launch, "device": "cpu",
        "torch": torch.__version__, "numpy": np.__version__, "python": sys.version, "threads": torch.get_num_threads(),
        "matchers": {mp.name: C.sha256(mp / "best.pt") for mp in MATCHERS}, "B": B, "seed": SEED,
        "end_time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=1))


# ------------------------------------------------------------------------------ preflight (metadata only)
def preflight(alog=None) -> dict:
    """G-PRE: manifests, synthetic keys, image-file existence and pool composition. Reads no pixel and no score:
    only manifest CSVs, file existence, and the test METADATA members of one dump per protocol."""
    alog = alog or C.AccessLog(EXP)
    out = {}
    tr = manifest("p1", "test", "0"); tp = [r["patch_id"] for r in tr]
    out["real_pid_unique"] = len(set(tp)) == len(tp)
    out["p2_reals_equal_p1"] = sorted(tp) == sorted(r["patch_id"] for r in manifest("p2", "test", "0"))
    out["real_files_exist"] = all((DATA / "p1" / c / r["source"] / f"{r['patch_id']}.png").exists()
                                  for r in tr for c in ("clean",) + PROBE_CONDS)
    for p in C.PROTOS:
        srows = [r for r in manifest(p, "test", "1") if r["source"] in C.UNSEEN[p]]
        keys = [(r["source"], r["patch_id"]) for r in srows]
        out[f"{p}|syn_key_unique"] = len(set(keys)) == len(keys)
        out[f"{p}|syn_files_exist"] = all((DATA / p / c / g / f"{pid}.png").exists() for g, pid in keys for c in PROBE_CONDS)
        z = C.read_members(C.dump_path("Rn", p, "leg", 0, C.ANCHOR), C.TEST_META_KEYS, alog)
        ty = z["test_y"]; src = z["test_source"].astype(str); pz = z["test_patch_id"].astype(str)
        out[f"{p}|pool_reals_equal_manifest"] = bool((ty == 0).sum() == len(tp) and set(pz[ty == 0]) == set(tp))
        S = (ty == 1) & np.isin(src, C.UNSEEN[p])
        out[f"{p}|pool_syn_equal_manifest"] = bool(S.sum() == len(keys) and set(zip(src[S], pz[S])) == set(keys))
        out[f"{p}|n_syn"] = int(S.sum())
    out["pass"] = all(v for k, v in out.items() if not k.endswith("n_syn"))
    return out


# ------------------------------------------------------------------------------ validation stage
def validation_stage():
    launch = launch_state()
    auth = check_validation_auth()
    outdir = OUT / f"validation_run_{time.strftime('%Y%m%d_%H%M%S')}"; outdir.mkdir(parents=True, exist_ok=False)
    ilog = E65.ImgLog(); rows = manifest("p1", "val", "0")
    ids = np.array([r["identity"] for r in rows]); pid = np.array([r["patch_id"] for r in rows])
    fingers = np.unique(ids); fidx = np.searchsorted(fingers, ids)
    gen, imp = pair_masks(ids, ids, pid, pid)
    excl = pid[:, None] == pid[None, :]
    taus, out, gd = {}, [], {}
    tau65 = json.loads(E065_TAU.read_text())
    for mp in MATCHERS:
        ms = seed_of(mp); model = E65.load_model(mp)
        ec = embed(model, "p1", rows, "clean", "val", ilog)
        for cond in PROBE_CONDS:
            e = embed(model, "p1", rows, cond, "val", ilog)
            S = e @ ec.T
            if cond == "degraded":                       # G-D: E065 tau_m recomputed from validation, exact
                imp65 = E65.impostor_max(e, ec, ids, ids)
                for fmr in FMRS:
                    t65 = float(np.quantile(imp65, 1.0 - fmr, method="higher"))
                    gd[f"s{ms}|{fmr}"] = {"recomputed": t65, "e065": tau65[f"s{ms}|{fmr}"], "equal": t65 == tau65[f"s{ms}|{fmr}"]}
            for fmr in FMRS:
                t = tau_from_pairs(S, imp, fmr)
                name = "anchor" if cond == "degraded" else f"matched|{cond}"
                taus[f"s{ms}|{name}|{fmr}"] = t
                m, f = per_probe_rates(S, gen, imp, t)
                out.append(dict(matcher_seed=ms, threshold=name, fmr=fmr, tau=t, n_imp_pairs=int(imp.sum()),
                                achieved_val_fmr_pairs=float((S[imp] > t).mean()),
                                val_fnmr_capture=float(np.nanmean(1 - m)), val_fmr_capture=float(np.nanmean(f))))
            if cond == "degraded":
                FM = finger_max(S, ids, fingers, exclude=excl)
                t = mt_tau(FM, fidx, PRIMARY_FMR)
                taus[f"s{ms}|mt|{PRIMARY_FMR}"] = t
                m, f = mt_rates(FM, fidx, t)
                out.append(dict(matcher_seed=ms, threshold="mt", fmr=PRIMARY_FMR, tau=t, n_imp_pairs=int(FM.shape[0] * (FM.shape[1] - 1)),
                                achieved_val_fmr_pairs=float(np.mean(f)), val_fnmr_capture=float(np.nanmean(1 - m)),
                                val_fmr_capture=float(np.mean(f))))
        print(f"  matcher s{ms}: anchor tau = {taus[f's{ms}|anchor|0.01']:.4f} (1e-2), "
              f"{taus[f's{ms}|anchor|0.001']:.4f} (1e-3)", flush=True)
    gates = {"G_D": {"pass": all(v["equal"] for v in gd.values()), "values": gd}}
    (outdir / "gates.json").write_text(json.dumps(gates, indent=1))
    if not gates["G_D"]["pass"]:
        (outdir / "access_log.json").write_text(json.dumps({"images_read": ilog.n, "test_scores_read": 0}, indent=1))
        write_prov(outdir, "validation(stopped)", launch)
        print("G-D FAILED: embeddings do not reproduce E065 bit-for-bit; no tau_v.json written"); return 2
    (outdir / "tau_v.json").write_text(json.dumps(taus, indent=1))
    with open(outdir / "tau_v.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    assert not any(k.startswith("test") for k in ilog.n), ilog.n
    (outdir / "access_log.json").write_text(json.dumps({"images_read": ilog.n, "authorisation": auth,
                                                        "test_scores_read": 0}, indent=1))
    write_prov(outdir, "validation", launch)
    print(f"validation done -> {outdir}")
    return 0


# ------------------------------------------------------------------------------ test stage
def test_stage():
    launch = launch_state()
    auth = check_test_auth()
    outdir = OUT / f"test_run_{time.strftime('%Y%m%d_%H%M%S')}"; outdir.mkdir(parents=True, exist_ok=False)
    logf = open(outdir / "run.log", "w")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()

    taus = json.loads(latest_tau().read_text())
    tau65 = json.loads(E065_TAU.read_text()); s65 = json.loads(E065_SUMMARY.read_text())
    ilog = E65.ImgLog(); alog = C.AccessLog(EXP, authorisation=auth)
    gates = {"G_PRE": preflight(alog)}
    if not gates["G_PRE"]["pass"]:
        (outdir / "gates.json").write_text(json.dumps(gates, indent=1, default=bool))
        alog.dump(outdir / "access_log_dumps.json"); write_prov(outdir, "test(stopped)", launch)
        log("G-PRE FAILED (metadata): stopping before any test pixel or score is read"); return 2
    trows = manifest("p1", "test", "0")
    tpid = np.array([r["patch_id"] for r in trows]); tids = np.array([r["identity"] for r in trows])
    fingers = np.unique(tids); tfidx = np.searchsorted(fingers, tids)
    gen, imp = pair_masks(tids, tids, tpid, tpid)
    excl = tpid[:, None] == tpid[None, :]
    has_pair = gen.sum(1) > 0
    # ---------------------------------------------------------------- embeddings (reals), G-E
    from diar.matcher_train import verification
    emb, seeds = {}, []
    for mp in MATCHERS:
        ms = seed_of(mp); seeds.append(ms); model = E65.load_model(mp)
        emb[ms] = {c: embed(model, "p1", trows, c, "test", ilog) for c in ("clean",) + PROBE_CONDS}
        v = verification(emb[ms]["clean"], list(tids))
        ref = json.loads((mp / "test_metrics.json").read_text())
        gates[f"G_E|s{ms}"] = {"pass": abs(v["eer"] - ref["eer"]) <= 0.005 and v["n_gen"] == ref["n_gen"],
                               "eer": v["eer"], "e018_eer": ref["eer"], "n_gen": v["n_gen"], "e018_n_gen": ref["n_gen"]}
        log(f"embedded test reals s{ms}; G-E eer {v['eer']} vs {ref['eer']}")
    # ---------------------------------------------------------------- G-X: E065 FNMR from the same embeddings
    gx = {}
    for fmr in FMRS:
        for c, key in (("degraded", None), ("restored_bilateral", "bil"), ("restored_tv", "tv")):
            vals = []
            for ms in seeds:
                G = E65.genuine_max(emb[ms][c], emb[ms]["clean"], tpid, tpid, tids, tids)
                ok = ~np.isnan(G)
                vals.append(float(np.mean(~(G[ok] > tau65[f"s{ms}|{fmr}"]))))
            mine = float(np.mean(vals))
            ref = (s65[f"dFNMR|{fmr}|bil"]["fnmr_degraded"] if key is None else s65[f"dFNMR|{fmr}|{key}"]["fnmr_restored"])
            gx[f"{fmr}|{c}"] = {"recomputed": mine, "e065": ref, "abs_dev": abs(mine - ref)}
    gates["G_X"] = {"pass": all(v["abs_dev"] <= GX_TOL for v in gx.values()), "values": gx}
    gates["G_PRE"] = {k: v for k, v in gates["G_PRE"].items()}
    gates["G_N"] = {"pass": int(has_pair.sum()) == EXPECTED_N["captures"]
                    and len(set(tids[has_pair])) == EXPECTED_N["credentials"],
                    "captures": int(has_pair.sum()), "credentials": int(len(set(tids[has_pair])))}
    (outdir / "gates.json").write_text(json.dumps(gates, indent=1, default=bool))
    if not all(g["pass"] for g in gates.values()):
        (outdir / "access_log.json").write_text(json.dumps({"images_read": ilog.n}, indent=1))
        write_prov(outdir, "test(stopped)", launch)
        log("GATE FAILED (G-E / G-X / G-N): stopping before the detector dumps are read"); return 2
    # ---------------------------------------------------------------- per-probe rates (manifest order)
    rates, per_seed = {}, []      # (ms, tname, fmr, cond) -> (m, f)
    accm = {}                     # (ms, cond) -> bool (probe, template) acceptance at the primary anchor threshold
    hists = {}                    # (ms, cond) -> (Hg, Hi) on eligible probes
    for ms in seeds:
        for c in PROBE_CONDS:
            S = emb[ms][c] @ emb[ms]["clean"].T
            tset = [("anchor", fmr, taus[f"s{ms}|anchor|{fmr}"]) for fmr in FMRS]
            if c != "degraded":
                tset += [("matched", fmr, taus[f"s{ms}|matched|{c}|{fmr}"]) for fmr in FMRS]
            else:
                tset += [("matched", fmr, taus[f"s{ms}|anchor|{fmr}"]) for fmr in FMRS]
            for tname, fmr, t in tset:
                m, f = per_probe_rates(S, gen, imp, t)
                f = np.where(has_pair, f, np.nan)
                if tname == "anchor" and fmr == PRIMARY_FMR:
                    accm[(ms, c)] = S > t
                rates[(ms, tname, fmr, c)] = (m, f)
                per_seed.append(dict(matcher_seed=ms, threshold=tname, fmr=fmr, condition=c, tau=t,
                                     fnmr_capture=float(np.nanmean(1 - m)), fmr_capture=float(np.nanmean(f)),
                                     fnmr_pair=float(1 - (S[gen] > t).mean()),
                                     fmr_pair=float((S[imp & has_pair[:, None]] > t).mean()), eer=float("nan")))
            FM = finger_max(S, tids, fingers, exclude=excl)
            m, f = mt_rates(FM, tfidx, taus[f"s{ms}|mt|{PRIMARY_FMR}"])
            rates[(ms, "mt", PRIMARY_FMR, c)] = (m, np.where(has_pair, f, np.nan))
            per_seed.append(dict(matcher_seed=ms, threshold="mt", fmr=PRIMARY_FMR, condition=c,
                                 tau=taus[f"s{ms}|mt|{PRIMARY_FMR}"], fnmr_capture=float(np.nanmean(1 - m)),
                                 fmr_capture=float(np.mean(f[has_pair])), fnmr_pair=float("nan"), fmr_pair=float("nan"),
                                 eer=float("nan")))
            hists[(ms, c)] = (row_hist(S[has_pair], gen[has_pair]), row_hist(S[has_pair], imp[has_pair]))
            per_seed.append(dict(matcher_seed=ms, threshold="none(EER)", fmr=float("nan"), condition=c, tau=float("nan"),
                                 fnmr_capture=float("nan"), fmr_capture=float("nan"), fnmr_pair=float("nan"),
                                 fmr_pair=float("nan"),
                                 eer=float(eer_from_hist(hists[(ms, c)][0].sum(0, keepdims=True),
                                                         hists[(ms, c)][1].sum(0, keepdims=True))[0])))
    log("per-probe rates and histograms done")
    # ---------------------------------------------------------------- synthetic zero-effort acceptance (V6)
    pools = POOL.load_pool("test", DETS, alog, authorised=True)
    syn_a = {}                    # (p, fmr, cond) -> matcher-seed-mean acceptance per pool synthetic (pool order)
    for p, pp in pools.items():
        srows_all = {(r["source"], r["patch_id"]): r for r in manifest(p, "test", "1")}
        srows = [srows_all[(str(g), str(x))] for x, g in zip(pp.syn_pid, pp.syn_gen)]
        assert all(r["source"] == str(g) for r, g in zip(srows, pp.syn_gen))
        for mp in MATCHERS:
            ms = seed_of(mp); model = E65.load_model(mp)
            for c in PROBE_CONDS:
                Ssyn = embed(model, p, srows, c, "test", ilog) @ emb[ms]["clean"].T
                for fmr in FMRS:
                    a = synthetic_accept(Ssyn, tids, fingers, taus[f"s{ms}|anchor|{fmr}"])
                    syn_a.setdefault((p, fmr, c), []).append(a)
        log(f"synthetic embeddings {p} done ({len(srows)} images)")
    syn_a = {k: np.mean(v, axis=0) for k, v in syn_a.items()}
    # ---------------------------------------------------------------- detector side and cascade precompute
    fits = json.loads(FIT.read_text())["fits"]
    shared = SharedUnits(pools)
    pos = {p: {pid: i for i, pid in enumerate(tpid)} for p in pools}
    for p, pp in pools.items():
        if sorted(pos[p][str(x)] for x in pp.real_pid) != list(range(len(tpid))):
            gates["G_N"]["pass"] = False; gates["G_N"]["pool_differs_from_manifest"] = p
            (outdir / "gates.json").write_text(json.dumps(gates, indent=1, default=bool))
            alog.dump(outdir / "access_log_dumps.json"); write_prov(outdir, "test(stopped)", launch)
            log(f"G-N FAILED: pool {p} real captures differ from the manifest test reals"); return 2
    cells = [(d, p, op) for d in DETS for p in C.PROTOS for op in C.OPS]
    X, cmask, SYN = {}, {}, {}
    for (det, p, op) in cells:
        pp = pools[p]; uu = pp.units
        ip = np.array([pos[p][str(x)] for x in pp.real_pid])
        keep = has_pair[ip].astype(float)
        nk = C.per_cred_mean(keep, uu)
        cmask[p] = (nk > 0).astype(float)
        T, pmask, Pidx, valid = fixed_template_tensors(uu, keep, ip, accm, seeds)
        for arm in C.ARMS:
            for pol in C.POLICIES:
                for s in C.SEEDS:
                    prm = fits[f"{det}|{p}|{arm}|{s}|{op}"]
                    z, restored = C.decide(pol, prm, pp.real[(det, arm, s, C.ANCHOR)], pp.real[(det, arm, s, C.OPS[op])])
                    rej = (z == C.REJ).astype(float)
                    rf_d = np.where(nk > 0, C.per_cred_mean(rej * keep, uu) / np.maximum(nk, 1e-12), 0.0)
                    lc1, lc3 = [], []
                    for ms in seeds:
                        l1, l3 = fixed_template_session(rej, restored, T[(ms, OPCOND[op])], T[(ms, "degraded")],
                                                        pmask, Pidx, valid)
                        lc1.append(l1); lc3.append(l3)
                    X[(det, p, arm, op, pol, s)] = np.stack([rf_d ** K, rf_d, np.mean(lc3, 0), np.mean(lc1, 0)], 1)
                    zs, rs = C.decide(pol, prm, pp.syn[(det, arm, s, C.ANCHOR)], pp.syn[(det, arm, s, C.OPS[op])])
                    pas = (zs == C.ACC) + (zs == C.STEP) * (1.0 - Q)
                    a_fw = np.where(rs, syn_a[(p, PRIMARY_FMR, OPCOND[op])], syn_a[(p, PRIMARY_FMR, "degraded")])
                    SYN[(det, p, arm, op, pol, s)] = np.stack([pas * a_fw, pas], 1)
    QN = ("Ldet3", "Ldet1", "Lcasc3", "Lcasc1")
    QS = ("SCA1", "PMP1")

    def cascade_components(d, m_seed):
        out = {}
        for (det, p, op) in cells:
            Wc = shared.cred_weights(d, p, pools[p].units) * cmask[p][None, :]
            Ws = shared.syn_weights(d, p, len(pools[p].syn_pid))
            for arm in C.ARMS:
                for pol in C.POLICIES:
                    M = np.stack([C.wmean(Wc, X[(det, p, arm, op, pol, s)]) for s in C.SEEDS], 2)    # (R, 4, 10)
                    Ms = np.stack([(Ws @ SYN[(det, p, arm, op, pol, s)]) / Ws.sum(1, keepdims=True) for s in C.SEEDS], 2)
                    for qi, qn in enumerate(QN):
                        out[(det, p, arm, op, pol, qn)] = C.seed_average(M[:, qi, :], m_seed[(det, p)])
                    for qi, qn in enumerate(QS):
                        out[(det, p, arm, op, pol, qn)] = C.seed_average(Ms[:, qi, :], m_seed[(det, p)])
        return out

    # ---------------------------------------------------------------- capture-level quantities
    p0 = "p1"; u0 = pools[p0].units
    idx0 = np.array([pos[p0][str(x)] for x in pools[p0].real_pid])
    sel = has_pair[idx0].astype(float)
    elig_rows = np.flatnonzero(has_pair)                     # rows of the histograms (manifest order)
    hist_row_of_pool = np.searchsorted(elig_rows, idx0)      # valid only where sel == 1

    def cap_quantities(Wcap):
        """Wcap (R, n_pool_real) -> dict of seed-mean FNMR/FMR per (tname, fmr, cond) and EER per cond."""
        Wd = Wcap * sel[None, :]
        den = Wd.sum(1)
        out = {}
        for (ms, tname, fmr, c), (m, f) in rates.items():
            mm = np.nan_to_num(m[idx0], nan=0.0); ff = np.nan_to_num(f[idx0], nan=0.0)
            out.setdefault(("FNMR", tname, fmr, c), []).append((Wd @ (1.0 - mm)) / den)
            out.setdefault(("FMR", tname, fmr, c), []).append((Wd @ ff) / den)
        We = np.zeros((Wcap.shape[0], len(elig_rows)))
        np.add.at(We.T, hist_row_of_pool[sel > 0], Wcap[:, sel > 0].T)
        for (ms, c), (Hg, Hi) in hists.items():
            out.setdefault(("EER", "none", 0.0, c), []).append(eer_from_hist(We @ Hg, We @ Hi))
        return {k: np.mean(v, axis=0) for k, v in out.items()}

    ones_m = {(d, p): np.ones((1, 10)) for d in DETS for p in C.PROTOS}
    d1 = ones_draw(shared)
    fullc = {k: float(v[0]) for k, v in cascade_components(d1, ones_m).items()}
    W1 = shared.cred_weights(d1, p0, u0)[:, u0.cred_of_capture]
    fullq = {k: float(v[0]) for k, v in cap_quantities(W1).items()}
    # G-C: detector-only L_det(pi0) reproduces E065 (same population, same F2, same formula)
    gc = {arm: float(np.median([fullc[(d, p, arm, op, "pi0", "Ldet3")] for (d, p, op) in cells])) for arm in C.ARMS}
    gates["G_C"] = {"pass": all(abs(gc[a] - s65["level|Ldet|pi0"][a]) <= GC_TOL for a in C.ARMS),
                    "recomputed": gc, "e065": s65["level|Ldet|pi0"]}
    (outdir / "gates.json").write_text(json.dumps(gates, indent=1))
    if not gates["G_C"]["pass"]:
        alog.dump(outdir / "access_log_dumps.json")
        (outdir / "access_log.json").write_text(json.dumps({"images_read": ilog.n}, indent=1))
        write_prov(outdir, "test(stopped)", launch)
        log("G-C FAILED: stopping before the bootstrap"); return 2
    # ---------------------------------------------------------------- bootstrap
    ss = np.random.SeedSequence(SEED).spawn(3)
    rng_seed, rng_subj, rng_syn = (np.random.default_rng(x) for x in ss)
    rep_c, rep_q = {}, {}
    t0 = time.time()
    for b0 in range(0, B, CHUNK):
        n = min(CHUNK, B - b0)
        m_seed = {(d, p): C.row_counts(rng_seed, n, 10) for d in DETS for p in C.PROTOS}
        dr = shared.draw(rng_subj, rng_syn, n)
        for k, v in cascade_components(dr, m_seed).items():
            rep_c.setdefault(k, []).append(v)
        Wcap = shared.cred_weights(dr, p0, u0)[:, u0.cred_of_capture]
        for k, v in cap_quantities(Wcap).items():
            rep_q.setdefault(k, []).append(v)
        log(f"  bootstrap {b0 + n}/{B} ({time.time() - t0:.0f}s)")
    rep_c = {k: np.concatenate(v) for k, v in rep_c.items()}; rep_q = {k: np.concatenate(v) for k, v in rep_q.items()}
    # ---------------------------------------------------------------- summaries
    def ci(v):
        return C.pctl(v, 0.95)

    def resolved(lohi):
        return bool(lohi[0] > 0 or lohi[1] < 0)

    summ = {"levels": {}, "deltas": {}, "cascade": {}, "synthetic": {}}
    for (qty, tname, fmr, c), v in fullq.items():
        summ["levels"][f"{qty}|{tname}|{fmr}|{c}"] = {"estimate": v, "ci95": ci(rep_q[(qty, tname, fmr, c)])}
    for (qty, tname, fmr, c) in fullq:
        if c == "degraded":
            continue
        k0 = (qty, tname, fmr, "degraded"); k1 = (qty, tname, fmr, c)
        lohi = ci(rep_q[k1] - rep_q[k0])
        summ["deltas"][f"d{qty}|{tname}|{fmr}|{c}"] = {"estimate": fullq[k1] - fullq[k0], "ci95": lohi,
                                                         "resolved": resolved(lohi)}
    def reading(d, e065):
        lo, hi = d["ci95"]
        rel = ("interval excludes E065's estimate %+.3f" % e065) if (hi < e065 or lo > e065) else \
              ("interval includes E065's estimate %+.3f" % e065)
        if lo > 0:
            return f"persists (resolved > 0); {rel}"
        if hi < 0:
            return f"reversed (resolved < 0; contradicts E065); {rel}"
        return f"not resolved at this threshold; {rel}"
    V1, A1 = {}, {}
    for c, key in (("restored_bilateral", "bil"), ("restored_tv", "tv")):
        e65 = s65[f"dFNMR|{PRIMARY_FMR}|{key}"]["estimate"]
        V1[c] = reading(summ["deltas"][f"dFNMR|anchor|{PRIMARY_FMR}|{c}"], e65)
        A1[c] = reading(summ["deltas"][f"dFNMR|mt|{PRIMARY_FMR}|{c}"], e65)
    summ["V1_reading"] = V1
    summ["A1_reading"] = A1
    for pol in C.POLICIES:
        for arm in C.ARMS:
            for qn in QN + QS:
                fv = [fullc[(d, p, arm, op, pol, qn)] for (d, p, op) in cells]
                rv = np.median(np.stack([rep_c[(d, p, arm, op, pol, qn)] for (d, p, op) in cells], 1), 1)
                summ["cascade"][f"level|{qn}|{pol}|{arm}"] = {"median": float(np.median(fv)), "ci95": ci(rv)}
                if pol != "pi0":
                    fd = [fullc[(d, p, arm, op, pol, qn)] - fullc[(d, p, arm, op, "pi0", qn)] for (d, p, op) in cells]
                    rd = np.median(np.stack([rep_c[(d, p, arm, op, pol, qn)] - rep_c[(d, p, arm, op, "pi0", qn)]
                                             for (d, p, op) in cells], 1), 1)
                    summ["cascade"][f"delta|{qn}|{pol}|{arm}"] = {"median": float(np.median(fd)), "ci95": ci(rd),
                                                                    "resolved": resolved(ci(rd))}
            if pol != "pi0":
                dd = summ["cascade"][f"delta|Ldet3|{pol}|{arm}"]
                if dd["resolved"]:
                    summ["cascade"][f"A|{pol}|{arm}"] = {"defined": True, "ratio_of_medians":
                                                         summ["cascade"][f"delta|Lcasc3|{pol}|{arm}"]["median"] / dd["median"]}
                else:
                    summ["cascade"][f"A|{pol}|{arm}"] = {"defined": False, "reason": "dL_det 95% interval includes 0"}
    for (p, fmr, c), a in syn_a.items():
        summ["synthetic"][f"SZA|{p}|{fmr}|{c}"] = {"estimate": float(a.mean()), "n": int(len(a))}
    summ["n"] = gates["G_N"]; summ["tau_v"] = taus; summ["B"] = B; summ["seed"] = SEED
    rows = [dict(det=d, protocol=p, op=op, arm=arm, policy=pol, quantity=qn, estimate=fullc[(d, p, arm, op, pol, qn)],
                 lo95=ci(rep_c[(d, p, arm, op, pol, qn)])[0], hi95=ci(rep_c[(d, p, arm, op, pol, qn)])[1])
            for (d, p, op) in cells for arm in C.ARMS for pol in C.POLICIES for qn in QN + QS]
    with open(outdir / "cascade.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(outdir / "per_seed.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_seed[0])); w.writeheader(); w.writerows(per_seed)
    (outdir / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
    alog.dump(outdir / "access_log_dumps.json")
    (outdir / "access_log.json").write_text(json.dumps({"images_read": ilog.n}, indent=1))
    write_prov(outdir, "test", launch)
    log(f"test done -> {outdir}; V1: {V1}; A1: {A1}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check-auth", action="store_true"); g.add_argument("--validation", action="store_true")
    g.add_argument("--test", action="store_true")
    g.add_argument("--preflight", action="store_true", help="metadata-only checks; no pixel, no score")
    a = ap.parse_args()
    torch.set_num_threads(16)
    if a.preflight:
        alog = C.AccessLog(EXP); r = preflight(alog)
        print(json.dumps(r, indent=1, default=bool)); print("metadata members read:", len(alog.entries))
        return 0 if r["pass"] else 2
    try:
        if a.check_auth:
            for f in (check_validation_auth, check_test_auth):
                try:
                    f(); print(f"{f.__name__}: OK")
                except C.TestAccessDenied as e:
                    print(f"{f.__name__}: REFUSED -- {e}")
            return 0
        return validation_stage() if a.validation else test_stage()
    except C.TestAccessDenied as e:
        print(f"authorisation: REFUSED -- {e}"); return 3


if __name__ == "__main__":
    sys.exit(main())
