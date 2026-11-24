#!/usr/bin/env python3
"""E065 -- matcher-aware cascade: genuine matching under restoration (M1) and detector -> matcher burden (M2).

Protocol: docs/experiments/E065_matcher_cascade_genuine.md (v3). ArcFace-88 (E018, 3 seeds) only; JIPNet is not run
(licence-header conflict awaiting the PI, checklist A7). Device: CPU when no GPU is free (CLAUDE.local.md); recorded.

    python experiments/E065_matcher_cascade_genuine/run_e065.py --check-auth
    python experiments/E065_matcher_cascade_genuine/run_e065.py --validation   # L2: tau_m from validation only
    python experiments/E065_matcher_cascade_genuine/run_e065.py --test         # L3: G-E, M1, M2 (one test read)

Vocabulary (binding): genuine matcher acceptance is measured here. Pre-matcher progression (PMP) is not estimated;
impostor acceptance of synthetics is E024's (IAR = 0, floor); attack success is NOT estimable and never reported.
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
from experiments.tcss_lib import core as C                      # noqa: E402
from experiments.tcss_lib import pool as POOL                   # noqa: E402
from experiments.tcss_lib.pool import SharedUnits, ones_draw    # noqa: E402

EXP = "E065"
OUT = C.ART / EXP
PROTOCOL = ROOT / "docs" / "experiments" / "E065_matcher_cascade_genuine.md"
FIT = C.ART / "E061" / "validation_fit.json"
FIT_SHA = "c3d01b7d84d8858bfd091e9bcdd8d8db4b24597fc4ccf8adeaf3554e9dae22e4"
IMG = ROOT / "data" / "track_a" / "p1"        # test/val reals and their conditions are identical in p1 and p2 (verified)
MATCHERS = [ROOT / "artifacts" / "runs" / "E018" / f"matcher_arcface88_tracka_s{s}" for s in (0, 1, 2)]
SOURCES = [HERE, ROOT / "experiments" / "tcss_lib" / "core.py", ROOT / "experiments" / "tcss_lib" / "pool.py",
           ROOT / "diar" / "models" / "matcher88.py", ROOT / "diar" / "matcher_train.py"]
FMRS = (1e-2, 1e-3)
B, CHUNK, SEED = 10_000, 500, 650065
DETS = C.HEADLINE
K = 3


# ------------------------------------------------------------------------------ authorisation
def _check(fname, scope_prefix, extra=()):
    f = OUT / fname
    if not f.exists():
        raise C.TestAccessDenied(f"{EXP}: no {C.rel(f)}")
    a = json.loads(f.read_text())
    miss = [k for k in ("authorised_by", "date", "scope", "freeze_commit", "protocol_sha256", "script_sha256") + tuple(extra)
            if not a.get(k)]
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
    return _check("VALIDATION_AUTHORISATION.json", "e065_validation")


def check_test_auth():
    a = _check("TEST_READ_AUTHORISATION.json", "e065_test", ("e059_post_test_reads_decision", "tau_m_sha256"))
    if C.sha256(FIT) != FIT_SHA:
        raise C.TestAccessDenied("validation_fit.json is not the frozen E061 F2 file")
    tau = latest_tau()
    if tau is None or C.sha256(tau) != a["tau_m_sha256"]:
        raise C.TestAccessDenied("tau_m.json differs from the authorised validation output")
    return a


def latest_tau():
    runs = sorted(OUT.glob("validation_run_*/tau_m.json"))
    return runs[-1] if runs else None


# ------------------------------------------------------------------------------ matcher helpers
class ImgLog:
    def __init__(self):
        self.n = {}
    def add(self, split, cond, k):
        self.n[f"{split}|{cond}"] = self.n.get(f"{split}|{cond}", 0) + k


def load_imgs(rows, cond, split, ilog):
    assert all(r["split"] == split for r in rows)
    imgs = [cv2.imread(str(IMG / cond / r["source"] / f"{r['patch_id']}.png"), 0) for r in rows]
    assert all(g is not None and g.shape == (88, 88) for g in imgs)
    ilog.add(split, cond, len(rows))
    return np.stack(imgs).astype(np.float32) / 255.0                       # matcher_train convention


@torch.no_grad()
def embed(model, rows, cond, split, ilog, bs=512):
    x = load_imgs(rows, cond, split, ilog)
    return np.concatenate([model(torch.from_numpy(x[i:i + bs]).unsqueeze(1)).float().numpy() for i in range(0, len(x), bs)])


def load_model(mp):
    from diar.models.matcher88 import load_matcher88
    torch.manual_seed(0); torch.use_deterministic_algorithms(True)
    return load_matcher88(str(mp / "best.pt"), "cpu")


def manifest_rows(split):
    return [r for r in csv.DictReader(open(IMG / "manifest.csv")) if r["split"] == split and r["label"] == "0"]


def impostor_max(probe_emb, gal_emb, probe_ids, gal_ids):
    """Leave-own-finger-out maximum cosine of each probe over the gallery's other fingers (E024 rule)."""
    S = probe_emb @ gal_emb.T
    S[probe_ids[:, None] == gal_ids[None, :]] = -2.0
    return S.max(1)


def genuine_max(probe_emb, gal_emb, probe_pid, gal_pid, probe_ids, gal_ids):
    """Max cosine over the clean captures of the SAME finger excluding the probe's own source image; NaN if none."""
    S = probe_emb @ gal_emb.T
    ok = (probe_ids[:, None] == gal_ids[None, :]) & (probe_pid[:, None] != gal_pid[None, :])
    S = np.where(ok, S, -np.inf)
    m = S.max(1)
    return np.where(np.isfinite(m), m, np.nan)


# ------------------------------------------------------------------------------ validation stage
def validation_stage():
    auth = check_validation_auth()
    outdir = OUT / f"validation_run_{time.strftime('%Y%m%d_%H%M%S')}"; outdir.mkdir(parents=True, exist_ok=False)
    ilog = ImgLog(); rows = manifest_rows("val")
    ids = np.array([r["identity"] for r in rows])
    taus, out = {}, []
    for mp in MATCHERS:
        ms = int(json.loads((mp / "meta.json").read_text())["seed"])
        model = load_model(mp)
        ed, ec = embed(model, rows, "degraded", "val", ilog), embed(model, rows, "clean", "val", ilog)
        imp = impostor_max(ed, ec, ids, ids)
        for fmr in FMRS:
            t = float(np.quantile(imp, 1.0 - fmr, method="higher"))
            taus[f"s{ms}|{fmr}"] = t
            out.append(dict(matcher_seed=ms, fmr=fmr, tau_m=t, n_val_probe=len(rows), n_val_fingers=len(set(ids)),
                            achieved_val_fmr=float((imp > t).mean())))
    (outdir / "tau_m.json").write_text(json.dumps(taus, indent=1))
    with open(outdir / "tau_m.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    assert not any(k.startswith("test") for k in ilog.n), ilog.n
    (outdir / "access_log.json").write_text(json.dumps({"images_read": ilog.n, "authorisation": auth}, indent=1))
    write_prov(outdir, "validation")
    print(f"validation done -> {outdir}\n" + "\n".join(str(r) for r in out))
    return 0


def write_prov(outdir, stage):
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    (outdir / "provenance.json").write_text(json.dumps({
        "experiment": EXP, "stage": stage, "git": git, "device": "cpu", "torch": torch.__version__,
        "numpy": np.__version__, "python": sys.version, "threads": torch.get_num_threads(),
        "sources": {C.rel(s): C.sha256(s) for s in SOURCES}, "protocol_sha256": C.sha256(PROTOCOL),
        "matchers": {mp.name: C.sha256(mp / "best.pt") for mp in MATCHERS}, "command": " ".join(sys.argv)}, indent=1))


# ------------------------------------------------------------------------------ test stage
def test_stage():
    auth = check_test_auth()
    outdir = OUT / f"test_run_{time.strftime('%Y%m%d_%H%M%S')}"; outdir.mkdir(parents=True, exist_ok=False)
    taus = json.loads(latest_tau().read_text())
    ilog = ImgLog(); alog = C.AccessLog(EXP, authorisation=auth)
    gates = {}
    trows = manifest_rows("test")
    tpid = np.array([r["patch_id"] for r in trows]); tids = np.array([r["identity"] for r in trows])
    # ---- G-E: E018 test_cap60 clean all-pairs EER per seed
    from diar.matcher_train import verification
    emb = {}
    for mp in MATCHERS:
        ms = int(json.loads((mp / "meta.json").read_text())["seed"])
        model = load_model(mp)
        emb[ms] = {c: embed(model, trows, c, "test", ilog) for c in ("clean", "degraded", "restored_bilateral", "restored_tv")}
        v = verification(emb[ms]["clean"], list(tids))
        ref = json.loads((mp / "test_metrics.json").read_text())
        gates[f"G_E|s{ms}"] = {"pass": abs(v["eer"] - ref["eer"]) <= 0.005 and v["n_gen"] == ref["n_gen"],
                               "eer": v["eer"], "e018_eer": ref["eer"], "n_gen": v["n_gen"], "e018_n_gen": ref["n_gen"]}
    (outdir / "gates.json").write_text(json.dumps(gates, indent=1))
    if not all(g["pass"] for g in gates.values()):
        (outdir / "access_log.json").write_text(json.dumps({"images_read": ilog.n}, indent=1))
        print("G-E FAILED: matcher pipeline does not reproduce E018; stopping"); return 2
    # ---- genuine scores per seed and condition (gallery clean, own image excluded)
    G = {ms: {c: genuine_max(emb[ms][c], emb[ms]["clean"], tpid, tpid, tids, tids)
              for c in ("degraded", "restored_bilateral", "restored_tv")} for ms in emb}
    has_pair = ~np.isnan(G[0]["degraded"])
    # ---- detector side: test pools (authorised), frozen F2, E061 decide()
    pools = POOL.load_pool("test", DETS, alog, authorised=True)
    fits = json.loads(FIT.read_text())["fits"]
    shared = SharedUnits(pools)
    match = {}   # (fmr, ms, cond) -> bool per test patch id order tpid
    for fmr in FMRS:
        for ms in G:
            for c in G[ms]:
                match[(fmr, ms, c)] = np.nan_to_num(G[ms][c], nan=-9.0) > taus[f"s{ms}|{fmr}"]
    pos = {p: {pid: i for i, pid in enumerate(tpid)} for p in pools}
    # ---- M1: FNMR by condition, subject-cluster bootstrap (capture level), seed mean
    p0 = "p1"; u = pools[p0].units
    idx_pool = np.array([pos[p0][str(x)] for x in pools[p0].real_pid])            # pool row -> manifest row
    sel = has_pair[idx_pool].astype(float)
    ss = np.random.SeedSequence(SEED).spawn(3)
    rng_seed, rng_subj, rng_syn = (np.random.default_rng(x) for x in ss)
    res, reps = {}, {}
    def fnmr_vec(fmr, c, Wcap):
        vals = [((~match[(fmr, ms, c)][idx_pool]) * sel) for ms in sorted(G)]
        return np.mean([(Wcap @ v) / (Wcap @ sel) for v in vals], axis=0)
    Wfull = np.ones((1, len(idx_pool)))
    for fmr in FMRS:
        for c in ("degraded", "restored_bilateral", "restored_tv"):
            res[f"FNMR|{fmr}|{c}"] = float(fnmr_vec(fmr, c, Wfull)[0])
    # ---- M2: cascade L per (cell, arm, policy) on the genuine-pair subset
    cells = [(d, p, op) for d in DETS for p in C.PROTOS for op in C.OPS]
    def precompute(fmr):
        """Per (cell, arm, policy, seed): credential-level L_det and matcher-seed-mean L_casc vectors (F,), and the
        credential mask of the genuine-pair subset. Weighted means are linear, so the seed mean commutes with them."""
        X, mask = {}, {}
        for (det, p, op) in cells:
            pp = pools[p]; uu = pp.units
            ip = np.array([pos[p][str(x)] for x in pp.real_pid])
            keep = has_pair[ip].astype(float)
            nk = C.per_cred_mean(keep, uu)
            mask[p] = (nk > 0).astype(float)
            for arm in C.ARMS:
                for pol in C.POLICIES:
                    for s in C.SEEDS:
                        prm = fits[f"{det}|{p}|{arm}|{s}|{op}"]
                        z, restored = C.decide(pol, prm, pp.real[(det, arm, s, C.ANCHOR)], pp.real[(det, arm, s, C.OPS[op])])
                        rej = (z == C.REJ)
                        rf_d = np.where(nk > 0, C.per_cred_mean(rej * keep, uu) / np.maximum(nk, 1e-12), 0.0)
                        lc = []
                        for ms in sorted(G):
                            mt = np.where(restored, match[(fmr, ms, C.OPS[op])][ip], match[(fmr, ms, "degraded")][ip])
                            rf_c = np.where(nk > 0, C.per_cred_mean((rej | ~mt) * keep, uu) / np.maximum(nk, 1e-12), 0.0)
                            lc.append(rf_c ** K)
                        X[(det, p, arm, op, pol, s)] = (rf_d ** K, np.mean(lc, axis=0))
        return X, mask

    X, cmask = precompute(1e-2)

    def cascade_components(W_by_p, m_seed):
        out = {}
        for (det, p, op) in cells:
            Wc = W_by_p[p] * cmask[p][None, :]
            for arm in C.ARMS:
                for pol in C.POLICIES:
                    Ld = np.stack([C.wmean(Wc, X[(det, p, arm, op, pol, s)][0][:, None])[:, 0] for s in C.SEEDS], 1)
                    Lc = np.stack([C.wmean(Wc, X[(det, p, arm, op, pol, s)][1][:, None])[:, 0] for s in C.SEEDS], 1)
                    out[(det, p, arm, op, pol, "Ldet")] = C.seed_average(Ld, m_seed[(det, p)])
                    out[(det, p, arm, op, pol, "Lcasc")] = C.seed_average(Lc, m_seed[(det, p)])
        return out
    ones_m = {(d, p): np.ones((1, 10)) for d in DETS for p in C.PROTOS}
    d1 = ones_draw(shared, with_syn=False)
    full = cascade_components({p: shared.cred_weights(d1, p, pools[p].units) for p in pools}, ones_m)
    full = {k: float(v[0]) for k, v in full.items()}
    rep_c, rep_f = {}, {}
    t0 = time.time()
    for b0 in range(0, B, CHUNK):
        n = min(CHUNK, B - b0)
        m_seed = {(d, p): C.row_counts(rng_seed, n, 10) for d in DETS for p in C.PROTOS}
        dr = shared.draw(rng_subj, rng_syn, n)
        Wp = {p: shared.cred_weights(dr, p, pools[p].units) for p in pools}
        for k, v in cascade_components(Wp, m_seed).items():
            rep_c.setdefault(k, []).append(v)
        Wcap = Wp[p0][:, u.cred_of_capture]
        for fmr in FMRS:
            for c in ("degraded", "restored_bilateral", "restored_tv"):
                rep_f.setdefault(f"FNMR|{fmr}|{c}", []).append(fnmr_vec(fmr, c, Wcap))
        print(f"  bootstrap {b0 + n}/{B} ({time.time() - t0:.0f}s)", flush=True)
    rep_c = {k: np.concatenate(v) for k, v in rep_c.items()}; rep_f = {k: np.concatenate(v) for k, v in rep_f.items()}
    # ---- summaries
    summ = {}
    for fmr in FMRS:
        for op, c in (("bil", "restored_bilateral"), ("tv", "restored_tv")):
            k1, k0 = f"FNMR|{fmr}|{c}", f"FNMR|{fmr}|degraded"
            dv = rep_f[k1] - rep_f[k0]
            summ[f"dFNMR|{fmr}|{op}"] = {"estimate": res[k1] - res[k0], "ci95": C.pctl(dv, 0.95),
                                         "fnmr_degraded": res[k0], "fnmr_restored": res[k1],
                                         "resolved": bool(C.pctl(dv, 0.95)[0] > 0 or C.pctl(dv, 0.95)[1] < 0)}
    for pol in C.POLICIES[1:]:
        for arm in C.ARMS:
            for q in ("Ldet", "Lcasc"):
                fv = [full[(d, p, arm, op, pol, q)] - full[(d, p, arm, op, "pi0", q)] for (d, p, op) in cells]
                rv = np.stack([rep_c[(d, p, arm, op, pol, q)] - rep_c[(d, p, arm, op, "pi0", q)] for (d, p, op) in cells], 1)
                med = np.median(rv, 1)
                summ[f"d{q}|{pol}|{arm}"] = {"median": float(np.median(fv)), "ci95": C.pctl(med, 0.95)}
            dd = summ[f"dLdet|{pol}|{arm}"]
            if dd["ci95"][0] > 0 or dd["ci95"][1] < 0:
                fvA = np.median([(full[(d, p, arm, op, pol, "Lcasc")] - full[(d, p, arm, op, "pi0", "Lcasc")]) for (d, p, op) in cells]) / dd["median"]
                summ[f"A|{pol}|{arm}"] = {"ratio_of_medians": float(fvA), "defined": True}
            else:
                summ[f"A|{pol}|{arm}"] = {"defined": False, "reason": "dL_det 95% interval includes 0"}
        for q in ("Ldet", "Lcasc"):
            summ[f"level|{q}|pi0"] = {arm: float(np.median([full[(d, p, arm, op, "pi0", q)] for (d, p, op) in cells])) for arm in C.ARMS}
    summ["n"] = {"test_real_captures": int(len(tpid)), "with_genuine_pair": int(has_pair.sum()),
                 "credentials": int(len(set(tids))), "credentials_with_pair": int(len(set(tids[has_pair])))}
    summ["tau_m"] = taus
    (outdir / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
    alog.dump(outdir / "access_log_dumps.json")
    (outdir / "access_log.json").write_text(json.dumps({"images_read": ilog.n}, indent=1))
    write_prov(outdir, "test")
    print(f"test done -> {outdir}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check-auth", action="store_true"); g.add_argument("--validation", action="store_true")
    g.add_argument("--test", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(16)
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
