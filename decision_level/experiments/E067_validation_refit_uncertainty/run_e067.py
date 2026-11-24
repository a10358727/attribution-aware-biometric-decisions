#!/usr/bin/env python3
"""E067 Stage V driver: validation-refitting uncertainty (post hoc; protocol docs/experiments/E067_*.md, v3).

    python experiments/E067_validation_refit_uncertainty/run_e067.py --check-auth     # reads no data
    python experiments/E067_validation_refit_uncertainty/run_e067.py --stage-v        # needs L1 + L2 file

Stage V reads VALIDATION keys only (`tcss_lib.pool.load_pool(mode="meta")`, which never requests `test_score`).
It refuses to read anything unless `artifacts/results/E067/VALIDATION_AUTHORISATION.json` exists, was written by
the PI, names the freeze commit, and pins the SHA-256 of the protocol and of every source below. Claude never
writes that file. Stage T (test) is a separate driver, added before its own authorisation.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE.parent))
from experiments.tcss_lib import core as C          # noqa: E402
from experiments.tcss_lib import pool as POOL       # noqa: E402
import refit as R                                   # noqa: E402

EXP = "E067"
OUT = C.ART / EXP
PROTOCOL = ROOT / "docs" / "experiments" / "E067_validation_refit_uncertainty.md"
FIT = C.ART / "E061" / "validation_fit.json"
FIT_SHA = "c3d01b7d84d8858bfd091e9bcdd8d8db4b24597fc4ccf8adeaf3554e9dae22e4"
SOURCES = [HERE, HERE.parent / "refit.py", ROOT / "experiments" / "tcss_lib" / "core.py",
           ROOT / "experiments" / "tcss_lib" / "pool.py"]
DETS = C.HEADLINE
B_V = 1000
SEED_V = 670067
REQUIRED = ("authorised_by", "date", "scope", "freeze_commit", "protocol_sha256", "script_sha256")


def check_validation_authorisation() -> dict:
    f = OUT / "VALIDATION_AUTHORISATION.json"
    if not f.exists():
        raise C.TestAccessDenied(f"{EXP}: no {C.rel(f)}; validation scores may not be read (L2 missing)")
    a = json.loads(f.read_text())
    miss = [k for k in REQUIRED if not a.get(k)]
    if miss:
        raise C.TestAccessDenied(f"{EXP}: authorisation lacks {miss}")
    if a.get("scope") != "stage_v_validation_only":
        raise C.TestAccessDenied(f"{EXP}: scope must be 'stage_v_validation_only'")
    if a["protocol_sha256"] != C.sha256(PROTOCOL):
        raise C.TestAccessDenied(f"{EXP}: protocol differs from the authorised version")
    for s in SOURCES:
        if a["script_sha256"].get(C.rel(s)) != C.sha256(s):
            raise C.TestAccessDenied(f"{EXP}: {C.rel(s)} differs from the authorised version")
    if C.sha256(FIT) != FIT_SHA:
        raise C.TestAccessDenied(f"{EXP}: validation_fit.json differs from the F2-frozen file")
    return a


def stage_v() -> int:
    auth = check_validation_authorisation()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    outdir = OUT / f"validation_run_{stamp}"
    outdir.mkdir(parents=True, exist_ok=False)
    alog = C.AccessLog(EXP, authorisation=auth)
    pools = POOL.load_pool("meta", DETS, alog)                       # validation keys + test metadata only
    frozen = json.loads(FIT.read_text())["fits"]          # F2 layout: {"fits": {key: params}, ...}
    gates, spread, shares = {"G_V0": {"pass": True, "mismatch": []}}, {}, {}
    for p, pp in pools.items():
        ident = POOL.val_identity_map((p,))
        # validation real rows in dump order: recover patch ids from the first dump read for this protocol
        f0 = C.dump_path(DETS[0], p, "leg", 0, C.ANCHOR)
        z = C.read_members(f0, ("val_patch_id", "val_y"), alog)
        vpid = z["val_patch_id"][z["val_y"] == 0]
        subj = np.array([C.subject_of(ident[str(x)]) for x in vpid])
        keys = {}
        for det in DETS:
            for arm in C.ARMS:
                for s in C.SEEDS:
                    for op, cond in C.OPS.items():
                        k = f"{det}|{p}|{arm}|{s}|{op}"
                        v0, v1 = pp.val_real[(det, arm, s, C.ANCHOR)], pp.val_real[(det, arm, s, cond)]
                        w0, w1 = pp.val_dev[(det, arm, s, C.ANCHOR)], pp.val_dev[(det, arm, s, cond)]
                        full = C.fit_policies(v0, v1, w0, w1)                     # G-V0: no resampling
                        for prm in R.PARAMS:
                            a, b = full[prm], frozen[k][prm]
                            if not (a == b or (np.isinf(a) and np.isinf(b) and np.sign(a) == np.sign(b))):
                                gates["G_V0"]["pass"] = False; gates["G_V0"]["mismatch"].append([k, prm, a, b])
                        keys[k] = (v0, v1, w0, w1)
        if not gates["G_V0"]["pass"]:
            break
        n_dev = len(next(iter(keys.values()))[2])
        rep = R.shared_replicates(keys, subj, n_dev, B_V, SEED_V + (0 if p == "p1" else 1))
        for k, rk in rep.items():
            spread[k] = R.summarise(rk, frozen[k])
            shares[k] = {"tau_b_neg_inf": float(np.mean(np.isneginf(rk["tau_b"]))),
                         "q_a_equals_frozen": float(np.mean(rk["q_a"] == frozen[k]["q_a"])),
                         "tau0_at_float32_one": float(np.mean(np.float32(rk["tau0"]) == np.float32(1.0))),
                         "tau0_distinct_values": int(np.unique(rk["tau0"]).size),
                         "nonfinite_share": float(np.mean(~rk["finite"]))}
    (outdir / "gates.json").write_text(json.dumps(gates, indent=1, default=float))
    alog.dump(outdir / "access_log.json")
    if not gates["G_V0"]["pass"]:
        print("G-V0 FAILED: full-sample refit does not reproduce validation_fit.json; stopping (no spread written)")
        return 2
    (outdir / "param_spread.json").write_text(json.dumps(spread, indent=1, default=float))
    (outdir / "shares.json").write_text(json.dumps(shares, indent=1))
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    (outdir / "provenance.json").write_text(json.dumps({
        "experiment": EXP, "stage": "V", "git": git, "B_V": B_V, "seed": SEED_V, "python": sys.version,
        "numpy": np.__version__, "sources": {C.rel(s): C.sha256(s) for s in SOURCES},
        "protocol_sha256": C.sha256(PROTOCOL), "validation_fit_sha256": FIT_SHA,
        "command": " ".join(sys.argv)}, indent=1))
    assert "test_score" not in json.dumps(json.loads((outdir / "access_log.json").read_text()))
    print(f"Stage V done -> {outdir}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check-auth", action="store_true")
    g.add_argument("--stage-v", action="store_true")
    a = ap.parse_args()
    try:
        if a.check_auth:
            check_validation_authorisation(); print("authorisation: OK"); return 0
        return stage_v()
    except C.TestAccessDenied as e:
        print(f"authorisation: REFUSED -- {e}"); return 3


if __name__ == "__main__":
    sys.exit(main())
