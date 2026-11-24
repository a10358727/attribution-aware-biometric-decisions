#!/usr/bin/env python3
"""E070 step 2: join the hashed NFIQ 2 tertiles (Q-B) with E069 outcome dumps; tertile contrast of dB.

    python e070_contrast.py --quality-dir <quality_dir> --outcome-dirs <E069 outcome_Rn> <outcome_Bs> [<outcome_Dt>]

Refuses unless the Q-B hashes match the files and the authorisation (artifacts/results/E070/AUTHORISATION.json).
With fewer than three families the readings are INTERIM.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve(); ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "E069_cross_sensor_bonafide_transfer")); sys.path.insert(0, str(ROOT))
import e069_common as E  # noqa: E402
from experiments.tcss_lib import core as C  # noqa: E402

OUT = ROOT / "artifacts" / "results" / "E070"
PROTOCOL = ROOT / "docs" / "experiments" / "E070_quality_conditioned_external.md"
SOURCES = [HERE, HERE.parent / "e070_quality.py", HERE.parents[1] / "E069_cross_sensor_bonafide_transfer" / "e069_common.py",
           HERE.parents[1] / "E069_cross_sensor_bonafide_transfer" / "e069_analyze.py",
           HERE.parents[1] / "E069_cross_sensor_bonafide_transfer" / "e069_score.py",
           ROOT / "experiments" / "tcss_lib" / "core.py"]
ELIG = (0.005, 0.95)       # contrast cell eligible iff the device's B(degraded) at tau0 (all eligible captures) is in range
B, CHUNK, SEED = 10_000, 500, 700070


def check(quality_dir: Path) -> dict:
    a = json.loads((OUT / "AUTHORISATION.json").read_text()) if (OUT / "AUTHORISATION.json").exists() else None
    if a is None:
        raise E.AccessDenied("E070: no AUTHORISATION.json")
    for k in ("authorised_by", "freeze_commit", "protocol_sha256", "script_sha256", "quality_hashes"):
        if not a.get(k):
            raise E.AccessDenied(f"E070 authorisation lacks {k}")
    for s in SOURCES:
        if a["script_sha256"].get(E.rel(s)) != E.sha256(s):
            raise E.AccessDenied(f"E070: {E.rel(s)} differs from the authorised version")
        if E.blob_sha(a["freeze_commit"], s) != E.sha256(s):
            raise E.AccessDenied(f"E070: {E.rel(s)} differs from its freeze-commit blob")
    E.verify_freeze(a["freeze_commit"], PROTOCOL, a["protocol_sha256"])
    h = json.loads((quality_dir / "hashes.json").read_text())
    for fn in ("nfiq2_scores.csv", "tertile_edges.json"):                 # the required set, not what hashes.json lists
        v = a["quality_hashes"].get(fn)
        if not v or h.get(fn) != v or E.sha256(quality_dir / fn) != v:
            raise E.AccessDenied(f"Q-B: {fn} hash missing or mismatched")
    # e070_quality.py opens no E069 dump by construction (asserted in its provenance); the binding guarantee of Q-B
    # ordering is the E069 L3 pin of these hashes, which the outcome read checks before any scoring.
    return a


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality-dir", required=True); ap.add_argument("--outcome-dirs", nargs="+", required=True)
    a = ap.parse_args()
    qd = Path(a.quality_dir)
    try:
        auth = check(qd)
    except E.AccessDenied as e:
        print(f"REFUSED -- {e}"); return 3
    launch = E.git_state()
    q = {r["patch_id"]: r for r in csv.DictReader(open(qd / "nfiq2_scores.csv")) if r["ok"] == "True"}
    edges = json.loads((qd / "tertile_edges.json").read_text())["edges"]
    rows = list(csv.DictReader(open(E.WIN / "manifest.csv")))
    pid = np.array([r["patch_id"] for r in rows]); pos = {p: i for i, p in enumerate(pid)}
    sys.path.insert(0, str(ROOT / "experiments" / "E069_cross_sensor_bonafide_transfer"))
    import e069_analyze as A69
    fams = [A69.verify_outcome_dir(Path(d)) for d in a.outcome_dirs]
    if len(set(fams)) != len(fams):
        print("REFUSED -- duplicate family"); return 3
    fail = json.loads((qd / "tertile_edges.json").read_text()).get("failures", {})
    deg = {r["patch_id"]: r.get("deg_type", "") for r in csv.DictReader(open(E.WIN / "manifest.csv"))}
    out_rows, tert_rows, readings = [], [], {}
    for dev, ed in sorted(edges.items()):
        gi = A69.DEVICE_ORDER.index(dev)
        rng_cl = np.random.default_rng(np.random.SeedSequence([SEED, gi]))
        rng_sd = {(f, p): np.random.default_rng(np.random.SeedSequence([SEED, gi, E.FAMILY_INDEX[f], pi]))
                  for f in fams for pi, p in enumerate(("p1", "p2"))}
        ids = [p for p, r in q.items() if r["device"] == dev]
        idx = np.array([pos[p] for p in ids]); qs = np.array([int(q[p]["nfiq2"]) for p in ids], float)
        b = np.where(qs <= ed["edges"][0], 0, np.where(qs <= ed["edges"][1], 1, 2))
        cl = np.array([q[p]["cluster"] for p in ids]); names, code = np.unique(cl, return_inverse=True); K = len(names)
        def csum(x, mask):
            return np.bincount(code[mask], weights=x[mask], minlength=K), np.bincount(code[mask], minlength=K).astype(float)
        reps, full = {}, {}
        dumps = {}
        for d, f in zip(a.outcome_dirs, fams):
            for ck in E.checkpoints((f,)):
                for cond in ("degraded",) + tuple(E.OPS.values()):
                    z = np.load(Path(d) / "dumps" / f"{ck['run_id']}__{cond}.npz", allow_pickle=False)
                    assert (z["patch_id"] == pid).all()
                    dumps[(f, ck["p"], ck["arm_code"], int(ck["seed"]), cond)] = (z["score"][idx] > float(z["tau0"])).astype(float)
        stats = {}
        for f in fams:
            for p in ("p1", "p2"):
                for arm in ("leg", "fix"):
                    for op, oc in E.OPS.items():
                        per = {}
                        for s in E.SEEDS:
                            dlt = dumps[(f, p, arm, s, oc)] - dumps[(f, p, arm, s, "degraded")]
                            per[s] = {k: csum(dlt, b == k) for k in (0, 2)}
                            per[s]["deg"] = {k: csum(dumps[(f, p, arm, s, "degraded")], b == k) for k in (0, 1, 2)}
                        stats[(f, p, arm, op)] = per
        rr = {k: [] for k in stats}
        for b0 in range(0, B, CHUNK):
            n = min(CHUNK, B - b0)
            W = C.row_counts(rng_cl, n, K)
            ms = {(f, p): C.row_counts(rng_sd[(f, p)], n, 10) for f in fams for p in ("p1", "p2")}
            for (f, p, arm, op), per in stats.items():
                vals = np.stack([(W @ per[s][0][0]) / np.maximum(W @ per[s][0][1], 1e-12) - (W @ per[s][2][0]) / np.maximum(W @ per[s][2][1], 1e-12)
                                 for s in E.SEEDS], 1)
                rr[(f, p, arm, op)].append(C.seed_average(vals, ms[(f, p)]))
        for k, per in stats.items():
            f, p, arm, op = k
            est = float(np.mean([per[s][0][0].sum() / per[s][0][1].sum() - per[s][2][0].sum() / per[s][2][1].sum() for s in E.SEEDS]))
            lo, hi = C.pctl(np.concatenate(rr[k]), 0.95)
            out_rows.append(dict(device=dev, family=f, protocol=p, arm=arm, op=op, contrast_low_minus_high=est, lo95=lo, hi95=hi,
                                 n_low=int((b == 0).sum()), n_high=int((b == 2).sum()), n_clusters=K))
            if op == "bil":
                for t in (0, 1, 2):
                    tert_rows.append(dict(device=dev, family=f, protocol=p, arm=arm, tertile=t,
                                          deg_mix=json.dumps(dict(sorted(__import__("collections").Counter(deg[pid] for pid, bb in zip(ids, b) if bb == t).items()))),
                                          B_degraded=float(np.mean([per[s]["deg"][t][0].sum() / max(per[s]["deg"][t][1].sum(), 1) for s in E.SEEDS])),
                                          n=int((b == t).sum())))
        bdeg = {(f, p): float(np.mean([dumps[(f, p, "leg", s, "degraded")].mean() for s in E.SEEDS])) for f in fams for p in ("p1", "p2")}
        fam_el = [f for f in fams if all(ELIG[0] <= bdeg[(f, p)] <= ELIG[1] for p in ("p1", "p2"))]
        for op in E.OPS:
            ok = {f: all(r["lo95"] > 0 for r in out_rows if r["device"] == dev and r["family"] == f and r["arm"] == "leg" and r["op"] == op) for f in fam_el}
            fam_ok = [f for f in fam_el if ok[f]]
            readings.setdefault(dev, {})[op] = {"families_supporting_both_models": fam_ok, "families_eligible": fam_el,
                                                "B_degraded_legacy": {f"{f}|{p}": v for (f, p), v in bdeg.items()},
                                                "Q1_supported": (len(fam_ok) >= 2) if len(fams) == 3 else None,
                                                "status": "final" if len(fams) == 3 else "INTERIM (D3 not yet scored)",
                                                "underpowered": bool(K < 30 or min(int((b == k).sum()) for k in range(3)) < 50),
                                                "unreliable_nfiq2_failures": bool(fail.get(dev, {}).get("unreliable", False)),
                                                "stratum": "secondary" if dev in A69.SECONDARY else "primary"}
    out = OUT / f"contrast_{'_'.join(fams)}_{time.strftime('%Y%m%d_%H%M%S')}"; out.mkdir(parents=True, exist_ok=False)
    for name, data in (("contrast.csv", out_rows), ("by_tertile.csv", tert_rows)):
        with open(out / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    counts = {}
    for op in E.OPS:
        for stratum in ("primary", "secondary"):
            adequate = [d for d, v in readings.items() if v[op]["stratum"] == stratum and not v[op]["underpowered"]
                        and not v[op]["unreliable_nfiq2_failures"] and len(v[op]["families_eligible"]) == len(fams)]
            counts[f"{op}|{stratum}"] = {"adequately_powered_eligible_devices": len(adequate),
                                         "Q1_supported_or_interim_both": sum(1 for d in adequate if readings[d][op]["Q1_supported"] or
                                             (readings[d][op]["Q1_supported"] is None and len(readings[d][op]["families_supporting_both_models"]) == len(fams)))}
    (out / "readings.json").write_text(json.dumps({"families": fams, "readings": readings, "counts": counts}, indent=1, default=bool))
    (out / "provenance.json").write_text(json.dumps({"launch": launch, "quality_dir": E.rel(qd), "authorisation": auth,
                                                     "sources": {E.rel(s): E.sha256(s) for s in SOURCES}}, indent=1))
    print(f"E070 contrast -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
