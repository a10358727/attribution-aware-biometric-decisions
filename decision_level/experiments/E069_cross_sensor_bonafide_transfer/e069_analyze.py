#!/usr/bin/env python3
"""E069 analysis of the external outcome dumps (protocol v4). Reads only outcome dumps written by e069_score.py
--outcome (which required L3-ext), the window manifest and the frozen E061 single-attempt B for the domain-shift
indicator.

    python e069_analyze.py --outcome-dirs <outcome_Rn_dir> <outcome_Bs_dir> [<outcome_Dt_dir>]

With fewer than three families the readings are written as INTERIM (decision rule not applied).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e069_common as E  # noqa: E402
sys.path.insert(0, str(E.ROOT))
from experiments.tcss_lib import core as C  # noqa: E402

B, CHUNK, SEED, SEED_RECAL = 10_000, 500, 690069, 690070
X3_MIN = 3
ELIG = (0.005, 0.95)          # cell eligible iff full-sample seed-mean B(degraded) in this range (CLAUDE.md 3.3)
DEVICE_ORDER = ["FVC2000_Db1", "FVC2000_Db2", "FVC2000_Db3", "FVC2002_Db1", "FVC2002_Db3", "FVC2004_DB1", "FVC2004_DB2",
                "SD302_C", "SD302_E", "SD302_G", "SD302_K", "SD302_M", "SD302_P", "SD302_S"]
SECONDARY = {"SD302_C", "SD302_E", "SD302_G", "SD302_S"}   # protocol v4 stratum: resolution not documented as captured at 500


def verify_outcome_dir(d: Path) -> str:
    """The directory must come from an authorised outcome read (access log with authorisation; provenance)."""
    acc = json.loads((d / "access_log.json").read_text()); prov = json.loads((d / "provenance.json").read_text())
    a = acc.get("authorisation") or {}
    if not a.get("freeze_commit") or not str(a.get("scope", "")).startswith("e069_outcome"):
        raise SystemExit(f"{d}: not an authorised E069 outcome read")
    if prov["launch"]["git"] != a["freeze_commit"] and not E.git("merge-base", "--is-ancestor", a["freeze_commit"], prov["launch"]["git"]).returncode == 0:
        raise SystemExit(f"{d}: outcome not run on or after the freeze commit")
    fam = d.name.split("_")[1]
    if fam not in E.FAMILY_INDEX:
        raise SystemExit(f"{d}: unknown family")
    return fam


def load(outcome_dirs):
    rows = list(csv.DictReader(open(E.WIN / "manifest.csv")))
    pid = np.array([r["patch_id"] for r in rows])
    rej, score, tau = {}, {}, {}
    fams = []
    for d in outcome_dirs:
        d = Path(d)
        fam = verify_outcome_dir(d)
        if fam in fams:
            raise SystemExit(f"duplicate family {fam}")
        fams.append(fam)
        for c in E.checkpoints((fam,)):
            for cond in E.CONDS:
                z = np.load(d / "dumps" / f"{c['run_id']}__{cond}.npz", allow_pickle=False)
                assert (z["patch_id"] == pid).all()
                key = (fam, c["p"], c["arm_code"], int(c["seed"]), cond)
                score[key] = z["score"].astype(np.float64); tau[key[:4]] = float(z["tau0"])
                rej[key] = (score[key] > tau[key[:4]]).astype(np.float64)
    return rows, fams, rej, score, tau


def device_index(rows):
    dev = defaultdict(list)
    for i, r in enumerate(rows):
        dev[r["source"]].append(i)
    return {g: np.array(v) for g, v in dev.items()}


def cluster_codes(rows, idx):
    cl = np.array([rows[i]["cluster"] for i in idx])
    names, code = np.unique(cl, return_inverse=True)
    return names, code


def sums(x, code, k):
    return np.bincount(code, weights=x, minlength=k)


def analyse(rows, fams, rej, score, tau, log):
    devs = device_index(rows)
    single = {(r["det"], r["protocol"], r["arm"]): float(r["estimate"]) for r in csv.DictReader(open(E.E061_SINGLE))
              if r["policy"] == "pi0" and r["quantity"] == "B" and r["op"] == "bil"}
    lvl_rows, d_rows, dd_rows, x3_rows, rc_rows = [], [], [], [], []
    readings = defaultdict(dict)
    for g in sorted(devs):
        gi = DEVICE_ORDER.index(g)
        rng_cl = np.random.default_rng(np.random.SeedSequence([SEED, gi]))
        rng_sd = {(f, p): np.random.default_rng(np.random.SeedSequence([SEED, gi, E.FAMILY_INDEX[f], pi]))
                  for f in fams for pi, p in enumerate(("p1", "p2"))}
        idx = devs[g]; names, code = cluster_codes(rows, idx); K = len(names)
        n_cl = sums(np.ones(len(idx)), code, K)
        # per (fam,p,arm,seed,cond): cluster sums of rejections
        S = {k: sums(v[idx], code, K) for k, v in rej.items()}
        full = {}
        reps = defaultdict(list)
        mseed = {}
        for b0 in range(0, B, CHUNK):
            n = min(CHUNK, B - b0)
            W = C.row_counts(rng_cl, n, K)                                   # cluster multiplicities (n, K)
            den = W @ n_cl
            for f in fams:
                for p in ("p1", "p2"):
                    m = C.row_counts(rng_sd[(f, p)], n, 10)                  # seeds per (family, protocol)
                    for a in ("leg", "fix"):
                        for cond in E.CONDS:
                            per = np.stack([(W @ S[(f, p, a, s, cond)]) / den for s in E.SEEDS], 1)
                            reps[(f, p, a, cond)].append(C.seed_average(per, m))
        reps = {k: np.concatenate(v) for k, v in reps.items()}
        for f in fams:
            for p in ("p1", "p2"):
                for a in ("leg", "fix"):
                    for cond in E.CONDS:
                        full[(f, p, a, cond)] = float(np.mean([S[(f, p, a, s, cond)].sum() / n_cl.sum() for s in E.SEEDS]))
        meta = {k: rows[idx[0]][k] for k in ("dataset", "technology", "impression")}
        for f in fams:
            for p in ("p1", "p2"):
                for a in ("leg", "fix"):
                    ref = single.get((f, p, a))
                    for cond in E.CONDS:
                        lo, hi = C.pctl(reps[(f, p, a, cond)], 0.95)
                        lvl_rows.append(dict(device=g, **meta, family=f, protocol=p, arm=a, condition=cond,
                                             cell_eligible=celig_full(full, f, p, a),
                                             n_windows=len(idx), n_clusters=K, B=full[(f, p, a, cond)], lo95=lo, hi95=hi,
                                             trackA_B_degraded=ref if cond == "degraded" else "",
                                             domain_shift_vs_trackA=(full[(f, p, a, cond)] - ref) if (cond == "degraded" and ref is not None) else ""))
                    for op, oc in E.OPS.items():
                        dv = reps[(f, p, a, oc)] - reps[(f, p, a, "degraded")]
                        lo, hi = C.pctl(dv, 0.95)
                        d_rows.append(dict(device=g, family=f, protocol=p, arm=a, op=op,
                                           dB=full[(f, p, a, oc)] - full[(f, p, a, "degraded")], lo95=lo, hi95=hi,
                                           resolved_pos=lo > 0, resolved_neg=hi < 0))
                for op, oc in E.OPS.items():
                    dl = reps[(f, p, "leg", oc)] - reps[(f, p, "leg", "degraded")]
                    dfx = reps[(f, p, "fix", oc)] - reps[(f, p, "fix", "degraded")]
                    lo, hi = C.pctl(dfx - dl, 0.95)
                    est = (full[(f, p, "fix", oc)] - full[(f, p, "fix", "degraded")]) - (full[(f, p, "leg", oc)] - full[(f, p, "leg", "degraded")])
                    dd_rows.append(dict(device=g, family=f, protocol=p, op=op, ddB=est, lo95=lo, hi95=hi,
                                        resolved_neg=hi < 0, resolved_pos=lo > 0))
        # ---- cell eligibility (floor/ceiling of B(degraded) at the frozen tau0)
        celig = {(f, p, a): ELIG[0] <= full[(f, p, a, "degraded")] <= ELIG[1] for f in fams for p in ("p1", "p2") for a in ("leg", "fix")}
        fam_el_leg = [f for f in fams if celig[(f, "p1", "leg")] and celig[(f, "p2", "leg")]]
        fam_el_all = [f for f in fam_el_leg if celig[(f, "p1", "fix")] and celig[(f, "p2", "fix")]]
        # ---- readings per operator
        for op in E.OPS:
            leg_ok = {f: all(r["resolved_pos"] for r in d_rows if r["device"] == g and r["family"] == f and r["arm"] == "leg" and r["op"] == op) for f in fam_el_leg}
            dd_ok = {f: all(r["resolved_neg"] for r in dd_rows if r["device"] == g and r["family"] == f and r["op"] == op) for f in fam_el_all}
            fam_tr = [f for f in fam_el_leg if leg_ok[f]]
            fam_nd = [f for f in fam_el_all if leg_ok.get(f) and dd_ok[f]]
            final = len(fams) == 3
            interpretable = len(fam_el_leg) >= (2 if final else len(fams))
            readings[g][op] = {"families_scored": fams, "families_eligible_legacy": fam_el_leg,
                               "families_eligible_both_arms": fam_el_all,
                               "families_legacy_dB_pos_both_models": fam_tr,
                               "families_also_ddB_neg_both_models": fam_nd,
                               "X2_transfers": (len(fam_tr) >= 2) if (final and interpretable) else None,
                               "normalization_dependence_transfers": (len(fam_nd) >= 2) if (final and len(fam_el_all) >= 2) else None,
                               "interim_both_families": (len(fam_tr) == len(fams)) if (not final and interpretable) else None,
                               "interim_both_families_norm_dependence": (len(fam_nd) == len(fams)) if (not final and len(fam_el_all) == len(fams)) else None,
                               "interpretable": interpretable,
                               "status": ("final" if final else "INTERIM (D3 not yet scored; rule not applied)") if interpretable
                                         else "NO INTERPRETABLE READING (cells outside the B(degraded) eligibility range)",
                               "stratum": "secondary" if g in SECONDARY else "primary",
                               "impression": rows[idx[0]]["impression"], "dataset": rows[idx[0]]["dataset"], "device": g,
                               "underpowered": bool(len(idx) < 200 or K < 30)}
        # ---- X3 (FVC credential burden)
        if rows[idx[0]]["dataset"] == "FVC":
            cred = np.array([rows[i]["identity"] for i in idx])
            cn, cc = np.unique(cred, return_inverse=True)
            nf = np.bincount(cc, minlength=len(cn))
            keep = nf >= X3_MIN
            rng_x = np.random.default_rng(np.random.SeedSequence([SEED, 3, len(idx)]))
            Wf = C.row_counts(rng_x, B, int(keep.sum()))
            for f in fams:
                for p in ("p1", "p2"):
                    for a in ("leg", "fix"):
                        vals = {}
                        for cond in ("degraded",) + tuple(E.OPS.values()):
                            pf = np.stack([np.bincount(cc, weights=rej[(f, p, a, s, cond)][idx], minlength=len(cn))[keep] / nf[keep]
                                           for s in E.SEEDS], 0)                                     # (10, F)
                            vals[cond] = {"Bcred": pf.mean(1), "C50": (pf >= 0.5).mean(1), "L3": (pf ** 3).mean(1),
                                          "rep": {q: np.stack([(Wf @ (fn(pf[s]))) / Wf.sum(1) for s in range(10)], 1).mean(1)
                                                  for q, fn in (("Bcred", lambda x: x), ("C50", lambda x: (x >= 0.5).astype(float)),
                                                                ("L3", lambda x: x ** 3))}}
                        for op, oc in E.OPS.items():
                            for q in ("Bcred", "C50", "L3"):
                                d = vals[oc]["rep"][q] - vals["degraded"]["rep"][q]
                                lo, hi = C.pctl(d, 0.95)
                                x3_rows.append(dict(device=g, family=f, protocol=p, arm=a, op=op,
                                                    quantity=q if q != "L3" else "L3 (modelled, M0)",
                                                    degraded=float(vals["degraded"][q].mean()), restored=float(vals[oc][q].mean()),
                                                    delta=float(vals[oc][q].mean() - vals["degraded"][q].mean()), lo95=lo, hi95=hi,
                                                    n_fingers=int(keep.sum()), n_excluded_lt3=int((~keep).sum()), bootstrap="fingers resampled, B=10000, seeds averaged"))
        # ---- recalibration diagnostic (point estimates)
        rngr = np.random.default_rng(np.random.SeedSequence([SEED_RECAL, gi]))
        cal = rngr.permutation(K)[: K // 2]
        is_cal = np.isin(code, cal)
        for f in fams:
            for p in ("p1", "p2"):
                for a in ("leg", "fix"):
                    for op, oc in E.OPS.items():
                        dbs = []
                        for s in E.SEEDS:
                            t = C.thr_e023(score[(f, p, a, s, "degraded")][idx][is_cal])
                            ev = ~is_cal
                            dbs.append(float((score[(f, p, a, s, oc)][idx][ev] > t).mean() - (score[(f, p, a, s, "degraded")][idx][ev] > t).mean()))
                        rc_rows.append(dict(device=g, family=f, protocol=p, arm=a, op=op, dB_recal_device=float(np.mean(dbs)),
                                            note="diagnostic; device-recalibrated 1% threshold on 50% clusters, evaluated on the rest"))
        log(f"  {g}: done")
    return lvl_rows, d_rows, dd_rows, x3_rows, rc_rows, readings


def celig_full(full, f, p, a):
    return ELIG[0] <= full[(f, p, a, "degraded")] <= ELIG[1]


def write(out, name, rows):
    if not rows:
        return
    with open(out / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outcome-dirs", nargs="+", required=True)
    a = ap.parse_args()
    launch = E.git_state()
    rows, fams, rej, score, tau = load(a.outcome_dirs)
    out = E.OUT / f"analysis_{'_'.join(fams)}_{time.strftime('%Y%m%d_%H%M%S')}"; out.mkdir(parents=True, exist_ok=False)
    logf = open(out / "run.log", "w")
    log = lambda m: (print(m, flush=True), logf.write(m + "\n"), logf.flush())
    lv, d, dd, x3, rc, rd = analyse(rows, fams, rej, score, tau, log)
    for n, r in (("rejection_by_device.csv", lv), ("deltaB.csv", d), ("deltaDeltaB.csv", dd), ("credential_fvc.csv", x3),
                 ("recal_diagnostic.csv", rc)):
        write(out, n, r)
    def tally(op, sel):
        gs = [g for g in rd if sel(rd[g][op])]
        return {"n_devices": len(gs), "interpretable": sum(rd[g][op]["interpretable"] for g in gs),
                "X2": sum(bool(rd[g][op]["X2_transfers"] or rd[g][op]["interim_both_families"]) for g in gs),
                "norm_dependence": sum(bool(rd[g][op]["normalization_dependence_transfers"] or rd[g][op]["interim_both_families_norm_dependence"]) for g in gs),
                "devices_X2": [g for g in gs if rd[g][op]["X2_transfers"] or rd[g][op]["interim_both_families"]]}
    strata = {"FVC (plain)": lambda v: v["dataset"] == "FVC",
              "SD302 primary (plain, aux devices K/M/P)": lambda v: v["dataset"] == "SD302" and v["stratum"] == "primary",
              "SD302 primary without K (sensitivity)": lambda v: v["dataset"] == "SD302" and v["stratum"] == "primary" and v["device"] != "SD302_K",
              "SD302 secondary: rolled-equivalent C/E/G": lambda v: v["dataset"] == "SD302" and v["impression"] == "rolled-equivalent",
              "SD302 secondary: segmented slap S": lambda v: v["dataset"] == "SD302" and v["impression"].startswith("4-4-2")}
    summary = {"families": fams, "final": len(fams) == 3, "readings": rd,
               "counts": {op: {k: tally(op, sel) for k, sel in strata.items()} for op in E.OPS},
               "eligibility_range_B_degraded": ELIG,
               "B": B, "seed": SEED, "outcome_dirs": [E.rel(Path(x)) for x in a.outcome_dirs]}
    (out / "readings.json").write_text(json.dumps(summary, indent=1, default=bool))
    (out / "provenance.json").write_text(json.dumps({"launch": launch, "sources": {E.rel(s): E.sha256(s) for s in E.SOURCES}}, indent=1))
    log(f"analysis -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
