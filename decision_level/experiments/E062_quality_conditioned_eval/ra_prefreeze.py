"""E062 v2 pre-freeze checks: finger feasibility (metadata only) and estimator simulation (synthetic outcomes).

Not evidence about E062. Two parts:

1. `finger`: from the protected q3_v2_all label CSV and the metadata columns of the Track A P1 manifest (the same
   inputs as `ra_audit.py`), counts per finger group x RA class, the feasibility of candidate finger sensitivity
   analyses (S-8 options), and the credential-majority definition problem (A-4). Aggregates only.
2. `sim`: Monte-Carlo study of the frozen-candidate estimator (`ra_analysis.run`). Rejection indicators are
   SYNTHETIC (latent-score generator); the design skeleton (subject, credential, RA class, finger group of each
   test capture) is either the real metadata skeleton ("plasmode") or fully synthetic. No detector score, logit,
   prediction or E060/E061 outcome is read, and no image is opened.

Outputs: JSON with counts and rates only (no subject, credential or filename). The skeleton is held in memory.

    python -m experiments.E062_quality_conditioned_eval.ra_prefreeze finger --labels <csv> --out <dir>
    python -m experiments.E062_quality_conditioned_eval.ra_prefreeze sim --labels <csv> --out <dir> --workers 28
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from experiments.E062_quality_conditioned_eval import ra_analysis as E
from experiments.E062_quality_conditioned_eval import ra_audit as RA
from experiments.tcss_lib import core as C

ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT / "config" / "splits" / "tracka_split_v1"
MANIFEST_P1 = ROOT / "data" / "track_a" / "p1" / "manifest.csv"
GROUPS = ("thumb", "non_thumb", "other")             # finger groups; "other" = non-standard finger name


def finger_group(credential: str) -> str:
    f = RA.finger_of(credential)
    return "other" if f == "other" else ("thumb" if f.endswith("Thumb") else "non_thumb")


# ------------------------------------------------------------------------------ skeleton (metadata only)
def skeleton(labels_csv: Path) -> dict:
    """Per test capture: subject, credential, RA class, finger group, vote flag, KNN support, severity.
    Order = the frozen test list. Held in memory only."""
    labels, dups, bad = RA.read_labels(labels_csv)
    if dups or bad:
        raise ValueError("label CSV has duplicates or invalid codes")
    files = RA.read_lines(SPLIT_DIR / "test_cap60_files.txt")
    sev = {Path(r["src_file"]).name: int(r["deg_severity"])
           for r in RA.read_manifest_real(MANIFEST_P1) if r["split"] == "test"}
    subj, cred = zip(*[RA.parse_name(f)[:2] for f in files])
    return {"subj": np.array([f"S{s}" for s in subj]), "cred": np.array(cred),
            "cls": np.array([labels[f][0] for f in files]),
            "finger": np.array([finger_group(c) for c in cred]),
            "vote": np.array([labels[f][1] for f in files]),
            "support": np.array([labels[f][2] if labels[f][2] is not None else 9 for f in files]),
            "severity": np.array([sev[f] for f in files])}


def _cls_summary(sk, sel) -> dict:
    out = {}
    for g, name in enumerate(E.CLASSES):
        m = sel & (sk["cls"] == g)
        sc = Counter(sk["subj"][m])
        top3 = sum(sorted(sc.values(), reverse=True)[:3])
        out[name] = {"captures": int(m.sum()), "subjects": len(sc), "credentials": int(len(np.unique(sk["cred"][m]))),
                     "top3_subject_share": round(top3 / m.sum(), 4) if m.sum() else None}
    caps = [v["captures"] for v in out.values()]
    subs = [v["subjects"] for v in out.values()]
    out["C-3"] = "PASS" if min(caps) >= RA.C3_MIN_CAPTURES else "FAIL"
    out["C-4"] = ("PASS" if min(subs) >= RA.C4_MIN_SUBJECTS else
                  ("WARNING" if min(subs) >= RA.C4_FLOOR else "FAIL"))
    out["C-5"] = "WARNING" if max(v["top3_subject_share"] or 0 for v in out.values() if isinstance(v, dict)
                                  ) > RA.C5_DOMINATED else "PASS"
    return out


def _v(sk, sel, by) -> float | None:
    t = defaultdict(Counter)
    for g, f in zip(sk["cls"][sel], by[sel]):
        t[g][f] += 1
    return RA._cramers_v(t)


def finger_report(sk) -> dict:
    n = len(sk["cls"])
    allm = np.ones(n, bool)
    fine = np.array([RA.finger_of(c) for c in sk["cred"]])
    rep = {"pool": {"captures": n, "subjects": int(len(np.unique(sk["subj"]))),
                    "credentials": int(len(np.unique(sk["cred"])))}}
    rep["by_group"] = {g: _cls_summary(sk, sk["finger"] == g) for g in GROUPS}
    rep["cramers_v"] = {"finger_10_plus_other_all": _v(sk, allm, fine), "finger_group_all": _v(sk, allm, sk["finger"])}
    # candidate finger sensitivity analyses (restrictions; the estimand is unchanged)
    mixed_creds = {c for c in np.unique(sk["cred"]) if len(np.unique(sk["cls"][sk["cred"] == c])) >= 2}
    cand = {"S-8a_no_thumb": sk["finger"] != "thumb",
            "S-8b_named_non_thumb": sk["finger"] == "non_thumb",
            "S-8c_mixed_credentials": np.isin(sk["cred"], list(mixed_creds))}
    for name, g in (("dry", 0), ("wet", 2)):
        both = {c for c in np.unique(sk["cred"]) if {g, 1} <= set(sk["cls"][sk["cred"] == c].tolist())}
        cand[f"S-8w_{name}_within_credential"] = np.isin(sk["cred"], list(both)) & np.isin(sk["cls"], (g, 1))
    rep["candidates"] = {}
    for k, m in cand.items():
        rep["candidates"][k] = {"captures": int(m.sum()), "subjects": int(len(np.unique(sk["subj"][m]))),
                                "classes": _cls_summary(sk, m),
                                "cramers_v_finger_residual": _v(sk, m, fine)}
    # finger-standardisation feasibility: (finger, class) cells with fewer than 10 captures
    cells = Counter(zip(fine, sk["cls"]))
    rep["standardisation"] = {"finger_categories": int(len(np.unique(fine))),
                              "cells_below_10_captures": sum(1 for f in np.unique(fine) for g in range(3)
                                                             if cells.get((f, g), 0) < 10),
                              "cells_total": int(len(np.unique(fine)) * 3)}
    # A-4: credential-majority class
    plural = Counter()
    for c in np.unique(sk["cred"]):
        cnt = Counter(sk["cls"][sk["cred"] == c]).most_common()
        k = len(np.unique(sk["cls"][sk["cred"] == c]))
        tie = len(cnt) > 1 and cnt[0][1] == cnt[1][1]
        plural[(k, "tie" if tie else "unique")] += 1
        if not tie:
            maj = cnt[0][1] / sum(v for _, v in cnt)
            plural[("strict_majority" if maj > 0.5 else "plurality_only", "")] += 1
    rep["credential_majority_A4"] = {f"{a}_classes_{b}" if b else str(a): v for (a, b), v in sorted(plural.items(),
                                                                                                 key=str)}
    return rep


# ------------------------------------------------------------------------------ synthetic outcome generator
SCEN = {   # anchor threshold t and restoration shift s of the latent generator (rates in the report)
    "standard": dict(t=2.0, s=0.8),
    "low_rate": dict(t=2.6, s=0.5),
}


def outcomes(design: dict, rng, t, s, eff=None, finger_eff=0.0, S=10) -> E.Data:
    """Latent z = subject + credential + capture components (independent of RA class unless `eff`), a
    detector/protocol capture response, a per-seed shift and noise. Rejection = latent > t.
    eff: {(arm, class): shift} added to the restored latent; finger_eff: added to restored latent of thumbs
    (a finger effect with NO RA effect, for the confounding scenario)."""
    eff = eff or {}
    subj, cred, cls, fg = design["subj"], design["cred"], design["cls"], design["finger"]
    n = len(cls)
    _, si = np.unique(subj, return_inverse=True)
    _, ci = np.unique(cred, return_inverse=True)
    z = rng.normal(0, 0.6, si.max() + 1)[si] + rng.normal(0, 0.5, ci.max() + 1)[ci] + rng.normal(0, 0.6, n)
    thumb = (fg == "thumb").astype(float)
    rej, el = {}, {}
    for det in E.HEADLINE:
        for p in C.PROTOS:
            dz = rng.normal(0, 0.3, n)
            for arm in C.ARMS:
                el[(det, p, arm)] = True
                e_arm = sum(v * (cls == g) for (a, g), v in eff.items() if a == arm) if eff else 0.0
                for cond in C.CONDS:
                    shift = 0.0 if cond == C.ANCHOR else s + e_arm + finger_eff * thumb
                    base = z + dz + shift
                    rej[(det, p, arm, cond)] = np.stack(
                        [((base + rng.normal(0, 0.25) + rng.normal(0, 0.2, n)) > t).astype(np.float64)
                         for _ in range(S)])
    attrs = {"vote": design.get("vote", np.ones(n, bool)), "support": design.get("support", np.full(n, 9)),
             "severity": design.get("severity", np.ones(n, int)), "finger": fg}
    return E.Data({p: subj for p in C.PROTOS}, {p: cred for p in C.PROTOS}, {p: cls for p in C.PROTOS}, rej, el,
                  {p: attrs for p in C.PROTOS})


def synthetic_design(rng, n_subj, creds=9, caps=6, p_cls=(1 / 3, 1 / 3, 1 / 3)) -> dict:
    subj = np.repeat([f"S{i}" for i in range(n_subj)], creds * caps)
    cred = np.repeat([f"S{i}_{f}" for i in range(n_subj) for f in range(creds)], caps)
    fg = np.repeat(np.tile(np.array(["thumb", "thumb"] + ["non_thumb"] * (creds - 2)), n_subj), caps)
    return {"subj": subj, "cred": cred, "cls": rng.choice(3, len(subj), p=p_cls), "finger": fg}


def restrict(design: dict, keep: np.ndarray) -> dict:
    return {k: v[keep] for k, v in design.items()}


def dry_from_k_subjects(design: dict, k: int, rng) -> dict:
    """Extreme imbalance: keep RA-dry captures from k randomly chosen subjects only (others' RA-dry dropped)."""
    dry_subj = np.unique(design["subj"][design["cls"] == 0])
    chosen = rng.choice(dry_subj, k, replace=False)
    keep = (design["cls"] != 0) | np.isin(design["subj"], chosen)
    return restrict(design, keep)


# ------------------------------------------------------------------------------ one replicate (worker)
_SK = None


def _init(sk):
    global _SK
    _SK = sk
    os.environ.setdefault("OMP_NUM_THREADS", "1")


def _design(spec, rng):
    kind = spec["design"]
    if kind == "plasmode":
        return _SK
    if kind.startswith("plasmode_dry"):
        return dry_from_k_subjects(_SK, int(kind.split("_")[-1]), rng)
    if kind.startswith("synthetic_"):
        n_subj = int(kind.split("_")[1])
        p_cls = (0.15, 0.40, 0.45) if kind.endswith("imbalanced") else (1 / 3, 1 / 3, 1 / 3)
        return synthetic_design(rng, n_subj, p_cls=p_cls)
    raise ValueError(kind)


def one(args):
    spec, k = args
    rng = np.random.default_rng([spec["gen_seed"], k])
    design = _design(spec, rng)
    data = outcomes(design, rng, **SCEN[spec["rate"]], eff=spec.get("eff"), finger_eff=spec.get("finger_eff", 0.0))
    if spec.get("full_only"):        # truth computation: full-sample point estimates only
        out = E.run(data, B=1, seed=k)
        return {n: v["median"] for n, v in out["primary"].items()}
    if spec.get("levels"):                           # record extra percentile levels from the same replicates
        base = E.interval

        def multi(rv, level, _base=base, _lv=tuple(spec["levels"])):
            iv = _base(rv, level)
            if level == E.LEVEL_PRIMARY and iv["reported"]:
                for lv in _lv:
                    iv[f"lohi_{lv}"] = list(C.pctl(rv, lv))
            return iv
        E.interval = multi
        try:
            out = E.run(data, B=spec["B"], seed=k)
        finally:
            E.interval = base
        return {"levels": {n: {str(lv): v.get(f"lohi_{lv}") for lv in spec["levels"]}
                           for n, v in out["primary"].items()}}
    res = {"primary": E.run(data, B=spec["B"], seed=k)}
    if spec.get("with_s8"):
        mk = E.masks(data)
        res["S-8a"] = E.run(data, B=spec["B"], seed=k, mask=mk["S-8a"], label="S-8a")
        w = {n: E.run(data, B=spec["B"], seed=k, mask=mk[f"S-8w_{n}"], weighting="credential", label=f"S-8w_{n}")
             for n in E.CONTRASTS}                          # within-credential: read only the matching contrast
        res["S-8w"] = {"primary": {m: w[m.split("_")[1].split("|")[0]]["primary"][m] for m in res["primary"]["primary"]}}
    rows = {}
    for lab, out in res.items():
        r = {}
        for n, v in out["primary"].items():
            r[n] = {"reading": v["reading"], "lo": v.get("lo"), "hi": v.get("hi"), "median": v.get("median")}
        r["_anchor_rate"] = float(np.mean([data.rej[key].mean() for key in data.rej if key[3] == C.ANCHOR]))
        r["_restored_rate"] = float(np.mean([data.rej[key].mean() for key in data.rej if key[3] != C.ANCHOR]))
        r["_dry_captures"] = int((design["cls"] == 0).sum())
        r["_dry_subjects"] = int(len(np.unique(design["subj"][design["cls"] == 0])))
        rows[lab] = r
    return rows


def wilson(x, n, z=1.96):
    if n == 0:
        return None
    p = x / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def summarise(rows, truth=None, lab="primary") -> dict:
    names = [k for k in rows[0][lab] if not k.startswith("_")]
    n = len(rows)
    reported = [all(r[lab][m]["reading"] in ("above 0", "below 0", "includes 0") for m in names) for r in rows]
    any_ex = [any(r[lab][m]["reading"] in ("above 0", "below 0") for m in names) for r in rows]
    out = {"datasets": n,
           "anchor_rate_mean": round(float(np.mean([r[lab]["_anchor_rate"] for r in rows])), 4),
           "restored_rate_mean": round(float(np.mean([r[lab]["_restored_rate"] for r in rows])), 4),
           "dry_captures_mean": round(float(np.mean([r[lab]["_dry_captures"] for r in rows])), 1),
           "dry_subjects_mean": round(float(np.mean([r[lab]["_dry_subjects"] for r in rows])), 1),
           "all_four_reported": sum(reported),
           "familywise_any_excludes_0": sum(any_ex), "fwer": round(sum(any_ex) / n, 4),
           "fwer_wilson95": wilson(sum(any_ex), n), "per_contrast": {}}
    for m in names:
        rd = Counter(r[lab][m]["reading"] for r in rows)
        rep = [r[lab][m] for r in rows if r[lab][m]["lo"] is not None]
        t = 0.0 if truth is None else truth[m]
        cov = sum(1 for v in rep if v["lo"] <= t <= v["hi"])
        out["per_contrast"][m] = {
            "readings": dict(rd), "truth": round(t, 5),
            "coverage": round(cov / len(rep), 4) if rep else None, "coverage_wilson95": wilson(cov, len(rep)),
            "excludes_0": rd.get("above 0", 0) + rd.get("below 0", 0),
            "mean_width": round(float(np.mean([v["hi"] - v["lo"] for v in rep])), 4) if rep else None,
            "mean_estimate": round(float(np.mean([v["median"] for v in rep if v["median"] is not None])), 5)
            if rep else None}
    return out


def level_summary(rows, levels) -> dict:
    """Null data: per-contrast and family-wise share of intervals excluding 0, at each percentile level."""
    names = list(rows[0]["levels"])
    out = {"per_contrast": {}, "familywise": {}}
    for lv in levels:
        ex = {m: [r["levels"][m][str(lv)] is not None and not (r["levels"][m][str(lv)][0] <= 0 <= r["levels"][m][str(lv)][1])
                  for r in rows] for m in names}
        out["per_contrast"][str(lv)] = {m: round(float(np.mean(v)), 4) for m, v in ex.items()}
        fw = sum(any(ex[m][i] for m in names) for i in range(len(rows)))
        out["familywise"][str(lv)] = {"fwer": round(fw / len(rows), 4), "wilson95": wilson(fw, len(rows)),
                                      "per_contrast_mean": round(float(np.mean([np.mean(v) for v in ex.values()])), 4)}
    return out


LEVELS = [0.9875, 0.99, 0.995, 0.9975]
LEVEL_SPECS = [   # level calibration on null data; the second generator is a hold-out check of the first
    dict(name="L1_levels_null_plasmode_standard", design="plasmode", rate="standard", n=1000, B=4000, gen_seed=101,
         levels=LEVELS, fixed_B=True),
    dict(name="L2_levels_null_plasmode_low_rate", design="plasmode", rate="low_rate", n=500, B=4000, gen_seed=102,
         levels=LEVELS, fixed_B=True),
    dict(name="L3_levels_null_synthetic_balanced_34", design="synthetic_34", rate="standard", n=500, B=4000,
         gen_seed=103, levels=LEVELS, fixed_B=True),
]

SPECS = [
    # FWER and coverage under the null, real design skeleton
    dict(name="N1_null_plasmode_standard", design="plasmode", rate="standard", n=1000, B=10_000, gen_seed=1),
    # bootstrap-size check: the formal B on a subset of N1's generator
    dict(name="N1b_null_plasmode_standard_B10000", design="plasmode", rate="standard", n=300, B=10_000, gen_seed=1,
         fixed_B=True),
    dict(name="N2_null_plasmode_low_rate", design="plasmode", rate="low_rate", n=1000, B=10_000, gen_seed=2),
    # extreme imbalance: RA-dry kept in 10 (C-4 threshold) or 5 (C-4 floor) subjects only
    dict(name="N3_null_dry_10_subjects", design="plasmode_dry_10", rate="standard", n=500, B=10_000, gen_seed=3),
    dict(name="N4_null_dry_5_subjects", design="plasmode_dry_5", rate="standard", n=500, B=10_000, gen_seed=4),
    # small samples, fully synthetic design
    dict(name="N5_null_synthetic_12_subjects", design="synthetic_12", rate="standard", n=500, B=10_000, gen_seed=5),
    dict(name="N6_null_synthetic_20_imbalanced", design="synthetic_20_imbalanced", rate="standard", n=500,
         B=10_000, gen_seed=6),
    # coverage under a planted RA-dry effect (legacy arm only) and power
    dict(name="A1_effect_dry_legacy", design="plasmode", rate="standard", n=500, B=10_000, gen_seed=7,
         eff={("leg", 0): 0.4}, truth_n=400),
    # confounding: thumb-specific restoration effect, NO RA effect; primary vs the S-8 restriction
    dict(name="F1_finger_effect_no_RA_effect", design="plasmode", rate="standard", n=300, B=10_000, gen_seed=8,
         finger_eff=0.6, with_s8=True, truth_n=400),
]


CONFIRM_SPECS = [   # final code (primary level read from ra_analysis: 0.995 after Amendment 1); new generator seeds
    dict(name="K1_null_plasmode_standard", design="plasmode", rate="standard", n=800, B=4000, gen_seed=301,
         fixed_B=True),
    dict(name="K2_null_plasmode_low_rate", design="plasmode", rate="low_rate", n=400, B=4000, gen_seed=302,
         fixed_B=True),
    dict(name="K3_effect_dry_legacy", design="plasmode", rate="standard", n=300, B=4000, gen_seed=303,
         eff={("leg", 0): 0.4}, truth_n=400, fixed_B=True),
    dict(name="K4_finger_effect_no_RA_effect", design="plasmode", rate="standard", n=200, B=4000, gen_seed=304,
         finger_eff=0.6, with_s8=True, truth_n=400, fixed_B=True),
]


def run_sims(sk, workers, only=None, scale=1.0, B=None, dump=None, levels_only=False, specs=None):
    from multiprocessing import get_context
    out = {}
    with get_context("fork").Pool(workers, initializer=_init, initargs=(sk,)) as pool:
        for spec in specs or (LEVEL_SPECS if levels_only else SPECS):
            if only and spec["name"] not in only:
                continue
            t0 = time.time()
            if B and not spec.get("fixed_B"):
                spec = dict(spec, B=B)
            n = max(4, int(spec["n"] * scale))
            truth = None
            if "truth_n" in spec:
                tspec = dict(spec, full_only=True)
                tr = pool.map(one, [(dict(tspec, gen_seed=spec["gen_seed"] + 10_000), k)
                                    for k in range(max(4, int(spec["truth_n"] * scale)))])
                truth = {m: float(np.mean([r[m] for r in tr])) for m in tr[0]}
            rows = pool.map(one, [(spec, k) for k in range(n)], chunksize=1)
            if spec.get("levels"):
                out[spec["name"]] = {"spec": spec, "n": n, "level_error": level_summary(rows, spec["levels"]),
                                     "seconds": round(time.time() - t0, 1)}
                print(spec["name"], json.dumps(out[spec["name"]]["level_error"]["familywise"]), flush=True)
                if dump:
                    dump(out)
                continue
            res = {"spec": {k: (str(v) if k == "eff" else v) for k, v in spec.items()}, "n": n,
                   "primary_level": E.LEVEL_PRIMARY, "primary": summarise(rows, truth)}
            if spec.get("with_s8"):
                res["S-8a"] = summarise(rows, truth=None, lab="S-8a")
                res["S-8w"] = summarise(rows, truth=None, lab="S-8w")
                res["note"] = ("no RA effect is planted; any interval excluding 0 is finger confounding. "
                               "'truth' of the primary estimand includes the finger composition")
            res["seconds"] = round(time.time() - t0, 1)
            out[spec["name"]] = res
            print(spec["name"], json.dumps({"fwer": res["primary"]["fwer"], "s": res["seconds"]}), flush=True)
            if dump:
                dump(out)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("finger", "sim", "levels", "confirm"))
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--scale", type=float, default=1.0, help="fraction of the planned number of datasets")
    ap.add_argument("--B", type=int, default=None, help="bootstrap replicates (overrides all but fixed_B specs)")
    a = ap.parse_args(argv)
    sk = skeleton(Path(a.labels))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"labels_csv_sha256": RA.sha256_file(a.labels), "manifest_p1_sha256": RA.sha256_file(MANIFEST_P1),
            "test_list_sha256": RA.sha256_file(SPLIT_DIR / "test_cap60_files.txt"),
            "ra_analysis_sha256": RA.sha256_file(E.HERE), "ra_prefreeze_sha256": RA.sha256_file(__file__),
            "note": "metadata skeleton + synthetic outcomes; no detector output read"}
    if a.what == "finger":
        rep = {"meta": meta, **finger_report(sk)}
        (out / "finger_feasibility.json").write_text(json.dumps(rep, indent=1, default=int))
        print(json.dumps(rep, indent=1, default=int))
    else:
        name = "simulation.json" if a.scale == 1.0 and not a.only else f"simulation_partial_{a.scale}.json"
        if a.what == "levels":
            name = "level_calibration.json" if a.scale == 1.0 else f"level_calibration_partial_{a.scale}.json"
        if a.what == "confirm":
            name = "confirmation_final_code.json" if a.scale == 1.0 else f"confirmation_partial_{a.scale}.json"
            meta["primary_level"] = E.LEVEL_PRIMARY
        meta["B_override"] = a.B

        def dump(results):
            rep = {"meta": meta, "generator": SCEN, "results": results}
            (out / name).write_text(json.dumps(rep, indent=1, default=str))
        run_sims(sk, a.workers, a.only, a.scale, a.B, dump, levels_only=a.what == "levels",
                 specs=CONFIRM_SPECS if a.what == "confirm" else None)


if __name__ == "__main__":
    main()
