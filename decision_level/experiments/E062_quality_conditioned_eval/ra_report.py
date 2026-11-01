"""E062 reporting: outcome-independent Supplementary tables and reading rules (Amendment 1 rev 3, D-28/D-29).

Takes the aggregate dict returned by `ra_analysis.run_all` and renders the pre-specified tables and the mechanical
readings. It never reads data and never chooses what to report: every analysis and every primary contrast is printed,
in a fixed order, whatever its value. Written and tested on synthetic output only, before any E062 outcome exists.

Reading rules
-------------
Statistical reading (v2): "above 0" / "below 0" / "includes 0" / "not reported" / "uninformative (floor)".

Practical reading (D-28; descriptive, not a hypothesis test). Reference scale R = 1 percentage point, the nominal
bona-fide rejection the validation-locked operating point is built to allow (E023 rule: 99th percentile of the
validation bona-fide scores; validation rejection 17/1,775 = 0.96 %, E061 record). For a contrast with interval [lo, hi]:
  P1  the whole interval lies beyond +R or beyond -R: "difference at least the size of the operating-point reference scale"
  P2  the interval excludes 0 but not ±R:              "difference detected; size relative to the reference scale unresolved"
  P3  the interval lies inside (-R, +R):                "any difference is smaller than the reference scale at this precision"
  P4  the interval includes 0 and reaches ±R:           "inconclusive about practical size"
R is an operating-point reference scale for reading effect sizes: it is not an acceptable-risk threshold for any
deployment and not an observed login-failure rate; E062 rates are capture-level detector rejections of synthetic
degradation/restoration conditions.

Primary vs S-8w concordance (D-29), per contrast and arm:
  C-same      both exclude 0 in the same direction
  C-null      both include 0
  C-primary   primary excludes 0, S-8w includes 0
  C-within    primary includes 0, S-8w excludes 0
  C-opposite  both exclude 0 in opposite directions
  C-na        either is not reported or floor-uninformative
"""
from __future__ import annotations

R_REF = 0.01                     # operating-point reference scale (1 percentage point)
ARMS = ("leg", "fix")
CONTRAST_NAMES = ("dry", "wet")
CLASSES = ("RA-dry", "RA-middle", "RA-wet")
ORDER = ("primary", "S-1", "S-2", "S-3a_sev12", "S-3b_sev34", "S-4", "S-5", "S-6_bil", "S-6_tv", "S-7",
         "S-8w_dry", "S-8w_wet")
LABEL = {"leg": "legacy (per-image min–max)", "fix": "fixed (/255)",
         "dry": "θ^dry = ΔB(RA-dry) − ΔB(RA-middle)", "wet": "θ^wet = ΔB(RA-wet) − ΔB(RA-middle)"}

PRACTICAL = {
    "P1": "difference at least the size of the operating-point reference scale (1 pp)",
    "P2": "difference detected; size relative to the 1 pp reference scale unresolved",
    "P3": "any difference is smaller than the 1 pp reference scale at this precision",
    "P4": "inconclusive about practical size",
    "P-na": "not interpretable (interval not reported or floor)",
}

CONCORDANCE = {
    "C-same": "difference detected in the primary analysis and within credentials (S-8w), same direction",
    "C-null": "no difference detected in the primary analysis or within credentials, at this precision",
    "C-primary": "primary and S-8w disagree: the primary difference is not detected within credentials; compatible "
                 "with between-subject or finger composition, not resolved by E062",
    "C-within": "primary and S-8w disagree: a within-credential difference is detected that the primary does not "
                "show; not resolved by E062",
    "C-opposite": "primary and S-8w disagree in direction; not resolved by E062",
    "C-na": "not interpretable (an interval is not reported or floor-uninformative)",
}


def _usable(j: dict) -> bool:
    return bool(j) and j.get("lo") is not None and not j.get("floor_uninformative", False)


def practical_reading(j: dict, r: float = R_REF) -> str:
    if not _usable(j):
        return "P-na"
    lo, hi = j["lo"], j["hi"]
    if lo >= r or hi <= -r:
        return "P1"
    if lo > 0 or hi < 0:
        return "P2"
    if lo > -r and hi < r:
        return "P3"
    return "P4"


def _dir(j: dict) -> int:
    return 1 if j["lo"] > 0 else (-1 if j["hi"] < 0 else 0)


def concordance(primary: dict, within: dict) -> str:
    if not (_usable(primary) and _usable(within)):
        return "C-na"
    a, b = _dir(primary), _dir(within)
    if a == 0 and b == 0:
        return "C-null"
    if a != 0 and b == 0:
        return "C-primary"
    if a == 0 and b != 0:
        return "C-within"
    return "C-same" if a == b else "C-opposite"


def pp(x, nd=2) -> str:
    return "—" if x is None else f"{100 * x:+.{nd}f}"


def ci(j: dict, nd=2) -> str:
    if not j or j.get("lo") is None:
        return "not reported"
    return f"[{pp(j['lo'], nd)}, {pp(j['hi'], nd)}]"


def table_groups(res: dict) -> list[str]:
    g = res["primary"]["groups"]
    rows = ["| RA class | Captures | Subjects | Credentials |", "|---|---|---|---|"]
    for c in CLASSES:
        v = g[f"p1|{c}"]
        rows.append(f"| {c} | {v['captures']} | {v['subjects']} | {v['credentials']} |")
    return rows


def table_descriptive(res: dict) -> list[str]:
    s = res["primary"]["secondary"]
    rows = ["| Arm | RA class | Anchor B (%) | Restored B (%) | ΔB (pp) [95 %] |", "|---|---|---|---|---|"]
    for arm in ARMS:
        for c in CLASSES:
            a, r, d = s[f"anc|{arm}|{c}"], s[f"res|{arm}|{c}"], s[f"dB|{arm}|{c}"]
            fa = "—" if a.get("median") is None else f"{100 * a['median']:.2f}"
            fr = "—" if r.get("median") is None else f"{100 * r['median']:.2f}"
            rows.append(f"| {LABEL[arm]} | {c} | {fa} | {fr} | {pp(d.get('median'))} {ci(d)} |")
    return rows


def table_primary(res: dict) -> list[str]:
    p = res["primary"]["primary"]
    rows = ["| Contrast | Arm | Cells | Median (pp) | 99.5 % CI (pp) | Statistical reading | Practical reading | "
            "S-8w median (pp) | S-8w 99.5 % CI | Concordance |", "|---|---|---|---|---|---|---|---|---|---|"]
    for name in CONTRAST_NAMES:
        for arm in ARMS:
            j = p[f"theta_{name}|{arm}"]
            w = res.get(f"S-8w_{name}", {}).get("primary", {}).get(f"theta_{name}|{arm}", {})
            rows.append(f"| {LABEL[name]} | {LABEL[arm]} | {j.get('n_cells', 0)} | {pp(j.get('median'))} | {ci(j)} | "
                        f"{j.get('reading', '—')} | {practical_reading(j)} | {pp(w.get('median'))} | {ci(w)} | "
                        f"{concordance(j, w)} |")
    return rows


def table_secondary(res: dict) -> list[str]:
    s = res["primary"]["secondary"]
    rows = ["| Quantity | Median (pp) | 97.5 % CI (pp) | Reading |", "|---|---|---|---|"]
    keys = [f"dtheta_{n}|fix-leg" for n in CONTRAST_NAMES] + [f"wet_minus_dry|{a}" for a in ARMS]
    for k in keys:
        j = s[k]
        rd = "not reported" if j.get("lo") is None else ("above 0" if j["lo"] > 0 else ("below 0" if j["hi"] < 0 else "includes 0"))
        rows.append(f"| {k} | {pp(j.get('median'))} | {ci(j)} | {rd} |")
    return rows


def table_sensitivity(res: dict) -> list[str]:
    rows = ["| Analysis | θ^dry legacy | θ^dry fixed | θ^wet legacy | θ^wet fixed |", "|---|---|---|---|---|"]
    for lab in ORDER[1:]:
        if lab not in res:
            rows.append(f"| {lab} | not run | not run | not run | not run |")
            continue
        out = res[lab]["primary"]
        only = res[lab].get("report_only")
        cells = []
        for name in CONTRAST_NAMES:
            for arm in ARMS:
                k = f"theta_{name}|{arm}"
                if only is not None and k not in only:
                    cells.append("n/a (not an estimand)")
                    continue
                j = out[k]
                tag = " (floor)" if j.get("floor_uninformative") else ""
                cells.append(f"{pp(j.get('median'))} {ci(j)}{tag}")
        rows.append(f"| {lab} | " + " | ".join(cells) + " |")
    return rows


def readings(res: dict) -> list[str]:
    p = res["primary"]["primary"]
    out = []
    for name in CONTRAST_NAMES:
        for arm in ARMS:
            j = p[f"theta_{name}|{arm}"]
            w = res.get(f"S-8w_{name}", {}).get("primary", {}).get(f"theta_{name}|{arm}", {})
            c = concordance(j, w)
            out.append(f"- {LABEL[name]}, {LABEL[arm]}: {CONCORDANCE[c]}; practical: {PRACTICAL[practical_reading(j)]}.")
    return out


def render(res: dict) -> str:
    parts = ["### Table S-E1. Test-pool composition by RA class", *table_groups(res), "",
             "### Table S-E2. Bona-fide rejection by RA class (median over eligible headline cells)",
             *table_descriptive(res), "",
             "### Table S-E3. Primary contrasts (computed per cell, then median over cells)", *table_primary(res), "",
             "### Table S-E4. Secondary quantities (paired per cell, then median)", *table_secondary(res), "",
             "### Table S-E5. Pre-specified sensitivity analyses (median, 99.5 % CI, pp)", *table_sensitivity(res), "",
             "### Mechanical readings", *readings(res)]
    return "\n".join(parts) + "\n"
