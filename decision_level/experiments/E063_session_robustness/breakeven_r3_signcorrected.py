#!/usr/bin/env python3
"""E063 Part A, r3: sign-corrected cost break-even from the frozen E061 cost grid (post hoc; descriptive; no test read).

Why r3: breakeven.py (r2, frozen in ad1e976 and kept unchanged) sets a pi1 crossover to "inf" whenever the pi1 change in
pre-matcher progression is >= 0 and reports such cells as "never cheaper". That is wrong when the remainder of the cost
difference is negative: pi1 is then cheaper than pi0 BELOW the crossover (fixed D3-P1-TV). r3 classifies every
pairwise comparison exactly, for every policy pi1..pi4 against pi0, and along c_A, q and c_U.

Independence from r2: components are recovered here directly from the E061 cost_grid.csv by exact least squares on the
design J = c_A*a + c_A*q*b + L + c_U*G + c_T*(N-1) (PMP = a + b q is linear in q), then checked against E061
session.csv. Every classification is then checked against the sign of the stored J differences at all 135 grid points
of every cell and arm, and the stored argmin is reproduced. Any mismatch stops the script (non-zero exit, nothing is
labelled as a result).

Values are full-sample point estimates. No interval is computed: intervals would need bootstrap replicates and a test
read, which E063 Part A does not do.

    python experiments/E063_session_robustness/breakeven_r3_signcorrected.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import statistics as st
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GRID = ROOT / "artifacts/results/E061/cost_grid.csv"
SESSION = ROOT / "artifacts/results/E061/session.csv"
R2 = ROOT / "artifacts/results/E063/breakeven_frozen_aggregates_r2/breakeven_per_cell.csv"
OUT = ROOT / "artifacts/results/E063/breakeven_frozen_aggregates_r3_signcorrected"
POL = ["pi0", "pi1", "pi2", "pi3", "pi4"]
CSTAR = {"c_A": 1.0, "c_U": 0.2, "c_T": 0.02, "q": 0.9}
EPS = 1e-10          # component differences below this are numerical zeros (e.g. pi1 dG, pi4 dL ~ 1e-15)
TIE = 1e-12          # stored J differences below this are ties (pi3 == pi0 exactly in some cells)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def classify(s: float, r: float, lo: float, hi: float):
    """Where is dJ(x) = s*x + r < 0 on the domain [lo, hi]?  Returns (case, crossover or None)."""
    s = 0.0 if abs(s) < EPS else s
    r = 0.0 if abs(r) < EPS else r
    if s == 0.0:
        return ("tie" if r == 0.0 else ("always_cheaper" if r < 0 else "never_cheaper")), None
    x = -r / s
    if s < 0:                                   # cheaper above x
        if x <= lo: return "always_cheaper", None
        if x >= hi: return "never_cheaper", None
        return "cheaper_above", x
    if x >= hi: return "always_cheaper", None   # cheaper below x
    if x <= lo: return "never_cheaper", None
    return "cheaper_below", x


def envelope(lines: dict, lo=1e-6, hi=1e6):
    """Exact lower envelope over c_A of J_pi = c_A*m_pi + k_pi; ties to the lower policy index (E061 rule: ties to pi0)."""
    pts = {lo, hi}
    names = list(lines)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            (ma, ka), (mb, kb) = lines[a], lines[b]
            if abs(ma - mb) > 1e-15:
                x = (kb - ka) / (ma - mb)
                if lo < x < hi: pts.add(x)
    pts = sorted(pts)
    segs = []
    for x0, x1 in zip(pts[:-1], pts[1:]):
        xm = np.sqrt(x0 * x1)
        J = {p: lines[p][0] * xm + lines[p][1] for p in names}
        best = min(J.values())
        win = next(p for p in names if J[p] - best <= TIE)
        if segs and segs[-1][2] == win: segs[-1][1] = x1
        else: segs.append([x0, x1, win])
    return segs


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=False)
    rows = list(csv.DictReader(open(GRID)))
    by = defaultdict(list)
    for r in rows:
        by[(r["det"], r["protocol"], r["op"], r["arm"])].append(r)
    sess = {}
    for r in csv.DictReader(open(SESSION)):
        sess[(r["det"], r["protocol"], r["op"], r["arm"], r["policy"], r["quantity"])] = float(r["estimate"])

    gates = {"lsq_max_residual": 0.0, "session_max_dev": 0.0, "grid_sign_mismatches": 0, "argmin_mismatches": 0,
             "n_grid_rows": len(rows), "n_cells": len(by)}
    comp, per_cell, env_rows = {}, [], []
    for cell, rs in sorted(by.items()):
        X = np.array([[float(r["c_A"]), float(r["c_A"]) * float(r["q"]), 1.0, float(r["c_U"]), float(r["c_T"])]
                      for r in rs])
        for p in POL:
            y = np.array([float(r[f"J_{p}"]) for r in rs])
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            gates["lsq_max_residual"] = max(gates["lsq_max_residual"], float(np.abs(X @ coef - y).max()))
            a, b, L, G, Nm1 = coef
            comp[cell + (p,)] = dict(a=a, b=b, L=L, G=G, Nm1=Nm1)
            chk = {"L": L, "G": G, "N": Nm1 + 1, "PMP_q0.9": a + 0.9 * b}
            for q, v in chk.items():
                k = cell + (p, q)
                if k in sess:
                    gates["session_max_dev"] = max(gates["session_max_dev"], abs(sess[k] - v))
        # grid cross-check: sign of every stored pairwise difference and the stored argmin
        for r in rs:
            cA, cU, cT, q = (float(r[k]) for k in ("c_A", "c_U", "c_T", "q"))
            Jhat = {}
            for p in POL:
                c = comp[cell + (p,)]
                Jhat[p] = cA * (c["a"] + c["b"] * q) + c["L"] + cU * c["G"] + cT * c["Nm1"]
            for p in POL[1:]:
                stored = float(r[f"J_{p}"]) - float(r["J_pi0"])
                pred = Jhat[p] - Jhat["pi0"]
                ss = 0 if abs(stored) < TIE else (1 if stored > 0 else -1)
                sp = 0 if abs(pred) < TIE else (1 if pred > 0 else -1)
                gates["grid_sign_mismatches"] += ss != sp
            best = min(Jhat.values())
            am = next(p for p in POL if Jhat[p] - best <= TIE)
            gates["argmin_mismatches"] += am != r["argmin"]
        # pairwise classification along each axis, other weights at c*
        c0 = comp[cell + ("pi0",)]
        for p in POL[1:]:
            c = comp[cell + (p,)]
            d = {k: c[k] - c0[k] for k in c}
            dPMPq = lambda q: d["a"] + d["b"] * q
            rest = d["L"] + CSTAR["c_U"] * d["G"] + CSTAR["c_T"] * d["Nm1"]
            dJ_cstar = CSTAR["c_A"] * dPMPq(CSTAR["q"]) + rest
            out = dict(det=cell[0], protocol=cell[1], op=cell[2], arm=cell[3], policy=p,
                       dPMP_q09=dPMPq(0.9), dL=d["L"], dG=d["G"], dNm1=d["Nm1"], dJ_cstar=dJ_cstar,
                       cheaper_than_pi0_at_cstar=bool(dJ_cstar < -TIE))
            out["cA_case"], out["cA_cross"] = classify(dPMPq(CSTAR["q"]), rest, 0.0, float("inf"))
            out["q_case"], out["q_cross"] = classify(CSTAR["c_A"] * d["b"], CSTAR["c_A"] * d["a"] + rest, 0.0, 1.0)
            out["cU_case"], out["cU_cross"] = classify(
                d["G"], CSTAR["c_A"] * dPMPq(CSTAR["q"]) + d["L"] + CSTAR["c_T"] * d["Nm1"], 0.0, float("inf"))
            per_cell.append(out)
        lines = {p: (comp[cell + (p,)]["a"] + comp[cell + (p,)]["b"] * CSTAR["q"],
                     comp[cell + (p,)]["L"] + CSTAR["c_U"] * comp[cell + (p,)]["G"]
                     + CSTAR["c_T"] * comp[cell + (p,)]["Nm1"]) for p in POL}
        for x0, x1, w in envelope(lines):
            env_rows.append(dict(det=cell[0], protocol=cell[1], op=cell[2], arm=cell[3], cA_from=x0, cA_to=x1,
                                 global_min=w))

    gates["pass"] = (gates["lsq_max_residual"] < 1e-9 and gates["session_max_dev"] < 1e-9
                     and gates["grid_sign_mismatches"] == 0 and gates["argmin_mismatches"] == 0)
    (OUT / "gates.json").write_text(json.dumps(gates, indent=1))
    if not gates["pass"]:
        print("GATE FAILURE", gates); return 2

    # r2 consistency: every r2 pi1 c_A* that is finite must equal the r3 cheaper_above crossover; list r2 relabels
    r2_cmp = []
    for r in csv.DictReader(open(R2)):
        o = next(x for x in per_cell if x["policy"] == "pi1" and (x["det"], x["protocol"], x["op"], x["arm"])
                 == (r["det"], r["protocol"], r["op"], r["arm"]))
        v2 = float(r["pi1_cAstar_cstar"])
        same = (o["cA_case"] == "cheaper_above" and np.isfinite(v2) and abs(v2 - o["cA_cross"]) < 1e-9 * max(1, v2)) \
            or (o["cA_case"] == "never_cheaper" and not np.isfinite(v2))
        r2_cmp.append(dict(cell=f'{r["det"]}-{r["protocol"]}-{r["op"]}-{r["arm"]}', r2=r["pi1_cAstar_cstar"],
                           r3_case=o["cA_case"], r3_cross=o["cA_cross"], agrees=bool(same)))

    # grid tallies from the stored grid (no model)
    tally = defaultdict(lambda: defaultdict(int))
    for r in rows:
        k = (r["arm"], float(r["c_A"]))
        tally[k]["cell_points"] += 1
        tally[k][f"argmin_{r['argmin']}"] += 1
        for p in POL[1:]:
            if float(r[f"J_{p}"]) - float(r["J_pi0"]) < -TIE:
                tally[k][f"{p}_lower_than_pi0"] += 1

    def summ(arm, pol, axis, case):
        v = [x[f"{axis}_cross"] for x in per_cell if x["arm"] == arm and x["policy"] == pol and x[f"{axis}_case"] == case]
        return {"n_cells": len(v), "median": st.median(v) if v else None, "min": min(v) if v else None,
                "max": max(v) if v else None}

    def cases(arm, pol, axis):
        c = defaultdict(list)
        for x in per_cell:
            if x["arm"] == arm and x["policy"] == pol:
                c[x[f"{axis}_case"]].append(f'{x["det"]}-{x["protocol"]}-{x["op"]}')
        return dict(c)

    summary = {}
    for arm in ("leg", "fix"):
        s = {}
        for pol in POL[1:]:
            s[pol] = {ax: {"cases": cases(arm, pol, ax),
                           "cheaper_above": summ(arm, pol, ax, "cheaper_above"),
                           "cheaper_below": summ(arm, pol, ax, "cheaper_below")} for ax in ("cA", "q", "cU")}
            s[pol]["cells_cheaper_than_pi0_at_cstar"] = [f'{x["det"]}-{x["protocol"]}-{x["op"]}' for x in per_cell
                                                         if x["arm"] == arm and x["policy"] == pol
                                                         and x["cheaper_than_pi0_at_cstar"]]
            s[pol]["median_dJ_cstar_point"] = st.median(x["dJ_cstar"] for x in per_cell
                                                         if x["arm"] == arm and x["policy"] == pol)
        sw = [e["cA_to"] for e in env_rows if e["arm"] == arm and e["global_min"] != "pi4"
              and any(f["global_min"] == "pi4" and f["cA_from"] == e["cA_to"] and f["arm"] == arm
                      and (f["det"], f["protocol"], f["op"]) == (e["det"], e["protocol"], e["op"]) for f in env_rows)]
        s["envelope_switch_to_pi4_cA"] = {"n_cells": len(sw), "min": min(sw) if sw else None,
                                          "max": max(sw) if sw else None,
                                          "median": st.median(sw) if sw else None}
        s["envelope_policies"] = sorted({e["global_min"] for e in env_rows if e["arm"] == arm})
        s["pi1_ever_global_min_on_envelope"] = any(e["global_min"] == "pi1" for e in env_rows if e["arm"] == arm)
        summary[arm] = s

    def wcsv(name, rs):
        with open(OUT / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rs[0])); w.writeheader(); w.writerows(rs)

    wcsv("pairwise_per_cell.csv", per_cell)
    wcsv("envelope_cA_at_cstar.csv", env_rows)
    wcsv("grid_tallies.csv", [dict(arm=a, c_A=c, **dict(v)) for (a, c), v in sorted(tally.items())])
    wcsv("r2_comparison.csv", r2_cmp)
    prov = {"experiment": "E063 Part A r3 (post hoc; descriptive; frozen aggregates only; no test read)",
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "script_sha256": sha(Path(__file__)),
            "inputs_sha256": {str(p.relative_to(ROOT)): sha(p) for p in (GRID, SESSION, R2)},
            "python": sys.version.split()[0], "numpy": np.__version__,
            "supersedes_labels_of": "breakeven_frozen_aggregates_r2 (kept unchanged)", "cstar": CSTAR,
            "eps_component": EPS, "tie": TIE}
    (OUT / "summary.json").write_text(json.dumps({"provenance": prov, "gates": gates, "summary": summary,
                                                  "r2_disagreements": [x for x in r2_cmp if not x["agrees"]]},
                                                 indent=1, default=float))
    (OUT / "README.txt").write_text(
        "E063 Part A r3: sign-corrected pairwise cost break-even from the frozen E061 cost grid (post hoc, descriptive).\n"
        "Full-sample point estimates; NO confidence intervals. No test score read. r2 is kept unchanged; r3 corrects\n"
        "its 'never cheaper' label for cells where pi1 is cheaper than pi0 below a crossover. Gates in gates.json.\n")
    print(json.dumps(gates)); print("r2 disagreements:", [x for x in r2_cmp if not x["agrees"]])
    return 0


if __name__ == "__main__":
    sys.exit(main())
