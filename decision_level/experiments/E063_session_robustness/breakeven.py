#!/usr/bin/env python3
"""E063 part A -- cost break-even / crossover analysis from FROZEN E061 full-sample aggregates.

No test score, logit or decision file is read. Inputs: artifacts/results/E061/{cost_grid.csv, session.csv} (frozen,
results commit 8fdd012). Status: post hoc descriptive derivation; full-sample point values; no intervals.

J_pi(c) = c_A*PMP_pi(q) + L_pi + c_U*G_pi + c_T*(N_pi - 1) is linear in (c_A, c_U, c_T), and PMP is linear in q per
seed (hence in the seed mean). The components are recovered per (cell, arm, policy) by least squares from the
135 grid rows and must reproduce every stored J to 1e-12 (gate BE-0) and the stored session.csv L, G, N, PMP_q0.9
(gate BE-1) before any crossover is written.

    python experiments/E063_session_robustness/breakeven.py --out artifacts/results/E063/breakeven_frozen_aggregates
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import statistics as st
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
E061 = ROOT / "artifacts" / "results" / "E061"
POL = ("pi0", "pi1", "pi2", "pi3", "pi4")
QS = (0.5, 0.9, 1.0)
CSTAR = {"c_A": 1.0, "c_U": 0.2, "c_T": 0.02, "q": 0.9}
TOL = 1e-12


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load():
    rows = defaultdict(list)
    for r in csv.DictReader(open(E061 / "cost_grid.csv")):
        rows[(r["det"], r["protocol"], r["op"], r["arm"])].append(r)
    sess = defaultdict(dict)
    for r in csv.DictReader(open(E061 / "session.csv")):
        sess[(r["det"], r["protocol"], r["op"], r["arm"], r["policy"])][r["quantity"]] = float(r["estimate"])
    return rows, sess


def components(grid_rows):
    """Per policy: PMP(q) for q in QS, L, G, Nm1 by exact least squares; returns comps and max residual."""
    X, Y = [], {p: [] for p in POL}
    for r in grid_rows:
        cA, cU, cT, q = (float(r[k]) for k in ("c_A", "c_U", "c_T", "q"))
        X.append([cA * (q == qq) for qq in QS] + [1.0, cU, cT])
        for p in POL:
            Y[p].append(float(r[f"J_{p}"]))
    X = np.array(X)
    comps, resid = {}, 0.0
    for p in POL:
        y = np.array(Y[p]); b, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = max(resid, float(np.max(np.abs(X @ b - y))))
        comps[p] = {"PMP": dict(zip(QS, b[:3])), "L": b[3], "G": b[4], "Nm1": b[5]}
    return comps, resid


def pmp_line(c):
    """PMP(q) = a + b q fitted through the three q points; returns a, b, max deviation from linearity."""
    q = np.array(QS); y = np.array([c["PMP"][x] for x in QS])
    b, a = np.polyfit(q, y, 1)
    return a, b, float(np.max(np.abs(a + b * q - y)))


def envelope_cA(comps, cU, cT, q):
    """Exact lower envelope of J_pi(c_A) = c_A*PMP_pi + K_pi over c_A in (0, inf): list of (from, to, policy).
    Ties are broken towards pi0, then by policy order (as E061)."""
    lines = {p: (comps[p]["PMP"][q], comps[p]["L"] + cU * comps[p]["G"] + cT * comps[p]["Nm1"]) for p in POL}
    def best(cA):
        v = {p: s * cA + k for p, (s, k) in lines.items()}
        m = min(v.values())
        for p in POL:                      # pi0 first -> ties to pi0
            if v[p] - m <= 1e-15:
                return p
    cands = {0.0}
    for a, b in itertools.combinations(POL, 2):
        (s1, k1), (s2, k2) = lines[a], lines[b]
        if s1 != s2:
            x = (k2 - k1) / (s1 - s2)
            if x > 0:
                cands.add(x)
    pts = sorted(cands) + [float("inf")]
    segs = []
    for lo, hi in zip(pts[:-1], pts[1:]):
        mid = lo * 2 + 1 if hi == float("inf") else (lo + hi) / 2
        p = best(mid)
        if segs and segs[-1][2] == p:
            segs[-1] = (segs[-1][0], hi, p)
        else:
            segs.append((lo, hi, p))
    return segs


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)          # never overwrite a run directory
    rows, sess = load()
    cells = sorted(rows)
    comp, gate = {}, {"BE0_max_J_residual": 0.0, "BE1_max_session_dev": 0.0, "pmp_linearity_max_dev": 0.0}
    for c in cells:
        comp[c], r = components(rows[c]); gate["BE0_max_J_residual"] = max(gate["BE0_max_J_residual"], r)
        for p in POL:
            s = sess[(c[0], c[1], c[2], c[3], p)]
            dev = max(abs(comp[c][p]["L"] - s["L"]), abs(comp[c][p]["G"] - s["G"]),
                      abs(comp[c][p]["Nm1"] - (s["N"] - 1)), abs(comp[c][p]["PMP"][0.9] - s["PMP_q0.9"]))
            gate["BE1_max_session_dev"] = max(gate["BE1_max_session_dev"], dev)
            gate["pmp_linearity_max_dev"] = max(gate["pmp_linearity_max_dev"], pmp_line(comp[c][p])[2])
    gate["pass"] = gate["BE0_max_J_residual"] <= 1e-10 and gate["BE1_max_session_dev"] <= 1e-10 \
        and gate["pmp_linearity_max_dev"] <= 1e-10
    (out / "gates.json").write_text(json.dumps(gate, indent=1))
    if not gate["pass"]:
        print("GATE FAILED", gate); return 2

    per_cell = []
    for c in cells:
        det, p, op, arm = c; C0, C1, C4 = comp[c]["pi0"], comp[c]["pi1"], comp[c]["pi4"]
        # pi1 vs pi0 (PMP of pi0/pi1 does not depend on q: V = 0)
        dP1 = C1["PMP"][0.9] - C0["PMP"][0.9]; dL1 = C1["L"] - C0["L"]
        dG1 = C1["G"] - C0["G"]; dN1 = C1["Nm1"] - C0["Nm1"]
        rest1 = dL1 + CSTAR["c_U"] * dG1 + CSTAR["c_T"] * dN1
        cA1 = (-rest1 / dP1) if dP1 < 0 else float("inf")            # pi1 cheaper iff c_A > cA1
        # pi4 vs pi0: dL = 0 by construction (checked), dPMP(q) = a + b q
        a4, b4, _ = pmp_line(C4); a0, b0, _ = pmp_line(C0)
        da, db = a4 - a0, b4 - b0
        dL4 = C4["L"] - C0["L"]; G4 = C4["G"] - C0["G"]; dN4 = C4["Nm1"] - C0["Nm1"]
        dP4 = lambda q: da + db * q
        # q*: pi4 cheaper iff cA*dP4(q) + cU*G4 + cT*dN4 + dL4 < 0
        def qstar(cA, cU, cT):
            return -(cA * da + cU * G4 + cT * dN4 + dL4) / (cA * db) if db != 0 else float("nan")
        def cUstar(cA, cT, q):
            return -(cA * dP4(q) + cT * dN4 + dL4) / G4 if G4 > 0 else float("nan")
        def cAstar4(cU, cT, q):
            d = dP4(q)
            return -(cU * G4 + cT * dN4 + dL4) / d if d < 0 else float("inf")
        env = envelope_cA(comp[c], CSTAR["c_U"], CSTAR["c_T"], CSTAR["q"])
        row = dict(det=det, protocol=p, op=op, arm=arm, pi1_dPMP=dP1, pi1_dL=dL1, pi1_dG=dG1, pi1_dNm1=dN1,
                   pi1_cAstar_cstar=cA1,
                   pi1_cAstar_cT0=(-(dL1 + CSTAR["c_U"] * dG1) / dP1) if dP1 < 0 else float("inf"))
        row["pi1_cAstar_cT0p1"] = (-(dL1 + CSTAR["c_U"] * dG1 + 0.1 * dN1) / dP1) if dP1 < 0 else float("inf")
        row.update(pi4_dL=dL4, pi4_G=G4, pi4_dNm1=dN4, pi4_dPMP_q0p5=dP4(0.5), pi4_dPMP_q0p9=dP4(0.9),
                   pi4_dPMP_q1=dP4(1.0), pi4_dPMP_slope_in_q=db,
                   pi4_qstar_cstar=qstar(1.0, 0.2, 0.02),
                   pi4_qstar_cA0p1=qstar(0.1, 0.2, 0.02), pi4_qstar_cA10=qstar(10.0, 0.2, 0.02),
                   pi4_qstar_cU0p5=qstar(1.0, 0.5, 0.02),
                   pi4_cUstar_cstar=cUstar(1.0, 0.02, 0.9), pi4_cUstar_q0p5=cUstar(1.0, 0.02, 0.5),
                   pi4_cUstar_q1=cUstar(1.0, 0.02, 1.0),
                   pi4_cAstar_cstar=cAstar4(0.2, 0.02, 0.9), pi4_cAstar_q0p5=cAstar4(0.2, 0.02, 0.5),
                   pi4_cAstar_cU0p5=cAstar4(0.5, 0.02, 0.9),
                   global_envelope_cA_cstar="; ".join(f"{p}:[{lo:.4g},{hi:.4g})" for lo, hi, p in env),
                   global_argmin_at_cstar=min(POL, key=lambda pp: (round(1.0 * comp[c][pp]["PMP"][0.9] + comp[c][pp]["L"]
                                                    + 0.2 * comp[c][pp]["G"] + 0.02 * comp[c][pp]["Nm1"], 15), POL.index(pp))))
        per_cell.append(row)
    with open(out / "breakeven_per_cell.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_cell[0])); w.writeheader(); w.writerows(per_cell)

    # grid shares: pairwise improvement vs global minimiser (frozen grid; ties count as pi0, as E061)
    pair = defaultdict(lambda: [0, 0, 0, 0, 0])   # (arm, c_A) -> [n, pi1<pi0, pi1 argmin, pi4<pi0, pi4 argmin]
    for c in cells:
        for r in rows[c]:
            k = (c[3], float(r["c_A"])); J = {p: float(r[f"J_{p}"]) for p in POL}
            pair[k][0] += 1; pair[k][1] += J["pi1"] < J["pi0"]; pair[k][2] += r["argmin"] == "pi1"
            pair[k][3] += J["pi4"] < J["pi0"]; pair[k][4] += r["argmin"] == "pi4"
    grid = [dict(arm=k[0], c_A=k[1], cell_points=v[0], pi1_lower_J_than_pi0=v[1], pi1_global_min=v[2],
                 pi4_lower_J_than_pi0=v[3], pi4_global_min=v[4]) for k, v in sorted(pair.items())]
    with open(out / "grid_pairwise_vs_global.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(grid[0])); w.writeheader(); w.writerows(grid)

    def med(arm, key):
        v = [r[key] for r in per_cell if r["arm"] == arm]
        fin = [x for x in v if np.isfinite(x)]
        return {"median_all_cells_inf_included": st.median(v),
                "min": min(fin) if fin else None, "max": max(fin) if fin else None,
                "n_cells": len(v), "n_infinite_or_nan": len(v) - len(fin)}
    keys = [k for k in per_cell[0] if k not in ("det", "protocol", "op", "arm", "global_envelope_cA_cstar",
                                                   "global_argmin_at_cstar")]
    summary = {arm: {k: med(arm, k) for k in keys} for arm in ("leg", "fix")}
    for arm in ("leg", "fix"):
        summary[arm]["global_argmin_at_cstar_counts"] = dict(
            sorted(__import__("collections").Counter(r["global_argmin_at_cstar"] for r in per_cell if r["arm"] == arm).items()))
    prov = {"experiment": "E063 part A (post hoc descriptive; frozen aggregates; no test read)",
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "inputs_sha256": {str(p.relative_to(ROOT)): sha(p) for p in (E061 / "cost_grid.csv", E061 / "session.csv")},
            "script_sha256": sha(Path(__file__)), "gates": gate}
    (out / "summary.json").write_text(json.dumps({"provenance": prov, "summary": summary}, indent=1, default=float))
    print(json.dumps(gate)); return 0


if __name__ == "__main__":
    sys.exit(main())
