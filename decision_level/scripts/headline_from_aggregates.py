#!/usr/bin/env python3
"""Print the headline E060 / E061 quantities of the TCSS manuscript from the frozen aggregate results in
tcss_cpsi/results/. Reads aggregates only; recomputes nothing from scores (none are released).

    python tcss_cpsi/scripts/headline_from_aggregates.py
"""
import csv
import json
import statistics
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results"


def fmt(x, nd=3):
    return f"{x:+.{nd}f}"


def ci(v, key):
    lo, hi = v[key]
    return f"[{lo:+.3f}, {hi:+.3f}]"


def main():
    o60 = json.loads((R / "E060" / "OUTCOME.json").read_text())
    s60 = json.loads((R / "E060" / "joint_summary.json").read_text())
    print(f"E060 outcome: {o60['outcome']}  (rule: {o60['rule']})")
    for k, lab in (("L3|leg", "dL3 legacy (H1)"), ("L3|fix", "dL3 fixed"), ("L3|dd", "ddL3 fixed-legacy (H2)"),
                   ("B_cred|leg", "dB_cred legacy"), ("C50|leg", "dC50 legacy"), ("L3_M1|leg", "dL3 legacy, M1")):
        v = s60[k]
        print(f"  {lab:24s} median {fmt(v['median'])}  95% {ci(v, 'ci95')}  97.5% {ci(v, 'ci975')}  cells={v['n_cells']}")
    # anchor and restored levels of L3 (median over the twelve cells), as in Table III
    rows = list(csv.DictReader(open(R / "E060" / "cells_modelled.csv")))
    for arm in ("leg", "fix"):
        lv = {kind: statistics.median(float(r["estimate"]) for r in rows
                                      if r["endpoint"] == "L3" and r["arm"] == arm and r["kind"] == kind)
              for kind in ("level_anchor", "level_restored")}
        print(f"  L3 level ({arm}): anchor {lv['level_anchor']:.4f} -> restored {lv['level_restored']:.4f}")

    o61 = json.loads((R / "E061" / "OUTCOME.json").read_text())
    s61 = json.loads((R / "E061" / "joint_summary.json").read_text())
    print(f"\nE061 outcome: {o61['outcome']}  (decision level {o61['decision_level']}; c* = {o61['c_star']})")
    print(f"  H3 classes: {o61['H3_class']}   H4: {o61['H4_supported']}")
    for pol in ("pi1", "pi2", "pi3", "pi4"):
        for arm in ("leg", "fix", "dd"):
            v = s61[f"dJ|{pol}|{arm}"]
            print(f"  dJ({pol}-pi0) {arm:3s} median {fmt(v['median'])}  95% {ci(v, 'ci95')}  "
                  f"{100 * v['decision_level']:.2f}% {ci(v, 'ci_decision')}")
    strict = sum(r["strict_pareto_improvement"] == "True" for r in csv.DictReader(open(R / "E061" / "pareto_strict.csv")))
    print(f"  strict Pareto improvements: {strict} of 96 policy-arm-cell comparisons")


if __name__ == "__main__":
    main()
