#!/usr/bin/env python3
"""Recompute the per-arm columns of Table I (detector-level deployment quantities) from the released
aggregates, and list the paired fixed-minus-legacy contrasts with their 95 % intervals.

Inputs: data/aggregate/headline_per_seed.csv (one row per detector, protocol, normalization policy, training
seed and condition; test-pool aggregates, no per-image score) and data/aggregate/policy_contrasts.csv (frozen
paired-bootstrap contrasts). Output: seed means of dAUC, dT_lock, dB_lock and T_recal for the twelve headline
configurations under both policies, then ddT and ddB with intervals. The d-tau column of Table I and the
count of seeds at the score ceiling are not recomputed (they need per-image scores).

    python scripts/table1_from_aggregates.py
"""
import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

AGG = Path(__file__).resolve().parents[1] / "data" / "aggregate"
rows = list(csv.DictReader((AGG / "headline_per_seed.csv").open()))
cells = defaultdict(list)
for r in rows:
    if r["condition"] in ("restored_bilateral", "restored_tv"):
        cells[(r["detector"], r["protocol"], r["condition"], r["policy"])].append(r)
print(f"{'detector':22} {'prot':4} {'operator':9} {'policy':7} {'dAUC':>8} {'dT_lock':>8} {'dB_lock':>8} {'T_recal':>8}  seeds")
for (det, prot, cond, pol), rs in sorted(cells.items()):
    m = lambda k: st.mean(float(x[k]) for x in rs)
    print(f"{det:22} {prot:4} {cond.replace('restored_', ''):9} {pol:7} {m('d_auc'):+8.4f} {m('d_tdr_locked'):+8.4f} "
          f"{m('d_rejection'):+8.4f} {m('tdr_recal'):8.4f}  {len(rs)}")

print(f"\n{'detector':22} {'prot':4} {'operator':9} {'ddT [95 %]':>26} {'ddB [95 %]':>26}")
seen = set()
for r in csv.DictReader((AGG / "policy_contrasts.csv").open()):
    key = (r["detector"], r["protocol"], r["operator"])
    if r["role"] != "headline" or key in seen:
        continue
    seen.add(key)
    f = lambda k: float(r[k])
    print(f"{r['detector']:22} {r['protocol']:4} {r['operator']:9} "
          f"{f('DDT'):+.3f} [{f('DDT_lo'):+.3f}, {f('DDT_hi'):+.3f}]   {f('DDB'):+.3f} [{f('DDB_lo'):+.3f}, {f('DDB_hi'):+.3f}]")
