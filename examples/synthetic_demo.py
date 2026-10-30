#!/usr/bin/env python3
"""End-to-end demonstration on synthetic, non-biometric data only.

1. A synthetic ridge-like 88 x 88 pattern (sinusoids + noise; not a fingerprint of any person) is
   degraded with the symmetric degradation bank and restored with each released operator; the
   normalization dose rho of each operator is printed.
2. Simulated detector scores for two normalization policies (no model, no real data) are evaluated with
   the deployment vector at a validation-locked threshold and with the paired policy contrast.
Every number printed is a property of the simulation, not a result of the paper.

    python examples/synthetic_demo.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from detector_attribution.degradation import apply_degradation          # noqa: E402
from detector_attribution.deployment import deployment_vector            # noqa: E402
from detector_attribution.normalization import get_normalizer, range_ratio  # noqa: E402
from detector_attribution.operators import RESTORERS, bandstop           # noqa: E402
from detector_attribution.paired_bootstrap import paired_contrast        # noqa: E402

rng = np.random.default_rng(0)
yy, xx = np.mgrid[0:88, 0:88]
pattern = 128 + 60 * np.sin(2 * np.pi * (xx * np.cos(0.5) + yy * np.sin(0.5)) / 9.0)
img = np.clip(pattern + rng.normal(0, 8, pattern.shape), 0, 255).astype(np.uint8)
deg = apply_degradation(img, "noise", 2, np.random.default_rng(1))
print("1) operators on a synthetic ridge pattern (rho = fraction of dynamic range kept)")
for name, op in RESTORERS.items():
    out = op([deg])[0]
    print(f"   {name:9s} rho = {range_ratio(deg, out):.3f}   "
          f"minmax range = [{get_normalizer('minmax')(out).min():.2f}, {get_normalizer('minmax')(out).max():.2f}]")
print(f"   bandstop  rho = {range_ratio(deg, bandstop(deg, 0.2, 9.0)):.3f}")

print("2) deployment vector and paired contrast on simulated scores")
n_real, n_syn, S = 400, 400, 5
y = np.r_[np.zeros(n_real, int), np.ones(n_syn, int)]
clusters = np.r_[np.repeat(np.arange(80), 5), np.zeros(n_syn, int)]
gens = np.r_[np.zeros(n_real, int), np.repeat(np.arange(4), n_syn // 4)]
arms = {}
for arm, shift in (("legacy", 0.8), ("fixed", 0.1)):   # simulated: a restoration shift that depends on the policy
    thr, A, C = [], [], []
    for s in range(S):
        r = np.random.default_rng(100 + s)
        val = r.normal(0, 1, 300)
        a = np.r_[r.normal(0, 1, n_real), r.normal(1.5, 1, n_syn)]
        c = a + shift + r.normal(0, 0.2, a.size)
        thr.append(np.quantile(val, 0.99, method="higher")); A.append(a); C.append(c)
        if s == 0:
            v = deployment_vector(val, val + shift, y, a, c)
            print(f"   {arm:6s} seed 0: " + ", ".join(f"{k}={v[k]:+.3f}" for k in
                  ("d_auc", "d_tdr_locked", "d_rejection", "dtau", "tdr_recal")))
    arms[arm] = {"thr": np.array(thr), "anchor": np.array(A), "cond": np.array(C)}
res = paired_contrast(arms, clusters, gens, y, n_boot=300, seed=1)
print(f"   DDT = {res['DDT']:+.3f} [{res['DDT_ci'][0]:+.3f}, {res['DDT_ci'][1]:+.3f}]   "
      f"DDB = {res['DDB']:+.3f} [{res['DDB_ci'][0]:+.3f}, {res['DDB_ci'][1]:+.3f}]  (simulation only)")
