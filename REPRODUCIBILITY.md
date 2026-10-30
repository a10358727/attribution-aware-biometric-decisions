# What this repository can and cannot reproduce

Verified on 2026-10-31 in a clean clone, with no access to fingerprint images, per-image scores or model weights.

## Can be run and checked

| Item | How | Verified result |
|---|---|---|
| Table I, per-arm ΔAUC, ΔT, ΔB, T(recal) (12 configurations × 2 policies) | `scripts/table1_from_aggregates.py` | 91 of 96 values equal Table I at 3 decimals; the other 5 differ by 0.001 in the last digit, because Table I was rounded from frozen 4-decimal values (double rounding) |
| Table I, paired contrasts ΔΔT, ΔΔB with 95 % intervals | same script (from `data/aggregate/policy_contrasts.csv`) | equal to Table I in all 12 cells |
| E060 outcome and headline values (Table III, §VI) | `decision_level/scripts/headline_from_aggregates.py` | outcome U-A; ΔL3 (legacy) +0.044, 97.5 % [0.028, 0.060]; ΔΔL3 −0.026 [−0.040, −0.013]; L3 0.0026 → 0.046 |
| E061 outcome and headline values (Table IV, §VII) | same script | outcome P-A; ΔJ(π1 − π0) legacy +0.030, 98.75 % [0.014, 0.049]; ΔΔJ −0.014 [−0.027, −0.004]; 0 of 96 strict Pareto improvements |
| E060 / E061 analysis core | `python -m pytest decision_level/tests` | 11 synthetic closed-form checks pass |
| E060 / E061 analysis sources | `sha256sum` (table in `README.md`) | byte-identical to the versions pinned for the test reads |
| Normalization policies, degradation bank, restoration operators, five deployment quantities, paired-contrast estimator | `examples/synthetic_demo.py` | run on synthetic data (illustration, not a paper result) |
| Detector architectures | `inference/score_patch.py` | build and run on a synthetic 88 × 88 input with random weights (checked 2026-10-17 for the earlier detector-level release; file unchanged) |
| Training settings of the fingerprint detector runs | `configs/` | copied unchanged from the run records (and from the earlier detector-level release) |

## Cannot be reproduced from this repository

| Item | Why |
|---|---|
| Training any detector; any per-image score | fingerprint corpus and weights not released |
| Locked thresholds, Δτ and its ceiling counts, bootstrap intervals and paired contrasts from scores | need per-image scores; only frozen aggregates are included |
| Re-running E060 / E061 | needs per-image score dumps and the detector-level paired-contrast module of the research code |
| Fig. 2 (score distributions) | needs per-image scores |
| Face and public-fingerprint studies | third-party corpora; study code not included |
| D3 detector | third-party code without a licence file |
