# What this repository can and cannot reproduce

Verified on 2026-10-31 (v1.0.0), 2026-11-02 (v1.1.0 release candidate) and 2026-11-24 (v1.1.0) in a clean clone, with no access to
fingerprint images, per-image labels, per-image scores or model weights.

## Can be run and checked

| Item | How | Verified result |
|---|---|---|
| Table I, per-arm ΔAUC, ΔT, ΔB, T(recal) (12 configurations × 2 policies) | `scripts/table1_from_aggregates.py` | 91 of 96 values equal Table I at 3 decimals; the other 5 differ by 0.001 in the last digit, because Table I was rounded from frozen 4-decimal values (double rounding) |
| Table I, paired contrasts ΔΔT, ΔΔB with 95 % intervals | same script (from `data/aggregate/policy_contrasts.csv`) | equal to Table I in all 12 cells |
| E060 outcome and headline values (Table III, §VI) | `decision_level/scripts/headline_from_aggregates.py` | outcome U-A; ΔL3 (legacy) +0.044, 97.5 % [0.028, 0.060]; ΔΔL3 −0.026 [−0.040, −0.013]; L3 0.0026 → 0.046 |
| E061 outcome and headline values (Table IV, §VII) | same script | outcome P-A; ΔJ(π1 − π0) legacy +0.030, 98.75 % [0.014, 0.049]; ΔΔJ −0.014 [−0.027, −0.004]; 0 of 96 strict Pareto improvements |
| E060 / E061 analysis core and E062 code | `python -m pytest decision_level/tests` | 48 synthetic checks pass (11 + 37) |
| E062 Supplementary Tables S14–S18 (primary, secondary, sensitivity S-1…S-7, S-8w) | `decision_level/scripts/e062_tables_from_aggregates.py` | identical, character by character, to the tables written by the authorised run |
| E062 method end to end | `decision_level/examples/e062_synthetic_demo.py` | grouping rule, estimator and tables run on synthetic data (illustration, not a paper result) |
| Pinned analysis sources (E060, E061, E062) | `python scripts/verify_frozen_sources.py` | byte-identical to the authorised / frozen versions |
| Privacy gate | `python scripts/privacy_check.py` | 0 problems |
| C1 macros and tables of §VIII, §X and Suppl. S7–S9 (E063–E071) | `decision_level/scripts/c1_check_macros.py` | all 7 files identical (comment lines excepted) to the manuscript's, from the released aggregates alone |
| E063–E071 analysis code | `python -m pytest decision_level` | 78 synthetic checks pass; 7 end-to-end driver tests skipped (need unreleased resources) |
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
| Re-running E062 | needs per-image score dumps and the per-capture appearance labels (not released) |
| Building the E062 appearance labels | needs the native fingerprint images; the Laplacian statistic of the label table is not verified; the released code reproduces the rule from statistics, which does not show that the classes are valid |
| E062 plasmode calibration (K1–K4) | uses the real design skeleton (class, credential and finger structure), which is not released; its summaries are included |
| Fig. 2 (score distributions) | needs per-image scores |
| Face and public-fingerprint studies | third-party corpora; study code not included |
| D3 detector | third-party code without a licence file |
| Re-running E063–E071 | need per-image score dumps, the research matcher, images (E065/E068/E071), the external captures and their windows (E069/E070) and the third-party enhancer / NFIQ 2 tools (E068/E070) |
