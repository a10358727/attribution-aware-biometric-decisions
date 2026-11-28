# Public release scope (v1.2.0, 2026-11-28; v1.1.0 plus the C5 additions in §2c)

Rule: built from an explicit allowlist; nothing is mirrored from the private research repository and no private Git
history is included. Every file below was copied by name, checked by `scripts/privacy_check.py` and, where pinned,
by `scripts/verify_frozen_sources.py`. The excluded categories and their reasons are in `PUBLIC_RELEASE_EXCLUSIONS.md`.

## 1. Already public in v1.0.0 (unchanged)

| Area | Content |
|---|---|
| RQ1 attribution (§V, Table I) | normalization policies, degradation bank, restoration operators, five deployment quantities at a validation-locked threshold, cluster and paired bootstrap, `scripts/table1_from_aggregates.py`, `data/aggregate/`, detector definitions (no weights), training configurations |
| RQ2 credential burden (§VI, Table III) | `tcss_lib/core.py`, `tcss_lib/pool.py`, `run_e060.py` (byte-identical to the authorised versions), aggregate results, 11 synthetic tests, `headline_from_aggregates.py` |
| RQ3 policy simulation (§VII, Table IV, Fig. 3) | `run_e061.py` (byte-identical), policies π0–π4, PMP, session cost J, Pareto readings, cost grid, aggregate results |

## 2. Added in this release candidate

| Path | What | Source in the research repository | Status |
|---|---|---|---|
| `decision_level/experiments/E062_quality_conditioned_eval/ra_analysis.py` | RA-class-stratified burden estimator: per-cell contrasts, median over 12 cells, subject-cluster × seed bootstrap, primary/secondary/S-1…S-7/S-8w (`run_all`); guarded real-data loader | same path | byte-identical (pinned by the E062 authorisation) |
| `…/ra_report.py` | Supplementary tables S-E1…S-E5 (manuscript S14–S18) and the pre-specified reading rules | same path | byte-identical (frozen with E062) |
| `…/ra_provenance_check.py` | the appearance-grouping rule: corpus-wide tertile votes, ≥ 3-of-4 agreement, unscaled Euclidean 9-NN (`reconstruct`); its `main` needs the private label table and images | same path | byte-identical |
| `…/ra_calibration.py` | toy null calibration of the estimator on synthetic latent scores | same path | byte-identical |
| `…/ra_prefreeze.py` | plasmode calibration (K1–K4, level calibration) on the real design skeleton | same path | byte-identical; **inspection only** (imports the private join module and needs the private label metadata) |
| `decision_level/tests/test_e062_ra_analysis.py`, `test_e062_report.py`, `test_e062_provenance.py` | 37 synthetic tests | `diar/tests/` | adapted: only the `sys.path` line differs |
| `decision_level/scripts/e062_tables_from_aggregates.py` | rebuilds S14–S18 from the released aggregate and checks them | new | public script |
| `decision_level/examples/e062_synthetic_demo.py` | grouping rule → estimator → tables on synthetic data | new | public script; output marked SYNTHETIC |
| `decision_level/results/E062/E062_formal_results.public.json` | aggregate output of the one authorised E062 run | standard-JSON copy of the frozen output (SHA-256 `6b7816ba…f770`): 288 non-estimand NaN entries → `null`, nothing else changed (`_source` records this) | smallest released group: 110 captures, 18 subjects |
| `decision_level/results/E062/E062_supplementary_tables.md` | the tables written by the run | byte copy | aggregate |
| `decision_level/results/E062/calibration/*.json` | synthetic-outcome calibration summaries (simulation, level calibration, K1–K4) and the label-rule reconstruction counts | byte copies | aggregates; no outcome of the study |
| `decision_level/results/E060_post_freeze_annotation/E060_M1_exclusion_counts.json` | credentials excluded from retry model M1 (60 for K = 3, 117 for K = 5), reconstructed from credential composition after the freeze | byte copy | post-freeze annotation, not a frozen result |
| `scripts/privacy_check.py`, `scripts/verify_frozen_sources.py`, `.github/workflows/ci.yml` | privacy gate, source verification, CI (no data download) | new | public tooling |

## 2b. Added for the C1 manuscript (2026-11-24)

Built by an allowlist export: every analysis source was checked against the SHA-256 pinned in its own authorisation
record (or freeze record / run output) **before** copying; `RELEASE_MANIFEST_C1.json` lists each file with its research
and public SHA-256. The only permitted change is a declared local-path redaction (one file).

| Path (under `decision_level/`) | What | Status |
|---|---|---|
| `experiments/E063_session_robustness/` | `run_e063.py` (M0/M1, M0 on M1's credentials), `session_models.py`, `breakeven_r3_signcorrected.py` (Part A r3, the version the manuscript reports), `breakeven.py` (r2, superseded), 1 test file | byte-identical, except `run_e063.py`: its private manifest path is redacted (research SHA-256 `0916ae34…`, pinned by the E063 authorisation) |
| `experiments/E064_session_persistence_K_attacker/` | `run_e064.py`, `session_m2.py` (persistent failures M2), 2 test files | byte-identical |
| `experiments/E065_matcher_cascade_genuine/run_e065.py`, `experiments/E071_matcher_verification/run_e071.py` | genuine-matching and cascade analysis; embedding calls the research matcher (not released) | byte-identical |
| `experiments/E067_validation_refit_uncertainty/` | `refit.py`, `run_e067.py`, `run_e067_stage_t.py`, 2 test files | byte-identical |
| `experiments/E068_fingerprint_enhancement_condition/e068_analyze.py`, `e068_common.py`; `experiments/E049_learned_restorer_sensitivity/analyze.py`; `experiments/E036_interaction_statistics/run_e036.py` | E068 analysis and its frozen dependencies | byte-identical; inspection only |
| `experiments/E069_cross_sensor_bonafide_transfer/e069_analyze.py`, `e069_common.py`; `experiments/E070_quality_conditioned_external/e070_contrast.py` | external-capture analysis and NFIQ 2 contrast | byte-identical; inspection only |
| `conftest.py` | skips 7 end-to-end driver tests that need unreleased resources | new public file |
| `results/E063/`, `E064/`, `E065/`, `E067/`, `E068/`, `E069/`, `E070/`, `E071/` | final aggregate outputs of the runs the manuscript reports (cell-, device- or tertile-level; no per-image, per-capture or per-subject row) | byte copies; `E068/analysis/class.public.json` is a standard-JSON copy with the launch/provenance keys removed |
| `scripts/c1_mk_sens.py`, `scripts/c1_mk_ext.py`, `scripts/c1_check_macros.py`, `results/C1_manuscript_macros/` | the manuscript's macro generators with only their input/output paths remapped, and the manuscript's macro/table files as expected outputs | public tooling; the check reproduces all 7 files from the released aggregates |

## 2c. Added for the C5 analyses (v1.2.0, 2026-11-28)

| Path (under `decision_level/`) | Content | Status |
|---|---|---|
| `experiments/c5_lib/stats.py` | finite-sample estimators, heterogeneity statistics, concentration and tail measures | byte-identical to the version pinned by the E072/E073 test-read authorisations |
| `experiments/c5_lib/tests/test_c5_stats.py` | 13 synthetic checks | byte-identical to the research file (pinned in the C5 freeze record) |
| `experiments/E072_burden_validity/run_e072.py`, `experiments/E073_distribution_policy/run_e073.py` | analysis drivers | byte-identical to the pinned versions; inspection only (need score dumps) |
| `protocols/C5/` | frozen protocol, registry with additive errata, freeze record | byte copies |
| `results/E072/`, `results/E073/` | final aggregate outputs (cell-, seed- or cost-grid-level; no per-image, per-capture or per-subject row) | byte copies |
| `scripts/c5_mk_macros.py`, `scripts/c5_check_macros.py`, `results/C5_manuscript_macros/` | the manuscript's C5 macro generator with only its input/output paths remapped, and the manuscript's macro/table files as expected outputs | public tooling; the check reproduces all 10 files |

## 3. Needs private dependencies — released for inspection only, or not at all

| Item | Handling |
|---|---|
| E062 real-data loader (`ra_analysis.load_real`) | released (byte-identical) but refuses to run without the private authorisation and score dumps |
| `ra_prefreeze.py` | released for inspection; not runnable here |
| E062 metadata join and audit (`ra_audit.py`), label-feature discrepancy audit, preflight and formal runner | not released: they depend on the private identity parser, the private label table, private folder structure and authorisation records |

## 4. Not released (see `PUBLIC_RELEASE_EXCLUSIONS.md`)

Fingerprint images of any kind; the q3_v2_all label table, its source file and any capture-to-class mapping;
per-image scores, logits, predictions, decisions and features; subject, finger and credential identifiers; splits and
manifests; finger-by-class and other small-cell metadata tables; model weights; authorisation, signature and
provenance-statement records; access logs and run locks; local paths; the training and dataset-building code.

## 5. What this means for reproducibility

Methods can be inspected and run on synthetic data; the paper's tables can be rebuilt from released aggregates; the
pinned analysis sources can be verified by hash. The experiments themselves cannot be re-run from this repository,
because the fingerprint corpus, per-image scores and model weights are not released.
