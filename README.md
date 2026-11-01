# Attribution-aware evaluation of image restoration for biometric decisions — code and aggregate results

Code and aggregate results accompanying the manuscript *Attribution-Aware Evaluation of Image Restoration
for Trustworthy Biometric Decisions in Cyber-Physical-Social Systems* (M.-H. Hsu, Y.-H. Shi; prepared for
submission to IEEE Transactions on Computational Social Systems, special issue on Cyber-Physical-Social
Intelligence). Not yet published.

MIT licence (`LICENSE`) for the authors' code, documentation and aggregate results; dependencies keep their
own licences and are not redistributed. The repository contains **no fingerprint images, no per-image scores,
no subject-level splits or manifests and no model weights** (`PUBLIC_RELEASE_EXCLUSIONS.md`).

## What the paper evaluates

A synthetic-image detector placed behind an image-restoration stage is one component of a biometric access
decision. The paper follows one chain — physical capture, cyber processing (restoration, normalization,
detector, a threshold locked on validation data), decision burden on legitimate claimants — in three steps:

| Step | Question | Code here | Evidence type |
|---|---|---|---|
| RQ1 Attribution (§V, Table I) | Is a detection gain at the locked threshold better discrimination, score displacement, bona-fide cost, threshold movement, or an effect of the normalization policy? | `detector_attribution/`, `scripts/table1_from_aggregates.py`, `data/aggregate/` | offline measurements |
| RQ2 Burden (§VI, Table III; E060) | How is the bona-fide cost distributed over credentials, and what repeated-attempt denial does it imply under an explicit retry model? | `decision_level/` | offline measurements; retries modelled (M0, K = 3) |
| RQ3 Policy (§VII, Table IV, Fig. 3; E061) | In a validation-fitted simulation, does any restoration-based policy improve on no restoration? | `decision_level/` | simulation under stated retry, attacker, step-up and cost assumptions |
| RQ2, exploratory supplement (Suppl. S5, Tables S14–S18; E062) | Does the restoration-induced bona-fide rejection differ between rule-based appearance classes of the native captures? | `decision_level/experiments/E062_quality_conditioned_eval/` | post hoc exploratory; offline measurements; not a contribution of the paper |

E060 and E061 are **post hoc secondary analyses** of a test pool already used by the detector studies; each
analysis plan was frozen before its single authorised test read. E062 is a **post hoc exploratory supplementary
analysis** on the same pool (frozen before its single authorised read): no difference between appearance classes was
detected at its 99.5 % level; its classes are rule-based, corpus-relative and transductive (cut points from the whole
corpus), not measured skin moisture or image quality, and the real per-capture labels are not released. Pre-matcher progression (PMP) is not access
or attack success; no matcher is modelled. No authentication session, retry or attack was observed.

## Repository structure

| Path | Contents |
|---|---|
| `detector_attribution/normalization.py` | the two normalization policies (per-image min–max, g/255); the normalization dose ρ |
| `detector_attribution/degradation.py`, `ridge_utils.py` | the symmetric study-degradation bank (six types; severity ladder 0–4, 0 = clean); ridge-period estimate |
| `detector_attribution/operators.py` | restoration operators (bilateral, total variation and others) |
| `detector_attribution/deployment.py` | locked threshold at 1 % FPR, AUC, the five deployment quantities, cluster bootstrap |
| `detector_attribution/paired_bootstrap.py` | bootstrap primitives and a reference estimator of the paired contrasts ΔΔT / ΔΔB |
| `inference/score_patch.py` | detector definitions and a patch-scoring script (no weights provided) |
| `configs/` | resolved training configuration, selection record and training curve of each fingerprint detector run (see `configs/README.md`) |
| `data/aggregate/` | detector-level test-pool aggregates: per-seed headline quantities, paired contrasts, cross-domain (face) signatures |
| `scripts/table1_from_aggregates.py` | recomputes the per-arm ΔAUC, ΔT, ΔB, T(recal) and lists ΔΔT / ΔΔB with 95 % intervals of Table I |
| `examples/synthetic_demo.py` | end-to-end run of the detector-level methods on synthetic, non-biometric data |
| `decision_level/experiments/tcss_lib/` | E060 / E061 analysis core: policies π0–π4 and their validation fitting, credential endpoints (B_cred, C50, D10, L_K, A_K), retry models M0 / M1, session quantities, PMP, session cost J, paired two-factor bootstrap, test-access guard |
| `decision_level/experiments/E060_credential_burden/`, `E061_policy_simulation/` | the two analysis drivers |
| `decision_level/tests/` | 11 closed-form checks of the analysis core on synthetic data |
| `decision_level/scripts/headline_from_aggregates.py` | prints the E060 / E061 outcomes and headline values |
| `decision_level/results/E060/`, `E061/` | aggregate results: outcomes, joint summaries (median over the twelve headline cells with intervals), per-cell tables, frozen validation-only policy parameters, gate records |
| `decision_level/experiments/E062_quality_conditioned_eval/` | E062: appearance-grouping rule (`ra_provenance_check.py`), stratified estimator with all pre-specified analyses (`ra_analysis.py`), table renderer (`ra_report.py`), synthetic calibration (`ra_calibration.py`), plasmode calibration (`ra_prefreeze.py`, inspection only) |
| `decision_level/tests/test_e062_*.py` | 37 synthetic checks of the E062 code |
| `decision_level/scripts/e062_tables_from_aggregates.py`, `decision_level/examples/e062_synthetic_demo.py` | rebuild of Supplementary Tables S14–S18 from the released aggregate; synthetic end-to-end run |
| `decision_level/results/E062/` | E062 aggregate output (standard-JSON copy), the tables of the run, synthetic calibration summaries |
| `decision_level/results/E060_post_freeze_annotation/` | M1 exclusion counts reconstructed after the freeze from credential composition |
| `scripts/privacy_check.py`, `scripts/verify_frozen_sources.py`, `.github/workflows/ci.yml` | privacy gate, pinned-source verification, CI on synthetic data and aggregates |
| `PUBLIC_RELEASE_SCOPE.md`, `PUBLIC_RELEASE_EXCLUSIONS.md`, `REPRODUCIBILITY.md`, `SHA256SUMS` | what is released and from where; what is excluded and why; what can be reproduced; checksums |

The analysis sources below are byte-identical to the versions pinned by SHA-256 in the test-read authorisations
(E062's renderer and calibration code: in its freeze record); `python scripts/verify_frozen_sources.py` checks them:

| File | SHA-256 |
|---|---|
| `decision_level/experiments/tcss_lib/core.py` | `ba15e60ea66ef1d7c4d1abfd71d1473940b0415f424fe776fa3409d7acf04158` |
| `decision_level/experiments/tcss_lib/pool.py` | `6ad1c002077f34a64022d4e2d34e58e6a1585d685f62722d7eb6b78eb21b1af6` |
| `decision_level/experiments/E060_credential_burden/run_e060.py` | `4f90df1a87f7e83fcfa56f01ee5f8ba6a9011d39e11af16a97cd3f141aac46f5` |
| `decision_level/experiments/E061_policy_simulation/run_e061.py` | `a5515674a1e7a9ee8cf720e0faa3428d4eb42bdd75bc920dcbc6e83660aa78e0` |
| `decision_level/experiments/E062_quality_conditioned_eval/ra_analysis.py` | `b52bb69e9f5b0c7b63c6bf3f747c9e99c825f5d33f4482a8c38a2314e826875d` |
| `decision_level/experiments/E062_quality_conditioned_eval/ra_report.py` | `638d035c5d1744de174dcd45f6a4940197010ae23e278a96c6f76aee8c60fa77` |

## Environment and use

Python 3.12 (tested); `pip install -r requirements.txt`. No GPU is needed.

| Command | Output |
|---|---|
| `python scripts/table1_from_aggregates.py` | per-arm detector-level quantities and paired contrasts of Table I |
| `python decision_level/scripts/headline_from_aggregates.py` | E060 outcome U-A and E061 outcome P-A with their headline medians and intervals |
| `python -m pytest decision_level/tests` | 48 synthetic checks (11 of the E060 / E061 core, 37 of E062) |
| `python decision_level/scripts/e062_tables_from_aggregates.py` | Supplementary Tables S14–S18 from the released E062 aggregate, checked against the run's tables |
| `python decision_level/examples/e062_synthetic_demo.py` | E062 grouping rule, estimator and tables on synthetic data (illustration only) |
| `python scripts/verify_frozen_sources.py` | SHA-256 of the pinned analysis sources |
| `python scripts/privacy_check.py` | privacy gate over every publishable file |
| `python examples/synthetic_demo.py` | the detector-level methods on simulated data (illustration only) |

The E060 / E061 drivers and the E062 real-data loader are released for inspection; they cannot run here, because
they read per-image score dumps and (E062) the per-capture labels, neither of which is released, and the E060 / E061
drivers call the detector-level paired-contrast module of the research code (not released).

## Labels used in the result files

Detectors `Rn` ResNet-18 (modified stem), `Bs` Benjamin et al. base detector, `Dt` D3, `Np` NPR
(supplemental). Protocols `p1`, `p2`: the two held-out-generator protocols. Operators `bil` bilateral, `tv`
total variation. Arms `leg` per-image min–max, `fix` g/255, `dd` fixed minus legacy.

## Limitations

All fingerprint evidence comes from one capacitive sensor model and 34 test subjects; the twelve headline
cells share checkpoints and one test pool. Burden is offline detector rejection, and retry consequences are
modelled; session results depend on assumed retry, attacker, step-up and cost models. Full reproduction
(training, scoring, thresholds and bootstrap intervals from data) is not possible from this repository; see
`REPRODUCIBILITY.md`.

## Citation

Cite this software release with `CITATION.cff`. The manuscript has no DOI yet.
