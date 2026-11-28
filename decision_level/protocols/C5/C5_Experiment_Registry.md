# C5 Experiment Registry: analyses, status, execution and deviations

**Index rows:** `docs/EXPERIMENT_INDEX.md`, E072–E074.
**Execution log:** `docs/EXECUTION_LOG.md`, entry 2026-11-27.
**Freeze record:** `docs/C5/C5_FREEZE.json`.

## Registered analyses

| ID / part | Analysis | Objective | Status | Outcome under the pre-specified rule | Output |
|---|---|---|---|---|---|
| E072 A1 | Φ and D10 against N0, N1 and N2; S-A1a, S-A1b, S-A1c | primary (N2, restored) | executed | **persists**: 12/12 both arms (ratio 1.81 leg, 1.53 fix) | `E072/a1_nulls.csv`, `joint_summary.json` |
| E072 A2-P1 | ΔL3 legacy, E3 unbiased | primary (97.5 %) | executed | **survives**: 0.017 [0.008, 0.028] | `E072/cells_boot.csv` |
| E072 A2-P2 | ΔL3 legacy, E4 empirical Bayes | primary (97.5 %) | executed | **survives**: 0.018 [0.009, 0.027] | same |
| E072 A2 bias simulation | plug-in against model truth for L3, C25/50/75, D10, CVaR | descriptive | executed | large upward bias of credential endpoints | `E072/a2_bias_sim.csv` |
| E072 A2 n_f strata | C25/50/75 by n_f, plug-in and E4 | descriptive | executed | C50 estimator-sensitive | `E072/a2_nf_strata.csv` |
| E072 A3 | nested GLMM, R1 and R2, LRT | exploratory | executed | restored: identifiable (σ_s ≈ 0.77, σ_c ≈ 0.9); anchor: **not identifiable** (53/60, 56/60 flagged) | `E072/a3_glmm_*.csv` |
| E072 B-P1 | ΔCVaR80 subject, legacy | primary (98.33 %) | executed | **increase**: 0.270 [0.196, 0.314] | `E072/cells_boot.csv` |
| E072 B-P2 | ΔΔCVaR80 | primary | executed | **reduced under fixed**: −0.128 [−0.170, −0.074] | same |
| E072 B-P3 | share worse off, legacy | primary (descriptive) | executed | 0.90 [0.82, 0.97] | same |
| E072 B other | quantiles, top-20 % share, paired Δ quantiles, S-B1–S-B3, LOO, variance components | descriptive | executed | see Results §2 | `E072/b_*.csv` |
| E072 CVaR90 rule | width > estimate in most cells? | rule | executed | not triggered; kept as sensitivity | `OUTCOME.json` |
| E073 G-R | reproduce E061 | gate | PASS | — | `E073/gates.json` |
| E073 C-P1 | change of the c* minimizer (E-1, E-6, E-7) | primary | executed | E-6 δ = 0: "changes selection" (**by construction for π4**); E-7 λ = 1: leg "marginal" (2/12), fix "no change"; E-1: 0/12 | `E073/minimizers_cstar.csv` |
| E073 C-P2 | π1 and π4 against π0: Δmean b, ΔCVaR80, worse-off share | primary (1 − 0.05/8) | executed | all eight Δmean and ΔCVaR80 tests above 0 | `E073/OUTCOME.json` |
| E073 E-5 | three-objective Pareto (full sample and bootstrap membership) | descriptive | executed | `E073/pareto3.csv` | — |
| E073 diagnostic | spread against interval width | rule | executed | discriminating (leg); weak (fix) | `OUTCOME.json` |
| E073 C-S1 | M1 | sensitivity | executed | c* minimizers unchanged | `E073/cs1_*.csv` |
| E073 C-S2 | cost grid | sensitivity | executed | E-7 differs at 12–14 % of points | `E073/cs2_*.csv` |
| E073 C-S3 | φ ∈ {0, 0.02, 0.05, 0.1, 0.2} (hypothetical) | sensitivity | executed | 0 sign flips of ΔJ | `E073/cs3_*.csv` |
| E073 C-S4 | same-image attacker | sensitivity | executed | λ = 1 removes π1 in 2 legacy cells | `E073/cs4_*.csv` |
| E074 | π5 recourse | gated | **not executed** | gate (i) not met | — |

## Not executed or dropped (with reasons)

- **Empirical-Bayes variant of the E073 session terms (former C-S1b):** dropped before the freeze. It would need an
  ad hoc joint reject/step-up prior. This is recorded in the protocol.
- **E074:** not executed, per its gate.
- **Age-stratified, demographic or session-persistence analyses:** not executed. They are not permitted, or the data
  are absent (protocol §1; audit §5).

## Deviations

- **None** after the freeze.
- **Pre-freeze change, recorded for transparency:** the E-6 δ = 0 by-construction note was added after the validation
  pilot and before the freeze.

## Execution record

| Item | Value |
|---|---|
| Freeze | 2026-11-27T00:56:37+0800; git HEAD at freeze a56033d (code uncommitted; hash-frozen) |
| E072 | one run, about 590 s, 32 CPU workers; B = 2,000; D_null = 1,000; SeedSequence 720701 |
| E073 | one run; B = 2,000; SeedSequence 730701 |
| GPU | none used |
| Frozen inputs | E060 dumps (360), E061 `validation_fit.json` (SHA-256 c3d01b7d…, matches the E061 authorization) |
| Access logs | `artifacts/results/E07{2,3}/access_log.json` |
| Provenance | `artifacts/results/E07{2,3}/provenance.json` |


## Reporting errata found at integration review (2026-11-27; no rerun, no change to frozen outputs)

1. **Anchor double count (E072 A1).** `run_e072.py` summarized the anchor over the 12 cell keys, but the anchor has
   only 6 instrument × protocol conditions per arm, so each was counted twice. The anchor "12/12" in `OUTCOME.json`,
   `joint_summary.json` (`a1_class`), the Results Report and the CURRENT_STATUS S40 row means 6/6 (and "10" means
   5/6). The restored and primary (N2) counts are unaffected (12 distinct cells). The manuscript and Supplementary now
   count from `a1_nulls.csv` directly (`tools/mk_c5.py`).
2. **Seed dependence in the A1 null.** The null draws are independent across the ten seeds, while the observed
   per-seed Φ share the same captures, so the null of the seed average is too narrow (identified by an independent
   statistical review). A conservative bound, the null quantile if the seeds were perfectly correlated
   (μ + √10 (q975 − μ), normal approximation from stored summaries), was added as a **post hoc reporting sensitivity**:
   - restored N2: 12/12 in both arms (unchanged);
   - anchor N1: 5/6 (legacy), 4/6 (fixed).

   The anchor heterogeneity is therefore reported as "suggested, not robust".
3. **E074 gate.** With G ≈ 0.026 and ΔJ ≈ −0.09, no φ ≤ 1 could reverse ΔJ(π4 − π0), so the pre-registered gate could
   not be met by design. This is now disclosed in the manuscript (Discussion, Limitations) and the Supplementary.
4. **C-P1 label.** The frozen labels ("changes selection" in both arms, driven by E-6 δ = 0 by construction; E-7 λ = 1
   "marginal" legacy and "no change" fixed) are now stated verbatim in the manuscript.
