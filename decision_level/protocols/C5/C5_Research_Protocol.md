# C5 Research Protocol: burden validity, individual burden and distribution-aware policy evaluation

- **Status:** FROZEN at Phase 1, before any C5 test-score read. The freeze record is `docs/C5/C5_FREEZE.json`
  (SHA-256 of this file and of the analysis scripts, with time).
- **Changes:** any change after the freeze is a dated **deviation** in `C5_Experiment_Registry.md`, never a silent
  edit.
- **Experiments:**
  - **E072**, burden validity (Task A) and individual and distributional burden (Task B);
  - **E073**, distribution-aware policy evaluation (Task C) and the hypothetical secondary-check failure
    sensitivity (C-S3);
  - **E074** (optional π5 recourse), registered with a go/no-go rule (§7); not executed unless that rule is met.
- **Evidence status of all C5 analyses: post hoc, exploratory.**
  - The test pool was already read by E060 and E061.
  - Freezing this plan before the new read protects against outcome-driven analysis. It does **not** make the
    results a prospective, preregistered confirmation or an independent validation.
  - No C5 result can upgrade an E060 or E061 outcome.

## 1. Common elements

| Element | Specification |
|---|---|
| Inputs | **E072:** the 360 frozen E060 dumps (`tcss_lib.core.SRC`, headline instruments Rn, Bs, Dt; P1, P2; arms leg, fix; seeds 0–9; anchor, bilateral, TV). **E073:** the same, plus the E061 F2 parameters (`artifacts/results/E061/validation_fit.json`) and the E023 unseen synthetic rows, exactly as E061 used them |
| Test rows | real test rows (1,653 captures; 309 credentials; 34 subjects); synthetic rows only in E073 for PMP |
| Thresholds | E060: τ_s = E023 rule on validation real anchor scores (= F2 τ0). E073: frozen F2 policy parameters. Nothing is refitted |
| Cells | 12 = instrument (3) × protocol (2) × operator (bilateral, TV), per arm. Eligibility is the frozen E060/E061 result (all eligible) |
| Strata z | degradation type (6) × severity (4) = 24, from the manifest |
| Seeds | endpoints per seed, then the seed-multiplicity-weighted mean (as E060) |
| Bootstrap | subjects (34, with all their credentials and captures) × training seeds (per instrument–protocol), paired across arms, conditions, operators, policies and protocols. E073 also resamples synthetic images per generator, as E061. **B = 2,000.** `SeedSequence(720701)` for E072, `SeedSequence(730701)` for E073 |
| Cross-cell summary | the median over the 12 cells, computed replicate by replicate. One summary of one dependent family; cells are never independent replicates |
| Interval level | 95 % percentile intervals. The pre-specified primary family per experiment is listed below, with Bonferroni-adjusted decision intervals. Everything else is descriptive |
| Test-access guard | `tcss_lib.core.read_members` with `artifacts/results/E07x/TEST_READ_AUTHORISATION.json`, pinning the SHA-256 of every source file run; access log written |
| Gates (stop on failure; never adjust code or data to pass) | **G-S:** synthetic unit tests. **G-R:** reproduction of frozen full-sample E060 (B_cred, C50, L3, D10) and E061 (L, G, N, PMP, J) values. **G-V:** validation-only pilot of the whole code path (not evidence) |
| Outputs | `artifacts/results/E072/`, `artifacts/results/E073/`; code in `experiments/E072_burden_validity/`, `experiments/E073_distribution_policy/`, shared code in `experiments/c5_lib/`. Frozen E060/E061 files are never written |

## 2. E072: Task A (burden validity)

### A1. Composition-conditioned reference models

- **Motivation:** E060's pooled-rate null ignores degradation composition, which varies by credential by chance.
- **Research question:** after conditioning on degradation type × severity (and, for restored inputs, on the
  capture's own anchor decision), does credential-level rejection remain more heterogeneous than the reference model
  implies?
- **Objective:** primary within E072 (post hoc, exploratory).
- **Unit:** credential (309), with capture counts n_f kept.
- **Eligibility:** all credentials. Credentials with V_f = 0 under a null contribute nothing to Φ.
- **Reference models.** Per (cell, arm, seed, condition k):

  | Model | Capture rejection probability |
  |---|---|
  | **N0** (pooled; reproduces E060) | iid Bernoulli(B_cap) |
  | **N1** (degradation-conditioned) | iid Bernoulli(π_z), where π_z is the stratum rejection rate |
  | **N2** (restored conditions only) | iid Bernoulli(π_{z,a}), where a_i ∈ {0,1} is the capture's anchor decision under the same instrument, protocol, arm and seed |

  N2 asks whether credential identity predicts restoration-induced rejection beyond stratum and the capture's own
  anchor outcome.
- **Statistics:**
  - **Φ** = Σ_f (k_f − E_f)² / Σ_f V_f, where E_f = Σ_{i∈f} π_i and V_f = Σ_{i∈f} π_i(1 − π_i). This is a
    credential-count dispersion ratio, about 1 under the model.
  - **D10** with unit credential weights (E060 definition).
- **Null distribution:** parametric bootstrap with 1,000 draws per seed.
  - In each draw, simulate every capture from π̂, **re-estimate π̂\* from the draw** (stratum, or stratum × anchor,
    rates), then recompute Φ\* and D10\*.
  - Draw j is paired across seeds.
  - T = seed mean of Φ_s; T\*_j = seed mean of Φ\*_{s,j}; likewise for D10.
- **Per arm–cell:**
  - exceedance = T > 97.5th percentile of T\*;
  - ratio R = T / mean(T\*).
- **Dependency:**
  - the subject × capture hierarchy is kept, because credentials are simulated within their subjects' composition;
  - seeds are paired by draw;
  - arms, operators and protocols are reported separately.
- **Primary readout:** the number of the 12 cells exceeding N2 (restored) per arm, and the median ratio R.
- **Decision rule** (per arm, N2, restored):
  - **"persists"** if at least 9 of 12 cells exceed;
  - **"explained"** if at most 3 of 12 exceed;
  - **"mixed"** otherwise.
- **Reported with it:** the same rule for N1 (anchor and restored), and N0 for continuity.
- **Sensitivity analyses:**
  - (S-A1a) strata = degradation type only;
  - (S-A1b) credentials with n_f ≥ 3 only;
  - (S-A1c) subject-level dispersion: the same Φ with subjects as units.
- **Interpretation boundaries:**
  - "persists" means heterogeneity beyond the conditioned model. That model controls the recorded composition, but
    not other capture properties (placement, pressure, skin state on the day).
  - It is **not** evidence of persistent individual vulnerability: there are no session data.
  - "Explained" is a substantive negative result and is reported as such.
- **Stopping:** none (a single run).

### A2. Finite-sample variability of burden estimators

- **Research question:** how much do plug-in endpoints, especially M0 L3, reflect finite-n_f estimation rather than
  credential rates?
- **Objective:** primary for the L3 comparison, descriptive for the rest.
- **Estimators of the M0 target θ_K = mean_f p_f^K** (K = 3; K = 5 as sensitivity):

  | Estimator | Definition | Population |
  |---|---|---|
  | **E1** plug-in | p̂_f^K (E060 M0) | all credentials |
  | **E2** restricted plug-in | p̂_f^K | n_f ≥ K |
  | **E3** unbiased binomial | C(k_f, K) / C(n_f, K) | n_f ≥ K |
  | **E4** empirical Bayes | beta-binomial posterior mean of p^K: Π_{j<K} (a + k + j) / (a + b + n + j) | all credentials |

  - E3 is unbiased for p_f^K under iid capture sampling with replacement. Its numeric form coincides with the M1
    session probability, but it is reported as an estimator of the M0 target, not as M1, and not as a new
    mathematical contribution.
  - E4 hyperparameters: the moment estimator per (cell, arm, seed, condition, replicate), m = Σk/Σn, with ρ by the
    beta-binomial ANOVA moment formula, clipped to [10⁻⁶, 0.999]; a = m(1/ρ − 1), b = (1 − m)(1/ρ − 1).
- **Endpoints:** levels, Δ (restored − anchor) per arm, and ΔΔ (fix − leg), all as medians over cells with 95 %
  intervals.
- **Primary family (m = 2, decided at 97.5 %):**
  - (A2-P1) median ΔL3 legacy under E3;
  - (A2-P2) median ΔL3 legacy under E4.
  - Decision: "survives finite-sample correction" if the lower bound is above 0.
  - These are compared with the frozen E060 H1 (E1).
- **Model-based bias check:**
  - From the full-sample E4 prior per (cell, arm, condition), seed 0, simulate 500 data sets: p_f ~ Beta(a, b) and
    k_f ~ Bin(n_f, p_f), with the observed n_f.
  - Report the relative bias of the plug-in against the simulated truth for L3, C25, C50, C75, D10, CVaR80 and CVaR90
    of the credential rates.
- **Descriptive:** C25, C50 and C75 by n_f stratum (1–2, 3–4, ≥ 5), plug-in and E4.
- **Sensitivity analysis:** K = 5.
- **Interpretation boundaries:** E3 and E4 change the estimator, not the retry model. They do not validate M0.

### A3. Hierarchical variance decomposition (exploratory)

- **Model** per (cell, arm, seed, condition), fitted by maximum likelihood:

  logit P(rej_i) = α + β_type(i) + γ_sev(i) + u_s(i) + v_f(i), with u ~ N(0, σ_s²) and v ~ N(0, σ_c²).

  - Severity enters as 4 levels.
  - Nested Gauss–Hermite quadrature, 15 nodes per level, optimized with L-BFGS-B on (log σ).
  - Variant **R2** (restored): adds the anchor decision a_i as a covariate.
- **Reported:**
  - medians across cells and seeds of σ_s, σ_c and the latent-scale shares σ²/(σ_s² + σ_c² + π²/3);
  - likelihood-ratio tests for σ_c = 0 given σ_s, and for σ_s = 0 given σ_c (boundary mixture ½χ²₀ + ½χ²₁);
  - the share of fits with p < 0.05 (descriptive, unadjusted).
- **Identifiability rule (fixed now):** a fit is flagged non-identifiable if any of the following holds:
  - the optimizer fails;
  - a σ goes to the boundary (< 0.01 or > 10);
  - the condition's rejection rate is below 2 %.

  If more than half the fits in a stratum are flagged, that stratum is reported as "not identifiable", with no
  variance shares.
- **Interpretation boundaries:**
  - The capture level is the binomial residual.
  - A credential variance above zero is heterogeneity between fingers in this corpus, not persistence over time.
  - No decision rule; exploratory only.

## 3. E072: Task B (individual and distributional burden)

- **Research question:** how are restoration-induced rejection burdens distributed across credentials and subjects?
- **Unit:** subject (34), primary for B. The credential is reported alongside.
- **Aggregation from credentials to subject** (defined, not assumed behaviour):

  | Variant | Definition |
  |---|---|
  | **Primary** | credential-averaged burden b_i = (1/\|F_i\|) Σ_{f∈F_i} x_f, with x_f a credential quantity (single-attempt rate p_f, or modelled L3 under E1 and E4) |
  | Sensitivity S-B1 | capture-pooled rate Σk / Σn |
  | Sensitivity S-B2 | best-credential bound min_f |
  | Sensitivity S-B3 | worst-credential bound max_f |

  - The primary variant describes the subject's enrolled credentials as evaluated, each counted once.
  - It is **not** an individual authentication failure rate and assumes no finger choice or switching.
- **Endpoints** (subject-weighted; also credential-weighted for comparison) per (cell, arm, condition):
  - **magnitude:** mean;
  - **distribution:** ECDF, Q25, Q50, Q75, Q90;
  - **concentration:** the share of total subject burden carried by the top 20 % of subjects;
  - **upper tail:**
    - CVaR80 = mean burden of the highest 20 % of subjects (fractional boundary weight, as D10), absolute scale;
    - CVaR90 as sensitivity;
  - **individual change:** paired Δ_i = b_i(restored) − b_i(anchor); its quantiles; share worse off,
    #{Δ_i > 0}/34 and #{Δ_i > 0.05}/34;
  - **variance components:**
    - between-subject variance of b_i;
    - mean within-subject variance of p_f across the subject's credentials;
    - the share of total credential variance between subjects.
- **Tail stability:**
  - leave-one-subject-out range of CVaR80 and CVaR90 (full sample, per cell);
  - the number of subjects defining each tail (6.8 and 3.4).
- **Primary family (m = 3, decided at 1 − 0.05/3):**
  - (B-P1) median over cells of ΔCVaR80 legacy, primary aggregation and p_f;
  - (B-P2) median ΔΔCVaR80 (fix − leg);
  - (B-P3) median share of subjects worse off (Δ_i > 0) legacy, reported against 0.5.
  - "Increase" if the lower bound is above 0; "reduced under fixed" if the ΔΔ upper bound is below 0; B-P3 is
    descriptive, with its interval.
- **Distinctions kept explicit in reporting:**
  1. magnitude (mean);
  2. concentration (top-20 % share; D10);
  3. individual increase (Δ_i);
  4. persistent vulnerability (**not identifiable**);
  5. modelled session denial (L3, M0/E4);
  6. experienced burden (**not measured**).
- **Interpretation boundaries:**
  - 34 subjects, one sensor;
  - tails rest on 3 to 7 people and are reported with leave-one-out ranges;
  - if the CVaR90 interval width exceeds the CVaR90 point estimate in most cells, CVaR90 is declared unstable and
    not used for conclusions.

## 4. E073: Task C (distribution-aware policy evaluation)

- **Research question:** does accounting for the distribution of legitimate-user burden across subjects change the
  evaluation or selection of access policies?
- **Baseline kept:**
  - J(π) = c_A·PMP(π) + L(π) + c_U·G(π) + c_T·(N(π) − 1), credential-weighted, exactly E061;
  - c* = (c_A 1, c_U 0.2, c_T 0.02, q 0.9, K 3).
- **Individual procedural burden:**
  - per credential under M0: L_f = r_f^K, G_f = Σ_{j<K} r_f^j u_f, N_f = Σ_{j<K} r_f^j;
    b_f = L_f + c_U·G_f + c_T·(N_f − 1);
  - per subject: b(π, i) = credential average over F_i (§3 primary).
  - PMP is unchanged from E061 (capture-level synthetic, fresh image, q).
- **Evaluations** per (cell, arm, operator), per policy π0–π4:

  | Code | Evaluation |
  |---|---|
  | **E-0** | J (E061, credential-weighted); gate G-R reproduces E061 |
  | **E-1** | J_subj = c_A·PMP + mean_i b(π, i) |
  | **E-2** | CVaR80_i b(π, i) (CVaR90 as sensitivity) |
  | **E-3** | share of subjects worse off relative to π0: b(π, i) − b(π0, i) > 0, and > 0.01 |
  | **E-4** | paired Δb_i quantiles (Q10, Q50, Q90) relative to π0 |
  | **E-5** | three-objective Pareto set on (PMP, mean b, CVaR80) |
  | **E-6** | tail-constrained: minimize J subject to CVaR80(π) ≤ CVaR80(π0) + δ, δ ∈ {0, 0.01} |
  | **E-7** | J_DA(λ) = c_A·PMP + mean b + λ·CVaR80, λ ∈ {0, 0.5, 1, 2, 5}. **Sensitivity only; λ is a normative assumption, not validated** |

- **Primary family:**
  - **(C-P1)** per arm: the number of the 12 cells in which the c* minimizer under E-1, E-6 (δ = 0) or E-7 (λ = 1)
    differs from the E-0 minimizer, with bootstrap support (the share of replicates in which the E-0 and alternative
    minimizers agree).
    - "**Changes selection**" if a change occurs in at least 4 of 12 cells under E-6 or E-7(λ = 1).
    - "**No change**" if 0 of 12.
    - Otherwise "**marginal**".
  - **(C-P2)** for π1 and π4 against π0, per arm: medians over cells of Δmean b, ΔCVaR80 and the worse-off share,
    with 95 % intervals. Bonferroni across the 2 policies × 2 arms × 2 endpoints (Δmean b, ΔCVaR80), so 8 tests,
    decided at 1 − 0.05/8.
- **Diagnostic when C-P1 shows no change:** the spread of mean b and CVaR80 across policies relative to their
  intervals, to tell "robust policy ranking" apart from "metrics cannot discriminate".
- **Sensitivity analyses** (descriptive):
  - **C-S1** retry model M1 (`E063 session_models.m1`, n_f ≥ 3). (An empirical-Bayes variant of the session terms was considered and dropped before the freeze: it needs a joint reject/step-up prior that would be ad hoc.)
  - **C-S2** cost grid of E061 (c_A, c_U, c_T, q): the minimizer under E-0, E-6 and E-7(λ = 1) at each grid point,
    full sample;
  - **C-S3** hypothetical legitimate failure of the secondary check, φ ∈ {0, 0.02, 0.05, 0.1, 0.2}: a failed check
    ends the session in denial, so L_f′ = r_f^K + φ·G_f for π4 only. The change in π4's mean b, CVaR80, worse-off
    share and J against π0 is reported as **hypothetical**;
  - **C-S4** attacker: same-image PMP (E061) in E-0, E-1 and E-7.
- **Invariances by construction (fixed before the read, after the validation pilot exposed the first):**
  - π4 uses π0's REJECT rule, so L_f and N_f are identical and G_f ≥ 0. Hence b(π4, i) ≥ b(π0, i) for every
    subject, and **E-6 with δ = 0 excludes π4 whenever any tail subject has a positive step-up probability.** That
    exclusion is mathematical, not empirical. E-6 (δ = 0.01) and the magnitudes (ΔCVaR80 against Δ c_A·PMP) carry
    the empirical content, and the reports must say so.
  - For π4, "worse off relative to π0" equals "positive modelled step-up probability".
- **Interpretation boundaries:**
  - b is a computational proxy of procedural burden, not experienced burden;
  - policy rankings are conditional on M0, K, the cost weights, the attacker model and the assumed check;
  - no "fair policy" claim;
  - λ and δ are normative choices.

## 5. Synthetic and validation gates

- **G-S unit tests** (`experiments/c5_lib/tests/`):
  - Φ about 1 under simulated homogeneous data;
  - Φ equals a hand-computed value on a toy case;
  - E3 is unbiased by exact enumeration;
  - the E4 moment estimator recovers (a, b) on large simulated data within tolerance;
  - the E4 posterior p^K by closed form against numeric integration;
  - CVaR with a fractional boundary;
  - subject aggregation;
  - the C-S3 formula against brute-force enumeration of sessions;
  - the GLMM recovers σ_s and σ_c on simulated data within tolerance (0.25 absolute at σ = 1).
- **G-R:**
  - E072 reproduces full-sample E060 `cells_observed.csv` B_cred, C50 and D10 and `cells_modelled.csv` L3, per
    arm–cell and condition, within 1e-9;
  - E073 reproduces E061 `session.csv` L, G, N and `PMP_q0.9` within 1e-9, and J.
- **G-V:** the complete code path on the validation pool as a stand-in test set (circular; code exercise only).

## 6. Reporting rules

- **Every** pre-specified endpoint is reported, including null, negative and unexpected results.
- Any analysis not executed is listed with its reason.
- No threshold, stratum, λ, δ, K, B or aggregation is changed after the read. A change is a dated deviation with a
  reason, and the original analysis is still reported.
- The same test corpus is never described as independent validation.

## 7. Optional E074 (π5 recourse): go/no-go rule

Execute only if **both** hold:
- (i) C-S3 shows that π4's advantage on J is sensitive within φ ≤ 0.1, i.e. the sign of ΔJ(π4 − π0) changes in at
  least 4 of 12 cells for some φ ≤ 0.1;
- (ii) the authors want a recourse design question in the paper.

Otherwise E074 is recorded as "not executed: C-S3 suffices".

If executed, π5 would add a rejection-triggered fallback with hypothetical legitimate pass, attacker-stop and cost
parameters, both paths modelled and every parameter swept. It would have its own frozen addendum before running.

## 8. Theory → analysis linkage (how the theory changes the design, not the vocabulary)

| Theory | Design consequence in C5 |
|---|---|
| Administrative burden: cost is borne through procedures | the burden metric is a procedural proxy (attempts, step-ups, denial) per person-level unit, b(π, i), rather than a capture error rate; experienced burden is explicitly out of scope |
| Distributive justice (worst-off and prioritarian concern) | tail and worse-off endpoints (CVaR80, share worse off, paired Δ_i) are evaluated alongside the mean; tail-constrained and multi-objective comparisons are preferred to one weighted sum; λ is left normative |
| Procedural fairness and recourse | the step-up is analysed as an added route with its own failure risk (C-S3), not as recourse; π5 is gated |
| Accountability and auditability | composition-conditioned nulls, finite-sample estimators and a variance decomposition test whether the distributional claims survive alternative explanations before any policy conclusion uses them |
| Small-sample statistics | 34 subject clusters drive every interval; tails are reported with leave-one-out ranges; CVaR90 is demoted if unstable |
