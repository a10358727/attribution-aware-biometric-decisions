### Table S-E1. Test-pool composition by RA class
| RA class | Captures | Subjects | Credentials |
|---|---|---|---|
| RA-dry | 403 | 29 | 148 |
| RA-middle | 591 | 34 | 215 |
| RA-wet | 659 | 34 | 208 |

### Table S-E2. Bona-fide rejection by RA class (median over eligible headline cells)
| Arm | RA class | Anchor B (%) | Restored B (%) | ΔB (pp) [95 %] |
|---|---|---|---|---|
| legacy (per-image min–max) | RA-dry | 1.67 | 21.51 | +19.90 [+14.75, +24.12] |
| legacy (per-image min–max) | RA-middle | 1.24 | 16.26 | +14.97 [+11.01, +18.14] |
| legacy (per-image min–max) | RA-wet | 1.05 | 11.68 | +10.64 [+7.78, +13.78] |
| fixed (/255) | RA-dry | 1.72 | 8.49 | +6.58 [+4.57, +9.44] |
| fixed (/255) | RA-middle | 1.28 | 5.41 | +3.97 [+3.02, +5.44] |
| fixed (/255) | RA-wet | 1.25 | 5.20 | +4.04 [+2.70, +5.39] |

### Table S-E3. Primary contrasts (computed per cell, then median over cells)
| Contrast | Arm | Cells | Median (pp) | 99.5 % CI (pp) | Statistical reading | Practical reading | S-8w median (pp) | S-8w 99.5 % CI | Concordance |
|---|---|---|---|---|---|---|---|---|---|
| θ^dry = ΔB(RA-dry) − ΔB(RA-middle) | legacy (per-image min–max) | 12 | +3.47 | [-1.10, +9.39] | includes 0 | P4 | +2.26 | [-2.77, +6.48] | C-null |
| θ^dry = ΔB(RA-dry) − ΔB(RA-middle) | fixed (/255) | 12 | +1.81 | [-0.80, +5.04] | includes 0 | P4 | +0.57 | [-2.09, +3.26] | C-null |
| θ^wet = ΔB(RA-wet) − ΔB(RA-middle) | legacy (per-image min–max) | 12 | -3.15 | [-6.40, +0.24] | includes 0 | P4 | -1.33 | [-4.65, +2.93] | C-null |
| θ^wet = ΔB(RA-wet) − ΔB(RA-middle) | fixed (/255) | 12 | -0.29 | [-2.17, +1.56] | includes 0 | P4 | +0.01 | [-2.01, +2.46] | C-null |

### Table S-E4. Secondary quantities (paired per cell, then median)
| Quantity | Median (pp) | 97.5 % CI (pp) | Reading |
|---|---|---|---|
| dtheta_dry|fix-leg | -2.22 | [-5.41, +0.95] | includes 0 |
| dtheta_wet|fix-leg | +3.36 | [+1.16, +5.32] | above 0 |
| wet_minus_dry|leg | -7.15 | [-11.59, -2.49] | below 0 |
| wet_minus_dry|fix | -3.04 | [-5.41, -0.26] | below 0 |

### Table S-E5. Pre-specified sensitivity analyses (median, 99.5 % CI, pp)
| Analysis | θ^dry legacy | θ^dry fixed | θ^wet legacy | θ^wet fixed |
|---|---|---|---|---|
| S-1 | +3.79 [-0.41, +9.17] | +2.05 [-0.37, +4.92] | -1.89 [-4.77, +1.15] | +0.27 [-1.62, +2.35] |
| S-2 | +3.17 [-1.44, +8.94] | +1.74 [-0.88, +4.90] | -2.40 [-5.97, +1.42] | +0.21 [-1.93, +2.30] |
| S-3a_sev12 | +5.42 [-1.73, +11.70] | +1.66 [-1.61, +4.64] | -3.07 [-6.14, +1.46] | +0.65 [-1.20, +2.99] |
| S-3b_sev34 | +1.91 [-2.35, +8.46] | +2.55 [-0.71, +7.06] | -4.09 [-8.83, +0.47] | -1.42 [-3.58, +1.49] |
| S-4 | +5.35 [-0.73, +11.43] | +2.93 [-0.05, +6.67] | -2.77 [-7.42, +1.33] | -0.29 [-2.41, +2.25] |
| S-5 | +12.36 [+3.57, +20.04] | +5.89 [+0.61, +11.10] | -5.23 [-10.91, -1.73] | -1.34 [-3.47, +0.23] |
| S-6_bil | +3.91 [-1.33, +10.30] | +2.20 [-0.63, +6.52] | -4.16 [-7.93, -0.03] | -0.87 [-3.40, +1.38] |
| S-6_tv | +3.46 [-1.00, +8.96] | +1.41 [-1.41, +4.45] | -2.32 [-5.34, +0.73] | +0.11 [-1.49, +2.30] |
| S-7 | +3.49 [-1.54, +9.86] | +1.82 [-1.03, +5.47] | -3.63 [-7.30, -0.13] | -0.16 [-2.26, +1.82] |
| S-8w_dry | +2.26 [-2.77, +6.48] | +0.57 [-2.09, +3.26] | n/a (not an estimand) | n/a (not an estimand) |
| S-8w_wet | n/a (not an estimand) | n/a (not an estimand) | -1.33 [-4.65, +2.93] | +0.01 [-2.01, +2.46] |

### Mechanical readings
- θ^dry = ΔB(RA-dry) − ΔB(RA-middle), legacy (per-image min–max): no difference detected in the primary analysis or within credentials, at this precision; practical: inconclusive about practical size.
- θ^dry = ΔB(RA-dry) − ΔB(RA-middle), fixed (/255): no difference detected in the primary analysis or within credentials, at this precision; practical: inconclusive about practical size.
- θ^wet = ΔB(RA-wet) − ΔB(RA-middle), legacy (per-image min–max): no difference detected in the primary analysis or within credentials, at this precision; practical: inconclusive about practical size.
- θ^wet = ΔB(RA-wet) − ΔB(RA-middle), fixed (/255): no difference detected in the primary analysis or within credentials, at this precision; practical: inconclusive about practical size.
