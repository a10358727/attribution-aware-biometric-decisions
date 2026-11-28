#!/usr/bin/env python3
"""Verify that the released analysis sources are byte-identical to the versions used for the authorised test reads.

Pinned = the SHA-256 recorded in the test-read authorisation of the analysis (E060-E073) or, for E062's renderer and
calibration code and E063 Part A, in the freeze record or run output. REDACTED lists the public copies that differ from
the pinned bytes only by a declared local-path redaction. Public adapters and tests are not listed: they differ
from the research versions by design (README, PUBLIC_RELEASE_SCOPE.md).
    python scripts/verify_frozen_sources.py      # exit 0 = all identical
"""
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DL = "decision_level/experiments"
PINNED = {
    f"{DL}/tcss_lib/core.py": ("ba15e60ea66ef1d7c4d1abfd71d1473940b0415f424fe776fa3409d7acf04158", "E060/E061/E062 authorisations"),
    f"{DL}/tcss_lib/pool.py": ("6ad1c002077f34a64022d4e2d34e58e6a1585d685f62722d7eb6b78eb21b1af6", "E060/E061/E062 authorisations"),
    f"{DL}/E060_credential_burden/run_e060.py": ("4f90df1a87f7e83fcfa56f01ee5f8ba6a9011d39e11af16a97cd3f141aac46f5", "E060/E062 authorisations"),
    f"{DL}/E061_policy_simulation/run_e061.py": ("a5515674a1e7a9ee8cf720e0faa3428d4eb42bdd75bc920dcbc6e83660aa78e0", "E061 authorisation"),
    f"{DL}/E062_quality_conditioned_eval/ra_analysis.py": ("b52bb69e9f5b0c7b63c6bf3f747c9e99c825f5d33f4482a8c38a2314e826875d", "E062 authorisation"),
    f"{DL}/E062_quality_conditioned_eval/ra_report.py": ("638d035c5d1744de174dcd45f6a4940197010ae23e278a96c6f76aee8c60fa77", "E062 freeze (renderer used by the run)"),
    f"{DL}/E062_quality_conditioned_eval/ra_prefreeze.py": ("b9ab68ffb8cf0b37477aec6d6b93e41385c9fc7a2e16b3021219698604905d4f", "E062 freeze record (K1-K4 calibration)"),
    f"{DL}/E062_quality_conditioned_eval/ra_calibration.py": ("41c86f8a81933df1db2c2dfe7f20b645f56daf4c219258c9f325772afe8eb645", "E062 freeze record"),
    f"{DL}/E062_quality_conditioned_eval/ra_provenance_check.py": ("b5bfaefedccf7eda7942ef644d47b0a42dcbf9aeccae69dfc47937bf0f96ff45", "E062 freeze record (label-rule reconstruction)"),
    # --- v1.1.0 additions (C1): byte-identical to the pinned versions
    f"{DL}/E063_session_robustness/session_models.py": ("0a64991988ebbe8b27b79e792a3da12e1f32b129111f0ecf2f6793504c5b537c", "E063 test-read authorisation"),
    f"{DL}/E063_session_robustness/breakeven.py": ("7163e10aa0f1f236796b64656de5cabde5f092b709c20876ea033e00bb4de086", "E063 freeze commit ad1e976 (Part A r2; superseded)"),
    f"{DL}/E063_session_robustness/breakeven_r3_signcorrected.py": ("2f4ec3e89ecdaa3596f9b20fc19e35766bd919d81f62b81b07a39f78733d332e", "script_sha256 recorded in the Part A r3 output (summary.json)"),
    f"{DL}/E064_session_persistence_K_attacker/run_e064.py": ("dc3c4313defd58d7c2e354519d773c02e39efbf0be0a2ebb7b7dd4e7bb8de084", "E064 test-read authorisation"),
    f"{DL}/E064_session_persistence_K_attacker/session_m2.py": ("73cbace43bb832eb7b6f3eadaf6405dbbbe2725c7e40520b25c2aad36f446abb", "E064 test-read authorisation"),
    f"{DL}/E065_matcher_cascade_genuine/run_e065.py": ("212a2663b8eb94fa1654a8c51ac513277510da557a5d9eae82ee2316d710e9ae", "E065 test-read authorisation"),
    f"{DL}/E067_validation_refit_uncertainty/refit.py": ("527c37d79f4c4363bbbe23c7dd3e24a2c5dca4a1f2f790fe2e786017380e4311", "E067 authorisations"),
    f"{DL}/E067_validation_refit_uncertainty/run_e067.py": ("8650bbbe8f23abfc2f09e5b5a960563e4a33f9902292526386c001f9c49a26f0", "E067 authorisations"),
    f"{DL}/E067_validation_refit_uncertainty/run_e067_stage_t.py": ("f93f8a421d9f066cdc57a53af40c3f01278f71762eab34192985ed5072fed638", "E067 authorisations"),
    f"{DL}/E071_matcher_verification/run_e071.py": ("9701048192540b11f421badfacc58b9dc8b34fba76d976bb2c8fbd5ce0925fde", "E071 test-read authorisation"),
    f"{DL}/E068_fingerprint_enhancement_condition/e068_common.py": ("2746175523bbe2ca777c8b5745e1653d376a0bfe6527f105c409e9cdda8a0e16", "E068 test-read authorisation (L3b)"),
    f"{DL}/E068_fingerprint_enhancement_condition/e068_analyze.py": ("dc495b7cb47bbb410b212baef477d44ba56529680ba782c7335c00f467c06ac1", "E068 test-read authorisation (L3b)"),
    f"{DL}/E049_learned_restorer_sensitivity/analyze.py": ("05cafa5f848a6b02ea4bf86279d4d8380dfce16427afdd1b0c5841b44e854e7a", "E068 test-read authorisation (L3b)"),
    f"{DL}/E036_interaction_statistics/run_e036.py": ("4ffac28c4b319ca5a12e8af5296e5a538adda39fa91518bfa72a8b62fbaa4de8", "E068 test-read authorisation (L3b)"),
    f"{DL}/E069_cross_sensor_bonafide_transfer/e069_common.py": ("cf5cdd73302f6d5b53939bd13aa412f2856ae6f6113785ab34d65d3c2fca4c40", "E068 test-read authorisation (L3b)"),
    f"{DL}/E069_cross_sensor_bonafide_transfer/e069_analyze.py": ("2c46e84eadfd479adce11418fb072b87ec3be7dc342e9ff1639a533492193363", "E069 outcome-read authorisation"),
    f"{DL}/E070_quality_conditioned_external/e070_contrast.py": ("48fa899dc647de1346ee1d2117a0b060f60af75464223ced041a139b19a976dd", "E070 authorisation"),
    # --- v1.2.0 additions (C5: E072, E073): byte-identical to the versions pinned by their test-read authorisations
    f"{DL}/c5_lib/stats.py": ("561c72b2998b1d09c3d801b787df19b752c33108596c8d854e5a6cde3ee6213c", "E072/E073 test-read authorisations"),
    f"{DL}/E072_burden_validity/run_e072.py": ("63877ae7ddc64782cb674bc5f80878e9260dab93d54c0a5c3ab6326d67847cdb", "E072 test-read authorisation"),
    f"{DL}/E073_distribution_policy/run_e073.py": ("3da37980e317fdb184a1062b44e3f1b33f8083e1ba36b5a6d52693e52ff509ee", "E073 test-read authorisation"),
}
# Public copies whose ONLY difference from the pinned research bytes is a declared local-path redaction
# (research SHA-256 pinned by the authorisation; public SHA-256 of the redacted copy; see RELEASE_MANIFEST_C1.json).
REDACTED = {
    f"{DL}/E063_session_robustness/run_e063.py": ("0916ae3479e983b75b7b431c1830efe98004156be62d288de814faf0b9450138", "4192bae95eefc43265ab97de40123e0a95eb38f134fb351cd7edb8598aad4992", "E063 test-read authorisation"),
}


def main():
    bad = 0
    for rel, (want, src) in PINNED.items():
        got = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
        ok = got == want
        bad += not ok
        print(f"{'OK  ' if ok else 'FAIL'} {rel}  {got[:12]}  ({src})")
    for rel, (research, public, src) in REDACTED.items():
        got = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
        ok = got == public
        bad += not ok
        print(f"{'OK  ' if ok else 'FAIL'} {rel}  {got[:12]}  (redacted copy of {research[:12]}, {src})")
    print("all pinned sources identical" if not bad else f"{bad} source(s) differ")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
