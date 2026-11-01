#!/usr/bin/env python3
"""Verify that the released analysis sources are byte-identical to the versions used for the authorised test reads.

Pinned = the SHA-256 recorded in the test-read authorisation of the analysis (E060, E061, E062) or, for E062's
renderer and calibration code, in the E062 freeze record. Public adapters and tests are not listed: they differ
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
}


def main():
    bad = 0
    for rel, (want, src) in PINNED.items():
        got = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
        ok = got == want
        bad += not ok
        print(f"{'OK  ' if ok else 'FAIL'} {rel}  {got[:12]}  ({src})")
    print("all pinned sources identical" if not bad else f"{bad} source(s) differ")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
