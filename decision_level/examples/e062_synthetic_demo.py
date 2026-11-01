#!/usr/bin/env python3
"""E062 end-to-end on SYNTHETIC data only (no fingerprint, label, identifier or score of the study is used).

1. Appearance grouping: four synthetic image statistics -> corpus-wide tertile votes -> training label when at least
   3 of 4 votes agree -> unscaled Euclidean 9-NN over all items (ra_provenance_check.reconstruct, the rule
   reconstructed from the study's label table). Reproducing a rule says nothing about whether the classes are valid.
2. Stratified burden: synthetic rejection indicators for 12 cells x 2 arms x 10 seeds, a planted RA-dry effect in the
   legacy arm; ra_analysis.run_all (primary, secondary, S-1...S-7, S-8w) with a small bootstrap.
3. Tables: ra_report.render, the renderer of the manuscript's Supplementary tables.

    python decision_level/examples/e062_synthetic_demo.py
Every number printed is SYNTHETIC and is not a result of the paper.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "E062_quality_conditioned_eval"))
sys.path.insert(0, str(ROOT / "tests"))
import ra_provenance_check as G  # noqa: E402
from experiments.E062_quality_conditioned_eval import ra_analysis as E  # noqa: E402
from experiments.E062_quality_conditioned_eval import ra_report as RP  # noqa: E402
from test_e062_ra_analysis import synth  # noqa: E402  (synthetic pool generator of the tests)


def main():
    print("=== SYNTHETIC DEMONSTRATION — not study data, not a paper result ===\n")
    rng = np.random.default_rng(2026)
    X = np.column_stack([rng.normal(77, 2, 3000), rng.normal(0.49, 0.03, 3000),
                         rng.normal(330, 20, 3000), rng.normal(15.5, 1.2, 3000)])   # four synthetic statistics
    g = G.reconstruct(X)
    print("1. Appearance grouping (synthetic):", np.bincount(g["pred"], minlength=3).tolist(),
          "items per class; vote-labelled:", int(g["train"].sum()), "\n")
    data = synth(n_subj=30, seed=7, effects={("leg", 0): 0.15})
    res = E.run_all(data, B=200, seed=E.SEED_DRAFT)
    print("2. Stratified burden (synthetic; planted RA-dry effect under legacy normalization)\n")
    print(RP.render(res))


if __name__ == "__main__":
    main()
