#!/usr/bin/env python3
"""Rebuild the E062 Supplementary tables (S14-S18 of the manuscript; template tables S-E1...S-E5) from the released
aggregate output only, and check them against the tables written by the authorised run.

Reads results/E062/E062_formal_results.public.json (standard-JSON copy of the frozen output; non-estimand entries
are null instead of NaN, nothing else changed) and renders it with the frozen renderer ra_report.render. Recomputes
nothing from scores (none are released).

    python decision_level/scripts/e062_tables_from_aggregates.py      # exit 0 when identical
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.E062_quality_conditioned_eval import ra_report as RP  # noqa: E402

R = ROOT / "results" / "E062"


def nan_back(o):
    """null -> NaN, as in the frozen output, so the renderer sees exactly what it saw in the authorised run."""
    if isinstance(o, dict):
        return {k: nan_back(v) for k, v in o.items()}
    if isinstance(o, list):
        return [nan_back(v) for v in o]
    return math.nan if o is None else o


def main():
    doc = json.loads((R / "E062_formal_results.public.json").read_text())
    text = RP.render(nan_back(doc["results"]))
    ref = (R / "E062_supplementary_tables.md").read_text()
    print(text)
    same = text == ref
    print("IDENTICAL to results/E062/E062_supplementary_tables.md" if same else "DIFFERS from the released tables")
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
