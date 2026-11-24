#!/usr/bin/env python3
"""Rebuild the C1 manuscript macros and tables (Sections VIII and X, Supplementary S7-S9) from the released aggregates and
compare them with the manuscript's files, ignoring comment lines (they record source paths and hashes)."""
import subprocess, sys
from pathlib import Path
R = Path(__file__).resolve().parents[1] / "results"
for s in ("c1_mk_sens.py", "c1_mk_ext.py"):
    subprocess.run([sys.executable, str(Path(__file__).parent / s)], check=True, capture_output=True)
bad = 0
for f in sorted((R / "C1_manuscript_macros").glob("*.tex")):
    want = [l for l in f.read_text().splitlines() if not l.startswith("%")]
    got = [l for l in (R / "C1_macros_rebuilt" / f.name).read_text().splitlines() if not l.startswith("%")]
    ok = want == got; bad += not ok
    print(f"{'OK  ' if ok else 'DIFF'} {f.name} ({len(want)} lines)")
print("all C1 macro files reproduced from released aggregates" if not bad else f"{bad} file(s) differ")
sys.exit(1 if bad else 0)
