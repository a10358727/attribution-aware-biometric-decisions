#!/usr/bin/env python3
"""Rebuild the C5 manuscript macros and tables (Sections VI-VII, Supplementary S10) from the released E072/E073
aggregates and compare them with the manuscript's files, ignoring comment lines (they record source paths)."""
import subprocess, sys
from pathlib import Path
R = Path(__file__).resolve().parents[1] / "results"
subprocess.run([sys.executable, str(Path(__file__).parent / "c5_mk_macros.py")], check=True, capture_output=True)
bad = 0
for f in sorted((R / "C5_manuscript_macros").glob("*.tex")):
    want = [l for l in f.read_text().splitlines() if not l.startswith("%")]
    got = [l for l in (R / "C5_macros_rebuilt" / f.name).read_text().splitlines() if not l.startswith("%")]
    ok = want == got; bad += not ok
    print(f"{'OK  ' if ok else 'DIFF'} {f.name} ({len(want)} lines)")
print("all C5 macro files reproduced from released aggregates" if not bad else f"{bad} file(s) differ")
sys.exit(1 if bad else 0)
