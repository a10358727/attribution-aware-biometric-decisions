"""[Public adaptation: only the sys.path setup differs from the research version.] E062 reporting rules and Supplementary tables: SYNTHETIC run_all output only."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]   # decision_level/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiments.E062_quality_conditioned_eval import ra_analysis as E  # noqa: E402
from experiments.E062_quality_conditioned_eval import ra_report as RP  # noqa: E402
from test_e062_ra_analysis import synth  # noqa: E402


def j(lo, hi, floor=False):
    return {"lo": lo, "hi": hi, "median": (lo + hi) / 2, "floor_uninformative": floor}


@pytest.mark.parametrize("lo,hi,want", [(0.012, 0.05, "P1"), (-0.05, -0.011, "P1"), (0.002, 0.03, "P2"),
                                        (-0.004, 0.006, "P3"), (-0.02, 0.004, "P4"), (-0.03, 0.03, "P4")])
def test_practical_reading(lo, hi, want):
    assert RP.practical_reading(j(lo, hi)) == want


def test_practical_reading_not_interpretable():
    assert RP.practical_reading({"lo": None, "hi": None}) == "P-na"
    assert RP.practical_reading(j(0.02, 0.05, floor=True)) == "P-na"


@pytest.mark.parametrize("p,w,want", [((0.01, 0.03), (0.005, 0.02), "C-same"), ((-0.01, 0.01), (-0.02, 0.02), "C-null"),
                                      ((0.01, 0.03), (-0.01, 0.02), "C-primary"), ((-0.01, 0.01), (0.001, 0.02), "C-within"),
                                      ((0.01, 0.03), (-0.03, -0.01), "C-opposite")])
def test_concordance(p, w, want):
    assert RP.concordance(j(*p), j(*w)) == want


def test_concordance_na():
    assert RP.concordance(j(0.01, 0.02), {"lo": None, "hi": None}) == "C-na"


def test_render_reports_everything_in_fixed_order():
    res = E.run_all(synth(seed=21, effects={("leg", 0): 0.2}), B=30, seed=0)
    text = RP.render(res)
    for t in ("S-E1", "S-E2", "S-E3", "S-E4", "S-E5", "Mechanical readings"):
        assert t in text
    pos = [text.index(f"| {lab} |") for lab in RP.ORDER[1:]]
    assert pos == sorted(pos)                                     # every sensitivity row, fixed order
    assert text.count("| θ^dry = ") == 2 and text.count("| θ^wet = ") == 2      # two arms per contrast
    assert "n/a (not an estimand)" in text                          # S-8w reports only its own contrast
    assert text.count("practical:") == 4
    assert "SUBJ" not in text and ".bmp" not in text
    assert "dry finger" not in text and "skin" not in text and "significant" not in text


def test_render_missing_analysis_is_shown_not_dropped():
    res = E.run_all(synth(seed=22), B=20, seed=0)
    res.pop("S-5")
    assert "| S-5 | not run |" in RP.render(res)
