"""Input normalization policies compared in the paper.

N_legacy (per-image min--max) and N_fixed (g / 255). The two functions and the registry are copied
verbatim from the research code; `range_ratio` is written for this release from the manuscript's
definition of the normalization dose rho(x) = a_x / a_{R(x)}.
"""
from __future__ import annotations

import numpy as np


def _minmax(g: np.ndarray) -> np.ndarray:
    lo, hi = float(g.min()), float(g.max())
    return (g.astype(np.float32) - lo) / max(hi - lo, 1e-6)


def _div255(g: np.ndarray) -> np.ndarray:
    """E027 N_fixed: the exact 8-bit digital-range mapping.

    No statistic is estimated from the image, the split, the condition, the class or the generator.
    """
    return g.astype(np.float32) / 255.0


NORMALIZERS = {"minmax": _minmax, "div255": _div255}


def get_normalizer(name: str = "minmax"):
    """E027. Default is `minmax`, the frozen legacy harness preprocessing: every existing run
    directory and every historical result loads through the unchanged default path."""
    if name not in NORMALIZERS:
        raise ValueError(f"unknown normalization {name!r}; known: {sorted(NORMALIZERS)}")
    return NORMALIZERS[name]


def range_ratio(x: np.ndarray, rx: np.ndarray) -> float:
    """Normalization dose rho(x) for per-image min--max: the fraction of the image's dynamic range that
    the operator keeps, (max R(x) - min R(x)) / (max x - min x). rho < 1 means N_legacy re-expands R(x)
    by 1 / rho. Written for this release from the manuscript's equation for rho; not a frozen script."""
    a = float(x.max()) - float(x.min())
    b = float(rx.max()) - float(rx.min())
    return b / max(a, 1e-6)
