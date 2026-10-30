"""Label-blind restoration operators applied to degraded 88 x 88 grey patches (uint8 in, uint8 out).

Copied verbatim from the research code: `r_nlm`, `r_bilateral`, `r_tv`, `r_wiener`, `r_median`
(condition builder) and `bandstop` (operator sweep). The learned restorer of the research code is not
included (its checkpoint and code are not released). `RESTORERS` lists the operators below only.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.signal import wiener as wiener_filter
from skimage.restoration import denoise_tv_chambolle


def r_nlm(batch: list[np.ndarray]) -> list[np.ndarray]:
    return [cv2.fastNlMeansDenoising(g, None, h=10, templateWindowSize=7, searchWindowSize=21)
            for g in batch]


def r_bilateral(batch: list[np.ndarray]) -> list[np.ndarray]:
    return [cv2.bilateralFilter(g, d=7, sigmaColor=45, sigmaSpace=7) for g in batch]


def r_tv(batch: list[np.ndarray]) -> list[np.ndarray]:
    return [(np.clip(denoise_tv_chambolle(g.astype(np.float32) / 255.0, weight=0.08), 0, 1) * 255)
            .astype(np.uint8) for g in batch]


def r_wiener(batch: list[np.ndarray]) -> list[np.ndarray]:
    out = []
    for g in batch:
        y = wiener_filter(g.astype(np.float64), mysize=5)
        out.append(np.clip(np.nan_to_num(y, nan=float(g.mean())), 0, 255).astype(np.uint8))
    return out


def r_median(batch: list[np.ndarray]) -> list[np.ndarray]:
    return [cv2.medianBlur(g, 3) for g in batch]


def bandstop(g: np.ndarray, half_width: float, period_px: float) -> np.ndarray:
    """Zero an annulus of radial frequency around f_ridge = 1/period (half-width relative to f_ridge)."""
    h, w = g.shape; f = np.fft.fftshift(np.fft.fft2(g.astype(np.float32)))
    fy = np.fft.fftshift(np.fft.fftfreq(h)); fx = np.fft.fftshift(np.fft.fftfreq(w)); r = np.hypot(fy[:, None], fx[None, :])
    fr = 1.0 / period_px; mask = (r >= fr * (1 - half_width)) & (r <= fr * (1 + half_width))
    f[mask] = 0
    y = np.real(np.fft.ifft2(np.fft.ifftshift(f)))
    return np.clip(y, 0, 255).astype(np.uint8)


RESTORERS = {"nlm": r_nlm, "bilateral": r_bilateral, "tv": r_tv,
             "wiener": r_wiener, "median": r_median}
