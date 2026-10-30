"""Ridge-period and foreground diagnostics used by the patch builders.

Ported verbatim (round_half_up, foreground_mask, estimate_ridge_period) from
`Controlnet/fingerprint_gen_v2/python/00_audit/preprocessing_common.py` so the
repo has no external code dependency. Matcher-independent; used only to
normalise every source to the reference ridge period and to pick foreground
crops (plan §10.1). Never a biometric-quality label.
"""
from __future__ import annotations

import math

import cv2
import numpy as np


def round_half_up(value: float) -> int:
    return int(math.floor(float(value) + 0.5))


def foreground_mask(gray: np.ndarray, border_fraction: float = 0.05, min_difference: float = 12.0) -> np.ndarray:
    """Frozen, matcher-independent foreground diagnostic based on border contrast."""
    image = np.asarray(gray, dtype=np.uint8)
    height, width = image.shape
    border = max(1, round_half_up(min(height, width) * border_fraction))
    pixels = np.concatenate((
        image[:border, :].ravel(), image[-border:, :].ravel(),
        image[:, :border].ravel(), image[:, -border:].ravel(),
    ))
    background = float(np.median(pixels))
    difference = np.abs(image.astype(np.float32) - background)
    threshold = max(float(min_difference), float(np.percentile(difference, 65)) * 0.45)
    mask = (difference >= threshold).astype(np.uint8)
    kernel_size = max(3, 2 * round_half_up(min(height, width) / 160) + 1)
    kernel_size = min(kernel_size, 15)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return mask.astype(bool)
    minimum = max(16, round_half_up(height * width * 0.0005))
    keep = np.zeros_like(mask, dtype=bool)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= minimum:
            keep |= labels == label
    return keep


def estimate_ridge_period(gray: np.ndarray, min_period: float = 3.0, max_period: float = 40.0,
                          max_side: int = 768) -> tuple[float | None, float]:
    """Apparent ridge period (px) and a confidence in [0,1] from a radial high-pass FFT."""
    image = np.asarray(gray, dtype=np.float32)
    height, width = image.shape
    analysis_scale = min(1.0, float(max_side) / max(height, width))
    if analysis_scale < 1.0:
        new_size = (max(16, round_half_up(width * analysis_scale)), max(16, round_half_up(height * analysis_scale)))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    if min(image.shape) < 32:
        return None, 0.0
    sigma = max(2.0, 0.35 * min_period * analysis_scale)
    highpass = image - cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    mask = foreground_mask(np.clip(image, 0, 255).astype(np.uint8))
    if mask.mean() < 0.01:
        mask = np.ones_like(image, dtype=bool)
    highpass *= mask.astype(np.float32)
    highpass -= float(highpass[mask].mean()) if np.any(mask) else float(highpass.mean())
    window = np.outer(np.hanning(image.shape[0]), np.hanning(image.shape[1])).astype(np.float32)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(highpass * window))) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(image.shape[0]))
    fx = np.fft.fftshift(np.fft.fftfreq(image.shape[1]))
    radius = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    low = analysis_scale / max_period
    high = min(0.48, analysis_scale / min_period)
    valid = (radius >= low) & (radius <= high)
    if not np.any(valid) or float(spectrum[valid].sum()) <= 0:
        return None, 0.0
    bins = np.linspace(low, high, 160)
    which = np.digitize(radius[valid], bins) - 1
    radial = np.bincount(which, weights=spectrum[valid], minlength=len(bins))[: len(bins) - 1]
    counts = np.bincount(which, minlength=len(bins))[: len(bins) - 1]
    radial = radial / np.maximum(counts, 1)
    peak = int(np.argmax(radial))
    frequency = float((bins[peak] + bins[peak + 1]) / 2.0)
    if frequency <= 0:
        return None, 0.0
    period_native = analysis_scale / frequency
    baseline = float(np.median(radial)) + 1e-12
    confidence = float(np.clip((radial[peak] / baseline - 1.0) / 10.0, 0.0, 1.0))
    return period_native, confidence
