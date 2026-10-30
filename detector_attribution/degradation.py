"""Symmetric acquisition-degradation operator bank (type x severity).

Applied identically to real and synthetic fingerprints — see
docs/history/plans/GenAI_Fingerprint_Modification_Plan.md §4.4 / §10.4 (symmetric degradation, §2.6). Nothing here may
branch on the authenticity label.

Severity 0 is the identity (clean); 1..4 increase in strength. Every operator
works on uint8 grayscale and returns uint8 grayscale of the same shape.

Ridge polarity: fingerprint ridges are dark on a light background, so "wetness"
(ridge merging from a larger contact area) expands dark regions, i.e. grayscale
erosion, while "dry" thins them.
"""
from __future__ import annotations

import cv2
import numpy as np

TYPES = ("wet", "blur", "noise", "smear", "dry", "contrast")
SEVERITIES = (0, 1, 2, 3, 4)

# per-type parameter ladders, index = severity
_P = {
    # ladders calibrated so that severity 1..4 costs roughly the same SSIM
    # (~0.97 / 0.92 / 0.85 / 0.76) for every type; see calibrate_degradation.py
    "wet":      dict(k=[0, 2, 2, 3, 3], a=[0.0, 0.18, 0.34, 0.52, 0.70], s=[1.0, 1.05, 1.15, 1.28, 1.45]),
    "blur":     dict(sigma=[0.0, 0.62, 0.92, 1.22, 1.55]),
    "noise":    dict(sigma=[0.0, 9.0, 16.0, 24.0, 34.0]),
    "smear":    dict(length=[0, 3, 3, 5, 5], a=[0.0, 0.30, 0.55, 0.62, 0.85], elastic=[0.0, 0.4, 0.9, 1.5, 2.2]),
    "dry":      dict(k=[0, 2, 2, 3, 3], a=[0.0, 0.38, 0.62, 0.82, 0.96], g=[1.0, 0.95, 0.90, 0.84, 0.77]),
    "contrast": dict(ratio=[1.0, 0.86, 0.74, 0.61, 0.48]),
}


def _lut_scurve(steepness: float) -> np.ndarray:
    x = (np.arange(256) - 128.0) / 128.0
    return np.clip(128 + 128 * np.tanh(steepness * x) / np.tanh(steepness), 0, 255).astype(np.uint8)


def _lut_gamma(gamma: float) -> np.ndarray:
    return np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)], np.uint8)


def _lut_contrast(ratio: float) -> np.ndarray:
    mid = 128.0
    return np.clip(mid + (np.arange(256) - mid) * ratio, 0, 255).astype(np.uint8)


def _motion_kernel(length: int, angle_deg: float) -> np.ndarray:
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0
    m = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle_deg, 1.0)
    k = cv2.warpAffine(k, m, (length, length))
    total = k.sum()
    return k / total if total > 0 else k


def apply_degradation(gray: np.ndarray, kind: str, severity: int,
                      rng: np.random.Generator | None = None) -> np.ndarray:
    """Apply one (type, severity) operator. `rng` makes stochastic ops reproducible."""
    if kind not in TYPES:
        raise ValueError(f"unknown degradation type {kind!r}")
    if severity not in SEVERITIES:
        raise ValueError(f"severity must be 0..4, got {severity}")
    if severity == 0:
        return gray.copy()
    rng = rng or np.random.default_rng(0)
    g = np.asarray(gray, dtype=np.uint8)
    p = _P[kind]

    if kind == "wet":
        # dark ridges spread and contrast compresses toward the dark end;
        # blended with the original so low severities stay gentle
        k = int(p["k"][severity])
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        a = float(p["a"][severity])
        out = cv2.addWeighted(g, 1.0 - a, cv2.erode(g, se), a, 0.0)
        return cv2.LUT(out, _lut_scurve(float(p["s"][severity])))

    if kind == "blur":
        return cv2.GaussianBlur(g, (0, 0), float(p["sigma"][severity]))

    if kind == "noise":
        n = rng.normal(0.0, float(p["sigma"][severity]), g.shape)
        return np.clip(g.astype(np.float32) + n, 0, 255).astype(np.uint8)

    if kind == "smear":
        length = int(p["length"][severity])
        angle = float(rng.uniform(0, 180))
        a = float(p["a"][severity])
        out = cv2.addWeighted(g, 1.0 - a, cv2.filter2D(g, -1, _motion_kernel(length, angle)), a, 0.0)
        amp = float(p["elastic"][severity])
        if amp > 0:
            h, w = out.shape
            dx = cv2.GaussianBlur(rng.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), 12)
            dy = cv2.GaussianBlur(rng.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), 12)
            norm = max(float(np.abs(dx).max()), float(np.abs(dy).max()), 1e-6)
            dx, dy = dx / norm * amp, dy / norm * amp
            gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            out = cv2.remap(out, gx + dx, gy + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return out

    if kind == "dry":
        k = int(p["k"][severity])
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        a = float(p["a"][severity])
        out = cv2.addWeighted(g, 1.0 - a, cv2.dilate(g, se), a, 0.0)   # ridges thin out
        return cv2.LUT(out, _lut_gamma(float(p["g"][severity])))

    if kind == "contrast":
        return cv2.LUT(g, _lut_contrast(float(p["ratio"][severity])))

    raise AssertionError("unreachable")


def sample_degradation(key: str, allow_clean: bool = False) -> tuple[str, int]:
    """Deterministic (type, severity) assignment from a string key (e.g. a patch id)."""
    h = abs(hash(key)) if False else int.from_bytes(
        __import__("hashlib").blake2b(key.encode(), digest_size=8).digest(), "big")
    kind = TYPES[h % len(TYPES)]
    lo = 0 if allow_clean else 1
    sev = lo + (h // len(TYPES)) % (5 - lo)
    return kind, sev


if __name__ == "__main__":
    import sys
    img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE) if len(sys.argv) > 1 else \
        (np.random.default_rng(0).integers(0, 255, (88, 88))).astype(np.uint8)
    print(f"{'type':9} {'sev':>3}  {'mean':>7} {'std':>7}  {'d_mean':>7}")
    base_m, base_s = img.mean(), img.std()
    print(f"{'clean':9} {0:>3}  {base_m:7.2f} {base_s:7.2f}  {0.0:7.2f}")
    for t in TYPES:
        for s in (1, 4):
            o = apply_degradation(img, t, s, np.random.default_rng(1))
            print(f"{t:9} {s:>3}  {o.mean():7.2f} {o.std():7.2f}  {o.mean()-base_m:+7.2f}")
