#!/usr/bin/env python3
"""Score 88x88 grey fingerprint patches with a released detector checkpoint.

Self-contained re-statement of the detector classes used in the paper (research repository
`diar/baselines/models.py`, `models_e025.py`, `models_e035.py`, `models_e031.py`): the parameter names
are identical, so a released `best.pt` loads with `strict=True`.

    python inference/score_patch.py --model-type npr --norm minmax --ckpt best.pt a.png b.png

`--norm` must be the arm the checkpoint was trained under:
    minmax   legacy  N_legacy : (g - min g) / max(max g - min g, 1e-6), per image
    div255   fixed   N_fixed  : g / 255
The two arms are not interchangeable; that is the paper's finding, not a nuisance.

Output: one line per image, `<path>\t<sigmoid score>`; higher = more likely synthetic. No threshold
is shipped (the paper's thresholds are validation quantiles of the research protocol).
"""
from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PATCH = 88


# ------------------------------------------------------------------ normalization arms
def norm_minmax(g: np.ndarray) -> np.ndarray:
    lo, hi = float(g.min()), float(g.max())
    return (g.astype(np.float32) - lo) / max(hi - lo, 1e-6)


def norm_div255(g: np.ndarray) -> np.ndarray:
    return g.astype(np.float32) / 255.0


NORMALIZERS = {"minmax": norm_minmax, "div255": norm_div255}


# ------------------------------------------------------------------ detector classes
def _resnet(name: str) -> nn.Module:
    import torchvision.models as tvm
    return {"resnet18": tvm.resnet18, "resnet50": tvm.resnet50}[name](weights=None)


def _to_gray_stem(m: nn.Module) -> nn.Module:
    conv = nn.Conv2d(1, m.conv1.out_channels, m.conv1.kernel_size, m.conv1.stride,
                     m.conv1.padding, bias=False)
    m.conv1 = conv
    return m


class _Logit(nn.Module):
    def logits(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def forward(self, x):
        return self.logits(x)

    @torch.no_grad()
    def score(self, x):
        return torch.sigmoid(self.logits(x))


class ResNet18E0(_Logit):
    """ResNet-18 with a 3x3 stride-1 stem and no max-pool (native 88 px), 1 channel, random init."""

    def __init__(self, in_ch: int = 1) -> None:
        super().__init__()
        from torchvision.models import resnet18
        m = resnet18(weights=None, num_classes=1)
        m.conv1 = nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        self.net = m

    def logits(self, x):
        return self.net(x).squeeze(1)


class BCESupCon(_Logit):
    """Benjamin et al.-style base detector: the E0 ResNet-18 trunk, a BCE head and a projection head
    (the projection only matters in training; it is kept so the state_dict loads strictly)."""

    def __init__(self, proj_dim: int = 128) -> None:
        super().__init__()
        m = ResNet18E0().net
        self.feat_dim = m.fc.in_features
        m.fc = nn.Identity()
        self.trunk = m
        self.head = nn.Linear(self.feat_dim, 1)
        self.proj = nn.Sequential(nn.Linear(self.feat_dim, 256), nn.GELU(), nn.Linear(256, proj_dim))

    def logits(self, x):
        return self.head(self.trunk(x)).squeeze(1)


class NPR(_Logit):
    """Neighbouring-pixel-relationship residual x - up(down(x)) (nearest, factor 2) -> ResNet-50."""

    def __init__(self, factor: int = 2) -> None:
        super().__init__()
        m = _resnet("resnet50")
        m.fc = nn.Linear(m.fc.in_features, 1)
        self.net = m
        self.factor = factor

    @staticmethod
    def npr_map(x: torch.Tensor, factor: int = 2) -> torch.Tensor:
        d = F.interpolate(x, scale_factor=1.0 / factor, mode="nearest")
        return x - F.interpolate(d, scale_factor=factor, mode="nearest")

    def logits(self, x):
        r = self.npr_map(x, self.factor)
        return self.net(r.expand(-1, 3, -1, -1)).squeeze(1)


class ResNetImageNet(_Logit):
    """torchvision ResNet-18/50 with the ImageNet stem collapsed to one channel (sum of RGB filters)."""

    def __init__(self, name: str) -> None:
        super().__init__()
        m = _to_gray_stem(_resnet(name))
        m.fc = nn.Linear(m.fc.in_features, 1)
        self.net = m

    def logits(self, x):
        return self.net(x).squeeze(1)


class EfficientNetB0ImageNet(_Logit):
    def __init__(self) -> None:
        super().__init__()
        import torchvision.models as tvm
        m = tvm.efficientnet_b0(weights=None)
        stem = m.features[0][0]
        m.features[0][0] = nn.Conv2d(1, stem.out_channels, stem.kernel_size, stem.stride,
                                     stem.padding, bias=stem.bias is not None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
        self.net = m

    def logits(self, x):
        return self.net(x).squeeze(1)


class ConvNeXtTinyImageNet(_Logit):
    def __init__(self) -> None:
        super().__init__()
        import torchvision.models as tvm
        m = tvm.convnext_tiny(weights=None)
        stem = m.features[0][0]
        m.features[0][0] = nn.Conv2d(1, stem.out_channels, stem.kernel_size, stem.stride,
                                     stem.padding, bias=stem.bias is not None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, 1)
        self.net = m

    def logits(self, x):
        return self.net(x).squeeze(1)


MODEL_TYPES = {
    "resnet18_e0": lambda: ResNet18E0(),
    "bcesupcon": lambda: BCESupCon(),
    "npr": lambda: NPR(),
    "resnet18_imagenet": lambda: ResNetImageNet("resnet18"),
    "resnet50_imagenet": lambda: ResNetImageNet("resnet50"),
    "efficientnet_b0_imagenet": lambda: EfficientNetB0ImageNet(),
    "convnext_tiny_imagenet": lambda: ConvNeXtTinyImageNet(),
    # "d3_84": depends on vendored third-party code (D3 + CLIP); not provided here.
}


# ------------------------------------------------------------------ I/O
def load_patch(path: str, norm: str) -> torch.Tensor:
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise SystemExit(f"cannot read {path}")
    if g.shape != (PATCH, PATCH):
        raise SystemExit(f"{path}: expected {PATCH}x{PATCH} grey, got {g.shape}; "
                         "the paper's detectors are native-resolution and are not resized")
    x = NORMALIZERS[norm](g)
    return torch.from_numpy(x)[None, None]  # (1,1,88,88)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-type", required=True, choices=sorted(MODEL_TYPES))
    ap.add_argument("--ckpt", required=True, help="released best.pt (a plain state_dict)")
    ap.add_argument("--norm", required=True, choices=sorted(NORMALIZERS),
                    help="the arm the checkpoint was trained under")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("images", nargs="+", help="88x88 grey PNG/BMP patches")
    a = ap.parse_args(argv)

    model = MODEL_TYPES[a.model_type]()
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=True)
    model.eval().to(a.device)

    for p in a.images:
        x = load_patch(p, a.norm).to(a.device)
        s = float(model.score(x)[0])
        print(f"{p}\t{s:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
