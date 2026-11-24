"""E068 shared constants, the FpEnhancer wrapper and authorisation guards.

Protocol: docs/experiments/E068_fingerprint_enhancement_condition.md (v4). The enhancer (FpEnhancer `Enhancer`, pinned
commit, authors' weights), the E049 instruments, the E061 headline detectors at the F2 tau0 and ArcFace-88 at E071's
tau_v are frozen artefacts used unchanged. Nothing is trained.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

EXP = "E068"
OUT = ROOT / "artifacts" / "results" / EXP
PROTOCOL = ROOT / "docs" / "experiments" / "E068_fingerprint_enhancement_condition.md"
DATA = ROOT / "data" / "track_a"
PROTOS = ("p1", "p2")
ANCHOR = "degraded"
COND = "restored_fpenh"

# ------------------------------------------------------------------ enhancer (G1, G2: pinned and hashed)
VENDOR = ROOT / "vendor" / "E068" / "FpEnhancer"
FPENH_COMMIT = "b8c6923ea4698bd6dd8c0ac054709156dd2477bc"
WEIGHTS = ROOT / "vendor" / "E068" / "weights_download" / "Enhancer" / "best.pth"
WEIGHTS_SHA = "4b1cd28d7cad0c868747c8c8d542e891838d8e164e95d6719db1d62b9c88461f"
CONFIG = ROOT / "vendor" / "E068" / "weights_download" / "Enhancer" / "config_enhancer.yaml"
VENDOR_SHA = {   # recorded 2026-11-23 from the clone at FPENH_COMMIT
    "models/Enhancer.py": "b41f9b4d3aff0d028a13d60d7d587c097160595af8f0ee7334e09812fd108424",
    "models/codebook.py": "bc364f66a992a7d92965e2bd10f53cefefa0d6f4c44f4a1e3ad36e37ac004fa4",
    "inference_enhancer.py": "6f0ac3b9920bf1366270104b9d11a9692a9c13d262d7c9ba830cda8293bf70b9",
    "utils/data_loader.py": "a3da39438f7e53f26387b056ed18129407d858a2b04d09bd4d5dfc8326f162e1",
    "LICENSE": "d039e1af5ef20f591d61ed86cb9670bd028532eead17315aeeb346722431d89f",
    "ckpts/Enhancer/config_enhancer.yaml": "770fc1bd9513530e0aa1fe7a3c1c4c8a4abe52c6d714a4f667cb4f576394daab",
}
CONFIG_SHA = "770fc1bd9513530e0aa1fe7a3c1c4c8a4abe52c6d714a4f667cb4f576394daab"
MODEL_CFG = dict(img_channel=1, width=32, mid_blk_num=2, enc_blk_nums=[2, 2, 2], dec_blk_nums=[2, 2, 2],
                 dw_expand=1, ffn_expand=2)                              # = config_enhancer.yaml MODEL

SOURCES = [HERE, HERE.parent / "e068_enhance.py", HERE.parent / "e068_score.py", HERE.parent / "e068_analyze.py",
           ROOT / "experiments" / "E049_learned_restorer_sensitivity" / "score.py",
           ROOT / "experiments" / "E049_learned_restorer_sensitivity" / "analyze.py",
           ROOT / "experiments" / "E036_interaction_statistics" / "run_e036.py",
           ROOT / "experiments" / "E069_cross_sensor_bonafide_transfer" / "e069_score.py",
           ROOT / "experiments" / "E069_cross_sensor_bonafide_transfer" / "e069_common.py",
           ROOT / "experiments" / "E071_matcher_verification" / "run_e071.py",
           ROOT / "experiments" / "E065_matcher_cascade_genuine" / "run_e065.py",
           ROOT / "experiments" / "tcss_lib" / "core.py",
           ROOT / "diar" / "evaluate.py", ROOT / "diar" / "zoo.py", ROOT / "diar" / "data" / "dataset.py",
           ROOT / "diar" / "models" / "matcher88.py"]


class AccessDenied(PermissionError):
    pass


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def rel(p) -> str:
    p = Path(p).resolve()
    return str(p.relative_to(ROOT)) if str(p).startswith(str(ROOT)) else str(p)


def git(*a) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True)


def git_state() -> dict:
    run = lambda *a: subprocess.run(a, cwd=ROOT, capture_output=True, text=True).stdout
    diff = git("diff", "HEAD").stdout
    return {"git": run("git", "rev-parse", "HEAD").strip(),
            "dirty_tracked": run("git", "status", "--porcelain", "--untracked-files=no").splitlines(),
            "git_diff_sha256": sha_bytes(diff), "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "command": " ".join(sys.argv), "sources": {rel(s): sha256(s) for s in SOURCES if s.exists()}}


def blob_sha(commit: str, path: Path) -> str | None:
    r = git("show", f"{commit}:{rel(path)}")
    return sha_bytes(r.stdout) if r.returncode == 0 else None


def verify_freeze(commit: str, protocol_sha: str) -> None:
    """Freeze commit is an ancestor of HEAD, holds the authorised protocol bytes; the working protocol only APPENDS."""
    if git("merge-base", "--is-ancestor", commit, "HEAD").returncode != 0:
        raise AccessDenied(f"freeze commit {commit} is not an ancestor of HEAD")
    frozen = git("show", f"{commit}:{rel(PROTOCOL)}").stdout
    if sha_bytes(frozen) != protocol_sha:
        raise AccessDenied(f"{rel(PROTOCOL)} at {commit} is not the authorised protocol")
    if not PROTOCOL.read_bytes().startswith(frozen):
        raise AccessDenied(f"{rel(PROTOCOL)} was edited, not appended to, after the freeze")


def check_auth(fname: str, scope_prefix: str, extra=()) -> dict:
    f = OUT / fname
    if not f.exists():
        raise AccessDenied(f"{EXP}: no {rel(f)}")
    a = json.loads(f.read_text())
    need = ("authorised_by", "date", "scope", "freeze_commit", "protocol_sha256", "script_sha256", "written_by") + tuple(extra)
    miss = [k for k in need if not a.get(k)]
    if miss:
        raise AccessDenied(f"{EXP}: {fname} lacks {miss}")
    if not str(a["scope"]).startswith(scope_prefix):
        raise AccessDenied(f"{EXP}: {fname} scope must start with {scope_prefix!r}")
    verify_freeze(a["freeze_commit"], a["protocol_sha256"])
    for s in SOURCES:
        if a["script_sha256"].get(rel(s)) != sha256(s):
            raise AccessDenied(f"{EXP}: {rel(s)} differs from the authorised version")
        if blob_sha(a["freeze_commit"], s) != sha256(s):
            raise AccessDenied(f"{EXP}: {rel(s)} differs from its freeze-commit blob")
    return a


# ------------------------------------------------------------------ enhancer
def vendor_hashes() -> dict:
    return {k: sha256(VENDOR / k) for k in VENDOR_SHA}


def check_vendor() -> dict:
    """G1/G2: pinned commit, unmodified code, the downloaded weights and config hashes."""
    head = subprocess.run(["git", "-C", str(VENDOR), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(VENDOR), "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    vh = vendor_hashes()
    g = {"commit": head, "commit_ok": head == FPENH_COMMIT, "worktree_clean": dirty == "",
         "code_sha256": vh, "code_ok": vh == VENDOR_SHA,
         "weights_sha256": sha256(WEIGHTS), "weights_ok": sha256(WEIGHTS) == WEIGHTS_SHA,
         "config_sha256": sha256(CONFIG), "config_ok": sha256(CONFIG) == CONFIG_SHA}
    g["pass"] = bool(g["commit_ok"] and g["worktree_clean"] and g["code_ok"] and g["weights_ok"] and g["config_ok"])
    return g


def build_enhancer(device: str):
    """The authors' `Enhancer` with config_enhancer.yaml; strict load of best.pth (G3). Single device, no DataParallel."""
    import torch
    g = check_vendor()
    if not g["pass"]:
        raise AccessDenied(f"G1/G2 failed: {g}")
    sys.path.insert(0, str(VENDOR))
    import models.Enhancer as ME          # noqa: E402  (vendor package; relative import of .codebook)
    assert Path(ME.__file__).resolve() == (VENDOR / "models" / "Enhancer.py").resolve(), ME.__file__
    net = ME.Enhancer(**MODEL_CFG)
    sd = torch.load(WEIGHTS, map_location="cpu")
    net.load_state_dict(sd, strict=True)
    return net.to(device).eval()


def enhance(net, g: np.ndarray, device: str) -> np.ndarray:
    """One 8-bit grey image -> the authors' float output clip(255*(1-pred), 0, 255) (before cv2.imwrite).

    Input exactly as utils/data_loader.load_dataset_test: (255.0 - img) / 255.0 in float64, then .float()."""
    import torch
    x = ((255.0 - g) / 255.0)[None, None]
    with torch.no_grad():
        y = net(torch.from_numpy(np.ascontiguousarray(x)).float().to(device))
    pred = y[0, 0].detach().cpu().numpy()
    assert pred.shape == g.shape, (pred.shape, g.shape)
    return np.clip(255 * (1 - pred), 0, 255)


def write_png(path: Path, out_float: np.ndarray) -> np.ndarray:
    """Stored exactly as the authors' script: cv2.imwrite of the float array (OpenCV converts to 8 bit). Returns
    the stored uint8 image read back."""
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), out_float):
        raise IOError(f"cv2.imwrite failed: {path}")
    back = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if back is None or back.dtype != np.uint8 or back.shape != out_float.shape:
        raise IOError(f"stored image is not 8-bit {out_float.shape}: {path}")
    return back


def effect_stats(g: np.ndarray, o: np.ndarray) -> dict:
    """E049 restorer-effect statistics (label-blind) and the E028 dose rho = s(Rx)/s(x), s = max - min."""
    gi, oi = g.astype(np.float64), o.astype(np.float64)
    mae = float(np.abs(oi - gi).mean()); rmse = float(np.sqrt(((oi - gi) ** 2).mean()))
    ri, ro = float(gi.max() - gi.min()), float(oi.max() - oi.min())
    return dict(mae=mae, rmse=rmse, psnr=float(20 * np.log10(255.0 / rmse)) if rmse > 0 else float("inf"),
                dyn_in=ri, dyn_out=ro, rho=ro / ri if ri > 0 else float("nan"),
                frac_extreme=float(((o == 0) | (o == 255)).mean()), constant=float(ro == 0),
                corr_in_out=float(np.corrcoef(gi.ravel(), oi.ravel())[0, 1]) if ri > 0 and ro > 0 else float("nan"))


def summarise_effect(stats: list[dict]) -> dict:
    mae = np.array([s["mae"] for s in stats]); rho = np.array([s["rho"] for s in stats])
    eff = dict(n_images=len(stats), median_mae=float(np.median(mae)), median_rmse=float(np.median([s["rmse"] for s in stats])),
               median_psnr=float(np.median([s["psnr"] for s in stats])), median_rho=float(np.nanmedian(rho)),
               median_abs_log_rho=float(np.nanmedian(np.abs(np.log(rho)))),
               frac_rho_changed=float(np.mean(np.abs(rho - 1) > 1e-9)),
               median_dyn_change=float(np.median([s["dyn_out"] - s["dyn_in"] for s in stats])),
               median_frac_extreme=float(np.median([s["frac_extreme"] for s in stats])),
               frac_constant_output=float(np.mean([s["constant"] for s in stats])),
               median_corr_in_out=float(np.nanmedian([s["corr_in_out"] for s in stats])),
               frac_negative_corr=float(np.mean([s["corr_in_out"] < 0 for s in stats])))
    eff["identity_like"] = bool(eff["median_mae"] < 1.0 and eff["median_abs_log_rho"] < 0.01)   # E049 definition
    return eff
