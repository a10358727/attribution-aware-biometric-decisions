"""E069 shared constants, checkpoint map, frozen thresholds and authorisation guards.

Protocol: docs/experiments/E069_cross_sensor_bonafide_transfer.md (v4). The detectors, F2 thresholds and inference code
are frozen artefacts used unchanged.
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

EXP = "E069"
OUT = ROOT / "artifacts" / "results" / EXP
PROTOCOL = ROOT / "docs" / "experiments" / "E069_cross_sensor_bonafide_transfer.md"
E070_PROTOCOL = ROOT / "docs" / "experiments" / "E070_quality_conditioned_external.md"
LOCAL = ROOT / "data" / "e069_external"
WIN = LOCAL / "win88"                                   # clean/degraded/restored_* /<device>/<patch_id>.png
FIT = ROOT / "artifacts" / "results" / "E061" / "validation_fit.json"
FIT_SHA = "c3d01b7d84d8858bfd091e9bcdd8d8db4b24597fc4ccf8adeaf3554e9dae22e4"
E051_MAP = ROOT / "artifacts" / "results" / "E051" / "checkpoint_to_dump_map.csv"
E051_VAL = ROOT / "artifacts" / "results" / "E051" / "val_logits"
E061_SINGLE = ROOT / "artifacts" / "results" / "E061" / "single_attempt.csv"
CONDS = ("clean", "degraded", "restored_bilateral", "restored_tv")
OPS = {"bil": "restored_bilateral", "tv": "restored_tv"}
FAMILY = {"Rn": "ResNet-18 E0", "Bs": "Benjamin", "Dt": "D3"}
ARM = {"legacy": "leg", "fixed": "fix"}
SEEDS = tuple(range(10))
SOURCES = [HERE, HERE.parent / "build_windows.py", HERE.parent / "e069_score.py", HERE.parent / "e069_analyze.py",
           ROOT / "experiments" / "tcss_lib" / "core.py", ROOT / "diar" / "evaluate.py", ROOT / "diar" / "zoo.py",
           ROOT / "diar" / "data" / "dataset.py", ROOT / "tools" / "dataset_build" / "make_conditions.py",
           ROOT / "preprocess" / "ridge_utils.py", ROOT / "preprocess" / "degradation_bank.py",
           HERE.parent / "stage0_manifest.py"]


class AccessDenied(PermissionError):
    pass


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def rel(p: Path) -> str:
    return str(Path(p).resolve().relative_to(ROOT)) if str(Path(p).resolve()).startswith(str(ROOT)) else str(p)


def checkpoints(families=("Rn", "Bs")) -> list[dict]:
    """E051 map rows of the E061 headline families (10 seeds x P1/P2 x legacy/fixed), CONFIRMED only."""
    want = {FAMILY[f]: f for f in families}
    rows = [r for r in csv.DictReader(open(E051_MAP)) if r["instrument"] in want]
    assert all(r["status"] == "CONFIRMED" for r in rows)
    for r in rows:
        r["family"] = want[r["instrument"]]
        r["p"] = r["protocol"].lower()
        r["arm_code"] = ARM[r["arm"]]
    assert len(rows) == 40 * len(families), len(rows)
    return rows


def tau0(fits: dict, c: dict) -> float:
    return float(fits[f"{c['family']}|{c['p']}|{c['arm_code']}|{int(c['seed'])}|bil"]["tau0"])


def load_fits() -> dict:
    if sha256(FIT) != FIT_SHA:
        raise AccessDenied("validation_fit.json is not the frozen E061 F2 file")
    return json.loads(FIT.read_text())["fits"]


def git_state() -> dict:
    run = lambda *a: subprocess.run(a, cwd=ROOT, capture_output=True, text=True).stdout
    return {"git": run("git", "rev-parse", "HEAD").strip(),
            "dirty_tracked": run("git", "status", "--porcelain", "--untracked-files=no").splitlines(),
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "command": " ".join(sys.argv)}


BATCH_BY_OWNER = {"E003": 512, "E009": 512, "E025": 512, "E026": 128, "E027": 128, "E035": 128}   # E051/E059 frozen
D3_PASS_SEED = 1000                                                  # E051 protocol d3_control.pass_seed
FAMILY_INDEX = {"Rn": 0, "Bs": 1, "Dt": 2}
D3_DEADLINE = "2026-12-23"                                           # D3 must be scored by then, whatever the interim says


def git(*a) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True)


def blob_sha(commit: str, path: Path) -> str | None:
    r = git("show", f"{commit}:{rel(path)}")
    return hashlib.sha256(r.stdout).hexdigest() if r.returncode == 0 else None


def verify_freeze(commit: str, protocol: Path, protocol_sha: str) -> None:
    """The freeze commit must exist, be an ancestor of HEAD, hold the authorised protocol bytes, and the working
    protocol may only APPEND to them (results sections are added after the freeze)."""
    if git("merge-base", "--is-ancestor", commit, "HEAD").returncode != 0:
        raise AccessDenied(f"freeze commit {commit} is not an ancestor of HEAD")
    frozen = git("show", f"{commit}:{rel(protocol)}").stdout
    if hashlib.sha256(frozen).hexdigest() != protocol_sha:
        raise AccessDenied(f"{rel(protocol)} at {commit} is not the authorised protocol")
    if not Path(protocol).read_bytes().startswith(frozen):
        raise AccessDenied(f"{rel(protocol)} was edited, not appended to, after the freeze")


def check_auth(fname: str, scope_prefix: str, extra=()) -> dict:
    f = OUT / fname
    if not f.exists():
        raise AccessDenied(f"{EXP}: no {rel(f)}")
    a = json.loads(f.read_text())
    need = ("authorised_by", "date", "scope", "freeze_commit", "protocol_sha256", "script_sha256") + tuple(extra)
    miss = [k for k in need if not a.get(k)]
    if miss:
        raise AccessDenied(f"{EXP}: {fname} lacks {miss}")
    if not str(a["scope"]).startswith(scope_prefix):
        raise AccessDenied(f"{EXP}: {fname} scope must start with {scope_prefix!r}")
    verify_freeze(a["freeze_commit"], PROTOCOL, a["protocol_sha256"])
    for s in SOURCES:
        if a["script_sha256"].get(rel(s)) != sha256(s):
            raise AccessDenied(f"{EXP}: {rel(s)} differs from the authorised version")
        if blob_sha(a["freeze_commit"], s) != sha256(s):
            raise AccessDenied(f"{EXP}: {rel(s)} differs from its freeze-commit blob")
    return a


def e070_frozen_at(commit: str, e070_protocol_sha: str) -> bool:
    """G-F: the E070 protocol at the freeze commit carries the freeze sentence and has the authorised hash."""
    r = git("show", f"{commit}:{rel(E070_PROTOCOL)}")
    return (r.returncode == 0 and b"Frozen by the commit" in r.stdout
            and hashlib.sha256(r.stdout).hexdigest() == e070_protocol_sha)


def checkpoint_ok(c: dict) -> bool:
    """best.pt SHA-256 equals the E051 map (frozen checkpoint)."""
    return sha256(ROOT / c["run_dir"] / "best.pt") == c["best_pt_sha256"]
