#!/usr/bin/env python3
"""Privacy gate for this repository: fails if any file looks like biometric data, a per-image record, an identifier
mapping, a model output dump, weights, a local path or a credential. Run before every release and in CI.

Checks every file that git would publish (tracked and untracked-but-not-ignored); does not rely on .gitignore.
    python scripts/privacy_check.py      # exit 0 = clean
"""
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DENY_EXT = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".gif", ".webp", ".npz", ".npy", ".pt", ".pth", ".ckpt",
            ".safetensors", ".onnx", ".h5", ".pkl", ".pickle", ".parquet", ".feather", ".numbers", ".xlsx", ".xls",
            ".zip", ".tar", ".gz", ".7z", ".pdf", ".mat", ".log"}
CONTENT_DENY = {
    "fingerprint capture filename": re.compile(r"\b\d{3}_(?:Left|Right)_(?:Thumb|Index|Middle|Ring|Little)\b|_l\d_Unknown"),
    "absolute local path": re.compile(r"(?:/home/[A-Za-z]|/Users/[A-Za-z]|[A-Z]:\\\\Users)"),
    "credential or token": re.compile(r"ghp_[A-Za-z0-9]{20,}|github_pat_|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|BEGIN [A-Z ]*PRIVATE KEY"),
    "authorisation or signature record": re.compile(r"TEST_READ_AUTHORISATION\.json\"?\s*[:=]|\"signed_by\"\s*:|\"stated_by\"\s*:"),
}
DATA_KEY_DENY = {"src_file", "patch_id", "identity", "subject_id", "subj", "file", "filename", "credential_id", "cred",
                 "image_path", "path", "score", "logit", "pred", "prediction"}
ALLOW_PATTERN_FILES = {"scripts/privacy_check.py", ".gitignore"}
ALLOW_KEY_PATHS = {("_source", "file")}       # provenance of a released copy (a repository path), not a per-item field
MAX_DATA_BYTES = 5_000_000


def files():
    out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--cached", "--others", "--exclude-standard"],
                         capture_output=True, text=True, check=True).stdout.split("\n")
    return [f for f in out if f and (ROOT / f).is_file()]


def data_keys(path: Path):
    if path.suffix == ".json":
        keys = set()

        def walk(o, trail):
            if isinstance(o, dict):
                for k, v in o.items():
                    if (trail[-1:] + (k,)) not in ALLOW_KEY_PATHS:
                        keys.add(k)
                    walk(v, trail + (k,))
            elif isinstance(o, list):
                for v in o:
                    walk(v, trail)
        walk(json.loads(path.read_text()), ())
        return keys
    if path.suffix == ".csv":
        with open(path, newline="") as fh:
            return set(next(csv.reader(fh), []))
    return set()


def main():
    problems = []
    fs = files()
    for f in fs:
        p = ROOT / f
        if p.suffix.lower() in DENY_EXT:
            problems.append((f, f"denied file type {p.suffix}"))
            continue
        if p.suffix in (".json", ".csv") and p.stat().st_size > MAX_DATA_BYTES:
            problems.append((f, "data file larger than 5 MB (possible per-item dump)"))
        try:
            text = p.read_text()
        except UnicodeDecodeError:
            problems.append((f, "binary file"))
            continue
        if f not in ALLOW_PATTERN_FILES:
            for name, rx in CONTENT_DENY.items():
                if rx.search(text):
                    problems.append((f, name))
        bad = data_keys(p) & DATA_KEY_DENY
        if bad:
            problems.append((f, f"per-item/identifier field(s) {sorted(bad)}"))
    for f, why in problems:
        print(f"PRIVACY FAIL  {f}: {why}")
    print(f"privacy check: {len(fs)} files, {len(problems)} problem(s)")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
