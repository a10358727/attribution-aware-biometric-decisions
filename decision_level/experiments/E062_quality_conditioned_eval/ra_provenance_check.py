"""E062 label-provenance check (P-2, P-3, P-4; pre-freeze, no detector outcome).

Reconstructs the q3_v2_all labelling from the label source itself and checks its image statistics against the native
88x88 captures. Reads: the `.numbers` label source (checksum-verified), the native corpus images, and the frozen test
list (for test-pool counts only). Never reads scores, logits, predictions, thresholds or E060/E061 results. Writes
aggregate counts only (no filename, label, subject ID or per-image value).

    python ra_provenance_check.py --source normal_all_v4_分類說明.numbers --native <normal_all_v4 dir> --out <dir>

Needs numbers_parser, numpy, scikit-learn and opencv (conversion-time dependencies, not part of the frozen estimator).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

LABEL_SHA256_Q3_V2_ALL = "0a964846931c49346aba693be4be9a1082e36c6433865838d38d952659e703cc"   # = ra_audit

COLS = {"file": "原始檔名", "final": "最終分類代碼", "train": "是否用於KNN訓練", "vote": "原始投票標籤代碼",
        "support": "KNN鄰居支持票數（共9票）", "changed": "KNN是否改變原始投票標籤",
        "mean": "灰階平均", "sd": "灰階標準差", "black": "黑色像素比例", "otsu": "Otsu黑色比例",
        "sobel": "Sobel梯度強度", "lap": "Laplacian特徵（原欄位lap_spectrum）",
        "v_sd": "灰階標準差投向等級", "v_otsu": "Otsu黑色比例投向等級", "v_sobel": "梯度投向等級（反向）",
        "v_lap": "Laplacian投向等級（反向）"}
STATS = ("sd", "otsu", "sobel", "lap")          # the four voting statistics
REVERSED = {"sobel", "lap"}                    # vote level 2 = lowest tertile
K = 9
TOL = 1e-6


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_lines(path):
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


def read_source(path: Path) -> dict:
    from numbers_parser import Document
    got = sha256_file(path)
    if got != LABEL_SHA256_Q3_V2_ALL:
        raise SystemExit(f"label source checksum {got} != q3_v2_all")
    rows = Document(str(path)).sheets[0].tables[0].rows(values_only=True)
    head, body = rows[0], rows[1:]
    out = {k: np.array([r[head.index(v)] for r in body]) for k, v in COLS.items()}
    for k in ("final", "support", *STATS, "mean", "black", "v_sd", "v_otsu", "v_sobel", "v_lap"):
        out[k] = out[k].astype(float)
    out["vote"] = np.array([np.nan if v is None else float(v) for v in out["vote"]])
    return out


def tertile_votes(X: np.ndarray, cuts=None):
    """Vote level per statistic from corpus tertiles (levels 0/1/2; reversed statistics flipped)."""
    if cuts is None:
        cuts = [np.quantile(X[:, j], [1 / 3, 2 / 3]) for j in range(X.shape[1])]
    V = np.stack([np.digitize(X[:, j], cuts[j]) for j in range(X.shape[1])], 1)
    for j, s in enumerate(STATS):
        if s in REVERSED:
            V[:, j] = 2 - V[:, j]
    return V, cuts


def reconstruct(X: np.ndarray, cuts=None):
    """Tertile votes -> training label when >= 3 of 4 votes agree -> 9-NN (unscaled Euclidean, uniform) on all images."""
    from sklearn.neighbors import KNeighborsClassifier
    V, cuts = tertile_votes(X, cuts)
    cnt = np.stack([(V == k).sum(1) for k in range(3)], 1)
    train = cnt.max(1) >= 3
    y = cnt.argmax(1)[train]
    m = KNeighborsClassifier(K).fit(X[train], y)
    pred = m.predict(X)
    nb = m.kneighbors(X, return_distance=False)
    support = (y[nb] == pred[:, None]).sum(1)
    return {"V": V, "train": train, "vote": np.where(train, cnt.argmax(1), -1), "pred": pred, "support": support,
            "cuts": cuts}


def native_stats(files, native: Path):
    import cv2
    out = {k: np.full(len(files), np.nan) for k in ("mean", "sd", "otsu", "sobel")}
    shapes = set()
    missing = 0
    for i, f in enumerate(files):
        p = native / f
        raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if raw is None:
            missing += 1
            continue
        shapes.add(tuple(raw.shape))
        g8 = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        g = g8.astype(np.float64)
        th, _ = cv2.threshold(g8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        sx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
        out["mean"][i], out["sd"][i] = g.mean(), g.std()
        out["otsu"][i] = (g <= th).mean()
        out["sobel"][i] = np.hypot(sx, sy).mean()
    return out, sorted(shapes), missing


def main(argv=None):
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[2]
    ap.add_argument("--source", required=True)
    ap.add_argument("--native", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    src = read_source(Path(a.source))
    n = len(src["file"])
    test = set(read_lines(root / "config" / "splits" / "tracka_split_v1" / "test_cap60_files.txt"))
    in_test = np.array([f in test for f in src["file"]])
    X = np.stack([src[s] for s in STATS], 1)
    rec = reconstruct(X)
    Vsrc = np.stack([src[f"v_{s}"] for s in STATS], 1).astype(int)
    vote_src = np.where(np.isnan(src["vote"]), -1, src["vote"]).astype(int)
    rep = {"inputs": {"label_source_sha256": LABEL_SHA256_Q3_V2_ALL, "rows": n, "test_pool_rows": int(in_test.sum())},
           "P2_reconstruction": {
               "vote_levels_equal_corpus_tertiles": int((rec["V"] == Vsrc).all(1).sum()),
               "tertile_cut_values": {s: [float(c) for c in rec["cuts"][j]] for j, s in enumerate(STATS)},
               "reversed_statistics": sorted(REVERSED),
               "training_rule_ge3_of_4_matches": int((rec["train"] == (src["train"] == "是")).sum()),
               "training_label_matches": int((rec["vote"] == vote_src).sum()),
               "knn_spec": "k=9, uniform weights, Euclidean, unscaled features (sd, otsu, sobel, lap), fit on vote-labelled"
                           " images, predicted for all images incl. the training images",
               "final_class_matches": int((rec["pred"] == src["final"]).sum()),
               "final_class_matches_test_pool": int((rec["pred"] == src["final"])[in_test].sum()),
               "support_matches": int((rec["support"] == src["support"]).sum()),
               "knn_changed_vote": int(((rec["pred"] != rec["vote"]) & rec["train"]).sum()),
               "knn_changed_vote_source": int((src["changed"] == "是").sum()),
           }}
    nat, shapes, missing = native_stats(src["file"], Path(a.native))
    diff = {k: np.abs(nat[k] - src[k]) for k in nat}
    mism = diff["sd"] > TOL
    Xn = X.copy()
    Xn[:, STATS.index("sd")] = nat["sd"]
    Xn[:, STATS.index("otsu")] = nat["otsu"]
    alt = reconstruct(Xn, rec["cuts"])
    rep["P4_native_statistics"] = {
        "native_files_read": n - missing, "missing": missing, "shapes": [list(s) for s in shapes],
        "recomputed": "cv2 IMREAD_GRAYSCALE; mean, population SD, Otsu (<= threshold) ratio, mean Sobel 3x3 magnitude",
        "exact_match": {k: int((d <= TOL).sum()) for k, d in diff.items()},
        "max_abs_diff": {k: float(np.nanmax(d)) for k, d in diff.items()},
        "discrepant_rows_mean_sd_otsu": int(mism.sum()),
        "discrepant_rows_in_test_pool": int((mism & in_test).sum()),
        "discrepant_same_rows_all_three": bool(((diff["mean"] > TOL) == mism).all() and ((diff["otsu"] > TOL) == mism).all()),
        "laplacian_statistic": "not reproduced (definition of lap_spectrum unknown)",
        "class_if_native_sd_otsu_with_source_cuts": {
            "differs": int((alt["pred"] != src["final"]).sum()),
            "differs_in_test_pool": int((alt["pred"] != src["final"])[in_test].sum())},
    }
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "label_provenance_check.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
