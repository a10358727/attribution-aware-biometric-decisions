"""E062 v2: null calibration and precision of the RA-class estimator on SYNTHETIC data (not evidence).

Latent-score generator: a persistent capture propensity (subject + credential + capture components), a
detector/protocol-specific capture response, a training-seed shift shared by all captures of a dump, and a
restoration shift identical for all RA classes under the null. Rejection = latent > 2. No DIAR data are read.

    python -m experiments.E062_quality_conditioned_eval.ra_calibration --n 250 --subjects 34
"""
from __future__ import annotations

import argparse

import numpy as np

from experiments.E062_quality_conditioned_eval import ra_analysis as E
from experiments.tcss_lib import core as C


def latent(n_subj, creds=9, caps=5, seed=0, S=10, eff=0.0):
    """eff adds to the restored latent of RA-dry under the legacy arm only (a planted interaction)."""
    rng = np.random.default_rng(seed)
    n = n_subj * creds * caps
    subj = np.repeat([f"S{i}" for i in range(n_subj)], creds * caps)
    cred = np.repeat([f"S{i}_{f}" for i in range(n_subj) for f in range(creds)], caps)
    cls = rng.integers(0, 3, n)
    z = (rng.normal(0, 1, n_subj).repeat(creds * caps) * 0.6 + rng.normal(0, 1, n_subj * creds).repeat(caps) * 0.5
         + rng.normal(0, 1, n) * 0.6)
    rej, el = {}, {}
    for det in E.HEADLINE:
        for p in C.PROTOS:
            dz = rng.normal(0, 0.3, n)
            for arm in C.ARMS:
                el[(det, p, arm)] = True
                for cond in C.CONDS:
                    shift = 0.0 if cond == C.ANCHOR else 0.8 + (eff if arm == "leg" else 0.0) * (cls == 0)
                    rej[(det, p, arm, cond)] = np.stack(
                        [((z + dz + shift + rng.normal(0, 0.25) + rng.normal(0, 0.2, n)) > 2.0).astype(float)
                         for _ in range(S)])
    return E.Data({p: subj for p in C.PROTOS}, {p: cred for p in C.PROTOS}, {p: cls for p in C.PROTOS}, rej, el)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--subjects", type=int, default=34)
    ap.add_argument("--B", type=int, default=1000)
    a = ap.parse_args(argv)
    primary = E.LEVEL_PRIMARY
    try:
        for level in (primary, E.LEVEL_DISPLAY):
            E.LEVEL_PRIMARY = level                  # the primary-family level is read at call time
            hits, widths = np.zeros(4), []
            for k in range(a.n):
                out = E.run(latent(a.subjects, seed=5000 + k), B=a.B, seed=k)
                for j, v in enumerate(out["primary"].values()):
                    hits[j] += v["reading"] in ("above 0", "below 0")
                widths.append(out["primary"]["theta_dry|leg"]["hi"] - out["primary"]["theta_dry|leg"]["lo"])
            print(f"level {level}: nominal two-sided error {1 - level:.4f}; observed {hits.sum() / (4 * a.n):.4f} "
                  f"(per contrast {np.round(hits / a.n, 3).tolist()}); mean width theta_dry|leg {np.mean(widths):.3f}")
    finally:
        E.LEVEL_PRIMARY = primary

if __name__ == "__main__":
    main()
