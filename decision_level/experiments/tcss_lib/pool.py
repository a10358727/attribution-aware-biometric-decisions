"""Score pools for E060/E061: aligned validation arrays, test metadata and (only when authorised)
test scores. mode='test' uses the real test rows and the E023 unseen synthetic group; mode='pilot'
uses the validation pool as a stand-in test set (code exercise only, not evidence)."""
from __future__ import annotations

import csv
from dataclasses import dataclass, field

import numpy as np

from .core import (ANCHOR, CONDS, DEV_GENERATOR, MANIFEST, SEEDS, SRC, TEST_META_KEYS, TEST_SCORE,
                   UNSEEN, VAL_KEYS, AccessLog, Units, dump_path, make_units, read_members)


def val_identity_map(protos) -> dict:
    ident = {}
    for p in protos:
        with open(MANIFEST[p]) as fh:
            for r in csv.DictReader(fh):
                if r["split"] == "val" and r["label"] == "0":
                    ident[r["patch_id"]] = r["identity"]
    return ident


@dataclass
class ProtoPool:
    p: str
    units: Units = None
    real_pid: np.ndarray = None
    syn_pid: np.ndarray = None
    syn_gen: np.ndarray = None
    real: dict = field(default_factory=dict)      # (det, arm, seed, cond) -> scores of real rows
    syn: dict = field(default_factory=dict)       # (det, arm, seed, cond) -> scores of synthetic rows
    val_real: dict = field(default_factory=dict)  # (det, arm, seed, cond) -> validation real scores
    val_dev: dict = field(default_factory=dict)   # (det, arm, seed, cond) -> validation wgangp scores
    checks: dict = field(default_factory=dict)


def load_pool(mode: str, dets, log: AccessLog, authorised: bool = False, src=None,
              protos=("p1", "p2")) -> dict:
    """mode: 'meta' (validation + test metadata, no test score), 'pilot', or 'test' (needs authorised)."""
    assert mode in ("meta", "pilot", "test")
    src = src or SRC
    vident = val_identity_map(protos) if mode == "pilot" else None
    pools = {}
    for p in protos:
        pp = ProtoPool(p)
        meta_ref = None
        val_ref = None
        for det in dets:
            for arm in ("leg", "fix"):
                for s in SEEDS:
                    for cond in CONDS:
                        f = dump_path(det, p, arm, s, cond) if src is SRC else _alt_path(src, det, p, arm, s, cond)
                        keys = list(VAL_KEYS) + list(TEST_META_KEYS)
                        if mode == "test":
                            keys.append(TEST_SCORE)
                        z = read_members(f, keys, log, authorised=authorised)
                        if meta_ref is None:
                            meta_ref = {k: z[k] for k in TEST_META_KEYS}
                            val_ref = {k: z[k] for k in ("val_patch_id", "val_y", "val_source")}
                        else:
                            for k in TEST_META_KEYS:
                                if not (z[k] == meta_ref[k]).all():
                                    raise AssertionError(f"{k} not aligned in {f}")
                            for k in ("val_patch_id", "val_y", "val_source"):
                                if not (z[k] == val_ref[k]).all():
                                    raise AssertionError(f"{k} not aligned in {f}")
                        vy = z["val_y"]; vs = z["val_source"]
                        pp.val_real[(det, arm, s, cond)] = z["val_score"][vy == 0].astype(np.float64)
                        pp.val_dev[(det, arm, s, cond)] = z["val_score"][(vy == 1) & (vs == DEV_GENERATOR)].astype(np.float64)
                        if mode == "pilot":
                            pp.real[(det, arm, s, cond)] = z["val_score"][vy == 0].astype(np.float64)
                            pp.syn[(det, arm, s, cond)] = z["val_score"][vy == 1].astype(np.float64)
                        elif mode == "test":
                            ty = meta_ref["test_y"]; R = ty == 0
                            S = (ty == 1) & np.isin(meta_ref["test_source"], UNSEEN[p])
                            pp.real[(det, arm, s, cond)] = z[TEST_SCORE][R].astype(np.float64)
                            pp.syn[(det, arm, s, cond)] = z[TEST_SCORE][S].astype(np.float64)
        ty = meta_ref["test_y"]; tsrc = meta_ref["test_source"]
        if mode == "pilot":
            vy = val_ref["val_y"]
            vpid = val_ref["val_patch_id"][vy == 0]
            pp.units = make_units(np.array([vident[str(x)] for x in vpid]))
            pp.real_pid = vpid
            pp.syn_pid = val_ref["val_patch_id"][vy == 1]
            pp.syn_gen = val_ref["val_source"][vy == 1]
        else:
            R = ty == 0
            S = (ty == 1) & np.isin(tsrc, UNSEEN[p])
            pp.units = make_units(meta_ref["test_identity"][R])
            pp.real_pid = meta_ref["test_patch_id"][R]
            pp.syn_pid = meta_ref["test_patch_id"][S]
            pp.syn_gen = tsrc[S]
        pp.checks = {"n_real": int(len(pp.real_pid)), "n_cred": pp.units.F, "n_subj": pp.units.U,
                     "n_syn": int(len(pp.syn_pid)),
                     "syn_by_gen": {str(g): int(n) for g, n in zip(*np.unique(pp.syn_gen, return_counts=True))},
                     "n_val_real": int((val_ref["val_y"] == 0).sum()),
                     "val_dev_n": int(((val_ref["val_y"] == 1) & (val_ref["val_source"] == DEV_GENERATOR)).sum())}
        pools[p] = pp
    return pools


def _alt_path(src, det, p, arm, s, cond):
    from .core import ART
    root, rn, _ = src[(det, p, arm)]
    return ART / root / f"{rn}__s{s}__{cond}.npz"


class SharedUnits:
    """Resampling units shared across protocols: subjects by name; synthetic images by
    (generator, patch_id) so that an image present in both protocols gets one draw (as E036)."""

    def __init__(self, pools: dict):
        names = sorted(set().union(*[set(pp.units.subj_names.tolist()) for pp in pools.values()]))
        pos = {n: i for i, n in enumerate(names)}
        self.subjects = names
        self.subj_col = {p: np.array([pos[n] for n in pp.units.subj_names]) for p, pp in pools.items()}
        gens = sorted(set().union(*[set(map(str, pp.syn_gen)) for pp in pools.values()]))
        self.gen_pids = {}
        for g in gens:
            ids = set()
            for pp in pools.values():
                ids |= set(map(str, pp.syn_pid[pp.syn_gen == g]))
            self.gen_pids[g] = sorted(ids)
        gpos = {g: {pid: i for i, pid in enumerate(v)} for g, v in self.gen_pids.items()}
        self.syn_map = {}
        for p, pp in pools.items():
            self.syn_map[p] = []
            gen = np.array([str(x) for x in pp.syn_gen])
            for g in sorted(set(gen.tolist())):
                idx = np.flatnonzero(gen == g)
                col = np.array([gpos[g][str(pid)] for pid in pp.syn_pid[idx]])
                self.syn_map[p].append((g, idx, col))

    def draw(self, rng_subj, rng_syn, n_rows: int) -> dict:
        from .core import row_counts
        d = {"subj": row_counts(rng_subj, n_rows, len(self.subjects))}
        if rng_syn is not None:
            for g in sorted(self.gen_pids):
                d[g] = row_counts(rng_syn, n_rows, len(self.gen_pids[g]))
        return d

    def cred_weights(self, d: dict, p: str, units: Units) -> np.ndarray:
        Wsubj = d["subj"][:, self.subj_col[p]]           # (R, U_p)
        return Wsubj[:, units.subj_of_cred]                # (R, F)

    def subj_weights(self, d: dict, p: str) -> np.ndarray:
        return d["subj"][:, self.subj_col[p]]

    def syn_weights(self, d: dict, p: str, n_syn: int) -> np.ndarray:
        n_rows = d["subj"].shape[0]
        Ws = np.zeros((n_rows, n_syn))
        for g, idx, col in self.syn_map[p]:
            Ws[:, idx] = d[g][:, col]
        return Ws


def ones_draw(shared: SharedUnits, with_syn: bool = True) -> dict:
    d = {"subj": np.ones((1, len(shared.subjects)))}
    if with_syn:
        for g, v in shared.gen_pids.items():
            d[g] = np.ones((1, len(v)))
    return d
