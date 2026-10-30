"""Shared core for E060 (credential burden) and E061 (policy simulation).

Frozen analysis plans: docs/experiments/E060_credential_burden.md, docs/experiments/E061_policy_simulation.md
(F1, commit 3ff5b85). Analysis of frozen score dumps only: no model, no image, no training.

Test-access rule (gate G-A): the key `test_score` is readable only through `read_members` with an
authorisation file `artifacts/results/<EXP>/TEST_READ_AUTHORISATION.json` naming the SHA-256 of every
analysis source file being run. Every member read is recorded in an AccessLog.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "results"
MANIFEST = {p: ROOT / "data" / "track_a" / p / "manifest.csv" for p in ("p1", "p2")}

PROTOS = ("p1", "p2")
SEEDS = tuple(range(10))
ANCHOR = "degraded"
OPS = {"bil": "restored_bilateral", "tv": "restored_tv"}
CONDS = (ANCHOR, OPS["bil"], OPS["tv"])
ARMS = ("leg", "fix")
HEADLINE = ("Rn", "Bs", "Dt")
SUPPLEMENTAL = ("Np",)
UNSEEN = {"p1": ("dcgan", "progan", "stylegan2ada", "stylegan3"),     # identical to E023 / E036
          "p2": ("adm", "dcgan", "stylegan2ada", "stylegan3")}
DEV_GENERATOR = "wgangp"

# (det, protocol, arm) -> (dump root under artifacts/results, run_name, scale). Primary scale (F1 §3):
# logit for ResNet-18 E0 / Benjamin et al. (E059 re-dumps), frozen sigmoid for D3. NPR is tier 2.
SRC = {}
for _n, _p in (("1", "p1"), ("2", "p2")):
    SRC[("Rn", _p, "leg")] = (f"E059/dumps_logit/E009/scores_tracka_p{_n}", f"bl_resnet18_e0_tp{_n}", "logit")
    SRC[("Rn", _p, "fix")] = (f"E059/dumps_logit/E027/scores_fixed_tracka_p{_n}", f"e027_resnet18_e0_tp{_n}_div255", "logit")
    SRC[("Bs", _p, "leg")] = (f"E059/dumps_logit/E025/scores_tracka_p{_n}_screened", f"bl_bcesupcon_tp{_n}", "logit")
    SRC[("Bs", _p, "fix")] = (f"E059/dumps_logit/E027/scores_fixed_tracka_p{_n}", f"e027_bcesupcon_tp{_n}_div255", "logit")
    SRC[("Dt", _p, "fix")] = (f"E027/scores_fixed_tracka_p{_n}", f"e027_d3_84_tp{_n}_div255", "sigmoid")
    SRC[("Np", _p, "fix")] = (f"E059/dumps_logit/E027/scores_fixed_tracka_p{_n}", f"e027_npr_tp{_n}_div255", "logit")
SRC[("Dt", "p1", "leg")] = ("E026/scores_tracka_p1_screened", "bl_d3_84_tp1", "sigmoid")
SRC[("Dt", "p2", "leg")] = ("E027/scores_legacy_tracka_p2", "e027_d3_84_tp2_minmax", "sigmoid")
SRC[("Np", "p1", "leg")] = ("E059/dumps_logit/E003/scores_tracka_p1", "bl_npr_tp1", "logit")
SRC[("Np", "p2", "leg")] = ("E059/dumps_logit/E027/scores_legacy_tracka_p2", "e027_npr_tp2_minmax", "logit")

VAL_KEYS = ("val_patch_id", "val_source", "val_y", "val_score")
TEST_META_KEYS = ("test_patch_id", "test_identity", "test_y", "test_source")
HEADER_KEYS = ("run_name", "seed", "condition")
TEST_SCORE = "test_score"


def dump_path(det: str, p: str, arm: str, seed: int, cond: str) -> Path:
    root, rn, _ = SRC[(det, p, arm)]
    return ART / root / f"{rn}__s{seed}__{cond}.npz"


def rel(path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


# ------------------------------------------------------------------------------ access guard
class TestAccessDenied(PermissionError):
    pass


@dataclass
class AccessLog:
    exp: str
    entries: list = field(default_factory=list)
    authorisation: dict | None = None

    def add(self, path: Path, key: str):
        self.entries.append({"file": rel(path), "key": key,
                             "t": time.strftime("%Y-%m-%dT%H:%M:%S%z")})

    def dump(self, out: Path):
        keys = {}
        for e in self.entries:
            keys[e["key"]] = keys.get(e["key"], 0) + 1
        out.write_text(json.dumps({"experiment": self.exp, "authorisation": self.authorisation,
                                   "reads_by_key": keys, "n_reads": len(self.entries),
                                   "first": self.entries[:3], "last": self.entries[-3:]}, indent=1))


def check_authorisation(exp: str, sources: list[Path]) -> dict:
    """Raise TestAccessDenied unless the PI authorisation file exists and pins every source file."""
    f = ART / exp / "TEST_READ_AUTHORISATION.json"
    if not f.exists():
        raise TestAccessDenied(f"{exp}: no {rel(f)}; test scores may not be read")
    auth = json.loads(f.read_text())
    pinned = auth.get("script_sha256", {})
    for s in sources:
        r = rel(s)
        if pinned.get(r) != sha256(Path(s)):
            raise TestAccessDenied(f"{exp}: {r} differs from the authorised version")
    return auth


def read_members(path: Path, keys, log: AccessLog, authorised: bool = False) -> dict:
    """Read named arrays from an .npz without opening any other member."""
    out = {}
    with zipfile.ZipFile(path) as z:
        for k in keys:
            if k == TEST_SCORE and not authorised:
                raise TestAccessDenied(f"{k} requested from {path.name} without authorisation")
            out[k] = np.load(io.BytesIO(z.read(k + ".npy")), allow_pickle=False)
            log.add(path, k)
    return out


def npy_header(path: Path, key: str):
    """Shape and dtype of one member, from its header only (no values)."""
    from numpy.lib import format as F
    with zipfile.ZipFile(path) as z, z.open(key + ".npy") as fh:
        v = F.read_magic(fh)
        shape, _, dt = (F.read_array_header_1_0 if v == (1, 0) else F.read_array_header_2_0)(fh)
    return shape, str(dt)


# ------------------------------------------------------------------------------ thresholds and policies
def thr_e023(real_scores: np.ndarray) -> float:
    """Frozen E023 rule: np.quantile(real, 0.99, method='higher')."""
    return float(np.quantile(np.asarray(real_scores, dtype=np.float64), 0.99, method="higher"))


def k_max(n: int) -> int:
    """Number of validation real scores strictly above the E023 threshold when there are no ties."""
    return n - 1 - math.ceil(0.99 * (n - 1))


Q_A_GRID = (0.50, 0.80, 0.90, 0.95, 0.99)
BETA = 0.02
ACC, REJ, STEP = 0, 1, 2
POLICIES = ("pi0", "pi1", "pi2", "pi3", "pi4")


def fit_policies(v0: np.ndarray, v1: np.ndarray, w0: np.ndarray, w1: np.ndarray, beta: float = BETA,
                 q_grid=Q_A_GRID) -> dict:
    """E061 F1 'Policies': all parameters from validation real (v0 degraded, v1 restored) and
    development-generator synthetic (w0, w1) scores. Returns the parameter record of one
    (instrument, protocol, arm, seed, operator)."""
    v0 = np.asarray(v0, np.float64); v1 = np.asarray(v1, np.float64)
    w0 = np.asarray(w0, np.float64); w1 = np.asarray(w1, np.float64)
    n = len(v0); km = k_max(n)
    tau0 = thr_e023(v0); tau1 = thr_e023(v1)
    table = []
    for qa in q_grid:
        ta = float(np.quantile(v0, qa, method="higher"))
        S = np.sort(v1[v0 > ta])
        tb = float(S[len(S) - km - 1]) if len(S) > km else -math.inf
        table.append({"q_a": qa, "tau_a": ta, "tau_b": tb,
                      "val_real_reject": float(np.mean((v0 > ta) & (v1 > tb))),
                      "val_dev_tdr": float(np.mean((w0 > ta) & (w1 > tb))),
                      "val_real_restored": float(np.mean(v0 > ta))})
    best = max(table, key=lambda r: (r["val_dev_tdr"], r["q_a"]))       # ties -> larger q_a
    x = np.sort(v0); c0 = int(np.sum(x <= tau0)); m = math.ceil(beta * n)
    tau_u = float(x[c0 - m - 1])
    rec = {"n_val_real": n, "k_max": km, "tau0": tau0, "tau1": tau1,
           "q_a_table": table, "q_a": best["q_a"], "tau_a": best["tau_a"], "tau_b": best["tau_b"],
           "tau_u": tau_u, "beta": beta,
           "val_real_reject_pi0": float(np.mean(v0 > tau0)),
           "val_real_reject_pi2": float(np.mean(v1 > tau1)),
           "val_real_reject_pi3": best["val_real_reject"],
           "val_real_stepup_pi4": float(np.mean((v0 > tau_u) & (v0 <= tau0))),
           "val_dev_tdr_pi0": float(np.mean(w0 > tau0)),
           "val_dev_tdr_pi3": best["val_dev_tdr"]}
    finite = all(math.isfinite(rec[k]) for k in ("tau0", "tau1", "tau_a", "tau_u"))
    rec["finite"] = finite                                    # tau_b = -inf is a valid pi3 outcome
    return rec


def decide(policy: str, prm: dict, s0: np.ndarray, s1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-attempt outcome codes (ACC/REJ/STEP) and the restored flag, for paired scores s0/s1."""
    s0 = np.asarray(s0, np.float64); s1 = np.asarray(s1, np.float64)
    z = np.zeros(len(s0), np.int8); restored = np.zeros(len(s0), bool)
    if policy == "pi0":
        z[s0 > prm["tau0"]] = REJ
    elif policy == "pi1":
        z[s1 > prm["tau0"]] = REJ; restored[:] = True
    elif policy == "pi2":
        z[s1 > prm["tau1"]] = REJ; restored[:] = True
    elif policy == "pi3":
        restored = s0 > prm["tau_a"]
        z[restored & (s1 > prm["tau_b"])] = REJ
    elif policy == "pi4":
        z[(s0 > prm["tau_u"]) & (s0 <= prm["tau0"])] = STEP
        z[s0 > prm["tau0"]] = REJ
    else:
        raise ValueError(policy)
    return z, restored


def eligible_tau(val_real: np.ndarray, tau: float) -> bool:
    """F1 E060 §4: finite tau and validation real rejection >= 0.5 %."""
    return bool(math.isfinite(tau) and np.mean(np.asarray(val_real, np.float64) > tau) >= 0.005)


# ------------------------------------------------------------------------------ units
@dataclass
class Units:
    """Credentials (one per real identity), their subjects, and the capture -> credential map."""
    cred_names: np.ndarray       # (F,)
    subj_of_cred: np.ndarray     # (F,) int index into subj_names
    subj_names: np.ndarray       # (U,)
    cred_of_capture: np.ndarray  # (n_capture,) int

    @property
    def F(self):
        return len(self.cred_names)

    @property
    def U(self):
        return len(self.subj_names)

    def n_f(self) -> np.ndarray:
        return np.bincount(self.cred_of_capture, minlength=self.F)


def subject_of(identity: str) -> str:
    """'selfcollected_88:56_Left_Middle' -> '56' (E052 Stage-0 parse rule)."""
    tail = identity.split(":", 1)[1]
    return tail.split("_", 1)[0]


def make_units(identities: np.ndarray) -> Units:
    ids = np.asarray([str(x) for x in identities])
    cred_names, cred_of_capture = np.unique(ids, return_inverse=True)
    subj = np.array([subject_of(c) for c in cred_names])
    subj_names, subj_of_cred = np.unique(subj, return_inverse=True)
    return Units(cred_names, subj_of_cred, subj_names, cred_of_capture)


def per_cred_mean(x: np.ndarray, units: Units) -> np.ndarray:
    """Per-credential mean of a capture-level quantity x (n_capture,) or (n_capture, k)."""
    n = units.n_f().astype(np.float64)
    if x.ndim == 1:
        return np.bincount(units.cred_of_capture, weights=x.astype(np.float64), minlength=units.F) / n
    return np.stack([per_cred_mean(x[:, j], units) for j in range(x.shape[1])], axis=1)


# ------------------------------------------------------------------------------ endpoints
def wmean(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Weighted means: W (R, F) weights, X (F, k) values -> (R, k)."""
    return (W @ X) / W.sum(axis=1, keepdims=True)


def denial(p: np.ndarray, K: int) -> np.ndarray:
    return p ** K


def attempts(p: np.ndarray, K: int) -> np.ndarray:
    return sum(p ** j for j in range(K))


def d10(p: np.ndarray, W: np.ndarray, frac: float = 0.10) -> np.ndarray:
    """Concentration of credential-normalized rejection mass (F1 E060 §2.1): share of sum_f w_f p_f
    carried by the highest-p credentials forming the top `frac` of total credential weight, with
    the boundary credential contributing the fractional part of its weight. p (F,), W (R, F).
    Returns (R,), NaN where the mass is zero."""
    order = np.argsort(-p, kind="mergesort")
    ps = p[order]; Ws = W[:, order]
    cw = np.cumsum(Ws, axis=1)
    cm = np.cumsum(Ws * ps[None, :], axis=1)
    tot_w = cw[:, -1]; tot_m = cm[:, -1]
    target = frac * tot_w
    j = (cw < target[:, None]).sum(axis=1)                  # first position with cw >= target
    j = np.minimum(j, len(p) - 1)
    rows = np.arange(W.shape[0])
    cw_prev = np.where(j > 0, cw[rows, np.maximum(j - 1, 0)], 0.0)
    cm_prev = np.where(j > 0, cm[rows, np.maximum(j - 1, 0)], 0.0)
    mass = cm_prev + (target - cw_prev) * ps[j]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = mass / tot_m
    out[tot_m <= 0] = np.nan
    return out


def d10_null(n_f: np.ndarray, b: float, rng: np.random.Generator, draws: int = 2000) -> np.ndarray:
    """Homogeneity null of D10 (full sample): each capture rejected iid with probability b."""
    k = rng.binomial(n_f[None, :].repeat(draws, 0), b)
    return d10_unit_rows(k / n_f[None, :])


def d10_unit_rows(P: np.ndarray, frac: float = 0.10) -> np.ndarray:
    """d10 with unit credential weights, one value per row of P (draws, F); equals d10(P[i], ones)."""
    S = -np.sort(-P, axis=1)
    t = frac * P.shape[1]
    j = int(math.floor(t))
    mass = S[:, :j].sum(axis=1) + (t - j) * S[:, j]
    tot = S.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = mass / tot
    out[tot <= 0] = np.nan
    return out


def without_replacement(rej_count: np.ndarray, n_f: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retry model M1 (E060 S1): attempts draw without replacement over a random ordering of the
    credential's captures. Returns (denial, attempts, eligible mask n_f >= K)."""
    elig = n_f >= K
    den = np.zeros(len(n_f)); att = np.zeros(len(n_f))
    for f in np.flatnonzero(elig):
        r, n = int(rej_count[f]), int(n_f[f])
        prob_first_j = [math.comb(r, j) / math.comb(n, j) for j in range(K + 1)]
        den[f] = prob_first_j[K]
        att[f] = sum(prob_first_j[:K])
    return den, att, elig


def session_legit(r: np.ndarray, u: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-credential session quantities under M0: denial r^K, step-up probability
    sum_{j<K} r^j u, expected attempts sum_{j<K} r^j."""
    geo = sum(r ** j for j in range(K))
    return r ** K, geo * u, geo


def pmp_fresh(P, V, q: float, K: int):
    """Pre-matcher progression, fresh image per attempt: sum_{j<K} (1-P-V)^j [P + V(1-q)]."""
    stay = 1.0 - P - V
    return sum(stay ** j for j in range(K)) * (P + V * (1.0 - q))


def pmp_same(P, V, q: float):
    """Same image reused on every attempt: the first outcome repeats, so retries add nothing."""
    return P + V * (1.0 - q)


def cost_J(pmp, L, G, N, c_A: float, c_U: float, c_T: float):
    return c_A * pmp + L + c_U * G + c_T * (N - 1.0)


C_STAR = {"c_A": 1.0, "c_U": 0.2, "c_T": 0.02, "q": 0.9, "K": 3}
COST_GRID = {"c_A": (0.01, 0.1, 1.0, 10.0, 100.0), "c_U": (0.05, 0.2, 0.5),
             "c_T": (0.0, 0.02, 0.1), "q": (0.5, 0.9, 1.0)}
TOL = {"pmp": 0.01, "L": 0.005}


# ------------------------------------------------------------------------------ bootstrap
def row_counts(rng: np.random.Generator, n_rows: int, n: int) -> np.ndarray:
    """n draws with replacement from n items per row -> multiplicities (n_rows, n)."""
    d = rng.integers(0, n, size=(n_rows, n))
    out = np.zeros((n_rows, n))
    np.add.at(out, (np.repeat(np.arange(n_rows), n), d.ravel()), 1.0)
    return out


def seed_average(E: np.ndarray, m: np.ndarray) -> np.ndarray:
    """E (R, 10) per-seed values, m (R, 10) seed multiplicities -> (R,), NaN-aware."""
    ok = ~np.isnan(E)
    num = np.where(ok, E, 0.0) * m
    den = (ok * m).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num.sum(axis=1) / den
    out[den == 0] = np.nan
    return out


def pctl(v: np.ndarray, level: float) -> tuple[float, float]:
    v = np.asarray(v, np.float64); v = v[~np.isnan(v)]
    if len(v) == 0:
        return (math.nan, math.nan)
    a = (1.0 - level) / 2.0
    return float(np.quantile(v, a)), float(np.quantile(v, 1.0 - a))
