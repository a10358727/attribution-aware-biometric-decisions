# [Public copy of paper_tcss/tools/mk_sens.py: only input/output paths are remapped to decision_level/results/.]
#!/usr/bin/env python3
"""Macros and tables for the post hoc sensitivity and matcher analyses (E063 Part B, E064, E067, E071).

Reads aggregate run outputs only (no score dump, image or replicate file). Writes
  paper_tcss/generated/numbers_sens.tex   (\\Sx... macros)
  paper_tcss/generated/tab_sens.tex       (main Table: session readings under each sensitivity analysis)
  paper_tcss/generated/tab_matcher.tex    (main Table: genuine matching by probe condition)
  paper_tcss/generated/supp_sens_grid.tex (supplementary: full K x attacker x rho reading grid, legacy R-H1/R-H2)
Run from the repository root:  python3 paper_tcss/tools/mk_sens.py
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
A = ROOT / "decision_level" / "results"   # [public copy: released aggregates]
OUT = ROOT / "decision_level" / "results" / "C1_macros_rebuilt"   # [public copy]
OUT.mkdir(parents=True, exist_ok=True)
SRC = {
    "E061": A / "E061" / "joint_summary.json",
    "E063": A / "E063" / "test_run" / "joint_summary.json",
    "E063R": A / "E063" / "test_run" / "RESULTS.json",
    "E067": A / "E067" / "test_run" / "intervals.csv",
    "E064": A / "E064" / "test_run" / "grid_readings.csv",
    "E064D": A / "E064" / "test_run" / "decomposition.json",
    "E064K": A / "E064" / "test_run" / "rankings.json",
    "E071": A / "E071" / "test_run" / "summary.json",
    "E071V": A / "E071" / "validation_run" / "tau_v.csv",
    "E071G": A / "E071" / "test_run" / "gates.json",
    "E065": A / "E065" / "test_run" / "summary.json",
}
LEVEL = "ci9875"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def num(v, d=3, plus=False):
    s = f"{abs(v):.{d}f}"
    if v < 0 and float(s) != 0:
        return "\\ensuremath{-}" + s
    return ("+" if plus and float(s) != 0 else "") + s


def iv(lo, hi, d=3):
    return f"[{num(lo, d)}, {num(hi, d)}]"


def cls(lo, hi):
    return "better" if hi < 0 else "worse" if lo > 0 else "ind."


# ------------------------------------------------------------------------------ load
e61 = json.loads(SRC["E061"].read_text())
e63 = json.loads(SRC["E063"].read_text())
e63r = json.loads(SRC["E063R"].read_text())
e67 = {r["statistic"]: r for r in csv.DictReader(open(SRC["E067"]))}
e64 = list(csv.DictReader(open(SRC["E064"])))
e64d = json.loads(SRC["E064D"].read_text())
e64k = json.loads(SRC["E064K"].read_text())["E064"]
e71 = json.loads(SRC["E071"].read_text())
e71v = list(csv.DictReader(open(SRC["E071V"])))


def g64(model, rho, K, att, reading):
    for r in e64:
        if r["model"] == model and r["K"] == str(K) and r["attacker"] == att and r["reading"] == reading and \
                (rho is None or (r["rho"] != "" and float(r["rho"]) == rho)):
            return float(r["median"]), float(r["lo9875"]), float(r["hi9875"]), r["label_vs_E061"]
    raise KeyError((model, rho, K, att, reading))


# reading -> (E061 key, E063 contrast key stem)
READ = [("R-H1", "pi1", "leg"), ("fixed_pi1", "pi1", "fix"), ("R-H2", "pi1", "dd"), ("R-H4|leg", "pi4", "leg"),
        ("R-H4|fix", "pi4", "fix")]


def row_e061():
    out = []
    for _, pol, arm in READ:
        d = e61[f"dJ|{pol}|{arm}"]; out.append((d["median"], *d["ci_decision"]))
    p2 = [cls(*e61[f"dJ|pi2|{a}"]["ci_decision"]) for a in ("leg", "fix")]
    return out, p2


def row_e067():
    out = []
    for _, pol, arm in READ:
        r = e67[f"dJ|{pol}|{arm}"]; out.append((float(r["full_sample"]), float(r["refit_on_lo"]), float(r["refit_on_hi"])))
    p2 = [cls(float(e67[f"dJ|pi2|{a}"]["refit_on_lo"]), float(e67[f"dJ|pi2|{a}"]["refit_on_hi"])) for a in ("leg", "fix")]
    return out, p2


def row_e063(model):
    out = []
    for _, pol, arm in READ:
        d = e63["contrasts"][f"dJ|{pol}|{arm}|{model}"]; out.append((d["median"], *d[LEVEL]))
    p2 = [cls(*e63["contrasts"][f"dJ|pi2|{a}|{model}"][LEVEL]) for a in ("leg", "fix")]
    return out, p2


def row_e064(rho, K, att, model="M2"):
    out = [g64(model, rho, K, att, r)[:3] for r, _, _ in READ]
    p2 = [cls(*g64(model, rho, K, att, f"R-H3|pi2|{a}")[1:3]) for a in ("leg", "fix")]
    return out, p2


ROWS_FULL = [
    ("E061 reference", "M0, all, $K{=}3$, fresh", row_e061()),
    ("+ refitting (E067)", "refitted thresholds/parameters", row_e067()),
    ("Selection control (E063)", "M0, $n_f\\ge3$", row_e063("M0e")),
    ("Without replacement (E063)", "M1, $n_f\\ge3$", row_e063("M1")),
    ("Persistence (E064)", "M2, $\\rho{=}0.25$", row_e064(0.25, 3, "fresh")),
    ("", "M2, $\\rho{=}0.5$", row_e064(0.5, 3, "fresh")),
    ("", "M2, $\\rho{=}0.75$", row_e064(0.75, 3, "fresh")),
    ("", "M2, $\\rho{=}1$", row_e064(1.0, 3, "fresh")),
    ("Attempt budget (E064)", "$K{=}1$ (fresh = same)", row_e064(0.0, 1, "fresh")),
    ("", "$K{=}2$", row_e064(0.0, 2, "fresh")),
    ("", "$K{=}4$", row_e064(0.0, 4, "fresh")),
    ("", "$K{=}5$", row_e064(0.0, 5, "fresh")),
    ("Attacker (E064)", "same image, $K{=}2$", row_e064(0.0, 2, "same")),
    ("", "same image, $K{=}3$", row_e064(0.0, 3, "same")),
    ("", "same image, $K{=}5$", row_e064(0.0, 5, "same")),
    ("", "worst generator, $K{=}1$", row_e064(0.0, 1, "worstgen")),
    ("", "worst generator, $K{=}3$", row_e064(0.0, 3, "worstgen")),
]
FULL_BREAKS = (2, 4, 8, 12)

ROWS = [
    ("E061 reference", "M0, all, $K{=}3$, fresh", row_e061()),
    ("+ refitting (E067)", "thresholds/parameters refitted", row_e067()),
    ("Selection control", "M0, $n_f\\ge3$", row_e063("M0e")),
    ("Without replacement", "M1, $n_f\\ge3$", row_e063("M1")),
    ("Persistence", "M2, $\\rho{=}1$", row_e064(1.0, 3, "fresh")),
    ("Attempt budget", "$K{=}1$ (any attacker)", row_e064(0.0, 1, "fresh")),
    ("", "$K{=}2$", row_e064(0.0, 2, "fresh")),
    ("Attacker", "same image, $K{=}3$", row_e064(0.0, 3, "same")),
    ("", "worst generator, $K{=}3$", row_e064(0.0, 3, "worstgen")),
]


def cell(m, lo, hi):
    s = f"{num(m, 3, plus=True)} {iv(lo, hi)}"
    return f"\\textbf{{{s}}}" if (lo > 0 or hi < 0) else s


with open(OUT / "tab_sens.tex", "w") as fh:
    fh.write("% AUTOGENERATED by paper_tcss/tools/mk_sens.py -- do not edit by hand\n")
    for k in ("E061", "E063", "E067", "E064"):
        fh.write(f"% source {SRC[k].relative_to(ROOT)} sha256 {sha(SRC[k])}\n")
    fh.write("\\begin{tabular}{@{}ll@{\\hspace{4pt}}cccc@{\\hspace{4pt}}c@{}}\n\\toprule\n")
    fh.write("& & \\multicolumn{3}{c}{$\\pi_1$ always restore} & $\\pi_4$ step-up & $\\pi_2$ \\\\\n")
    fh.write("\\cmidrule(lr){3-5}\\cmidrule(lr){6-6}\n")
    fh.write("Analysis & Model & $\\Delta J$, $N_{\\text{legacy}}$ (H1) & $\\Delta J$, $N_{\\text{fixed}}$ & "
             "$\\Delta\\Delta J$ (H2) & $\\Delta J$, $N_{\\text{legacy}}$ & leg./fix. \\\\\n\\midrule\n")
    for i, (a, m, (vals, p2)) in enumerate(ROWS):
        if i in (2, 4, 5, 7):
            fh.write("\\midrule\n")
        fh.write(f"{a} & {m} & " + " & ".join(cell(*v) for v in vals[:4]) + f" & {p2[0]}/{p2[1]} \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")

with open(OUT / "supp_tab_sens.tex", "w") as fh:
    fh.write("% AUTOGENERATED by paper_tcss/tools/mk_sens.py -- do not edit by hand\n")
    for k in ("E061", "E063", "E067", "E064"):
        fh.write(f"% source {SRC[k].relative_to(ROOT)} sha256 {sha(SRC[k])}\n")
    fh.write("\\begin{tabular}{@{}ll@{\\hspace{4pt}}ccccc@{\\hspace{4pt}}c@{}}\n\\toprule\n")
    fh.write("& & \\multicolumn{3}{c}{$\\pi_1$ always restore} & \\multicolumn{2}{c}{$\\pi_4$ step-up band} & $\\pi_2$ \\\\\n")
    fh.write("\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}\n")
    fh.write("Analysis & Model & $\\Delta J$, legacy (H1) & $\\Delta J$, fixed & $\\Delta\\Delta J$ (H2) & "
             "$\\Delta J$, legacy & $\\Delta J$, fixed & leg./fix. \\\\\n\\midrule\n")
    for i, (a, m, (vals, p2)) in enumerate(ROWS_FULL):
        if i in FULL_BREAKS:
            fh.write("\\midrule\n")
        fh.write(f"{a} & {m} & " + " & ".join(cell(*v) for v in vals) + f" & {p2[0]}/{p2[1]} \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------------------ matcher table
L, D = e71["levels"], e71["deltas"]
CONDS = [("degraded", "Degraded (anchor)"), ("restored_bilateral", "Bilateral"), ("restored_tv", "Total variation")]
with open(OUT / "tab_matcher.tex", "w") as fh:
    fh.write("% AUTOGENERATED by paper_tcss/tools/mk_sens.py -- do not edit by hand\n")
    fh.write(f"% source {SRC['E071'].relative_to(ROOT)} sha256 {sha(SRC['E071'])}\n")
    fh.write("\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n")
    fh.write("& \\multicolumn{3}{c}{FNMR, validation FMR $10^{-2}$} & & \\\\\n\\cmidrule(lr){2-4}\n")
    fh.write("Probe & locked $\\tau_v$ & $\\Delta$ [95\\,\\%] & $\\Delta$, matched & EER & FMR \\\\\n\\midrule\n")
    for c, name in CONDS:
        f = L[f"FNMR|anchor|0.01|{c}"]["estimate"]; e = L[f"EER|none|0.0|{c}"]["estimate"]
        fm = L[f"FMR|anchor|0.01|{c}"]["estimate"]
        if c == "degraded":
            fh.write(f"{name} & {num(f)} & --- & --- & {num(e)} & {num(fm, 4)} \\\\\n")
        else:
            d = D[f"dFNMR|anchor|0.01|{c}"]; dm = D[f"dFNMR|matched|0.01|{c}"]
            fh.write(f"{name} & {num(f)} & {num(d['estimate'], 3, True)} {iv(*d['ci95'])} & "
                     f"{num(dm['estimate'], 3, True)} & {num(e)} & {num(fm, 4)} \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------------------ macros
M = {}
# E063 Part B
for mo, tag in (("M1", "Mone"), ("M0e", "Mzeroe")):
    for (_, pol, arm), nm in zip(READ, ("OneLeg", "OneFix", "OneDD", "FourLeg", "FourFix")):
        d = e63["contrasts"][f"dJ|{pol}|{arm}|{mo}"]
        M[f"Sx{tag}{nm}"] = num(d["median"], 4, True); M[f"Sx{tag}{nm}Lo"] = num(d[LEVEL][0], 4); M[f"Sx{tag}{nm}Hi"] = num(d[LEVEL][1], 4)
for eff, nm in (("selection", "Sel"), ("model", "Mod"), ("total", "Tot")):
    for q in ("dJ", "dL"):
        d = e63["effects"][f"{eff}|{q}|pi1|leg"]
        M[f"Sx{nm}{q[1]}"] = num(d["median"], 4, True)
        M[f"Sx{nm}{q[1]}Lo"] = num(d["ci9875"][0], 4); M[f"Sx{nm}{q[1]}Hi"] = num(d["ci9875"][1], 4)
M["SxMoneExclCred"] = str(e63r["gates"]["elig_counts"]["p1"]["excluded_K3"])
# E067
rat = [float(e67[k]["width_ratio"]) for k in ("dJ|pi1|leg", "dJ|pi1|fix", "dJ|pi1|dd", "dJ|pi4|leg", "dJ|pi4|fix", "E060_dL3|leg")]
M["SxRefitRatioMin"] = f"{min(rat):.2f}"; M["SxRefitRatioMax"] = f"{max(rat):.1f}"
r = e67["E060_dL3|leg"]
M["SxRefitEsixLo"] = num(float(r["refit_on_lo"]), 3); M["SxRefitEsixHi"] = num(float(r["refit_on_hi"]), 3)
r = e67["dJ|pi1|leg"]; M["SxRefitOneLegLo"] = num(float(r["refit_on_lo"]), 3); M["SxRefitOneLegHi"] = num(float(r["refit_on_hi"]), 3)
# E064
same = [g64("M2", 0.0, k, "same", "R-H1") for k in (2, 3, 4, 5)]
M["SxSameOneLegMin"] = num(min(s[0] for s in same), 3, True); M["SxSameOneLegMax"] = num(max(s[0] for s in same), 3, True)
samef = [g64("M2", 0.0, k, "same", "fixed_pi1") for k in (2, 3, 4, 5)]
M["SxSameOneFixMin"] = num(min(s[0] for s in samef), 3, True); M["SxSameOneFixMax"] = num(max(s[0] for s in samef), 3, True)
M["SxSameOneFixResolvedN"] = str(sum(s[1] > 0 for s in samef))
p4 = [g64("M2", None, 0, "", "")] if False else None
p4l = [float(rw["median"]) for rw in e64 if rw["model"] == "M2" and rw["reading"] == "R-H4|leg"]
p4f = [float(rw["median"]) for rw in e64 if rw["model"] == "M2" and rw["reading"] == "R-H4|fix"]
M["SxFourRangeLo"] = num(min(p4l + p4f), 3, True); M["SxFourRangeHi"] = num(max(p4l + p4f), 3, True)
M["SxFourUnchanged"] = str(sum(rw["label_vs_E061"] == "unchanged" for rw in e64 if rw["model"] == "M2" and rw["reading"].startswith("R-H4")))
M["SxFourTotal"] = str(sum(1 for rw in e64 if rw["model"] == "M2" and rw["reading"].startswith("R-H4")))
cnt = lambda rd, lab: sum(1 for rw in e64 if rw["model"] == "M2" and rw["reading"] == rd and rw["label_vs_E061"] == lab)
for rd, nm in (("R-H1", "One"), ("R-H2", "DD")):
    for lab, ln in (("unchanged", "Unch"), ("narrowed", "Narr"), ("reversed", "Rev")):
        M[f"Sx{nm}{ln}N"] = str(cnt(rd, lab))
M["SxConfigN"] = str(len({(rw["rho"], rw["K"], rw["attacker"]) for rw in e64 if rw["model"] == "M2"}))
M["SxTwoWorseN"] = str(sum(1 for rw in e64 if rw["model"] == "M2" and rw["reading"] == "R-H3|pi2|leg" and rw["label_vs_E061"] == "newly resolved worse"))
p2same = [g64("M2", None if False else 0.0, k, "same", "R-H3|pi2|leg")[0] for k in (2, 3, 4, 5)] + \
         [g64("M2", 0.0, k, "same", "R-H3|pi2|fix")[0] for k in (2, 3, 4, 5)] + \
         [g64("M2", 0.0, 1, "fresh", f"R-H3|pi2|{a}")[0] for a in ("leg", "fix")]
M["SxTwoWorseMin"] = num(min(p2same), 2, True); M["SxTwoWorseMax"] = num(max(p2same), 2, True)
for k in (2, 3, 4, 5):
    for q, nm in (("selection", "Sel"), ("without_replacement", "Wr")):
        d = e64d[f"K{k}|dJ_pi1_leg|{q}"]
        M[f"Sx{nm}K{'abcde'[k-1]}"] = num(d["median"], 3, True)
for k in (4, 5):
    v = g64("M1|E_K", None, k, "fresh", "R-H1")
    M[f"SxMoneK{'abcde'[k-1]}"] = num(v[0], 4, True); M[f"SxMoneK{'abcde'[k-1]}Lo"] = num(v[1], 4); M[f"SxMoneK{'abcde'[k-1]}Hi"] = num(v[2], 4)
rk = e64k
M["SxFourShareKc"] = f"{rk['M2|rho=0.0|K=3|fresh|leg']['grid_share']['pi4']:.2f}"
M["SxFourShareKa"] = f"{rk['M2|rho=0.0|K=1|fresh|leg']['grid_share']['pi4']:.2f}"
M["SxSameOneWinsMax"] = str(max(v["argmin_cstar"]["pi1"] for k, v in rk.items() if "|same|" in k))
# E071
taus = [float(r["tau"]) for r in e71v if r["threshold"] == "anchor" and r["fmr"] == "0.01"]
M["SxTauLo"] = f"{min(taus):.2f}"; M["SxTauHi"] = f"{max(taus):.2f}"
for c, nm in (("restored_bilateral", "Bil"), ("restored_tv", "Tv")):
    for t, tn in (("anchor", "V"), ("mt", "A"), ("matched", "M")):
        d = D[f"dFNMR|{t}|0.01|{c}"]
        M[f"Sx{tn}{nm}"] = num(d["estimate"], 3, True); M[f"Sx{tn}{nm}Lo"] = num(d["ci95"][0], 3); M[f"Sx{tn}{nm}Hi"] = num(d["ci95"][1], 3)
    d = D[f"dEER|none|0.0|{c}"]; M[f"SxEer{nm}"] = num(L[f"EER|none|0.0|{c}"]["estimate"], 3)
    d = D[f"dFMR|anchor|0.01|{c}"]; M[f"SxFmrD{nm}"] = num(d["estimate"], 4, True); M[f"SxFmrD{nm}Lo"] = num(d["ci95"][0], 4); M[f"SxFmrD{nm}Hi"] = num(d["ci95"][1], 4)
g71 = json.loads(SRC["E071G"].read_text())
M["SxEerClean"] = f"{sum(g71[f'G_E|s{i}']['eer'] for i in range(3)) / 3:.2f}"
M["SxFnmrEsixfive"] = f"{json.loads(SRC['E065'].read_text())['dFNMR|0.01|bil']['fnmr_degraded']:.2f}"
M["SxVBilShort"] = f"{D['dFNMR|anchor|0.01|restored_bilateral']['estimate']:.2f}"
M["SxVTvShort"] = f"{D['dFNMR|anchor|0.01|restored_tv']['estimate']:.2f}"
M["SxEerDeg"] = num(L["EER|none|0.0|degraded"]["estimate"], 3)
M["SxFnmrDeg"] = num(L["FNMR|anchor|0.01|degraded"]["estimate"], 2)
M["SxFnmrDegMt"] = num(L["FNMR|mt|0.01|degraded"]["estimate"], 2)
M["SxCapN"] = str(e71["n"]["captures"]); M["SxCredN"] = str(e71["n"]["credentials"])
C = e71["cascade"]
for arm, an in (("leg", "Leg"), ("fix", "Fix")):
    for pol, pn in (("pi1", "One"), ("pi2", "Two")):
        d = C[f"delta|Lcasc3|{pol}|{arm}"]
        M[f"SxCasc{pn}{an}"] = num(d["median"], 3, True); M[f"SxCasc{pn}{an}Lo"] = num(d["ci95"][0], 3); M[f"SxCasc{pn}{an}Hi"] = num(d["ci95"][1], 3)
        d = C[f"delta|Ldet3|{pol}|{arm}"]; M[f"SxDet{pn}{an}"] = num(d["median"], 3, True)
    M[f"SxAtten{an}"] = f"{C[f'A|pi1|{arm}']['ratio_of_medians']:.0f}"
lv = C["level|Lcasc3|pi0|leg"]; M["SxCascZero"] = num(lv["median"], 2); M["SxCascZeroLo"] = num(lv["ci95"][0], 2); M["SxCascZeroHi"] = num(lv["ci95"][1], 2)
lv = C["level|Lcasc1|pi0|leg"]; M["SxCascZeroOne"] = num(lv["median"], 2)
M["SxDetZero"] = num(C["level|Ldet3|pi0|leg"]["median"], 4)
sza = e71["synthetic"]
M["SxSzaDeg"] = f"{100 * sza['SZA|p1|0.01|degraded']['estimate']:.1f}"
M["SxSzaRestMax"] = f"{100 * max(sza[f'SZA|{p}|0.01|{c}']['estimate'] for p in ('p1', 'p2') for c in ('restored_bilateral', 'restored_tv')):.1f}"
M["SxScaZero"] = num(C["level|SCA1|pi0|leg"]["median"], 4)
M["SxScaTwo"] = num(C["delta|SCA1|pi2|leg"]["median"], 4, True)

with open(OUT / "numbers_sens.tex", "w") as fh:
    fh.write("% AUTOGENERATED by paper_tcss/tools/mk_sens.py -- do not edit by hand\n")
    for k, p in SRC.items():
        fh.write(f"% source {p.relative_to(ROOT)} sha256 {sha(p)}\n")
    for k, v in M.items():
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

# ------------------------------------------------------------------------------ supplementary: full grid (legacy R-H1 / R-H2)
with open(OUT / "supp_sens_grid.tex", "w") as fh:
    fh.write("% AUTOGENERATED by paper_tcss/tools/mk_sens.py -- do not edit by hand\n")
    fh.write(f"% source {SRC['E064'].relative_to(ROOT)} sha256 {sha(SRC['E064'])}\n")
    fh.write("\\begin{tabular}{@{}lc" + "c" * 5 + "@{}}\n\\toprule\n")
    fh.write("Attacker & $K$ & " + " & ".join(f"$\\rho={r}$" for r in ("0", "0.25", "0.5", "0.75", "1")) + " \\\\\n\\midrule\n")
    lab = {"unchanged": "u", "narrowed": "n", "reversed": "r"}
    for att, an in (("fresh", "fresh"), ("same", "same image"), ("worstgen", "worst gen.")):
        for k in (1, 2, 3, 4, 5):
            cells = []
            for rho in (0.0, 0.25, 0.5, 0.75, 1.0):
                m1, _, _, l1 = g64("M2", rho, k, att, "R-H1"); m2, _, _, l2 = g64("M2", rho, k, att, "R-H2")
                cells.append(f"{num(m1, 3, True)}\\,{lab[l1]} / {num(m2, 3, True)}\\,{lab[l2]}")
            fh.write(f"{an if k == 1 else ''} & {k} & " + " & ".join(cells) + " \\\\\n")
        fh.write("\\midrule\n" if att != "worstgen" else "")
    fh.write("\\bottomrule\n\\end{tabular}\n")
print(f"wrote {len(M)} macros, tab_sens.tex, tab_matcher.tex, supp_sens_grid.tex")
