# [Public copy of paper_tcss/tools/mk_ext.py: only input/output paths are remapped to decision_level/results/.]
#!/usr/bin/env python3
"""Macros and Supplementary tables for the post hoc external bona-fide analyses (E069, E070) and the
fingerprint-enhancement condition (E068).

Reads AGGREGATE outputs only (device-level CSV/JSON; no window, capture, score dump or per-capture quality file).
Writes
  paper_tcss/generated/numbers_ext.tex  (\\Ext... macros used by the main text)
  paper_tcss/generated/supp_ext.tex     (Supplementary section: full E069/E070 tables)
Run from the repository root:  python3 paper_tcss/tools/mk_ext.py
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[2]
A = ROOT / "decision_level" / "results"   # [public copy: released aggregates]
OUT = ROOT / "decision_level" / "results" / "C1_macros_rebuilt"   # [public copy]
OUT.mkdir(parents=True, exist_ok=True)
E69 = A / "E069" / "analysis"
E70 = A / "E070" / "contrast"
E70Q = A / "E070" / "quality" / "tertile_edges.json"
E68 = A / "E068" / "analysis"
SRC = {"readings": E69 / "readings.json", "rej": E69 / "rejection_by_device.csv", "dB": E69 / "deltaB.csv",
       "q_readings": E70 / "readings.json", "q_contrast": E70 / "contrast.csv", "q_edges": E70Q,
       "e68_class": E68 / "class.public.json", "e68_cells": E68 / "cells_subject.csv", "e68_head": E68 / "headline_dB.csv"}
NAME = {"FVC2000_Db1": "FVC2000 DB1", "FVC2000_Db2": "FVC2000 DB2", "FVC2000_Db3": "FVC2000 DB3",
        "FVC2002_Db1": "FVC2002 DB1", "FVC2002_Db3": "FVC2002 DB3", "FVC2004_DB1": "FVC2004 DB1",
        "FVC2004_DB2": "FVC2004 DB2", "SD302_C": "SD302 C", "SD302_E": "SD302 E", "SD302_G": "SD302 G",
        "SD302_K": "SD302 K", "SD302_M": "SD302 M", "SD302_P": "SD302 P", "SD302_S": "SD302 S"}
FAM = ("Rn", "Bs", "Dt")


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rows(p):
    return list(csv.DictReader(open(p)))


def pct(x, d=1):
    return f"{100 * x:.{d}f}\\,\\%"


def main() -> None:
    R = json.loads(SRC["readings"].read_text())
    devs = list(R["readings"])
    assert len(devs) == 14
    # ------------------------------------------------ X1: domain shift of the locked threshold (degraded)
    rej = [r for r in rows(SRC["rej"]) if r["condition"] == "degraded"]
    assert len(rej) == 168, len(rej)
    shift = [float(r["domain_shift_vs_trackA"]) for r in rej]
    twice = sum(float(r["B"]) > 2 * float(r["trackA_B_degraded"]) for r in rej)
    ta = sorted({round(float(r["trackA_B_degraded"]), 6) for r in rej})
    # ------------------------------------------------ X2 under the rule (frozen readings) and without it
    dB = rows(SRC["dB"])
    pos = {(r["device"], r["family"], r["protocol"], r["arm"], r["op"]): r["resolved_pos"] == "True" for r in dB}
    x2 = {op: [g for g in devs if R["readings"][g][op]["X2_transfers"]] for op in ("bil", "tv")}
    interp = {op: [g for g in devs if R["readings"][g][op]["interpretable"]] for op in ("bil", "tv")}
    noeli = {op: [g for g in devs if sum(pos[(g, f, "p1", "leg", op)] and pos[(g, f, "p2", "leg", op)] for f in FAM) >= 2]
             for op in ("bil", "tv")}
    assert x2["bil"] == ["SD302_S"] and sorted(x2["tv"]) == ["SD302_M", "SD302_S"], x2                        # E069 record
    assert len(noeli["bil"]) == 7 and len(noeli["tv"]) == 8, noeli                 # E069 errata item 5
    n_interp_any = len([g for g in devs if R["readings"][g]["bil"]["interpretable"] and R["readings"][g]["tv"]["interpretable"]])
    neg = {(r["device"], r["family"], r["protocol"], r["arm"], r["op"]): r["resolved_neg"] == "True" for r in dB}
    dt_neg = {op: [g for g in devs if neg[(g, "Dt", "p1", "leg", op)] and neg[(g, "Dt", "p2", "leg", op)]] for op in ("bil", "tv")}
    assert len(dt_neg["bil"]) == 6 and len(dt_neg["tv"]) == 5, dt_neg                # E069 record + errata items 1-2
    # ------------------------------------------------ E070
    Q = json.loads(SRC["q_readings"].read_text())
    q_dev = list(Q["readings"])
    q_sup = [g for g in q_dev if any(Q["readings"][g][op]["Q1_supported"] for op in ("bil", "tv"))]
    q_full = [g for g in q_dev if all(len(Q["readings"][g][op]["families_eligible"]) == 3 and not Q["readings"][g][op]["underpowered"]
                                      for op in ("bil", "tv"))]
    assert q_sup == [] and len(q_full) == 1, (q_sup, q_full)
    elig = {(r["device"], r["family"], r["protocol"], r["arm"]): r["cell_eligible"] == "True" for r in rej}
    excl = [k for k, v in elig.items() if not v]
    excl_pos = [k for k in excl if pos[k + ("bil",)] and pos[k + ("tv",)]]
    excl_neg = [k for k in excl if neg[k + ("bil",)] or neg[k + ("tv",)]]
    assert len(excl) == 20 and len(excl_pos) == 17 and not excl_neg, (len(excl), len(excl_pos), excl_neg)   # E069 errata 4
    # ------------------------------------------------ E068
    C68 = json.loads(SRC["e68_class"].read_text())
    cells68 = [r for r in rows(SRC["e68_cells"]) if r["role"] == "primary"]
    head68 = rows(SRC["e68_head"])
    assert C68["R1_subject"]["S_fix"]["S_fix"] == 8 and C68["R1_subject"]["outcome"] == "B"
    assert len(head68) == 12 and all(float(r["B_fpenh"]) == 0.0 and float(r["dB_hi"]) < 0 for r in head68)
    dT68 = [float(r[f"dT_{a}"]) for r in cells68 for a in ("leg", "fix")]
    assert all(v < 0 for v in dT68)
    r3 = C68["R3"]["0.01"]
    qc_all = rows(SRC["q_contrast"])
    fd = [r for r in qc_all if r["device"] == q_full[0] and (float(r["lo95"]) > 0 or float(r["hi95"]) < 0)]
    n_res, n_res_neg = len(fd), sum(float(r["hi95"]) < 0 for r in fd)
    assert (n_res, n_res_neg) == (11, 10), (n_res, n_res_neg)                  # independent C1 audit (E1)
    m = {
        "ExtQRes": n_res, "ExtQResNeg": n_res_neg,
        "ExtNExcl": len(excl), "ExtNExclPos": len(excl_pos),
        "EnhDTmin": f"{max(dT68):.2f}".replace("-", "$-$"), "EnhDTmax": f"{min(dT68):.2f}".replace("-", "$-$"),
        "EnhHeadB": f"{max(float(r['B_degraded']) for r in head68) * 100:.1f}\\,\\%",
        "EnhHeadBmin": f"{min(float(r['B_degraded']) for r in head68) * 100:.1f}",
        "EnhFnmrDeg": f"{r3['FNMR_degraded']:.3f}", "EnhFnmr": f"{r3['FNMR_fpenh']:.3f}",
        "EnhDFnmr": f"{r3['dFNMR']:+.3f}", "EnhDFnmrLo": f"{r3['ci95'][0]:+.3f}", "EnhDFnmrHi": f"{r3['ci95'][1]:+.3f}",
        "ExtNDev": len(devs), "ExtNFvc": sum(g.startswith("FVC") for g in devs), "ExtNSd": sum(g.startswith("SD302") for g in devs),
        "ExtNCells": len(rej), "ExtNCellsTwice": twice,
        "ExtShiftMed": f"{median(shift):+.3f}", "ExtShiftMin": f"{min(shift):+.3f}", "ExtShiftMax": f"{max(shift):+.3f}",
        "ExtTrackAMin": pct(min(ta)), "ExtTrackAMax": pct(max(ta)),
        "ExtXtwoBil": len(x2["bil"]), "ExtXtwoTv": len(x2["tv"]),
        "ExtXtwoDevs": " and ".join(NAME[g] for g in sorted(set(x2["bil"]) | set(x2["tv"]))),
        "ExtInterp": n_interp_any, "ExtNoRuleBil": len(noeli["bil"]), "ExtNoRuleTv": len(noeli["tv"]),
        "ExtQDev": len(q_dev), "ExtQFull": len(q_full), "ExtQFullDev": NAME[q_full[0]],
    }
    lines = ["% Generated by paper_tcss/tools/mk_ext.py from aggregate E069/E070 outputs. Do not edit.",
             *[f"% {k}: {v.relative_to(ROOT)} sha256 {sha(v)[:16]}" for k, v in SRC.items()]]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items()]
    (OUT / "numbers_ext.tex").write_text("\n".join(lines) + "\n")

    # ------------------------------------------------ Supplementary section
    def cell(g, op):
        r = R["readings"][g][op]
        if not r["interpretable"]:
            return "n.i."
        return "yes" if r["X2_transfers"] else "no"

    def fams_pos(g, op):
        f = [x for x in FAM if pos[(g, x, "p1", "leg", op)] and pos[(g, x, "p2", "leg", op)]]
        return ",".join(f) if f else "--"

    def shift_med(g):
        v = [float(r["domain_shift_vs_trackA"]) for r in rej if r["device"] == g]
        return f"{median(v):+.3f}"

    def n_win(g):
        r = next(r for r in rej if r["device"] == g)
        return r["n_windows"], r["n_clusters"]

    S = [r"% Generated by paper_tcss/tools/mk_ext.py. Do not edit.",
         r"\section{External bona-fide captures (post hoc): does the operating point or the restoration cost transfer?}",
         r"\label{sec:s8}",
         r"These analyses (experiment records E069 and E070) were designed after the Track A results were known and use",
         r"public captures from other sensors: the FVC2000, FVC2002 and FVC2004 sets (Maltoni et al.) and NIST Special",
         r"Database 302 (the Nail to Nail Fingerprint Challenge; NIST TN 2007). They are post hoc external evidence, not",
         r"independent prospective validation. Only bona-fide captures are used: no synthetic image is generated for",
         r"these sensors, so only bona-fide rejection is measured. One $88\times88$ window per capture is taken about the",
         r"foreground centroid (both the 88 and 176 windows at least 80\,\% foreground); the Track A degradation",
         r"conditions and restoration operators are applied unchanged; every detector, normalization arm and threshold",
         r"$\tau_0$ is the frozen Track A one. Intervals are 95\,\% cluster-bootstrap intervals ($B=10{,}000$) with fingers",
         r"(FVC; no volunteer identifiers are released) or subjects (SD302; the same participants recur across devices)",
         r"as clusters, and devices are never pooled. Only aggregates are reported; no SD302 image or subject-level value is",
         r"redistributed.",
         "",
         rf"\paragraph{{The operating point does not transfer}} At $\tau_0$, the frozen detectors reject degraded external",
         rf"bona-fide windows at rates that differ from Track A's ({m['ExtTrackAMin']}--{m['ExtTrackAMax']}) by a median of",
         rf"{m['ExtShiftMed']} (range {m['ExtShiftMin']} to {m['ExtShiftMax']}); {twice} of {len(rej)} device $\times$",
         r"detector $\times$ protocol $\times$ arm cells exceed twice the Track A rate (Table~\ref{tab:s_ext}).",
         "",
         r"\paragraph{The restoration cost transfers selectively} The preregistered rule asked whether the legacy-arm",
         r"increase in bona-fide rejection ($\Delta B>0$, interval excluding 0, under both protocol models) appears for at",
         r"least two of the three detector families, counting only cells whose degraded rejection lay in $[0.005,0.95]$.",
         r"It held on SD302 S for both operators and on SD302 M for TV only, and on no FVC",
         rf"device; {len(devs) - n_interp_any} of {len(devs)} devices had no interpretable reading for at least one operator,",
         r"mostly because protocol-2 legacy cells fell below the floor. The floor rule also removed cells in which",
         r"$\Delta B$ was resolved above zero; without it (not preregistered; descriptive) the rule would hold on",
         rf"{m['ExtNoRuleBil']} devices for bilateral and {m['ExtNoRuleTv']} for TV. The two pixel-domain detectors show the",
         r"Track A direction on most SD302 devices and two FVC sets, whereas D3's legacy $\Delta B$ is resolved below zero under both protocol models on",
         rf"{len(dt_neg['bil'])} devices (bilateral) and {len(dt_neg['tv'])} (TV): outside Track A, the sign of the",
         r"restoration effect on bona-fide rejection depends on the instrument and the device.",
         "",
         r"\paragraph{Capture quality: inconclusive} Among in-domain captures for NIST NFIQ~2 (v2.3.0; plain optical",
         rf"captures at 500\,ppi with the ISO minimum area; {m['ExtQDev']} devices), the preregistered contrast (lowest minus",
         r"highest quality tertile of $\Delta B$, legacy arm) was not supported on any device. Only",
         rf"{m['ExtQFullDev']} had all three detector families eligible and adequate power, and {n_res_neg} of its {n_res} resolved",
         r"contrasts (all arms and operators) pointed the other way (the high-quality tertile showed the larger change). The result is therefore",
         r"reported as inconclusive: no quality moderation was detected, which is not evidence of its absence.",
         "",
         r"\begin{table*}[!t]\centering\scriptsize",
         r"\caption{External bona-fide captures (post hoc). Windows and clusters per device; median domain shift of the",
         r"degraded bona-fide rejection at $\tau_0$ relative to Track A over the 12 detector $\times$ protocol $\times$ arm",
         r"cells; the preregistered transfer reading per operator (yes / no / n.i.\ = no interpretable reading); and, without",
         r"the floor rule (descriptive), the families with legacy $\Delta B$ resolved $>0$ under both protocol models. SD302 P (28 subjects) is underpowered.}",
         r"\label{tab:s_ext}",
         r"\begin{tabular}{lrrrcccc}\toprule",
         r"Device & Windows & Clusters & Median shift & Rule: bil. & Rule: TV & No rule: bil. & No rule: TV\\\midrule"]
    for g in devs:
        w, c = n_win(g)
        S.append(f"{NAME[g]} & {w} & {c} & {shift_med(g)} & {cell(g, 'bil')} & {cell(g, 'tv')} & {fams_pos(g, 'bil')} & {fams_pos(g, 'tv')}\\\\")
    S += [r"\bottomrule\end{tabular}\end{table*}", ""]
    edges = json.loads(SRC["q_edges"].read_text())
    qc = rows(SRC["q_contrast"])
    S += [r"\begin{table}[!t]\centering\scriptsize",
          r"\caption{Capture-quality contrast (post hoc, NFIQ~2 domain): legacy-arm $\Delta B$ in the lowest minus the highest",
          r"NFIQ~2 tertile, bilateral, by detector family and protocol model (95\,\% cluster-bootstrap interval).}",
          r"\label{tab:s_ext_q}",
          r"\begin{tabular}{lccc}\toprule Device (P1 / P2) & Rn & Bs & Dt\\\midrule"]
    for g in q_dev:
        cells = []
        elig_f = set(Q["readings"][g]["bil"]["families_eligible"])
        for f in FAM:
            v = []
            if f not in elig_f:
                cells.append("-- / --"); continue
            for p in ("p1", "p2"):
                r = next((r for r in qc if r["device"] == g and r["family"] == f and r["protocol"] == p and r["arm"] == "leg"
                          and r["op"] == "bil"), None)
                v.append("--" if r is None else f"{float(r['contrast_low_minus_high']):+.3f}" + (
                    "$^\\ast$" if float(r["lo95"]) > 0 or float(r["hi95"]) < 0 else ""))
            cells.append(" / ".join(v))
        S.append(f"{NAME.get(g, g)} & " + " & ".join(cells) + r"\\")
    S += [r"\bottomrule\multicolumn{4}{l}{$^\ast$ interval excludes 0. -- = family not eligible (rejection-floor rule).}\\",
          r"\multicolumn{4}{l}{SD302 P is underpowered; SD302 S is a secondary stratum.}\end{tabular}\end{table}", ""]
    S += [r"\paragraph{Floor rule (audit)} The rule's floor excluded " + f"{len(excl)} of {len(rej)}" + r" cells, all protocol-2 cells",
          r"with degraded rejection below 0.005; " + f"{len(excl_pos)}" + r" of them had $\Delta B$ resolved above zero under both operators",
          r"and none below zero. The rule was added on review, before any outcome was read; the rule-based counts therefore",
          r"understate the descriptive pattern, and the without-rule counts are a sensitivity analysis, not a registered result.",
          ""]
    NM = {"d3": "D3", "resnet18_imagenet": "ResNet-18 (ImageNet)", "resnet50_imagenet": "ResNet-50 (ImageNet)",
          "efficientnet_b0_imagenet": "EfficientNet-B0 (ImageNet)"}
    S += [r"\section{A fingerprint-specific enhancer as the upstream transformation (post hoc)}",
          r"\label{sec:s9}",
          r"Experiment record E068 replaced the generic operators by FpEnhancer (Guan et al.; UNet variant, authors' weights,",
          r"MIT licence; \url{https://github.com/XiongjunGuan/FpEnhancer}), a learned enhancer whose output is a near-binary ridge map, applied to the degraded test and",
          r"validation images of both protocols and scored once by the frozen detectors and the matcher; it is post hoc and",
          r"is one enhancer, not an operator class, applied outside its training distribution (degraded crops rather than",
          r"rolled prints). The detector cells are the E049 instrument set (D3 and three ImageNet",
          r"backbones); the headline detectors are read separately below. All reproduction gates passed. Intervals are 95\,\% bootstrap intervals",
          r"over seeds, the 34 test subjects and synthetic images within generator ($B=10{,}000$).",
          "",
          r"\begin{table}[!t]\centering\scriptsize",
          r"\caption{Fingerprint-specific enhancement versus the degraded anchor: change in locked detection rate",
          r"($\Delta T$) and bona-fide rejection ($\Delta B$) at the validation 1\,\% threshold, legacy / fixed",
          r"normalization, mean of 10 seeds; $^\ast$ interval excludes 0.}",
          r"\label{tab:s_enh}",
          r"\begin{tabular}{llcc}\toprule Instrument & Prot. & $\Delta T$ (leg / fix) & $\Delta B$ (leg / fix)\\\midrule"]
    for r in cells68:
        def f(k):
            v, lo, hi = float(r[k]), float(r[k + "_lo"]), float(r[k + "_hi"])
            return f"{v:+.3f}".replace("-", "$-$") + ("$^\\ast$" if lo > 0 or hi < 0 else "")
        S.append(f"{NM[r['inst']]} & {r['protocol']} & {f('dT_leg')} / {f('dT_fix')} & {f('dB_leg')} / {f('dB_fix')}\\\\")
    S += [r"\bottomrule\end{tabular}\end{table}", "",
          r"At the frozen thresholds $\tau_0$ of the three headline detectors, bona-fide rejection of the 1,653 test reals fell",
          r"from " + f"{m['EnhHeadBmin']}--{m['EnhHeadB']}" + r" to zero in all 12 detector $\times$ protocol $\times$ arm cells. At the",
          r"matcher's validation threshold (FMR $10^{-2}$), the single-template false non-match rate of enhanced probes",
          r"against clean templates rose from " + m["EnhFnmrDeg"] + " to " + m["EnhFnmr"] + r" (change " + m["EnhDFnmr"] + r", interval",
          m["EnhDFnmrLo"] + " to " + m["EnhDFnmrHi"] + r"). The detector shifts persisted under fixed normalization in all 8 cells",
          r"(primary readout $S_\mathrm{fix}=8/8$; E049 class B), AUC fell in 7 of 8 cells (to $-0.37$), and the per-image min--max",
          r"range was essentially unchanged (median ratio 1.00), consistent with a shift that does not run through per-image",
          r"normalization. Templates were not enhanced; an enhance-both pipeline",
          r"was not evaluated, and the lower synthetic detection rate is a measurement of this transformation, not an",
          r"evaluated attack.", ""]
    (OUT / "supp_ext.tex").write_text("\n".join(S) + "\n")
    del edges
    print(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
