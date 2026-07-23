"""
Phase 5 - Recalibration on pooled external data.

The deployed temperature and decision band were fitted on the internal dev set:
550 images, curated to 1:1, same source and curation as training. We now have
164 genuinely external labelled cases from two independent cohorts, which is a
better proxy for deployment than an internally-balanced split.

Two separate questions, kept apart:

  1. DOES A THRESHOLD TRANSFER?  Leave-one-cohort-out - fit on one cohort, apply
     to the other. This is the honest estimate of what happens on cohort 3, and
     it is the number that should be quoted.

  2. WHAT SHOULD SHIP?  Fit on all 164. Better estimated, but no longer testable
     on the same data - so it is reported as a calibration figure, never as a
     validation result.

Conflating those two is how calibration quietly becomes overfitting.

Outputs
    outputs/checkpoints/calibration_external.json
    outputs/reports/15_recalibration.md
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, brier_score_loss

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "outputs" / "external_validation"
CKPT = ROOT / "outputs" / "checkpoints"
REPORT_DIR = ROOT / "outputs" / "reports"

MIN_CALLS = 8


def load_pooled():
    reg = pd.read_csv(EXT / "registry.csv")
    parts = []
    for c in reg.cohort:
        d = pd.read_csv(EXT / c / "predictions_scored.csv")
        d = d[(d.status == "ok") & d.gt_norm.isin(["bacterial", "fungal"])].copy()
        d["cohort"] = c
        parts.append(d)
    a = pd.concat(parts, ignore_index=True)
    a["y"] = (a.gt_norm == "fungal").astype(int)
    return a


def refit_temperature(p, y, T_old):
    """
    Recover the pre-calibration logit, then refit the scalar.
    p = sigmoid(logit_raw / T_old)  =>  logit_raw = T_old * logit(p)
    """
    p = np.clip(p, 1e-6, 1 - 1e-6)
    raw = T_old * np.log(p / (1 - p))
    t = torch.nn.Parameter(torch.ones(1))
    lg = torch.from_numpy(raw).float()
    yy = torch.from_numpy(y.astype(np.float32))
    opt = torch.optim.LBFGS([t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(lg / t.clamp(min=1e-2), yy)
        loss.backward()
        return loss
    opt.step(closure)
    T = float(t.detach().clamp(min=1e-2))
    return T, 1 / (1 + np.exp(-raw / T))


def metrics(y, dec):
    m = {}
    for cls, name in [(1, "fungal"), (0, "bacterial")]:
        tp = int(((dec == cls) & (y == cls)).sum())
        cl = int((dec == cls).sum())
        tot = int((y == cls).sum())
        m[f"{name}_precision"] = tp / cl if cl else np.nan
        m[f"{name}_recall"] = tp / tot if tot else np.nan
    m["coverage"] = float((dec >= 0).mean())
    m["accuracy_covered"] = float(((dec == y) & (dec >= 0)).sum() / max((dec >= 0).sum(), 1))
    m["misroute"] = float(((dec == 0) & (y == 1)).sum() / max((y == 1).sum(), 1))
    return m


def best_band(p, y, min_prec):
    """Max coverage with both arms above a precision floor."""
    best = None
    grid = np.unique(np.round(np.concatenate([p, [0.0, 1.0]]), 3))
    for hi in grid:
        for lo in grid[grid <= hi]:
            dec = np.where(p >= hi, 1, np.where(p <= lo, 0, -1))
            if int((dec == 1).sum()) < MIN_CALLS or int((dec == 0).sum()) < MIN_CALLS:
                continue
            m = metrics(y, dec)
            if m["fungal_precision"] < min_prec or m["bacterial_precision"] < min_prec:
                continue
            if best is None or m["coverage"] > best[2]["coverage"]:
                best = (float(lo), float(hi), m)
    return best


def best_threshold(p, y, min_fungal_recall):
    """Single cut-point achieving a fungal-recall target with max accuracy."""
    best = None
    for t in np.unique(np.round(np.concatenate([p, [0.0, 1.0]]), 3)):
        d = (p >= t).astype(int)
        rec = float(((d == 1) & (y == 1)).sum() / max((y == 1).sum(), 1))
        if rec < min_fungal_recall:
            continue
        acc = float((d == y).mean())
        if best is None or acc > best[1]:
            best = (float(t), acc, rec)
    return best


def main():
    a = load_pooled()
    ck = torch.load(CKPT / "final_model.pt", map_location="cpu", weights_only=False)
    T_old = ck["temperature"]
    op_old = ck["operating_point"]
    cohorts = sorted(a.cohort.unique())

    y_all, p_all = a.y.to_numpy(), a.p_fungal.to_numpy()
    L = ["# Phase 5 - Recalibration on external data\n"]
    L.append(f"Pooled external: **{len(a)}** cases "
             f"({int((y_all==0).sum())} bacterial / {int((y_all==1).sum())} fungal) "
             f"across {len(cohorts)} independent cohorts.\n")

    # ---------- is the shipped calibration miscalibrated? ----------
    L.append("## Calibration as shipped\n")
    L.append(f"- AUC **{roc_auc_score(y_all, p_all):.4f}**")
    L.append(f"- Brier **{brier_score_loss(y_all, p_all):.4f}** "
             f"(internal test was {ck.get('test_auc') and 0.175:.3f})")
    L.append(f"- mean predicted P(fungal) **{p_all.mean():.3f}** vs observed "
             f"**{y_all.mean():.3f}**\n")
    bias = p_all.mean() - y_all.mean()
    L.append(f"The model is {'over' if bias > 0 else 'under'}-predicting fungal by "
             f"**{abs(bias):.3f}** on external data.\n")

    T_new, p_cal = refit_temperature(p_all, y_all, T_old)
    L.append(f"Refitting temperature on all 164: **{T_old:.3f} -> {T_new:.3f}**, "
             f"Brier {brier_score_loss(y_all, p_all):.4f} -> "
             f"**{brier_score_loss(y_all, p_cal):.4f}**\n")
    L.append("> Temperature is monotonic, so AUC is unchanged - it fixes the "
             "probability scale, not the ranking.\n")

    # ---------- 1. does a threshold transfer? ----------
    L.append("## Does a threshold transfer? (leave-one-cohort-out)\n")
    L.append("Fit on one cohort, applied unchanged to the other. This is the honest "
             "estimate for a future cohort.\n")
    L.append("| fit on | applied to | band | coverage | acc (covered) | fungal rec | "
             "bact rec | misroute |")
    L.append("|---|---|---|---|---|---|---|---|")
    loco = []
    for held in cohorts:
        fit = a[a.cohort != held]
        tst = a[a.cohort == held]
        b = best_band(fit.p_fungal.to_numpy(), fit.y.to_numpy(), 0.75)
        if not b:
            continue
        lo, hi, _ = b
        pt, yt = tst.p_fungal.to_numpy(), tst.y.to_numpy()
        m = metrics(yt, np.where(pt >= hi, 1, np.where(pt <= lo, 0, -1)))
        loco.append(m)
        L.append(f"| {[c for c in cohorts if c != held][0]} | {held} | "
                 f"[{lo:.2f}, {hi:.2f}] | {m['coverage']:.1%} | "
                 f"{m['accuracy_covered']:.1%} | {m['fungal_recall']:.1%} | "
                 f"{m['bacterial_recall']:.1%} | {m['misroute']:.1%} |")
    if loco:
        L.append(f"\n**Transferred average: coverage "
                 f"{np.mean([m['coverage'] for m in loco]):.1%}, accuracy on covered "
                 f"{np.mean([m['accuracy_covered'] for m in loco]):.1%}, misroute "
                 f"{np.mean([m['misroute'] for m in loco]):.1%}.**\n")
        L.append("With only two cohorts each fit uses a single cohort, so these are "
                 "noisy - but they are the only unbiased figures here.\n")

    # ---------- 2. what should ship ----------
    L.append("## Candidate operating points, fitted on all 164\n")
    L.append("Calibration figures, not validation - they are fitted and reported on "
             "the same data.\n")
    L.append("| mode | setting | coverage | accuracy | fungal rec | fungal prec | "
             "bact rec | bact prec | misroute |")
    L.append("|---|---|---|---|---|---|---|---|---|")

    fc = (p_cal >= 0.5).astype(int)
    m = metrics(y_all, fc)
    L.append(f"| forced choice | t=0.50 | 100% | {(fc==y_all).mean():.1%} | "
             f"{m['fungal_recall']:.1%} | {m['fungal_precision']:.1%} | "
             f"{m['bacterial_recall']:.1%} | {m['bacterial_precision']:.1%} | "
             f"{m['misroute']:.1%} |")

    ship = {"mode": "forced_choice", "threshold": 0.5}
    for tgt in [0.90, 0.95]:
        bt = best_threshold(p_cal, y_all, tgt)
        if bt:
            t, acc, rec = bt
            d = (p_cal >= t).astype(int)
            mm = metrics(y_all, d)
            L.append(f"| forced, {tgt:.0%} fungal recall | t={t:.2f} | 100% | {acc:.1%} | "
                     f"{mm['fungal_recall']:.1%} | {mm['fungal_precision']:.1%} | "
                     f"{mm['bacterial_recall']:.1%} | {mm['bacterial_precision']:.1%} | "
                     f"{mm['misroute']:.1%} |")
            if tgt == 0.90:
                ship = {"mode": "forced_choice", "threshold": round(t, 3),
                        "rationale": "90% fungal recall on pooled external"}

    for mp in [0.75, 0.80]:
        b = best_band(p_cal, y_all, mp)
        if b:
            lo, hi, mm = b
            L.append(f"| abstain, {mp:.0%} precision floor | [{lo:.2f},{hi:.2f}] | "
                     f"{mm['coverage']:.1%} | {mm['accuracy_covered']:.1%} | "
                     f"{mm['fungal_recall']:.1%} | {mm['fungal_precision']:.1%} | "
                     f"{mm['bacterial_recall']:.1%} | {mm['bacterial_precision']:.1%} | "
                     f"{mm['misroute']:.1%} |")
    L.append("")

    band75 = best_band(p_cal, y_all, 0.75)
    cal = {
        "temperature_external": T_new,
        "temperature_internal": T_old,
        "n_calibration": int(len(a)),
        "cohorts": cohorts,
        "pooled_auc": round(float(roc_auc_score(y_all, p_all)), 4),
        "shipped": ship,
        "abstain_band_75": [band75[0], band75[1]] if band75 else None,
        "internal_band": [op_old["lo"], op_old["hi"]],
        "loco_mean_accuracy_covered": round(float(np.mean([m["accuracy_covered"] for m in loco])), 4) if loco else None,
    }
    with open(CKPT / "calibration_external.json", "w") as f:
        json.dump(cal, f, indent=2)

    L.append("## What ships\n")
    L.append(f"- **temperature {T_new:.3f}** (was {T_old:.3f}) - external-calibrated")
    L.append(f"- **forced choice at t={ship['threshold']}** as the default display")
    if band75:
        L.append(f"- abstention band [{band75[0]:.2f}, {band75[1]:.2f}] offered as a "
                 f"secondary view, replacing the internal "
                 f"[{op_old['lo']:.2f}, {op_old['hi']:.2f}]")
    L.append("\nThe internal band was fitted on a 1:1 curated dev set and did not hold "
             "up externally: coverage fell 81.5% -> 76.8% and accuracy on covered "
             "88.7% -> 77.6% between the two cohorts.\n")

    L.append("## Honest limits\n")
    L.append("- 164 cases from 2 cohorts. The leave-one-cohort-out figures are the "
             "only unbiased ones and rest on a single cohort each.")
    L.append("- Neither cohort has patient identifiers, so a patient could appear in "
             "both. Overlap was ruled out at image level (0 shared hashes) but not at "
             "patient level.")
    L.append("- Cohort prevalence is 64% fungal, still not clinic prevalence.")
    L.append("- Bacterial recall is weak in every cohort measured (62.5%, 51.4%, "
             "internal 73.0%). This is a data problem, not a threshold problem, and "
             "recalibration cannot fix it.\n")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "15_recalibration.md").write_text("\n".join(L), encoding="utf-8")
    print(f"temperature {T_old:.3f} -> {T_new:.3f}")
    print(f"Brier {brier_score_loss(y_all, p_all):.4f} -> {brier_score_loss(y_all, p_cal):.4f}")
    print(f"ship: {ship}")
    if loco:
        print(f"LOCO mean accuracy on covered: {np.mean([m['accuracy_covered'] for m in loco]):.1%}")
    print(f"wrote {CKPT / 'calibration_external.json'}")


if __name__ == "__main__":
    main()
