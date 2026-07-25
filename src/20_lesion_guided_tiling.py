"""
Experiment - Does lesion-guided tiling beat whole-cornea tiling?

Everything so far tiles the whole cornea because we had no lesion mask. Now we
do (infiltrate Dice 0.88, hypopyon 0.86). This tests whether focusing the bag on
the disease - the infiltrate, its margin, and the hypopyon - beats sampling the
whole cornea, at the same 3.67 mm scale and same MIL recipe.

Tilings compared (all 896 px native = 3.67 mm, fed at 448, mean-pooled MIL):

    whole_cornea   the current baseline (0.791)               [reuse]
    lesion         tiles overlapping infiltrate u hypopyon
    lesion_rim     lesion + a dilated margin band (the feathery-vs-compact edge)
    rim_only       only the margin band around the infiltrate

If a lesion-focused bag matches or beats 0.791 with far fewer tiles, the disease
region is what carries the signal - and the model becomes both better and more
interpretable. Images where no infiltrate is found fall back to whole-cornea so
no case is dropped.

Outputs
    data/processed/tile_embeddings_lesionguided.npy (+ index)
    outputs/reports/20_lesion_guided_tiling.md
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import _exp_common as C

ROOT = C.ROOT
PROC = ROOT / "data" / "processed"
MASKS = ROOT / "data" / "interim" / "lesion_masks"
LESION_CKPT = ROOT / "models" / "lesion_seg" / "lesion_unetpp.pth"
REPORT = ROOT / "outputs" / "reports" / "20_lesion_guided_tiling.md"

CROP = 896
STRIDE = 448
MIN_LIMBUS_FRAC = 0.50
MAX_GLARE_FRAC = 0.60
MAX_TILES = 24
RIM_MM = 1.5           # margin band width
MM_PER_PX_FALLBACK = 0.00409

import torch  # noqa
_CLASSES = torch.load(LESION_CKPT, map_location="cpu", weights_only=False)["config"]["classes"]
CID = {c: i for i, c in enumerate(_CLASSES)}


def load_masks(image_id, H, W):
    p = MASKS / f"{image_id}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    lab = cv2.resize(z["label512"], (W, H), interpolation=cv2.INTER_NEAREST)
    return {c: (lab == CID[c]).astype(np.uint8) for c in ("infiltrate", "hypopyon", "cellularity")}


def load_limbus_mask(image_id, H, W):
    lb = C.load_limbus(image_id)
    if lb is None:
        return None
    contour, _ = lb
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [contour.astype(np.int32)], 1)
    return m


def tiles_over(region, glare, limbus, H, W, need_frac=0.10):
    """896-px tiles whose overlap with `region` >= need_frac, inside limbus, low glare."""
    if region.sum() == 0:
        return []
    ii = cv2.integral(region)
    gg = cv2.integral(glare) if glare is not None else None
    lg = cv2.integral(limbus) if limbus is not None else None
    ys, xs = np.where(region > 0)
    y0, y1 = max(0, ys.min() - CROP), min(H - CROP, ys.max())
    x0, x1 = max(0, xs.min() - CROP), min(W - CROP, xs.max())
    out = []
    for y in range(y0, max(y0 + 1, y1 + 1), STRIDE):
        for x in range(x0, max(x0 + 1, x1 + 1), STRIDE):
            if y + CROP > H or x + CROP > W:
                continue
            s = ii[y+CROP, x+CROP] - ii[y, x+CROP] - ii[y+CROP, x] + ii[y, x]
            if s / (CROP*CROP) < need_frac:
                continue
            if lg is not None:
                lf = (lg[y+CROP, x+CROP] - lg[y, x+CROP] - lg[y+CROP, x] + lg[y, x]) / (CROP*CROP)
                if lf < MIN_LIMBUS_FRAC:
                    continue
            if gg is not None:
                gf = (gg[y+CROP, x+CROP] - gg[y, x+CROP] - gg[y+CROP, x] + gg[y, x]) / (CROP*CROP)
                if gf > MAX_GLARE_FRAC:
                    continue
            out.append((x, y, float(s/(CROP*CROP))))
    out.sort(key=lambda t: -t[2])
    return [(x, y) for x, y, _ in out[:MAX_TILES]]


def build_plan(df, variant):
    """Return a tile plan DataFrame for a variant, with whole-cornea fallback."""
    rows, fell_back = [], 0
    for r in df.itertuples():
        lb = C.load_limbus(r.image_id)
        if lb is None:
            continue
        contour, (H, W) = lb
        limbus = np.zeros((H, W), np.uint8); cv2.fillPoly(limbus, [contour.astype(np.int32)], 1)
        m = load_masks(r.image_id, H, W)
        rim_px = int(RIM_MM / MM_PER_PX_FALLBACK)

        picks = []
        if m is not None and variant != "whole_cornea":
            infl = m["infiltrate"]; hypo = m["hypopyon"]
            lesion = ((infl | hypo) > 0).astype(np.uint8)
            if lesion.sum() > 0:
                if variant == "lesion":
                    picks = tiles_over(lesion, None, limbus, H, W)
                elif variant == "lesion_rim":
                    dil = cv2.dilate(lesion, np.ones((rim_px, rim_px), np.uint8))
                    picks = tiles_over(dil, None, limbus, H, W)
                elif variant == "rim_only":
                    dil = cv2.dilate(infl, np.ones((rim_px, rim_px), np.uint8))
                    rim = ((dil > 0) & (infl == 0)).astype(np.uint8)
                    picks = tiles_over(rim if rim.sum() > 0 else lesion, None, limbus, H, W)

        if not picks:                                   # fallback: whole cornea
            fell_back += 1 if variant != "whole_cornea" else 0
            ii = cv2.integral(limbus)
            ys, xs = np.where(limbus > 0)
            cand = []
            for y in range(int(ys.min()), int(ys.max()-CROP+2), STRIDE):
                for x in range(int(xs.min()), int(xs.max()-CROP+2), STRIDE):
                    if y+CROP > H or x+CROP > W:
                        continue
                    s = ii[y+CROP, x+CROP]-ii[y, x+CROP]-ii[y+CROP, x]+ii[y, x]
                    f = s/(CROP*CROP)
                    if f >= MIN_LIMBUS_FRAC:
                        cand.append((x, y, f))
            cand.sort(key=lambda t: -t[2])
            picks = [(x, y) for x, y, _ in cand[:MAX_TILES]]

        for x, y in picks:
            rows.append({"image_id": r.image_id, "label": r.label,
                         "patient_key": r.patient_key, "x": x, "y": y, "crop_px": CROP})
    return pd.DataFrame(rows), fell_back


def main():
    df = pd.read_csv(C.MANIFEST)
    variants = ["lesion", "lesion_rim", "rim_only"]

    L = ["# Experiment - Lesion-guided tiling\n"]
    L.append("All 896 px (3.67 mm) tiles, fed 448, mean-pooled MIL, repeated "
             "patient-grouped CV. Baseline = whole-cornea tiling.\n")
    L.append("| tiling | tiles/img | images | fallback | AUC |\n|---|---|---|---|---|")

    # whole-cornea baseline (reuse existing s896 embeddings)
    ms = pd.read_csv(PROC / "tile_index_ms.csv"); ms = ms[ms.scale == "s896"].reset_index(drop=True)
    base_emb = np.load(PROC / "tile_embeddings_ms.npy").astype(np.float32)[ms.emb_row.to_numpy()]
    ms["emb_row"] = np.arange(len(ms))
    Xb, Mb, meta_b = C.build_bags(ms, base_emb)
    base_mean, base_sd = C.eval_cv(Xb, Mb, meta_b)
    L.append(f"| whole cornea (baseline) | {Mb.sum(1).mean():.1f} | {meta_b.shape[0]} | - | "
             f"**{base_mean:.4f} ± {base_sd:.4f}** |")
    print(f"whole_cornea {base_mean:.4f}")

    results = {"whole_cornea": (base_mean, base_sd)}
    for v in variants:
        plan, fb = build_plan(df, v)
        cache = PROC / f"tile_emb_{v}.npy"
        if cache.exists() and len(np.load(cache)) == len(plan):
            emb = np.load(cache).astype(np.float32)
        else:
            emb = C.embed_tiles(plan, desc=v).astype(np.float32)
            np.save(cache, emb)
        plan = plan.reset_index(drop=True); plan["emb_row"] = np.arange(len(plan))
        X, M, meta = C.build_bags(plan, emb)
        mean, sd = C.eval_cv(X, M, meta)
        results[v] = (mean, sd)
        L.append(f"| {v} | {M.sum(1).mean():.1f} | {meta.shape[0]} | {fb} | "
                 f"{mean:.4f} ± {sd:.4f} |")
        print(f"{v:12s} {mean:.4f} +/-{sd:.4f}  (fallback {fb}, {M.sum(1).mean():.1f} tiles/img)")

    best = max(results.items(), key=lambda kv: kv[1][0])
    L.append("")
    L.append("## Reading\n")
    d = best[1][0] - base_mean
    if best[0] == "whole_cornea" or d < 0.01:
        L.append(f"Whole-cornea tiling ({base_mean:.3f}) is not beaten by lesion focusing "
                 f"(best {best[0]} {best[1][0]:.3f}). The signal is spread across the cornea; "
                 f"restricting to the lesion loses as much as it gains. Whole-cornea stays the "
                 f"design - but lesion masks remain valuable for biomarkers and interpretability.")
    else:
        L.append(f"**{best[0]}** reaches **{best[1][0]:.4f}** vs {base_mean:.4f} whole-cornea "
                 f"(**{d:+.4f}**), often with fewer tiles. Focusing the bag on the disease and "
                 f"its margin helps - the model gets better *and* more interpretable.")
    L.append("")
    REPORT.write_text("\n".join(L), encoding="utf-8")
    print(f"\nbest: {best[0]} {best[1][0]:.4f}  vs whole {base_mean:.4f}")
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
