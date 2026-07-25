"""
Experiment 1 - Does the hypopyon carry signal, and does adding it help?

The hypopyon (pus level in the anterior chamber) is clinically discriminative:
thick and immobile in fungal disease, thin and mobile in bacterial. It settles
inferiorly. The model so far tiles the whole limbus and has never been given the
hypopyon region explicitly.

Two tests, both on the winning 3.67 mm config:

  1a  Regional ablation (free - reuses existing embeddings).
      Split the corneal tiles by angle into the inferior third (where the
      hypopyon sits) vs the superior third vs the whole cornea. If the inferior
      region carries more signal, the hypopyon / inferior infiltrate is
      contributing.

  1b  Add the hypopyon band (needs new tiles).
      Extend sampling into the inferior anterior chamber - a band below the
      limbus, where a hypopyon shows below the corneal circle - and compare the
      full-limbus bag against full-limbus + hypopyon-band.

Outputs
    outputs/reports/16_experiment_hypopyon.md
    data/processed/tile_embeddings_hypoband.npy
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
REPORT = ROOT / "outputs" / "reports" / "16_experiment_hypopyon.md"

CROP = 896            # 3.67 mm - the winning scale
STRIDE = 224          # denser, to recover the thin inferior meniscus
# The hypopyon sits at the BOTTOM of the cornea, inside the limbus. Visual QC
# showed a below-limbus band lands on eyelid/lashes, not hypopyon. The tiles
# that actually hold the hypopyon meniscus are inferior and STRADDLE the lower
# limbus, so they were dropped by the baseline's >=50%-cornea rule. This band
# recovers exactly those: inferior, cornea fraction in [0.20, 0.50].
BAND_FRAC_LO = 0.20
BAND_FRAC_HI = 0.50
BAND_THETA = (55, 125)   # inferior sector (y grows downward)
MAX_BAND = 8


def region_bags(ms, emb, mask_fn):
    sub = ms[mask_fn(ms)].copy()
    # re-pack embeddings contiguously so build_bags indices are local
    sub = sub.reset_index(drop=True)
    local = emb[sub.emb_row.to_numpy()]
    sub["emb_row"] = np.arange(len(sub))
    return C.build_bags(sub, local)


def plan_hypopyon_band(man):
    """
    Inferior tiles that straddle the lower limbus (cornea fraction 0.20-0.50) -
    the hypopyon meniscus the baseline's >=50%-cornea rule dropped. These sit ON
    the cornea's lower edge, not below it on the lid.
    """
    rows = []
    for r in man.itertuples():
        lb = C.load_limbus(r.image_id)
        if lb is None:
            continue
        contour, (H, W) = lb
        mask, cx, cy, rad = C.limbus_mask_and_centre(contour, H, W)
        ii = cv2.integral(mask)
        xs, ys = contour[:, 0], contour[:, 1]
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(cy), min(H - CROP, int(ys.max()) + CROP // 2)   # lower half only
        cand = []
        for y in range(max(0, y0), max(1, y1 + 1), STRIDE):
            for x in range(x0, max(x0 + 1, x1 - CROP + 2), STRIDE):
                if y + CROP > H or x + CROP > W:
                    continue
                s = ii[y + CROP, x + CROP] - ii[y, x + CROP] - ii[y + CROP, x] + ii[y, x]
                frac = s / (CROP * CROP)
                if not (BAND_FRAC_LO <= frac <= BAND_FRAC_HI):
                    continue
                tcx, tcy = x + CROP / 2, y + CROP / 2
                theta = np.degrees(np.arctan2(tcy - cy, tcx - cx)) % 360
                if not (BAND_THETA[0] <= theta <= BAND_THETA[1]):
                    continue
                cand.append((frac, x, y))
        cand.sort(reverse=True)                    # most-cornea straddlers first
        for frac, x, y in cand[:MAX_BAND]:
            rows.append({"image_id": r.image_id, "label": r.label,
                         "patient_key": r.patient_key, "x": x, "y": y, "crop_px": CROP})
    return pd.DataFrame(rows)


def main():
    ms = pd.read_csv(PROC / "tile_index_ms.csv")
    ms = ms[ms.scale == "s896"].copy()
    emb = np.load(PROC / "tile_embeddings_ms.npy").astype(np.float32)
    man = pd.read_csv(C.MANIFEST)

    L = ["# Experiment 1 - Hypopyon\n"]
    L.append(f"Winning 3.67 mm config, {len(ms):,} corneal tiles / "
             f"{ms.image_id.nunique()} images. Repeated patient-grouped 5-fold CV.\n")

    # ---------- 1a regional ablation (free) ----------
    L.append("## 1a. Which region carries the signal? (free ablation)\n")
    regions = {
        "whole cornea (baseline)": lambda d: d.index == d.index,
        "inferior third (hypopyon zone)": lambda d: (d.theta > 50) & (d.theta < 130),
        "superior third": lambda d: (d.theta > 230) & (d.theta < 310),
        "nasal+temporal sides": lambda d: (d.theta <= 50) | (d.theta >= 310) | ((d.theta >= 130) & (d.theta <= 230)),
    }
    rows = []
    for name, fn in regions.items():
        X, M, meta = region_bags(ms, emb, fn)
        n_tiles = int(fn(ms).sum())
        mean, sd = C.eval_cv(X, M, meta)
        rows.append({"region": name, "tiles": n_tiles,
                     "img": int(meta.shape[0]), "AUC": round(mean, 4), "sd": round(sd, 4)})
        print(f"  {name:34s} n={n_tiles:5d}  AUC={mean:.4f} +/-{sd:.4f}")
    reg = pd.DataFrame(rows)
    L.append(reg.to_markdown(index=False))
    inf = reg.loc[reg.region.str.startswith("inferior"), "AUC"].iloc[0]
    sup = reg.loc[reg.region.str.startswith("superior"), "AUC"].iloc[0]
    base = reg.loc[reg.region.str.startswith("whole"), "AUC"].iloc[0]
    L.append(f"\n- inferior (hypopyon zone) **{inf:.3f}** vs superior **{sup:.3f}** "
             f"vs whole cornea **{base:.3f}**")
    if inf > sup + 0.02:
        L.append("- The inferior cornea carries **more** signal than the superior — consistent "
                 "with the hypopyon and inferior infiltrate being discriminative.")
    elif abs(inf - sup) <= 0.02:
        L.append("- Inferior and superior carry **similar** signal — the discriminative "
                 "texture is spread around the cornea, not concentrated at the hypopyon.")
    else:
        L.append("- The superior cornea carries more — the hypopyon zone is not the driver.")
    L.append("")

    # ---------- 1b add hypopyon band ----------
    L.append("## 1b. Does adding the hypopyon band help?\n")
    band_cache = PROC / "tile_embeddings_hypoband.npy"
    band_idx_cache = PROC / "tile_index_hypoband.csv"
    if band_cache.exists() and band_idx_cache.exists():
        band = pd.read_csv(band_idx_cache)
        band_emb = np.load(band_cache).astype(np.float32)
        print("  [band] cached")
    else:
        band = plan_hypopyon_band(man)
        band_emb = C.embed_tiles(band, desc="hypopyon band").astype(np.float32)
        band.to_csv(band_idx_cache, index=False)
        np.save(band_cache, band_emb)
    print(f"  band tiles: {len(band)}  (mean {band.groupby('image_id').size().mean():.1f}/img)")

    # baseline = full cornea
    base_sub = ms.reset_index(drop=True).copy()
    base_local = emb[base_sub.emb_row.to_numpy()]
    base_sub["emb_row"] = np.arange(len(base_sub))
    Xb, Mb, meta_b = C.build_bags(base_sub, base_local)
    base_mean, base_sd = C.eval_cv(Xb, Mb, meta_b)

    # cornea + band: concat embeddings per image
    combined = pd.concat([
        base_sub[["image_id", "label", "patient_key"]].assign(src="cornea",
            _e=list(base_local)),
        band[["image_id", "label", "patient_key"]].assign(src="band",
            _e=list(band_emb)),
    ], ignore_index=True)
    combined = combined[combined.image_id.isin(base_sub.image_id)]     # same image set
    comb_emb = np.stack(combined["_e"].to_numpy())
    combined = combined.drop(columns="_e").reset_index(drop=True)
    combined["emb_row"] = np.arange(len(combined))
    Xc, Mc, meta_c = C.build_bags(combined, comb_emb)
    comb_mean, comb_sd = C.eval_cv(Xc, Mc, meta_c)

    L.append(f"| bag | tiles/img | AUC |\n|---|---|---|")
    L.append(f"| cornea only (baseline) | {Mb.sum(1).mean():.1f} | {base_mean:.4f} ± {base_sd:.4f} |")
    L.append(f"| cornea + hypopyon band | {Mc.sum(1).mean():.1f} | {comb_mean:.4f} ± {comb_sd:.4f} |")
    d = comb_mean - base_mean
    L.append(f"\n- change: **{d:+.4f}**")
    if d > 0.012:
        L.append("- Adding the inferior anterior-chamber band **helps** — the hypopyon region "
                 "holds signal the corneal tiles were missing.")
    elif d < -0.012:
        L.append("- Adding the band **hurts** — the below-limbus region is mostly iris/sclera "
                 "and dilutes the bag. The hypopyon that matters is already seen through the "
                 "inferior cornea.")
    else:
        L.append("- **No material change.** Whatever hypopyon signal exists is already captured "
                 "by the inferior corneal tiles; a dedicated band adds nothing here.")
    L.append("")

    L.append("## Reading\n")
    L.append("The model tiles the cornea; the hypopyon is seen through the inferior cornea, so "
             "it is already partly in view. These tests measure whether it is a *distinct* "
             "lever. Note the caveat: there is no hypopyon segmenter for this cohort, so 1b "
             "adds a geometric band, not a detected hypopyon — a true hypopyon-mask test needs "
             "the 7-class segmentation that lives on Azure.\n")

    REPORT.write_text("\n".join(L), encoding="utf-8")
    print(f"\nbaseline {base_mean:.4f} | +band {comb_mean:.4f} | inferior {inf:.4f} sup {sup:.4f}")
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
