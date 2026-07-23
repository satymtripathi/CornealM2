"""
Phase 2b - Native-resolution tiling inside the limbus, and per-tile embeddings.

The central claim under test:

    Whole-eye pooled representations plateau at AUC ~0.745 (Phase 1c). Every
    route to that plateau - more resolution, better framing, both - lands in the
    same place, because global pooling averages fine texture away. If the
    discriminative signal really is localised margin texture, then per-tile
    embeddings with attention must clear 0.747. If they do not, the entire
    localisation premise is wrong.

Design decisions and why:

  tile size 224 px, taken at NATIVE resolution and fed at 224
      1:1. No resampling at any point. At 4.09 um/px a tile spans 0.92 mm and a
      50-200 um fungal filament is 12-49 px inside it - fully resolved. The
      incumbent's tiles were cropped from a 1024 canvas, i.e. 5.3x downsampled,
      leaving the same filament at 2-9 px.

  grid over the limbus, not the frame
      Phase 2a validated the limbus to 1.5% of a perfect ellipse, so we can
      restrict to cornea and discard lid, lashes and sclera outright.

  no lesion detector
      There are no lesion masks for this cohort and none can be validated, so a
      lesion segmenter would be an unverifiable component in the middle of the
      pipeline. Instead the bag covers the whole cornea densely and attention
      is left to find the lesion. If attention concentrates on the infiltrate
      margin, that IS the localisation - learned, and inspectable.

  nothing about the label touches tiling
      This is the specific defect that invalidated the incumbent, whose bag
      builder added an extra lesion tile when the label was bacterial and an
      extra hypopyon tile when it was fungal, in train AND validation.

Outputs
    data/processed/tile_index.csv        one row per tile, with polar position
    data/processed/tile_embeddings.npy   float16 (n_tiles, 384), row-aligned
    outputs/reports/06_tiling.md
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import timm
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "outputs" / "manifests" / "manifest.csv"
LIMBUS_DIR = ROOT / "data" / "interim" / "limbus"
OUT_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "outputs" / "reports"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "vit_small_patch14_dinov2.lvd142m"

TILE = 224                 # native px, fed at 224 -> 1:1, no resampling
STRIDE = 224               # non-overlapping first pass
MIN_LIMBUS_FRAC = 0.50     # tile must be at least half cornea
MAX_GLARE_FRAC = 0.60      # drop tiles that are mostly specular reflection
MAX_TILES = 96             # bound per-image bag size and compute
BATCH = 48

IMNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMNET_STD = np.array([0.229, 0.224, 0.225], np.float32)


# =====================================================
# TILE PLANNING
# =====================================================
def plan_tiles(limbus_mask: np.ndarray, cx: float, cy: float, radius: float):
    """
    Regular grid over the limbus bounding box, keeping tiles that are mostly
    cornea. Returns (x, y, limbus_frac, r_norm, theta) per tile.

    r_norm is distance from the limbus centre in units of limbus radius, so
    0 = centre of cornea, 1 = limbal edge. That makes "rim" addressable later
    without ever having segmented a lesion.
    """
    ys, xs = np.where(limbus_mask > 0)
    if len(xs) == 0:
        return []
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())

    # integral image -> exact tile coverage in O(1) per tile
    ii = cv2.integral(limbus_mask.astype(np.uint8))
    H, W = limbus_mask.shape
    area = float(TILE * TILE)

    out = []
    for y in range(y0, max(y0 + 1, y1 - TILE + 2), STRIDE):
        for x in range(x0, max(x0 + 1, x1 - TILE + 2), STRIDE):
            if y + TILE > H or x + TILE > W:
                continue
            s = (ii[y + TILE, x + TILE] - ii[y, x + TILE]
                 - ii[y + TILE, x] + ii[y, x])
            frac = s / area
            if frac < MIN_LIMBUS_FRAC:
                continue
            tcx, tcy = x + TILE / 2.0, y + TILE / 2.0
            dx, dy = tcx - cx, tcy - cy
            out.append({
                "x": x, "y": y,
                "limbus_frac": float(frac),
                "r_norm": float(np.hypot(dx, dy) / max(radius, 1e-6)),
                "theta": float(np.degrees(np.arctan2(dy, dx)) % 360.0),
            })
    return out


def glare_fraction(tile_rgb: np.ndarray) -> float:
    hsv = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2HSV)
    S = hsv[..., 1].astype(np.float32) / 255.0
    V = hsv[..., 2].astype(np.float32) / 255.0
    return float(((V > 0.94) & (S < 0.15)).mean())


def load_limbus(image_id: str):
    p = LIMBUS_DIR / f"{image_id}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return z["limbus_contour"], z["native_hw"]


# =====================================================
# MAIN
# =====================================================
def main():
    df = pd.read_csv(MANIFEST)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model = timm.create_model(MODEL, pretrained=True, num_classes=0, img_size=TILE)
    model.eval().to(DEVICE)
    print(f"device={DEVICE}  tile={TILE}px native (1:1)  stride={STRIDE}")

    index_rows, emb_chunks = [], []
    row = 0
    skipped = 0

    for r in tqdm(list(df.itertuples()), desc="tiling"):
        lb = load_limbus(r.image_id)
        if lb is None:
            skipped += 1
            continue
        contour, (H0, W0) = lb[0], lb[1]

        mask = np.zeros((int(H0), int(W0)), np.uint8)
        cv2.fillPoly(mask, [contour.astype(np.int32)], 1)

        M = cv2.moments(contour.astype(np.int32))
        if M["m00"] <= 0:
            skipped += 1
            continue
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        radius = float(np.sqrt(M["m00"] / np.pi))     # equivalent-circle radius

        plan = plan_tiles(mask, cx, cy, radius)
        if not plan:
            skipped += 1
            continue

        # deterministic, label-free selection: most-cornea tiles first
        plan.sort(key=lambda t: -t["limbus_frac"])
        plan = plan[:MAX_TILES]

        with Image.open(ROOT / r.rel_path) as im:
            rgb = np.asarray(im.convert("RGB"))       # full decode - native res

        kept, batch = [], []
        for t in plan:
            crop = rgb[t["y"]:t["y"] + TILE, t["x"]:t["x"] + TILE]
            if crop.shape[:2] != (TILE, TILE):
                continue
            g = glare_fraction(crop)
            if g > MAX_GLARE_FRAC:
                continue
            t["glare_frac"] = g
            t["mean_L"] = float(cv2.cvtColor(crop, cv2.COLOR_RGB2LAB)[..., 0].mean())
            kept.append(t)
            x = crop.astype(np.float32) / 255.0
            batch.append(((x - IMNET_MEAN) / IMNET_STD).transpose(2, 0, 1))

        if not kept:
            skipped += 1
            continue

        feats = []
        with torch.no_grad():
            for i in range(0, len(batch), BATCH):
                t_in = torch.from_numpy(np.stack(batch[i:i + BATCH])).to(DEVICE)
                with torch.autocast("cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):
                    f = model(t_in)
                feats.append(f.float().cpu().numpy())
        feats = np.concatenate(feats, 0).astype(np.float16)
        emb_chunks.append(feats)

        for k, t in enumerate(kept):
            index_rows.append({
                "image_id": r.image_id, "label": r.label, "class_name": r.class_name,
                "patient_key": r.patient_key, "split": r.split, "fold": r.fold,
                "tile_i": k, "emb_row": row + k,
                "x": t["x"], "y": t["y"], "tile_px": TILE,
                "limbus_frac": round(t["limbus_frac"], 4),
                "glare_frac": round(t["glare_frac"], 4),
                "r_norm": round(t["r_norm"], 4),
                "theta": round(t["theta"], 2),
                "mean_L": round(t["mean_L"], 2),
            })
        row += len(kept)

    idx = pd.DataFrame(index_rows)
    emb = np.concatenate(emb_chunks, 0)
    assert len(idx) == len(emb), f"index/embedding mismatch {len(idx)} vs {len(emb)}"

    idx.to_csv(OUT_DIR / "tile_index.csv", index=False)
    np.save(OUT_DIR / "tile_embeddings.npy", emb)

    # ---------------- report ----------------
    per_img = idx.groupby("image_id").size()
    L = ["# Phase 2b - Native-Resolution Tiling\n"]
    L.append(f"- images tiled: **{idx.image_id.nunique()}** (skipped {skipped})")
    L.append(f"- total tiles: **{len(idx):,}**")
    L.append(f"- tile: **{TILE}px at native resolution, fed at {TILE}** - no resampling")
    L.append(f"- at 4.09 um/px a tile spans **{TILE * 0.00409:.2f} mm**\n")

    L.append("## Bag sizes\n")
    L.append(f"| | tiles per image |\n|---|---|")
    for q in ["min", "p25", "median", "p75", "max"]:
        v = {"min": per_img.min(), "p25": per_img.quantile(.25),
             "median": per_img.median(), "p75": per_img.quantile(.75),
             "max": per_img.max()}[q]
        L.append(f"| {q} | {v:.0f} |")
    L.append("")

    L.append("## Bag size by class\n")
    bs = idx.groupby(["class_name", "image_id"]).size().groupby("class_name").agg(["mean", "median"])
    L.append(bs.round(2).to_markdown())
    L.append("\nBag size is driven by limbus area and framing, not by pathology, so these "
             "should be close. A large gap would mean bag size itself leaks the label.\n")

    L.append("## Tile composition\n")
    L.append(f"- median limbus coverage per tile: **{idx.limbus_frac.median():.3f}**")
    L.append(f"- median glare fraction: **{idx.glare_frac.median():.4f}**")
    L.append(f"- tiles dropped for glare > {MAX_GLARE_FRAC:.0%}: applied at extraction")
    L.append(f"- radial spread (`r_norm`, 0 = corneal centre, 1 = limbal edge): "
             f"p25 {idx.r_norm.quantile(.25):.2f}, median {idx.r_norm.median():.2f}, "
             f"p75 {idx.r_norm.quantile(.75):.2f}\n")

    L.append("## Storage\n")
    L.append(f"- `tile_embeddings.npy` float16 {emb.shape} = "
             f"{emb.nbytes / 1e6:.0f} MB")
    L.append("- `tile_index.csv` carries position, polar coordinates and quality per tile, "
             "so attention can be mapped back onto the cornea for inspection.\n")

    L.append("## Label hygiene\n")
    L.append("Tile planning uses only the limbus mask, tile geometry and glare. **No "
             "label, and no label-derived quantity, enters bag construction** - the "
             "defect that invalidated the incumbent's reported numbers.\n")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "06_tiling.md").write_text("\n".join(L), encoding="utf-8")

    print(f"\ntiles: {len(idx):,}  |  images: {idx.image_id.nunique()}  |  skipped: {skipped}")
    print(f"bag size median {per_img.median():.0f}  (min {per_img.min()}, max {per_img.max()})")
    print(f"embeddings {emb.shape} = {emb.nbytes/1e6:.0f} MB")
    print(f"wrote {REPORT_DIR / '06_tiling.md'}")


if __name__ == "__main__":
    main()
