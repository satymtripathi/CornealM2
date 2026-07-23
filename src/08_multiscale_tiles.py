"""
Phase 2c - Multi-scale tiling. Correcting a tile-size error.

Phase 3 found that native-resolution 224px tiles gained nothing over the
whole-eye view (0.746 vs 0.747). The likely cause is that the tiles were too
SMALL, not too coarse.

The reasoning, from the annotation geometry in the CornealAI ROI analysis:

    feathery_margin   median Feret 3.44 mm, median 4 separate components

The diagnostic feature is not an individual hypha - it is the branching PATTERN
those hyphae form, and that pattern spans ~3.4 mm. Phase 2b used 224 px tiles
at 4.09 um/px = 0.92 mm, so every tile saw about a quarter of the pattern's
width and a fourteenth of its area. The structure that makes it diagnostic was
cut apart before the encoder ever saw it.

This retiles at two larger scales, chosen against that measurement:

    s448   448 px crop -> 448 input   1.83 mm   1:1, half the pattern
    s896   896 px crop -> 448 input   3.67 mm   2x down, fits the whole pattern

s896 is the one the hypothesis actually predicts should work: it is the smallest
tile that can contain a median feathery margin whole. Filaments remain 6-24 px
inside it, so they are still resolved.

Tiling stays label-free: geometry, limbus coverage and glare only.

Outputs
    data/processed/tile_index_ms.csv
    data/processed/tile_embeddings_ms.npy
    outputs/reports/08_multiscale_tiles.md
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
PROC = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "outputs" / "reports"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "vit_small_patch14_dinov2.lvd142m"
INPUT = 448                       # both scales feed at 448 -> one model
UM_PER_PX = 0.00409               # measured in Phase 2a

SCALES = [
    # name   crop_px  stride_px  max_tiles
    ("s448",     448,       448,       48),
    ("s896",     896,       448,       24),   # 50% overlap - fewer positions
]

MIN_LIMBUS_FRAC = 0.50
MAX_GLARE_FRAC = 0.60
BATCH = 12

IMNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMNET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def plan(mask, cx, cy, radius, crop, stride):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return []
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    ii = cv2.integral(mask.astype(np.uint8))
    H, W = mask.shape
    area = float(crop * crop)

    out = []
    for y in range(y0, max(y0 + 1, y1 - crop + 2), stride):
        for x in range(x0, max(x0 + 1, x1 - crop + 2), stride):
            if y + crop > H or x + crop > W:
                continue
            s = ii[y + crop, x + crop] - ii[y, x + crop] - ii[y + crop, x] + ii[y, x]
            frac = s / area
            if frac < MIN_LIMBUS_FRAC:
                continue
            dx, dy = x + crop / 2.0 - cx, y + crop / 2.0 - cy
            out.append({"x": x, "y": y, "limbus_frac": float(frac),
                        "r_norm": float(np.hypot(dx, dy) / max(radius, 1e-6)),
                        "theta": float(np.degrees(np.arctan2(dy, dx)) % 360.0)})
    return out


def glare_frac(t):
    hsv = cv2.cvtColor(t, cv2.COLOR_RGB2HSV)
    return float(((hsv[..., 2] / 255.0 > 0.94) & (hsv[..., 1] / 255.0 < 0.15)).mean())


def main():
    df = pd.read_csv(MANIFEST)
    PROC.mkdir(parents=True, exist_ok=True)

    model = timm.create_model(MODEL, pretrained=True, num_classes=0, img_size=INPUT)
    model.eval().to(DEVICE)
    print(f"device={DEVICE} input={INPUT}")
    for n, c, s, m in SCALES:
        print(f"  {n}: {c}px crop = {c*UM_PER_PX:.2f} mm, stride {s}, cap {m}")

    rows, chunks, row = [], [], 0

    for r in tqdm(list(df.itertuples()), desc="multiscale"):
        p = LIMBUS_DIR / f"{r.image_id}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        contour, (H0, W0) = z["limbus_contour"], z["native_hw"]

        mask = np.zeros((int(H0), int(W0)), np.uint8)
        cv2.fillPoly(mask, [contour.astype(np.int32)], 1)
        M = cv2.moments(contour.astype(np.int32))
        if M["m00"] <= 0:
            continue
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        radius = float(np.sqrt(M["m00"] / np.pi))

        with Image.open(ROOT / r.rel_path) as im:
            rgb = np.asarray(im.convert("RGB"))

        for sname, crop, stride, cap in SCALES:
            tiles = plan(mask, cx, cy, radius, crop, stride)
            tiles.sort(key=lambda t: -t["limbus_frac"])
            tiles = tiles[:cap]

            kept, batch = [], []
            for t in tiles:
                c = rgb[t["y"]:t["y"] + crop, t["x"]:t["x"] + crop]
                if c.shape[:2] != (crop, crop):
                    continue
                g = glare_frac(c)
                if g > MAX_GLARE_FRAC:
                    continue
                t["glare_frac"] = g
                if crop != INPUT:
                    c = cv2.resize(c, (INPUT, INPUT), interpolation=cv2.INTER_AREA)
                kept.append(t)
                x = c.astype(np.float32) / 255.0
                batch.append(((x - IMNET_MEAN) / IMNET_STD).transpose(2, 0, 1))

            if not kept:
                continue

            feats = []
            with torch.no_grad():
                for i in range(0, len(batch), BATCH):
                    tt = torch.from_numpy(np.stack(batch[i:i + BATCH])).to(DEVICE)
                    with torch.autocast("cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):
                        f = model(tt)
                    feats.append(f.float().cpu().numpy())
            feats = np.concatenate(feats, 0).astype(np.float16)
            chunks.append(feats)

            for k, t in enumerate(kept):
                rows.append({
                    "image_id": r.image_id, "label": r.label, "class_name": r.class_name,
                    "patient_key": r.patient_key, "split": r.split, "fold": r.fold,
                    "scale": sname, "crop_px": crop, "mm": round(crop * UM_PER_PX, 3),
                    "tile_i": k, "emb_row": row + k,
                    "x": t["x"], "y": t["y"],
                    "limbus_frac": round(t["limbus_frac"], 4),
                    "glare_frac": round(t["glare_frac"], 4),
                    "r_norm": round(t["r_norm"], 4), "theta": round(t["theta"], 2),
                })
            row += len(kept)

    idx = pd.DataFrame(rows)
    emb = np.concatenate(chunks, 0)
    assert len(idx) == len(emb)
    idx.to_csv(PROC / "tile_index_ms.csv", index=False)
    np.save(PROC / "tile_embeddings_ms.npy", emb)

    L = ["# Phase 2c - Multi-scale Tiling\n"]
    L.append("Phase 2b used 224 px = **0.92 mm** tiles. The `feathery_margin` annotation "
             "geometry gives a median Feret of **3.44 mm**, so those tiles held roughly a "
             "quarter of the pattern's width. This retiles at scales that can contain it.\n")
    L.append("| scale | crop | field of view | fed at | filament width |\n|---|---|---|---|---|")
    L.append(f"| s448 | 448 px | **1.83 mm** | 448 (1:1) | 12-49 px |")
    L.append(f"| s896 | 896 px | **3.67 mm** | 448 (2x down) | 6-24 px |")
    L.append("")
    per = idx.groupby(["scale", "image_id"]).size().groupby("scale").agg(["mean", "median", "min", "max"])
    L.append("## Bag sizes\n")
    L.append(per.round(1).to_markdown())
    L.append(f"\n- total tiles: **{len(idx):,}** across {idx.image_id.nunique()} images")
    L.append(f"- embeddings: {emb.shape} float16 = {emb.nbytes/1e6:.0f} MB\n")
    L.append("## Bag size by class\n")
    bs = idx.groupby(["scale", "class_name", "image_id"]).size().groupby(["scale", "class_name"]).mean()
    L.append(bs.round(2).to_markdown())
    L.append("\nDriven by limbus area and framing, not pathology - a gap here would mean bag "
             "size itself leaks the label.\n")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "08_multiscale_tiles.md").write_text("\n".join(L), encoding="utf-8")

    print(f"\ntiles {len(idx):,} | images {idx.image_id.nunique()} | {emb.nbytes/1e6:.0f} MB")
    print(per.to_string())
    print(f"wrote {REPORT_DIR / '08_multiscale_tiles.md'}")


if __name__ == "__main__":
    main()
