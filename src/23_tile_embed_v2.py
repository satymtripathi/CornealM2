"""
Tile + embed the combined 3-class cohort (manifest_v2) at the winning config:
896 px native tiles (3.67 mm), fed at 448, frozen DINOv2 ViT-S/14.

New images have no cached limbus mask, so the limbus UNet++ is run on the fly
(and cached). Tiling is identical to the binary model and class-agnostic - no
label touches it.

Outputs
    data/interim/limbus/<image_id>.npz     (new masks added)
    data/processed/tile_index_v2.csv
    data/processed/tile_embeddings_v2.npy
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import timm
import segmentation_models_pytorch as smp
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT / "outputs" / "manifests" / "manifest_v2.csv"
LIMBUS_DIR = ROOT / "data" / "interim" / "limbus"
LIMBUS_CKPT = ROOT / "models" / "limbus_seg" / "model_limbus_crop_unetpp_weighted.pth"
PROC = ROOT / "data" / "processed"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL = "vit_small_patch14_dinov2.lvd142m"
CROP, STRIDE, INPUT = 896, 448, 448
MIN_LIMBUS_FRAC, MAX_GLARE_FRAC, MAX_TILES = 0.50, 0.60, 24
BATCH = 8
IMN_M = np.array([0.485, 0.456, 0.406], np.float32)
IMN_S = np.array([0.229, 0.224, 0.225], np.float32)


def load_limbus_model():
    ck = torch.load(LIMBUS_CKPT, map_location=DEVICE, weights_only=False)
    cfg = ck.get("config", {})
    tg = cfg.get("target_list", [{"label": "crop"}, {"label": "limbus"}])
    labs = [t["label"].strip().lower() for t in tg]
    i_l = labs.index("limbus") if "limbus" in labs else 1
    m = smp.UnetPlusPlus(cfg.get("encoder_name", "timm-efficientnet-b0"),
                         encoder_weights=None, in_channels=3, classes=len(tg), activation=None)
    m.load_state_dict(ck["state_dict"]); m.eval().to(DEVICE)
    return m, i_l, tuple(cfg.get("img_size", (512, 512)))


def get_contour(image_id, rgb, seg, i_l, size):
    cache = LIMBUS_DIR / f"{image_id}.npz"
    if cache.exists():
        z = np.load(cache); return z["limbus_contour"], tuple(int(v) for v in z["native_hw"])
    H, W = rgb.shape[:2]
    x = cv2.resize(rgb, size[::-1], interpolation=cv2.INTER_LINEAR)
    x = ((x.astype(np.float32) / 255. - IMN_M) / IMN_S).transpose(2, 0, 1)
    with torch.no_grad():
        prob = torch.sigmoid(seg(torch.from_numpy(x).unsqueeze(0).to(DEVICE)))[0, i_l].cpu().numpy()
    work = 2048; s = work / max(H, W)
    pm = cv2.resize(prob, (int(round(W*s)), int(round(H*s))), interpolation=cv2.INTER_LINEAR)
    mm = cv2.morphologyEx((pm > 0.5).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, (H, W)
    c = np.round(max(cnts, key=cv2.contourArea).astype(np.float64) / s).astype(np.int32).reshape(-1, 2)
    LIMBUS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, limbus_contour=c, native_hw=np.array([H, W], np.int32))
    return c, (H, W)


def plan(mask, H, W):
    ys, xs = np.where(mask > 0)
    if not len(xs): return []
    ii = cv2.integral(mask)
    out = []
    for y in range(int(ys.min()), max(int(ys.min())+1, int(ys.max())-CROP+2), STRIDE):
        for x in range(int(xs.min()), max(int(xs.min())+1, int(xs.max())-CROP+2), STRIDE):
            if y+CROP > H or x+CROP > W: continue
            s = ii[y+CROP, x+CROP]-ii[y, x+CROP]-ii[y+CROP, x]+ii[y, x]
            f = s/(CROP*CROP)
            if f >= MIN_LIMBUS_FRAC:
                out.append((x, y, f))
    out.sort(key=lambda t: -t[2])
    return out[:MAX_TILES]


def main():
    df = pd.read_csv(MAN)
    seg, i_l, ssize = load_limbus_model()
    bb = timm.create_model(MODEL, pretrained=True, num_classes=0, img_size=INPUT).eval().to(DEVICE)
    print(f"{len(df)} images -> tiling at {CROP}px/{INPUT}in")

    rows, embs, row, skipped = [], [], 0, 0
    for r in tqdm(list(df.itertuples()), desc="tile+embed"):
        with Image.open(ROOT / r.rel_path) as im:
            rgb = np.asarray(im.convert("RGB"))
        c, (H, W) = get_contour(r.image_id, rgb, seg, i_l, ssize)
        if c is None: skipped += 1; continue
        mask = np.zeros((H, W), np.uint8); cv2.fillPoly(mask, [c.astype(np.int32)], 1)
        tp = plan(mask, H, W)
        if not tp: skipped += 1; continue
        crops = []
        for x, y, f in tp:
            cr = rgb[y:y+CROP, x:x+CROP]
            if cr.shape[:2] != (CROP, CROP): continue
            hsv = cv2.cvtColor(cv2.resize(cr, (256, 256), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2HSV)
            if float(((hsv[..., 2]/255. > .94) & (hsv[..., 1]/255. < .15)).mean()) > MAX_GLARE_FRAC:
                continue
            cc = cv2.resize(cr, (INPUT, INPUT), interpolation=cv2.INTER_AREA)
            crops.append(((cc.astype(np.float32)/255. - IMN_M)/IMN_S).transpose(2, 0, 1))
            rows.append({"image_id": r.image_id, "label": r.label, "class_name": r.class_name,
                         "patient_key": r.patient_key, "split": r.split, "fold": r.fold,
                         "emb_row": row}); row += 1
        if not crops: skipped += 1; continue
        with torch.no_grad():
            for i in range(0, len(crops), BATCH):
                t = torch.from_numpy(np.stack(crops[i:i+BATCH])).to(DEVICE)
                with torch.autocast("cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):
                    embs.append(bb(t).float().cpu().numpy().astype(np.float16))
    idx = pd.DataFrame(rows)
    emb = np.concatenate(embs, 0)
    assert len(idx) == len(emb)
    PROC.mkdir(parents=True, exist_ok=True)
    idx.to_csv(PROC / "tile_index_v2.csv", index=False)
    np.save(PROC / "tile_embeddings_v2.npy", emb)
    print(f"\ntiles {len(idx):,} | images {idx.image_id.nunique()} | skipped {skipped}")
    print(f"bag size median {idx.groupby('image_id').size().median():.0f}")
    print(f"wrote {PROC/'tile_embeddings_v2.npy'} {emb.shape}")


if __name__ == "__main__":
    main()
