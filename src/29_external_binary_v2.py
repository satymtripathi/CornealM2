"""
Step 2 - External validation of the retrained binary model on both cohorts.

Reuses the self-contained preprocessing (limbus + DINOv2 tiling from Pipeline3)
with the v2 binary heads + v2 calibration. Scored on bacterial/fungal GT only
(the binary model's scope); "Others" reported separately as out-of-scope.

    python src/29_external_binary_v2.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None

import cv2
from inference_3class import Pipeline3, CROP, INPUT, IMNET_MEAN, IMNET_STD, MAX_GLARE_FRAC

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "outputs" / "checkpoints"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COHORTS = [
    ("cohort_01", r"C:\Users\satyam.tripathi\Desktop\Model2Image_ReviewR\images", "outputs/review_gt.tsv"),
    ("cohort_02", r"C:\Users\satyam.tripathi\Desktop\Model2ImageReview\images", "outputs/review_gt_cohort02.tsv"),
]
NORM = {"bacterial": "Bacterial", "fungal": "Fungal", "others": "Other", "other": "Other"}


class MILMean(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(384, 192), nn.LayerNorm(192), nn.ReLU(), nn.Dropout(0.25))
        self.head = nn.Linear(192, 1)

    def forward(self, x, m):
        h = self.proj(x); mm = m.unsqueeze(-1).float()
        return self.head((h * mm).sum(1) / mm.sum(1).clamp(min=1)).squeeze(-1)


def read_gt(path):
    sep = "\t" if str(path).endswith((".tsv", ".txt")) else ","
    df = pd.read_csv(ROOT / path, sep=sep)
    fcol = next(c for c in df.columns if any(k in c.lower() for k in ("file", "name", "image")))
    lcol = next(c for c in df.columns if c != fcol and any(k in c.lower() for k in ("gt", "label", "truth", "class")))
    out = df[[fcol, lcol]].copy(); out.columns = ["filename", "gt"]
    out["gt_norm"] = out["gt"].astype(str).str.strip().str.lower().map(NORM)
    return out


def main():
    base = Pipeline3.get()          # limbus + backbone (reuse preprocessing)
    ck = torch.load(CKPT / "final_model_v2.pt", map_location=DEVICE, weights_only=False)
    cal = json.load(open(CKPT / "calibration_binary_v2.json"))
    T = cal["temperature"]; fs_t = cal["modes"]["fungal_safety"]["t"]
    heads = []
    for sd in ck["state_dicts"]:
        m = MILMean().to(DEVICE); m.load_state_dict(sd); m.eval(); heads.append(m)

    def p_fungal(rgb):
        contour, mask = base._segment_limbus(rgb)
        if contour is None: return None
        tiles = []
        for t in base._plan(mask):
            c = rgb[t["y"]:t["y"]+CROP, t["x"]:t["x"]+CROP]
            if c.shape[:2] != (CROP, CROP): continue
            hsv = cv2.cvtColor(cv2.resize(c, (256, 256), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2HSV)
            if float(((hsv[..., 2]/255. > .94) & (hsv[..., 1]/255. < .15)).mean()) > MAX_GLARE_FRAC: continue
            cc = cv2.resize(c, (INPUT, INPUT), interpolation=cv2.INTER_AREA)
            tiles.append(((cc.astype(np.float32)/255. - IMNET_MEAN)/IMNET_STD).transpose(2, 0, 1))
        if not tiles: return None
        with torch.no_grad():
            feats = []
            for i in range(0, len(tiles), 8):
                bt = torch.from_numpy(np.stack(tiles[i:i+8])).to(DEVICE)
                with torch.autocast("cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):
                    feats.append(base.backbone(bt).float())
            F_ = torch.cat(feats, 0).unsqueeze(0); msk = torch.ones(1, F_.shape[1], dtype=torch.bool, device=DEVICE)
            lg = np.mean([h(F_, msk).item() for h in heads])
        return 1/(1+np.exp(-lg/T))

    L = ["# Binary v2 — external validation\n",
         f"Retrained model, dev AUC {cal['dev_auc']:.3f}. Scored on bacterial/fungal GT. "
         f"fungal_safety t={fs_t:.2f}.\n"]
    pooled_y, pooled_p = [], []
    for name, img_dir, gt in COHORTS:
        g = read_gt(gt)
        files = sorted(p for p in Path(img_dir).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        rows = []
        for p in tqdm(files, desc=name, leave=False):
            pf = None
            try: pf = p_fungal(np.asarray(Image.open(p).convert("RGB")))
            except Exception: pass
            rows.append({"filename": p.name, "p_fungal": pf})
        d = pd.DataFrame(rows).merge(g, on="filename", how="left")
        b = d[(d.gt_norm.isin(["Bacterial", "Fungal"])) & d.p_fungal.notna()].copy()
        y = (b.gt_norm == "Fungal").astype(int).to_numpy(); pv = b.p_fungal.to_numpy()
        pooled_y += list(y); pooled_p += list(pv)
        auc = roc_auc_score(y, pv)
        L.append(f"## {name} (n={len(b)}: {int((y==0).sum())} bac / {int((y==1).sum())} fun)\n")
        L.append("| mode | AUC | fungal recall | bacterial recall | misroute |\n|---|---|---|---|---|")
        for nm, t in [("balanced", 0.5), ("fungal_safety", fs_t)]:
            dd = (pv >= t).astype(int); fung = y == 1
            L.append(f"| {nm} | {auc:.3f} | {(dd[fung]==1).mean():.0%} | "
                     f"{(dd[~fung]==0).mean():.0%} | {(dd[fung]==0).mean():.0%} |")
        L.append("")
        print(f"{name}: AUC {auc:.3f} (n={len(b)})")
    y = np.array(pooled_y); pv = np.array(pooled_p)
    auc = roc_auc_score(y, pv)
    L.append(f"## Pooled external (n={len(y)})\n")
    L.append(f"- **AUC {auc:.4f}** (binary v1 external was 0.839)")
    for nm, t in [("balanced", 0.5), ("fungal_safety", fs_t)]:
        dd = (pv >= t).astype(int); fung = y == 1
        L.append(f"- {nm}: fungal recall {(dd[fung]==1).mean():.0%}, "
                 f"bacterial recall {(dd[~fung]==0).mean():.0%}, misroute {(dd[fung]==0).mean():.0%}")
    (ROOT / "outputs" / "reports" / "29_external_binary_v2.md").write_text("\n".join(L), encoding="utf-8")
    print(f"\nPOOLED external AUC {auc:.4f} (v1 was 0.839)")
    print("wrote outputs/reports/29_external_binary_v2.md")


if __name__ == "__main__":
    main()
