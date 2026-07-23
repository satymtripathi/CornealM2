"""
Phase 3c - s896 tiles at FULL resolution (896 input, no downsampling).

Phase 3b found the best configuration to be 896px native crops (3.67 mm field of
view) - but those crops were downsampled 2x to 448 before the backbone. The
s448 tiles, by contrast, were fed 1:1.

So s896 beat s448 by +0.032 while throwing away half its pixels. Field of view
beat pixel fidelity. That leaves one configuration untested: the same 3.67 mm
field of view at full fidelity.

    s448 @ 448   1.83 mm, 1:1      0.7593
    s896 @ 448   3.67 mm, 2x down  0.7907   <- current best
    s896 @ 896   3.67 mm, 1:1      ???

This reuses the EXACT tile positions from Phase 3b (tile_index_ms.csv, scale
s896), so the only variable that changes is input resolution. Same crops, same
bags, same folds.

Outputs
    data/processed/tile_embeddings_s896full.npy
    outputs/reports/10_s896_fullres.md
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "outputs" / "reports"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL = "vit_small_patch14_dinov2.lvd142m"
INPUT = 896                    # 64x64 = 4096 patches - heavy, keep batch small
BATCH = 2
EMB_PATH = PROC / "tile_embeddings_s896full.npy"

N_FOLDS, N_REPEATS = 5, 8
EPOCHS, PATIENCE, LR, WD, MB = 60, 12, 3e-4, 1e-2, 32
D_HID, D_ATT, DROPOUT = 192, 128, 0.25

IMNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMNET_STD = np.array([0.229, 0.224, 0.225], np.float32)

BASELINE = {"s896 @448 (2x down)": 0.7907, "s448 @448 (1:1)": 0.7593,
            "whole-eye (Phase 1c)": 0.747}


# =====================================================
# EMBEDDING
# =====================================================
def extract(idx: pd.DataFrame, man: pd.DataFrame) -> np.ndarray:
    if EMB_PATH.exists():
        print("cached")
        return np.load(EMB_PATH)

    model = timm.create_model(MODEL, pretrained=True, num_classes=0, img_size=INPUT)
    model.eval().to(DEVICE)
    paths = dict(zip(man.image_id, man.rel_path))

    out = np.zeros((len(idx), 384), np.float16)
    pos = 0
    for iid, g in tqdm(idx.groupby("image_id", sort=False), desc=f"s896@{INPUT}"):
        with Image.open(ROOT / paths[iid]) as im:
            rgb = np.asarray(im.convert("RGB"))

        crops = []
        for t in g.itertuples():
            c = rgb[t.y:t.y + t.crop_px, t.x:t.x + t.crop_px]
            if c.shape[:2] != (t.crop_px, t.crop_px):
                c = cv2.resize(c, (t.crop_px, t.crop_px), interpolation=cv2.INTER_AREA)
            if t.crop_px != INPUT:
                c = cv2.resize(c, (INPUT, INPUT), interpolation=cv2.INTER_AREA)
            x = c.astype(np.float32) / 255.0
            crops.append(((x - IMNET_MEAN) / IMNET_STD).transpose(2, 0, 1))

        feats = []
        with torch.no_grad():
            for i in range(0, len(crops), BATCH):
                t_in = torch.from_numpy(np.stack(crops[i:i + BATCH])).to(DEVICE)
                with torch.autocast("cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):
                    f = model(t_in)
                feats.append(f.float().cpu().numpy())
        f = np.concatenate(feats, 0).astype(np.float16)
        out[pos:pos + len(f)] = f
        pos += len(f)

    del model
    torch.cuda.empty_cache()
    np.save(EMB_PATH, out)
    return out


# =====================================================
# MIL (identical to Phase 3 so numbers are comparable)
# =====================================================
class MIL(nn.Module):
    def __init__(self, d_in, pooling):
        super().__init__()
        self.pooling = pooling
        self.proj = nn.Sequential(nn.Linear(d_in, D_HID), nn.LayerNorm(D_HID),
                                  nn.ReLU(), nn.Dropout(DROPOUT))
        if pooling == "gated":
            self.V = nn.Linear(D_HID, D_ATT)
            self.U = nn.Linear(D_HID, D_ATT)
            self.w = nn.Linear(D_ATT, 1)
        self.head = nn.Linear(D_HID, 1)

    def forward(self, x, mask):
        h = self.proj(x)
        m = mask.unsqueeze(-1).float()
        if self.pooling == "mean":
            z = (h * m).sum(1) / m.sum(1).clamp(min=1)
        elif self.pooling == "max":
            z = h.masked_fill(~mask.unsqueeze(-1), -1e4).max(1).values
        else:
            a = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))).squeeze(-1)
            a = torch.softmax(a.masked_fill(~mask, -1e4), 1)
            z = (a.unsqueeze(-1) * h).sum(1)
        return self.head(z).squeeze(-1)


def run(model, opt, X, M, y, train):
    model.train(train)
    n = len(X)
    order = np.random.permutation(n) if train else np.arange(n)
    out = np.zeros(n)
    for i in range(0, n, MB):
        b = order[i:i + MB]
        xb = torch.from_numpy(X[b]).to(DEVICE)
        mb = torch.from_numpy(M[b]).to(DEVICE)
        yb = torch.from_numpy(y[b].astype(np.float32)).to(DEVICE)
        with torch.set_grad_enabled(train):
            lg = model(xb, mb)
            loss = F.binary_cross_entropy_with_logits(lg, yb)
        if train:
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        out[b] = torch.sigmoid(lg).detach().cpu().numpy()
    return out


def fit_predict(Xtr, Mtr, ytr, gtr, Xte, Mte, pooling, seed, d):
    torch.manual_seed(seed); np.random.seed(seed)
    tr_i, va_i = next(GroupShuffleSplit(n_splits=1, test_size=0.2,
                                        random_state=seed).split(Xtr, ytr, groups=gtr))
    model = MIL(d, pooling).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    best, state, bad = -np.inf, None, 0
    for _ in range(EPOCHS):
        run(model, opt, Xtr[tr_i], Mtr[tr_i], ytr[tr_i], True)
        pv = run(model, opt, Xtr[va_i], Mtr[va_i], ytr[va_i], False)
        a = roc_auc_score(ytr[va_i], pv) if len(np.unique(ytr[va_i])) > 1 else .5
        if a > best:
            best, bad = a, 0
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if state:
        model.load_state_dict(state)
    return run(model, opt, Xte, Mte, np.zeros(len(Xte)), False)


# =====================================================
# MAIN
# =====================================================
def main():
    man = pd.read_csv(ROOT / "outputs" / "manifests" / "manifest.csv")
    ms = pd.read_csv(PROC / "tile_index_ms.csv")
    idx = ms[ms.scale == "s896"].sort_values(["image_id", "tile_i"]).reset_index(drop=True)
    print(f"{len(idx):,} s896 tiles / {idx.image_id.nunique()} images -> input {INPUT}")

    emb = extract(idx, man).astype(np.float32)

    images = idx.image_id.unique()
    max_t = int(idx.groupby("image_id").size().max())
    X = np.zeros((len(images), max_t, 384), np.float32)
    M = np.zeros((len(images), max_t), bool)
    meta = []
    start = 0
    for i, im in enumerate(images):
        n = int((idx.image_id == im).sum())
        X[i, :n] = emb[start:start + n]
        M[i, :n] = True
        g = idx[idx.image_id == im].iloc[0]
        meta.append({"label": g.label, "patient_key": g.patient_key})
        start += n
    meta = pd.DataFrame(meta)
    y, groups = meta.label.to_numpy(), meta.patient_key.to_numpy()

    rows = []
    for pooling in ["mean", "max", "gated"]:
        aucs = []
        for seed in range(N_REPEATS):
            cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
            oof = np.zeros(len(y))
            for tr, te in cv.split(X, y, groups):
                oof[te] = fit_predict(X[tr], M[tr], y[tr], groups[tr],
                                      X[te], M[te], pooling, seed, 384)
            aucs.append(roc_auc_score(y, oof))
        a = np.array(aucs)
        rows.append({"pooling": pooling, "auc": round(a.mean(), 4), "sd": round(a.std(), 4)})
        print(f"  {pooling:6s} AUC = {a.mean():.4f} +/- {a.std():.4f}")

    res = pd.DataFrame(rows)
    best = float(res.auc.max())
    prev = BASELINE["s896 @448 (2x down)"]

    L = ["# Phase 3c - s896 tiles at full resolution\n"]
    L.append(f"Identical tile positions to Phase 3b (scale `s896`, {len(idx):,} tiles). "
             f"The only change is input resolution: **448 -> {INPUT}**, i.e. the 3.67 mm "
             f"crop is now fed 1:1 instead of downsampled 2x.\n")
    L.append(res.to_markdown(index=False))
    L.append("\n## Against the ladder\n")
    L.append("| configuration | field of view | fidelity | AUC |\n|---|---|---|---|")
    L.append(f"| whole eye | 11.7 mm | heavy downsample | {BASELINE['whole-eye (Phase 1c)']:.4f} |")
    L.append(f"| s448 @448 | 1.83 mm | 1:1 | {BASELINE['s448 @448 (1:1)']:.4f} |")
    L.append(f"| s896 @448 | 3.67 mm | 2x down | {prev:.4f} |")
    L.append(f"| **s896 @896** | **3.67 mm** | **1:1** | **{best:.4f}** |")
    L.append("")

    d = best - prev
    L.append("## Reading\n")
    if d > 0.015:
        L.append(f"Full fidelity adds **{d:+.4f}**. Field of view and pixel fidelity are "
                 f"*both* contributing, so the two axes are complementary rather than "
                 f"substitutes. Larger crops at full resolution are worth their compute, "
                 f"and the scale sweep should be run at 1:1.")
    elif d > -0.015:
        L.append(f"Full fidelity changes little (**{d:+.4f}**). **Field of view is what "
                 f"matters; pixel fidelity is not.** The 2x-downsampled version is "
                 f"statistically equivalent at a quarter of the compute, so the cheaper "
                 f"configuration should be kept and the scale sweep run at 448 input.")
    else:
        L.append(f"Full fidelity is **{d:+.4f}** WORSE. Likely the position-embedding "
                 f"interpolation to {INPUT} degrading a backbone trained at 518, the same "
                 f"effect seen at eye@896 in Phase 1c. Keep the downsampled configuration.")
    L.append("")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "10_s896_fullres.md").write_text("\n".join(L), encoding="utf-8")
    print(f"\nbest {best:.4f} vs s896@448 {prev:.4f}  ->  {d:+.4f}")
    print(f"wrote {REPORT_DIR / '10_s896_fullres.md'}")


if __name__ == "__main__":
    main()
