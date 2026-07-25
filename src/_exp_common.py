"""
Shared helpers for the hypopyon and camera-vs-pathology experiments.

Keeps both experiments using the EXACT training recipe - frozen DINOv2 ViT-S/14
at 448, mean-pooled MIL, repeated patient-grouped CV - so any difference is
attributable to the tiles fed in, not to a changed evaluation.
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
MANIFEST = ROOT / "outputs" / "manifests" / "manifest.csv"
LIMBUS_DIR = ROOT / "data" / "interim" / "limbus"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL = "vit_small_patch14_dinov2.lvd142m"
INPUT = 448
EMB_BATCH = 8

IMNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMNET_STD = np.array([0.229, 0.224, 0.225], np.float32)

N_FOLDS, N_REPEATS = 5, 8
EPOCHS, PATIENCE, LR, WD, MB = 60, 12, 3e-4, 1e-2, 32
D_HID, DROPOUT = 192, 0.25


# ---------------- limbus ----------------
def load_limbus(image_id):
    p = LIMBUS_DIR / f"{image_id}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return z["limbus_contour"], tuple(int(v) for v in z["native_hw"])


def limbus_mask_and_centre(contour, H, W):
    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask, [contour.astype(np.int32)], 1)
    M = cv2.moments(contour.astype(np.int32))
    if M["m00"] <= 0:
        return mask, W / 2, H / 2, 0.0
    return mask, M["m10"] / M["m00"], M["m01"] / M["m00"], float(np.sqrt(M["m00"] / np.pi))


# ---------------- embedding ----------------
_backbone = None


def _get_backbone():
    global _backbone
    if _backbone is None:
        _backbone = timm.create_model(MODEL, pretrained=True, num_classes=0, img_size=INPUT)
        _backbone.eval().to(DEVICE)
    return _backbone


def embed_tiles(plan_df, transform=None, desc="embedding"):
    """
    plan_df: rows with image_id, x, y, crop_px. Returns (N, 384) float16 aligned
    to plan_df order. `transform(rgb_uint8_tile)->rgb_uint8_tile` is applied to
    each crop before normalisation (used for photometric perturbations).
    """
    man = pd.read_csv(MANIFEST)
    paths = dict(zip(man.image_id, man.rel_path))
    model = _get_backbone()

    out = np.zeros((len(plan_df), 384), np.float16)
    pos = 0
    for iid, g in tqdm(plan_df.groupby("image_id", sort=False), desc=desc, leave=False):
        with Image.open(ROOT / paths[iid]) as im:
            rgb = np.asarray(im.convert("RGB"))
        H, W = rgb.shape[:2]
        crops = []
        for t in g.itertuples():
            cp = int(t.crop_px)
            y0, x0 = int(t.y), int(t.x)
            c = rgb[max(0, y0):y0 + cp, max(0, x0):x0 + cp]
            if c.shape[0] < 8 or c.shape[1] < 8:
                c = np.zeros((cp, cp, 3), np.uint8)
            if c.shape[:2] != (cp, cp):
                c = cv2.resize(c, (cp, cp), interpolation=cv2.INTER_AREA)
            if transform is not None:
                c = transform(c)
            if cp != INPUT:
                c = cv2.resize(c, (INPUT, INPUT), interpolation=cv2.INTER_AREA)
            x = c.astype(np.float32) / 255.0
            crops.append(((x - IMNET_MEAN) / IMNET_STD).transpose(2, 0, 1))
        feats = []
        with torch.no_grad():
            for i in range(0, len(crops), EMB_BATCH):
                tin = torch.from_numpy(np.stack(crops[i:i + EMB_BATCH])).to(DEVICE)
                with torch.autocast("cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):
                    f = model(tin)
                feats.append(f.float().cpu().numpy())
        f = np.concatenate(feats, 0).astype(np.float16)
        out[pos:pos + len(f)] = f
        pos += len(f)
    return out


# ---------------- bags ----------------
def build_bags(index_df, emb):
    """index_df needs image_id, label, patient_key, emb_row. Returns X, M, meta."""
    idx = index_df.sort_values(["image_id", "emb_row"])
    images = idx.image_id.unique()
    max_t = max(1, int(idx.groupby("image_id").size().max()))
    X = np.zeros((len(images), max_t, emb.shape[1]), np.float32)
    Msk = np.zeros((len(images), max_t), bool)
    rows = []
    for i, im in enumerate(images):
        g = idx[idx.image_id == im]
        r = g.emb_row.to_numpy()
        X[i, :len(r)] = emb[r]
        Msk[i, :len(r)] = True
        rows.append({"image_id": im, "label": int(g.label.iloc[0]),
                     "patient_key": g.patient_key.iloc[0]})
    return X, Msk, pd.DataFrame(rows)


# ---------------- MIL ----------------
class MILMean(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d_in, D_HID), nn.LayerNorm(D_HID),
                                  nn.ReLU(), nn.Dropout(DROPOUT))
        self.head = nn.Linear(D_HID, 1)

    def forward(self, x, mask):
        h = self.proj(x)
        m = mask.unsqueeze(-1).float()
        z = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(z).squeeze(-1)


def _run(model, opt, X, M, y, train):
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


def _fit_predict(Xtr, Mtr, ytr, gtr, Xte, Mte, seed, d):
    torch.manual_seed(seed); np.random.seed(seed)
    tr, va = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed)
                  .split(Xtr, ytr, groups=gtr))
    model = MILMean(d).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    best, state, bad = -np.inf, None, 0
    for _ in range(EPOCHS):
        _run(model, opt, Xtr[tr], Mtr[tr], ytr[tr], True)
        pv = _run(model, opt, Xtr[va], Mtr[va], ytr[va], False)
        a = roc_auc_score(ytr[va], pv) if len(np.unique(ytr[va])) > 1 else .5
        if a > best:
            best, bad = a, 0
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if state:
        model.load_state_dict(state)
    return _run(model, opt, Xte, Mte, np.zeros(len(Xte)), False)


def eval_cv(X, M, meta, n_repeats=N_REPEATS, return_oof=False):
    """Repeated patient-grouped 5-fold CV. Returns (mean, sd) or with OOF preds."""
    y = meta.label.to_numpy()
    groups = meta.patient_key.to_numpy()
    d = X.shape[2]
    aucs, last = [], None
    for seed in range(n_repeats):
        cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        oof = np.zeros(len(y))
        for tr, te in cv.split(X, y, groups):
            oof[te] = _fit_predict(X[tr], M[tr], y[tr], groups[tr], X[te], M[te], seed, d)
        aucs.append(roc_auc_score(y, oof))
        last = oof
    a = np.array(aucs)
    if return_oof:
        return float(a.mean()), float(a.std()), last, y
    return float(a.mean()), float(a.std())
