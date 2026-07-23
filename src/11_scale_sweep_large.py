"""
Phase 3d - Locating the peak of the scale curve.

The curve so far, all mean-pooled, all fed at 448 input:

    0.92 mm  (s224)      0.7451
    1.83 mm  (s448)      0.7593
    3.67 mm  (s896)      0.7907   <- best measured
    ...
    11.7 mm  (whole eye) 0.7470

It rises to 3.67 mm and has fallen away by 11.7 mm, so the maximum lies
somewhere in between and has never been sampled. This fills the gap with
5.5 mm and 7.3 mm crops.

Phase 3c established that pixel fidelity contributes nothing (s896 fed at 896
scored -0.009 against the same crops fed at 448), so everything here is fed at
448. That keeps compute low and matches the configuration that actually won.

Geometry has produced both of the real gains so far. This is the last untested
geometric knob; after it, the remaining levers are all about the representation.

Outputs
    data/processed/tile_index_large.csv
    data/processed/tile_embeddings_large.npy
    outputs/reports/11_scale_sweep_large.md
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
LIMBUS_DIR = ROOT / "data" / "interim" / "limbus"
REPORT_DIR = ROOT / "outputs" / "reports"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL = "vit_small_patch14_dinov2.lvd142m"
INPUT = 448
BATCH = 8

# (name, crop_px, stride_px, max_tiles)
SCALES = [
    ("s1344", 1344, 672, 16),
    ("s1792", 1792, 896, 12),
]

MIN_LIMBUS_FRAC = 0.50
MAX_GLARE_FRAC = 0.60
MM_PER_PX = 0.00409

N_FOLDS, N_REPEATS = 5, 8
EPOCHS, PATIENCE, LR, WD, MB = 60, 12, 3e-4, 1e-2, 32
D_HID, D_ATT, DROPOUT = 192, 128, 0.25

IMNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMNET_STD = np.array([0.229, 0.224, 0.225], np.float32)

LADDER = [("0.92 mm  s224", 0.7451), ("1.83 mm  s448", 0.7593),
          ("3.67 mm  s896", 0.7907), ("11.7 mm  whole eye", 0.7470)]


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
            f = s / area
            if f < MIN_LIMBUS_FRAC:
                continue
            dx, dy = x + crop / 2 - cx, y + crop / 2 - cy
            out.append({"x": x, "y": y, "limbus_frac": float(f),
                        "r_norm": float(np.hypot(dx, dy) / max(radius, 1e-6))})
    return out


def build():
    idx_p = PROC / "tile_index_large.csv"
    emb_p = PROC / "tile_embeddings_large.npy"
    if idx_p.exists() and emb_p.exists():
        print("cached")
        return pd.read_csv(idx_p), np.load(emb_p)

    man = pd.read_csv(ROOT / "outputs" / "manifests" / "manifest.csv")
    model = timm.create_model(MODEL, pretrained=True, num_classes=0, img_size=INPUT)
    model.eval().to(DEVICE)

    rows, chunks, row = [], [], 0
    for r in tqdm(list(man.itertuples()), desc="large scales"):
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

        rgb = None
        for sname, crop, stride, cap in SCALES:
            tp = plan(mask, cx, cy, radius, crop, stride)
            if not tp:
                continue
            tp.sort(key=lambda t: -t["limbus_frac"])
            tp = tp[:cap]

            if rgb is None:
                with Image.open(ROOT / r.rel_path) as im:
                    rgb = np.asarray(im.convert("RGB"))

            kept, batch = [], []
            for t in tp:
                c = rgb[t["y"]:t["y"] + crop, t["x"]:t["x"] + crop]
                if c.shape[:2] != (crop, crop):
                    continue
                hsv = cv2.cvtColor(cv2.resize(c, (256, 256), interpolation=cv2.INTER_AREA),
                                   cv2.COLOR_RGB2HSV)
                g = float(((hsv[..., 2] / 255. > .94) & (hsv[..., 1] / 255. < .15)).mean())
                if g > MAX_GLARE_FRAC:
                    continue
                # downsize immediately - never hold a batch of 1792^2 arrays
                c = cv2.resize(c, (INPUT, INPUT), interpolation=cv2.INTER_AREA)
                t["glare_frac"] = g
                kept.append(t)
                x = c.astype(np.float32) / 255.0
                batch.append(((x - IMNET_MEAN) / IMNET_STD).transpose(2, 0, 1))

            if not kept:
                continue
            feats = []
            with torch.no_grad():
                for i in range(0, len(batch), BATCH):
                    t_in = torch.from_numpy(np.stack(batch[i:i + BATCH])).to(DEVICE)
                    with torch.autocast("cuda", dtype=torch.float16,
                                        enabled=(DEVICE == "cuda")):
                        f = model(t_in)
                    feats.append(f.float().cpu().numpy())
            f = np.concatenate(feats, 0).astype(np.float16)
            chunks.append(f)
            for k, t in enumerate(kept):
                rows.append({"image_id": r.image_id, "label": r.label,
                             "patient_key": r.patient_key, "scale": sname,
                             "crop_px": crop, "mm": round(crop * MM_PER_PX, 3),
                             "tile_i": k, "emb_row": row + k,
                             "x": t["x"], "y": t["y"],
                             "limbus_frac": round(t["limbus_frac"], 4),
                             "r_norm": round(t["r_norm"], 4)})
            row += len(kept)

    del model
    torch.cuda.empty_cache()
    idx = pd.DataFrame(rows)
    emb = np.concatenate(chunks, 0)
    idx.to_csv(idx_p, index=False)
    np.save(emb_p, emb)
    return idx, emb


# ---------------- MIL (identical to Phase 3/3b/3c) ----------------
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


def run(model, opt, X, Msk, y, train):
    model.train(train)
    n = len(X)
    order = np.random.permutation(n) if train else np.arange(n)
    out = np.zeros(n)
    for i in range(0, n, MB):
        b = order[i:i + MB]
        xb = torch.from_numpy(X[b]).to(DEVICE)
        mb = torch.from_numpy(Msk[b]).to(DEVICE)
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


def evaluate(idx, emb, scales):
    sub = idx[idx.scale.isin(scales)].sort_values(["image_id", "scale", "tile_i"])
    images = sub.image_id.unique()
    max_t = int(sub.groupby("image_id").size().max())
    d = emb.shape[1]
    X = np.zeros((len(images), max_t, d), np.float32)
    Msk = np.zeros((len(images), max_t), bool)
    meta = []
    for i, im in enumerate(images):
        g = sub[sub.image_id == im]
        rws = g.emb_row.to_numpy()
        X[i, :len(rws)] = emb[rws]
        Msk[i, :len(rws)] = True
        meta.append({"label": g.label.iloc[0], "patient_key": g.patient_key.iloc[0]})
    meta = pd.DataFrame(meta)
    y, groups = meta.label.to_numpy(), meta.patient_key.to_numpy()

    res = {}
    for pooling in ["mean", "max"]:
        aucs = []
        for seed in range(N_REPEATS):
            cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
            oof = np.zeros(len(y))
            for tr, te in cv.split(X, y, groups):
                oof[te] = fit_predict(X[tr], Msk[tr], y[tr], groups[tr],
                                      X[te], Msk[te], pooling, seed, d)
            aucs.append(roc_auc_score(y, oof))
        a = np.array(aucs)
        res[pooling] = (a.mean(), a.std())
    return res, max_t, len(images)


def main():
    idx, emb = build()
    print(f"{len(idx):,} tiles | {idx.image_id.nunique()} images")
    print(idx.groupby("scale").size().to_string())

    rows = []
    for sname, crop, _, _ in SCALES:
        r, max_t, n_img = evaluate(idx, emb, [sname])
        rows.append({"scale": sname, "mm": round(crop * MM_PER_PX, 2),
                     "images": n_img, "max_tiles": max_t,
                     "mean": round(r["mean"][0], 4), "mean_sd": round(r["mean"][1], 4),
                     "max": round(r["max"][0], 4), "max_sd": round(r["max"][1], 4)})
        print(f"  {sname} ({crop*MM_PER_PX:.2f} mm)  mean={r['mean'][0]:.4f} "
              f"+/-{r['mean'][1]:.4f}   max={r['max'][0]:.4f}")

    res = pd.DataFrame(rows)
    best_new = float(res["mean"].max())

    L = ["# Phase 3d - Scale sweep, large crops\n"]
    L.append(f"All fed at {INPUT} input (Phase 3c showed pixel fidelity contributes "
             f"nothing). Same MIL, same folds, {N_REPEATS} fold assignments.\n")
    L.append(res.to_markdown(index=False))

    L.append("\n## Full scale curve (mean pooling)\n")
    L.append("| field of view | AUC |\n|---|---|")
    curve = LADDER[:3] + [(f"{r['mm']:.2f} mm  {r['scale']}", r["mean"]) for _, r in res.iterrows()] \
            + [LADDER[3]]
    for name, v in sorted(curve, key=lambda t: float(t[0].split()[0])):
        L.append(f"| {name} | {v:.4f} |")
    L.append("")

    L.append("## Reading\n")
    if best_new > 0.7907 + 0.012:
        pk = res.loc[res["mean"].idxmax()]
        L.append(f"The peak is larger than 3.67 mm: **{pk['mm']:.2f} mm reaches "
                 f"{pk['mean']:.4f}** (+{best_new - 0.7907:+.4f} over s896). Geometry still "
                 f"has room, and the winning configuration changes.")
    elif best_new > 0.7907 - 0.012:
        L.append(f"Best large crop reaches **{best_new:.4f}** against **0.7907** for 3.67 mm - "
                 f"statistically indistinguishable. **The curve is flat between roughly 3.7 and "
                 f"7 mm**, then falls away by whole-eye. Geometry is exhausted: s896 stays the "
                 f"configuration, and further gains must come from the representation.")
    else:
        L.append(f"Best large crop reaches only **{best_new:.4f}** against **0.7907** for "
                 f"3.67 mm. **The peak is at or near 3.67 mm** and larger crops dilute the "
                 f"pattern, converging toward the whole-eye result. Geometry is settled; "
                 f"s896 is the configuration to carry forward.")
    L.append("")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "11_scale_sweep_large.md").write_text("\n".join(L), encoding="utf-8")
    print(f"\nbest new {best_new:.4f} vs s896 0.7907 -> {best_new - 0.7907:+.4f}")
    print(f"wrote {REPORT_DIR / '11_scale_sweep_large.md'}")


if __name__ == "__main__":
    main()
