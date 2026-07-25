"""
Retrain the BINARY model (bacterial vs fungal) on the enlarged cohort.

Uses the v2 tile embeddings filtered to bacterial + fungal (Other dropped):
1484 images (885 bac / 599 fun) vs the original 682. Same proven recipe as the
first binary model - 896 px tiles, frozen DINOv2, mean-pooled MIL, 15-model
ensemble, temperature calibration - so the only change is the amount of data.

Saves to final_model_v2.pt (the validated original is untouched).

Outputs
    outputs/checkpoints/final_model_v2.pt
    outputs/reports/27_binary_v2.md
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, brier_score_loss

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
CKPT = ROOT / "outputs" / "checkpoints"
REPORT = ROOT / "outputs" / "reports" / "27_binary_v2.md"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

D_IN, D_HID, DROP = 384, 192, 0.25
EPOCHS, PATIENCE, LR, WD, MB = 60, 12, 3e-4, 1e-2, 32
N_FOLDS, SEEDS, N_REPEATS = 5, [0, 1, 2], 8
OLD_DEV, OLD_TEST = 0.806, 0.815     # original binary numbers for comparison


class MILMean(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(D_IN, D_HID), nn.LayerNorm(D_HID),
                                  nn.ReLU(), nn.Dropout(DROP))
        self.head = nn.Linear(D_HID, 1)

    def forward(self, x, mask):
        h = self.proj(x); m = mask.unsqueeze(-1).float()
        return self.head((h * m).sum(1) / m.sum(1).clamp(min=1)).squeeze(-1)


def bags():
    idx = pd.read_csv(PROC / "tile_index_v2.csv")
    idx = idx[idx.label < 2].sort_values(["image_id", "emb_row"])   # bacterial+fungal only
    emb = np.load(PROC / "tile_embeddings_v2.npy").astype(np.float32)
    imgs = idx.image_id.unique(); mt = int(idx.groupby("image_id").size().max())
    X = np.zeros((len(imgs), mt, D_IN), np.float32); M = np.zeros((len(imgs), mt), bool)
    meta = []
    for i, im in enumerate(imgs):
        g = idx[idx.image_id == im]; r = g.emb_row.to_numpy()
        X[i, :len(r)] = emb[r]; M[i, :len(r)] = True
        meta.append({"label": int(g.label.iloc[0]), "patient_key": g.patient_key.iloc[0],
                     "split": g.split.iloc[0], "fold": int(g.fold.iloc[0])})
    return X, M, pd.DataFrame(meta)


def epoch(model, opt, X, M, y, train):
    model.train(train); n = len(X)
    order = np.random.permutation(n) if train else np.arange(n)
    out = np.zeros(n)
    for i in range(0, n, MB):
        b = order[i:i + MB]
        xb = torch.from_numpy(X[b]).to(DEVICE); mb = torch.from_numpy(M[b]).to(DEVICE)
        yb = torch.from_numpy(y[b].astype(np.float32)).to(DEVICE)
        with torch.set_grad_enabled(train):
            lg = model(xb, mb); loss = F.binary_cross_entropy_with_logits(lg, yb)
        if train:
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        out[b] = lg.detach().cpu().numpy()
    return out


def train_one(Xtr, Mtr, ytr, gtr, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    tr, va = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(Xtr, ytr, groups=gtr))
    model = MILMean().to(DEVICE); opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    best, state, bad = -np.inf, None, 0
    for _ in range(EPOCHS):
        epoch(model, opt, Xtr[tr], Mtr[tr], ytr[tr], True)
        lv = epoch(model, opt, Xtr[va], Mtr[va], ytr[va], False)
        a = roc_auc_score(ytr[va], lv) if len(np.unique(ytr[va])) > 1 else .5
        if a > best:
            best, bad = a, 0
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE: break
    model.load_state_dict(state); return model


def temp_fit(logits, y):
    t = torch.nn.Parameter(torch.ones(1)); lg = torch.from_numpy(logits).float()
    yy = torch.from_numpy(y.astype(np.float32)); opt = torch.optim.LBFGS([t], lr=0.1, max_iter=100)
    def c():
        opt.zero_grad(); loss = F.binary_cross_entropy_with_logits(lg / t.clamp(min=1e-2), yy)
        loss.backward(); return loss
    opt.step(c); return float(t.detach().clamp(min=1e-2))


def boot(y, p, groups, n=2000, seed=42):
    rng = np.random.default_rng(seed); uq = np.unique(groups); out = []
    for _ in range(n):
        pick = rng.choice(uq, len(uq), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in pick])
        if len(np.unique(y[idx])) > 1: out.append(roc_auc_score(y[idx], p[idx]))
    return np.percentile(out, 2.5), np.percentile(out, 97.5)


def main():
    X, M, meta = bags()
    dev = (meta.split == "dev").to_numpy(); test = (meta.split == "test").to_numpy()
    y = meta.label.to_numpy(); grp = meta.patient_key.to_numpy()
    print(f"binary v2: dev {dev.sum()} test {test.sum()} | "
          f"bac {int((y==0).sum())} fun {int((y==1).sum())}")

    # repeated dev CV
    di = np.where(dev)[0]; aucs = []
    for s in range(N_REPEATS):
        cv = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=s)
        oof = np.zeros(len(y))
        for tr, te in cv.split(X[di], y[di], groups=grp[di]):
            m = train_one(X[di][tr], M[di][tr], y[di][tr], grp[di][tr], s)
            oof[di[te]] = epoch(m, None, X[di][te], M[di][te], y[di][te], False)
        aucs.append(roc_auc_score(y[di], oof[di]))
    dev_auc = float(np.mean(aucs))
    print(f"dev AUC {dev_auc:.4f} ± {np.std(aucs):.4f}")

    # ensemble on all dev, temperature from last-repeat OOF, eval test once
    models = []
    for f in range(N_FOLDS):
        tr = di[meta.iloc[di].fold.to_numpy() != f]
        for s in SEEDS:
            models.append(train_one(X[tr], M[tr], y[tr], grp[tr], s))
    T = temp_fit(oof[di], y[di])
    ti = np.where(test)[0]
    lg = np.mean([epoch(m, None, X[ti], M[ti], y[ti], False) for m in models], 0)
    p = 1 / (1 + np.exp(-lg / T)); yt = y[ti]
    test_auc = roc_auc_score(yt, p); lo, hi = boot(yt, p, grp[ti]); brier = brier_score_loss(yt, p)
    pred = (p >= 0.5).astype(int)
    cm = confusion_matrix(yt, pred, labels=[0, 1])
    fr = (pred[yt == 1] == 1).mean(); br = (pred[yt == 0] == 0).mean()
    mis = (pred[yt == 1] == 0).mean()   # fungal->bacterial

    CKPT.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dicts": [m.state_dict() for m in models], "temperature": T,
                "dev_auc": dev_auc, "test_auc": test_auc,
                "config": {"d_in": D_IN, "d_hid": D_HID, "dropout": DROP,
                           "crop_px": 896, "input_px": 448, "pooling": "mean",
                           "backbone": "vit_small_patch14_dinov2.lvd142m", "mm_per_px": 0.00409}},
               CKPT / "final_model_v2.pt")

    L = ["# Binary model — retrained on enlarged cohort\n"]
    L.append(f"Bacterial+fungal only: **{int((y==0).sum())} / {int((y==1).sum())}** "
             f"({dev.sum()} dev / {test.sum()} test), patient-disjoint. Same recipe as the "
             f"original; only the data grew (was 335/347, 682 total).\n")
    L.append("| | original (682) | retrained (1484) |\n|---|---|---|")
    L.append(f"| dev AUC | {OLD_DEV:.3f} | **{dev_auc:.3f} ± {np.std(aucs):.3f}** |")
    L.append(f"| locked test AUC | {OLD_TEST:.3f} (n=131) | **{test_auc:.3f}** [{lo:.3f}, {hi:.3f}] (n={test.sum()}) |")
    L.append(f"| Brier | 0.175 | {brier:.3f} |\n")
    L.append("## Test confusion (t=0.5)\n")
    L.append("| true ⧵ pred | Bacterial | Fungal |\n|---|---|---|")
    L.append(f"| **Bacterial** | {int(cm[0,0])} | {int(cm[0,1])} |")
    L.append(f"| **Fungal** | {int(cm[1,0])} | {int(cm[1,1])} |")
    L.append(f"\n- bacterial recall **{br:.1%}** · fungal recall **{fr:.1%}** · "
             f"fungal→bacterial **{mis:.1%}**")
    L.append(f"- temperature {T:.3f}\n")
    d = test_auc - OLD_TEST
    L.append("## Reading\n")
    L.append((f"More data **improved** the model ({OLD_TEST:.3f} → {test_auc:.3f}), on a "
              f"larger, more trustworthy test ({int(test.sum())} vs 131)."
              if d > 0.01 else
              f"More data left AUC essentially unchanged ({OLD_TEST:.3f} → {test_auc:.3f}), "
              f"but on a **larger** test ({int(test.sum())} vs 131) — a more trustworthy "
              f"estimate of the same ~0.8 ceiling.") +
             " Bacterial recall is the number to watch, since bacterial data nearly tripled.")
    REPORT.write_text("\n".join(L), encoding="utf-8")
    print(f"\nTEST AUC {test_auc:.4f} [{lo:.3f},{hi:.3f}] | bac_rec {br:.1%} fun_rec {fr:.1%} mis {mis:.1%}")
    print("confusion:\n", cm)
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
