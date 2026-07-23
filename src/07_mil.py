"""
Phase 3 - Attention MIL over native-resolution tile embeddings.

Tests the claim left open by Phase 1c: whole-eye pooled representations plateau
at AUC 0.747 +/- 0.013, and only LOCALISED representations can go past it.

The pooling ablation is the point. A gain over 0.747 could come from either of
two very different sources, and mean-pooling separates them:

    mean pooling beats 0.747
        -> the win is native RESOLUTION. Tiles at 1:1 simply carry texture that
           a downsampled whole-eye view destroys, and no localisation is needed.

    only attention beats 0.747
        -> the win is LOCALISATION. The signal sits in a few tiles and gets
           diluted by averaging over the whole cornea - which is precisely why
           the global view plateaued.

    neither beats 0.747
        -> the localisation premise is wrong and the design needs rethinking.

Evaluation matches Phase 1c exactly so the numbers are comparable: patient
grouped, repeated outer CV over several fold assignments, and early stopping on
an inner split carved from the training folds only - never on the test fold.

Outputs
    outputs/manifests/mil_predictions.npz
    outputs/reports/07_mil.md
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
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "outputs" / "reports"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_FOLDS = 5
N_REPEATS = 10
EPOCHS = 60
PATIENCE = 12
LR = 3e-4
WD = 1e-2
BATCH = 32
D_HID = 192
D_ATT = 128
DROPOUT = 0.25
TOPK = 8

POOLINGS = ["mean", "max", "topk", "gated"]


# =====================================================
# MODEL
# =====================================================
class MIL(nn.Module):
    def __init__(self, d_in, pooling="gated"):
        super().__init__()
        self.pooling = pooling
        self.proj = nn.Sequential(
            nn.Linear(d_in, D_HID), nn.LayerNorm(D_HID), nn.ReLU(), nn.Dropout(DROPOUT)
        )
        if pooling in ("gated", "topk"):
            self.V = nn.Linear(D_HID, D_ATT)
            self.U = nn.Linear(D_HID, D_ATT)
            self.w = nn.Linear(D_ATT, 1)
        self.head = nn.Linear(D_HID, 1)

    def scores(self, h, mask):
        a = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))).squeeze(-1)
        return a.masked_fill(~mask, -1e4)

    def forward(self, x, mask):
        h = self.proj(x)                                  # B,T,D
        m = mask.unsqueeze(-1).float()
        att = None

        if self.pooling == "mean":
            z = (h * m).sum(1) / m.sum(1).clamp(min=1)
        elif self.pooling == "max":
            z = h.masked_fill(~mask.unsqueeze(-1), -1e4).max(1).values
        elif self.pooling == "topk":
            a = self.scores(h, mask)
            k = min(TOPK, int(mask.sum(1).min().item()))
            idx = a.topk(k, dim=1).indices
            z = torch.gather(h, 1, idx.unsqueeze(-1).expand(-1, -1, h.size(-1))).mean(1)
            att = torch.softmax(a, 1)
        else:                                             # gated attention
            a = self.scores(h, mask)
            att = torch.softmax(a, 1)
            z = (att.unsqueeze(-1) * h).sum(1)

        return self.head(z).squeeze(-1), att


# =====================================================
# DATA
# =====================================================
def load_bags():
    idx = pd.read_csv(PROC / "tile_index.csv")
    emb = np.load(PROC / "tile_embeddings.npy").astype(np.float32)

    order = idx.sort_values(["image_id", "tile_i"])
    images = order.image_id.unique()
    max_t = int(order.groupby("image_id").size().max())
    d = emb.shape[1]

    X = np.zeros((len(images), max_t, d), np.float32)
    M = np.zeros((len(images), max_t), bool)
    meta = []
    for i, im in enumerate(images):
        g = order[order.image_id == im]
        rows = g.emb_row.to_numpy()
        X[i, :len(rows)] = emb[rows]
        M[i, :len(rows)] = True
        meta.append({"image_id": im, "label": g.label.iloc[0],
                     "patient_key": g.patient_key.iloc[0],
                     "split": g.split.iloc[0], "n_tiles": len(rows)})
    return X, M, pd.DataFrame(meta), max_t, d


def run_epoch(model, opt, X, M, y, train=True):
    model.train(train)
    n = len(X)
    order = np.random.permutation(n) if train else np.arange(n)
    tot, out = 0.0, np.zeros(n)
    for i in range(0, n, BATCH):
        b = order[i:i + BATCH]
        xb = torch.from_numpy(X[b]).to(DEVICE)
        mb = torch.from_numpy(M[b]).to(DEVICE)
        yb = torch.from_numpy(y[b].astype(np.float32)).to(DEVICE)
        with torch.set_grad_enabled(train):
            logit, _ = model(xb, mb)
            loss = F.binary_cross_entropy_with_logits(logit, yb)
        if train:
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        tot += float(loss) * len(b)
        out[b] = torch.sigmoid(logit).detach().cpu().numpy()
    return tot / n, out


def fit_predict(Xtr, Mtr, ytr, gtr, Xte, Mte, pooling, seed, d):
    """Early stopping on an inner split carved from TRAIN only."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_i, va_i = next(gss.split(Xtr, ytr, groups=gtr))

    model = MIL(d, pooling).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    best_auc, best_state, bad = -np.inf, None, 0
    for ep in range(EPOCHS):
        run_epoch(model, opt, Xtr[tr_i], Mtr[tr_i], ytr[tr_i], train=True)
        _, pv = run_epoch(model, opt, Xtr[va_i], Mtr[va_i], ytr[va_i], train=False)
        auc = roc_auc_score(ytr[va_i], pv) if len(np.unique(ytr[va_i])) > 1 else 0.5
        if auc > best_auc:
            best_auc, bad = auc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    if best_state:
        model.load_state_dict(best_state)
    dummy = np.zeros(len(Xte))
    _, pte = run_epoch(model, opt, Xte, Mte, dummy, train=False)
    return pte


# =====================================================
# MAIN
# =====================================================
def main():
    X, M, meta, max_t, d = load_bags()
    y = meta.label.to_numpy()
    groups = meta.patient_key.to_numpy()
    print(f"{len(meta)} bags | max {max_t} tiles | dim {d} | device {DEVICE}")

    results, preds = [], {}
    for pooling in POOLINGS:
        aucs = []
        for seed in range(N_REPEATS):
            cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
            oof = np.zeros(len(y))
            for tr, te in cv.split(X, y, groups):
                oof[te] = fit_predict(X[tr], M[tr], y[tr], groups[tr],
                                      X[te], M[te], pooling, seed, d)
            aucs.append(roc_auc_score(y, oof))
            if seed == 0:
                preds[pooling] = oof
        a = np.array(aucs)
        results.append({"pooling": pooling, "auc": round(a.mean(), 4),
                        "sd": round(a.std(), 4),
                        "lo": round(a.min(), 4), "hi": round(a.max(), 4)})
        print(f"  {pooling:6s} AUC = {a.mean():.4f} +/- {a.std():.4f}")

    res = pd.DataFrame(results)
    np.savez(ROOT / "outputs" / "manifests" / "mil_predictions.npz",
             y=y, image_id=meta.image_id.to_numpy(), **preds)

    BASE = 0.747
    BASE_SD = 0.013
    best = res.loc[res.auc.idxmax()]

    L = ["# Phase 3 - Attention MIL over Native-Resolution Tiles\n"]
    L.append(f"{len(meta)} bags, up to {max_t} tiles each, {d}-d frozen DINOv2 embeddings.")
    L.append(f"Patient-grouped {N_FOLDS}-fold CV repeated over {N_REPEATS} fold "
             f"assignments; early stopping on an inner split of the training folds only.\n")
    L.append(res.to_markdown(index=False))
    L.append("")
    L.append(f"**Baseline to beat (Phase 1c, whole-eye pooled): {BASE:.3f} +/- {BASE_SD:.3f}**\n")

    mean_auc = float(res.loc[res.pooling == "mean", "auc"].iloc[0])
    att_auc = float(res.loc[res.pooling == "gated", "auc"].iloc[0])
    L.append("## Reading\n")
    if best.auc <= BASE + BASE_SD:
        L.append(f"Best pooling reaches **{best.auc:.3f} +/- {best.sd:.3f}**, which does not "
                 f"clear the whole-eye baseline of {BASE:.3f}. **The localisation premise is "
                 f"not supported by this experiment.** Either the signal is genuinely global, "
                 f"or 224px tiles are the wrong scale, or dense uniform coverage dilutes the "
                 f"lesion too far for attention to recover it. Next step is a scale sweep "
                 f"before adding any more machinery.")
    elif mean_auc > BASE + BASE_SD and att_auc - mean_auc < 0.01:
        L.append(f"Mean pooling alone reaches **{mean_auc:.3f}**, above the {BASE:.3f} "
                 f"baseline, and attention adds little on top ({att_auc:.3f}). "
                 f"**The win is native resolution, not localisation.** Tiles at 1:1 carry "
                 f"texture that a downsampled whole-eye view destroys; the model does not "
                 f"need to know where to look.")
    else:
        L.append(f"Attention reaches **{att_auc:.3f}** against **{mean_auc:.3f}** for mean "
                 f"pooling and **{BASE:.3f}** for the whole-eye baseline. **The win is "
                 f"localisation.** The signal sits in a minority of tiles and is diluted by "
                 f"averaging - which explains directly why the global view plateaued at "
                 f"{BASE:.3f} regardless of resolution or framing.")
    L.append("")
    L.append("## Next\n")
    L.append("- map attention back onto the cornea via `tile_index.csv` (x, y, r_norm, theta) "
             "and check whether it concentrates on the infiltrate margin. If it does, that is "
             "lesion localisation obtained without a lesion segmenter.")
    L.append("- sweep tile scale, and add overlap, before adding model capacity.")
    L.append("- only then: calibration, abstention, and the operating point at 90% fungal recall.\n")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "07_mil.md").write_text("\n".join(L), encoding="utf-8")
    print(f"\n{res.to_string(index=False)}")
    print(f"\nwrote {REPORT_DIR / '07_mil.md'}")


if __name__ == "__main__":
    main()
