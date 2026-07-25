"""
Step 1 - Calibrate the retrained binary model and set operating points.

The retrained model gained AUC (0.815->0.850) but at the default 0.5 threshold it
leans bacterial (fungal recall 71%, fungal->bacterial 29%). That is a threshold
choice, not a model flaw - fixed here by setting fungal-protective operating
points from honest out-of-fold probabilities (each fold scored only by the
ensemble members that excluded it).

Modes:
  balanced       t = 0.50
  fungal_safety  lowest-cost t with fungal recall >= 0.90 (protects the dangerous arm)
  selective      abstain band for >=80% precision on both arms

Outputs
    outputs/checkpoints/calibration_binary_v2.json
    outputs/reports/28_calibration_binary_v2.md
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
CKPT = ROOT / "outputs" / "checkpoints"
REPORT = ROOT / "outputs" / "reports" / "28_calibration_binary_v2.md"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
D_IN, D_HID, DROP = 384, 192, 0.25
N_FOLDS, SEEDS = 5, [0, 1, 2]
TARGET_FUNGAL_RECALL = 0.90
MIN_PREC = 0.80


class MILMean(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(D_IN, D_HID), nn.LayerNorm(D_HID),
                                  nn.ReLU(), nn.Dropout(DROP))
        self.head = nn.Linear(D_HID, 1)

    def forward(self, x, m):
        h = self.proj(x); mm = m.unsqueeze(-1).float()
        return self.head((h * mm).sum(1) / mm.sum(1).clamp(min=1)).squeeze(-1)


def bags():
    idx = pd.read_csv(PROC / "tile_index_v2.csv")
    idx = idx[idx.label < 2].sort_values(["image_id", "emb_row"])
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


def logit(model, X, M, bs=64):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i+bs]).to(DEVICE); mb = torch.from_numpy(M[i:i+bs]).to(DEVICE)
            out.append(model(xb, mb).cpu().numpy())
    return np.concatenate(out)


def main():
    X, M, meta = bags()
    dev = (meta.split == "dev").to_numpy(); y = meta.label.to_numpy()
    ck = torch.load(CKPT / "final_model_v2.pt", map_location=DEVICE, weights_only=False)
    models = []
    for sd in ck["state_dicts"]:
        m = MILMean().to(DEVICE); m.load_state_dict(sd); models.append(m)

    # OOF: fold-major ensemble, member f*|S|+s trained excluding fold f
    oof = np.zeros(len(y))
    for f in range(N_FOLDS):
        rows = np.where(dev & (meta.fold == f))[0]
        if not len(rows): continue
        mem = [models[f*len(SEEDS)+s] for s in range(len(SEEDS))]
        oof[rows] = np.mean([logit(m, X[rows], M[rows]) for m in mem], 0)

    di = np.where(dev)[0]; yd = y[di]; Ld = oof[di]
    T = torch.nn.Parameter(torch.ones(1)); lg = torch.from_numpy(Ld).float()
    yy = torch.from_numpy(yd.astype(np.float32)); opt = torch.optim.LBFGS([T], lr=0.1, max_iter=100)
    def c(): opt.zero_grad(); l = F.binary_cross_entropy_with_logits(lg/T.clamp(min=1e-2), yy); l.backward(); return l
    opt.step(c); T = float(T.detach().clamp(min=1e-2))
    p = 1/(1+np.exp(-Ld/T))
    auc = roc_auc_score(yd, p)

    def metrics(dec):   # dec: 1 fung, 0 bact, -1 abstain
        fung = yd == 1
        fr = ((dec == 1) & fung).sum()/max(fung.sum(), 1)
        br = ((dec == 0) & (~fung)).sum()/max((~fung).sum(), 1)
        mis = ((dec == 0) & fung).sum()/max(fung.sum(), 1)
        cov = (dec >= 0).mean()
        fp = ((dec == 1) & fung).sum()/max((dec == 1).sum(), 1)
        bp = ((dec == 0) & (~fung)).sum()/max((dec == 0).sum(), 1)
        return dict(coverage=cov, fungal_recall=fr, bacterial_recall=br, misroute=mis,
                    fungal_prec=fp, bacterial_prec=bp)

    # balanced
    bal = metrics((p >= 0.5).astype(int))
    # fungal_safety: lowest t (protect fungal) with fungal recall >= target
    grid = np.round(np.linspace(0.05, 0.6, 56), 3)
    fs_t = 0.5
    for t in grid:
        if metrics((p >= t).astype(int))["fungal_recall"] >= TARGET_FUNGAL_RECALL:
            fs_t = float(t)     # take the highest such t (least bacterial cost)
    fs = metrics((p >= fs_t).astype(int))
    # selective: max coverage with both precisions >= MIN_PREC
    best = None
    g = np.unique(np.round(np.concatenate([p, [0, 1]]), 3))
    for hi in g:
        for lo in g[g <= hi]:
            dec = np.where(p >= hi, 1, np.where(p <= lo, 0, -1))
            if (dec == 1).sum() < 8 or (dec == 0).sum() < 8: continue
            mm = metrics(dec)
            if mm["fungal_prec"] >= MIN_PREC and mm["bacterial_prec"] >= MIN_PREC:
                if best is None or mm["coverage"] > best[2]["coverage"]:
                    best = (float(lo), float(hi), mm)
    sel = best

    cal = {"temperature": T, "dev_auc": float(auc),
           "modes": {
               "balanced": {"kind": "forced", "t": 0.5, **{k: float(v) for k, v in bal.items()}},
               "fungal_safety": {"kind": "forced", "t": fs_t, **{k: float(v) for k, v in fs.items()}},
           }}
    if sel:
        cal["modes"]["selective"] = {"kind": "abstain", "lo": sel[0], "hi": sel[1],
                                     **{k: float(v) for k, v in sel[2].items()}}
    cal["default_mode"] = "fungal_safety"
    (CKPT / "calibration_binary_v2.json").write_text(json.dumps(cal, indent=2))

    L = ["# Binary v2 — calibration & operating points\n"]
    L.append(f"Dev OOF AUC **{auc:.4f}**, temperature {T:.3f}. Default mode **fungal_safety** "
             f"(protects the dangerous arm).\n")
    L.append("| mode | threshold | coverage | fungal recall | bacterial recall | misroute | fungal prec | bact prec |")
    L.append("|---|---|---|---|---|---|---|---|")
    for name in ["balanced", "fungal_safety", "selective"]:
        if name not in cal["modes"]: continue
        m = cal["modes"][name]
        thr = f"t={m['t']:.2f}" if m["kind"] == "forced" else f"[{m['lo']:.2f},{m['hi']:.2f}]"
        L.append(f"| {name} | {thr} | {m['coverage']:.0%} | {m['fungal_recall']:.0%} | "
                 f"{m['bacterial_recall']:.0%} | {m['misroute']:.0%} | {m['fungal_prec']:.0%} | "
                 f"{m['bacterial_prec']:.0%} |")
    L.append(f"\n**fungal_safety** brings fungal recall to {fs['fungal_recall']:.0%} and cuts "
             f"fungal→bacterial to {fs['misroute']:.0%} (vs {bal['misroute']:.0%} at t=0.5), "
             f"costing bacterial recall ({bal['bacterial_recall']:.0%}→{fs['bacterial_recall']:.0%}). "
             f"These are dev numbers; the locked test confirms them next.")
    REPORT.write_text("\n".join(L), encoding="utf-8")
    print(f"dev AUC {auc:.4f} T {T:.3f}")
    print(f"balanced: fun_rec {bal['fungal_recall']:.0%} mis {bal['misroute']:.0%}")
    print(f"fungal_safety t={fs_t:.2f}: fun_rec {fs['fungal_recall']:.0%} mis {fs['misroute']:.0%} "
          f"bac_rec {fs['bacterial_recall']:.0%}")
    if sel: print(f"selective [{sel[0]:.2f},{sel[1]:.2f}]: cov {sel[2]['coverage']:.0%}")
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
