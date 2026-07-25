"""
Can we get BOTH classes to ~85% recall AND ~85% precision at once?

Rebuilds the pooled honest predictions (dev OOF + locked test + external = 1644,
same set the final calibration used), then searches every abstention band
(lo, hi) to see the best simultaneously-achievable point.

Reports recall two ways, because with abstention they diverge:
  recall_all      = correct-class calls / all of that class      (abstained hurt)
  recall_covered  = correct-class calls / that class among covered cases
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json, numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
from sklearn.metrics import roc_auc_score

ROOT = Path(".")
CKPT = ROOT / "outputs" / "checkpoints"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [0, 1, 2]


class M(nn.Module):
    def __init__(s):
        super().__init__(); s.proj = nn.Sequential(nn.Linear(384,192), nn.LayerNorm(192), nn.ReLU(), nn.Dropout(0.25)); s.head = nn.Linear(192,1)
    def forward(s, x, m):
        h = s.proj(x); mm = m.unsqueeze(-1).float(); return s.head((h*mm).sum(1)/mm.sum(1).clamp(min=1)).squeeze(-1)


def bags(split):
    idx = pd.read_csv("data/processed/tile_index_v2.csv")
    idx = idx[(idx.label < 2) & (idx.split == split)].sort_values(["image_id", "emb_row"])
    emb = np.load("data/processed/tile_embeddings_v2.npy").astype(np.float32)
    return [(emb[g.emb_row.to_numpy()], int(g.label.iloc[0]), int(g.fold.iloc[0]))
            for _, g in idx.groupby("image_id")]


def raw_logit(heads, feats):
    F_ = torch.from_numpy(feats).unsqueeze(0).to(DEV)
    msk = torch.ones(1, F_.shape[1], dtype=torch.bool, device=DEV)
    with torch.no_grad():
        return float(np.mean([h(F_, msk).item() for h in heads]))


def pooled():
    ck = torch.load(CKPT / "final_model_v2.pt", map_location=DEV, weights_only=False)
    cal = json.load(open(CKPT / "calibration_binary_v2.json")); T = cal["temperature"]
    heads = []
    for sd in ck["state_dicts"]:
        m = M().to(DEV); m.load_state_dict(sd); m.eval(); heads.append(m)
    y, lg = [], []
    for feats, lab, fold in bags("dev"):
        mem = [heads[fold*len(SEEDS)+s] for s in range(len(SEEDS))]
        y.append(lab); lg.append(raw_logit(mem, feats))
    for feats, lab, _ in bags("test"):
        y.append(lab); lg.append(raw_logit(heads, feats))
    ext = pd.read_csv(ROOT / "outputs" / "_ext_pred_v2.csv")
    Told = cal["temperature"]  # ext cache already at this T; recover logit then re-apply T
    for _, r in ext.iterrows():
        p = min(max(float(r.p), 1e-6), 1-1e-6); y.append(int(r.y)); lg.append(Told*np.log(p/(1-p)))
    y = np.array(y); p = 1/(1+np.exp(-np.array(lg)/T))
    return y, p


def metrics(y, p, lo, hi):
    dec = np.where(p >= hi, 1, np.where(p <= lo, 0, -1))
    fung = y == 1
    cov = (dec >= 0)
    nf, nb = fung.sum(), (~fung).sum()
    nf_cov, nb_cov = (fung & cov).sum(), ((~fung) & cov).sum()
    tp_f, tp_b = ((dec == 1) & fung).sum(), ((dec == 0) & (~fung)).sum()
    return dict(
        lo=lo, hi=hi, coverage=cov.mean(),
        fun_r_all=tp_f/nf, fun_r_cov=tp_f/max(nf_cov, 1),
        fun_p=tp_f/max((dec == 1).sum(), 1),
        bac_r_all=tp_b/nb, bac_r_cov=tp_b/max(nb_cov, 1),
        bac_p=tp_b/max((dec == 0).sum(), 1),
        misroute=((dec == 0) & fung).sum()/nf,
        n_f_call=(dec == 1).sum(), n_b_call=(dec == 0).sum())


def main():
    y, p = pooled()
    print(f"pooled n={len(y)}  ({(y==0).sum()} bac / {(y==1).sum()} fun)  AUC {roc_auc_score(y,p):.4f}\n")

    grid = np.round(np.unique(np.concatenate([p, [0.0, 1.0]])), 3)
    TARGET = 0.85
    MINCALL = 10

    # ---- (A) no abstention: single threshold, recall_all == recall_cov ----
    print("A) single threshold (no abstain) — best min-of-4 metric:")
    bestA = None
    for t in grid:
        m = metrics(y, p, t, t)
        four = min(m["fun_r_all"], m["fun_p"], m["bac_r_all"], m["bac_p"])
        if bestA is None or four > bestA[0]:
            bestA = (four, t, m)
    m = bestA[1]; mm = bestA[2]
    print(f"   t={m:.3f}  fun_r {mm['fun_r_all']:.0%} fun_p {mm['fun_p']:.0%} "
          f"bac_r {mm['bac_r_all']:.0%} bac_p {mm['bac_p']:.0%}  "
          f"-> worst arm {bestA[0]:.0%}\n")

    # ---- (B) abstain: max coverage s.t. all four (recall_all) >= 0.85 ----
    print(f"B) abstain band with ALL FOUR (recall-over-all) >= {TARGET:.0%}, max coverage:")
    bestB = None
    for hi in grid:
        for lo in grid[grid <= hi]:
            m = metrics(y, p, lo, hi)
            if m["n_f_call"] < MINCALL or m["n_b_call"] < MINCALL: continue
            if min(m["fun_r_all"], m["fun_p"], m["bac_r_all"], m["bac_p"]) >= TARGET:
                if bestB is None or m["coverage"] > bestB["coverage"]:
                    bestB = m
    if bestB:
        print(f"   band [{bestB['lo']:.2f},{bestB['hi']:.2f}]  coverage {bestB['coverage']:.0%}  "
              f"fun_r {bestB['fun_r_all']:.0%} fun_p {bestB['fun_p']:.0%} "
              f"bac_r {bestB['bac_r_all']:.0%} bac_p {bestB['bac_p']:.0%}")
    else:
        print("   NOT ACHIEVABLE at any band (recall counted over all cases).")

    # ---- (C) abstain: all four with recall-OVER-COVERED >= 0.85, max coverage
    print(f"\nC) abstain band with ALL FOUR (recall-over-covered) >= {TARGET:.0%}, max coverage:")
    bestC = None
    for hi in grid:
        for lo in grid[grid <= hi]:
            m = metrics(y, p, lo, hi)
            if m["n_f_call"] < MINCALL or m["n_b_call"] < MINCALL: continue
            if min(m["fun_r_cov"], m["fun_p"], m["bac_r_cov"], m["bac_p"]) >= TARGET:
                if bestC is None or m["coverage"] > bestC["coverage"]:
                    bestC = m
    if bestC:
        print(f"   band [{bestC['lo']:.2f},{bestC['hi']:.2f}]  coverage {bestC['coverage']:.0%}  "
              f"fun_r(cov) {bestC['fun_r_cov']:.0%} fun_p {bestC['fun_p']:.0%} "
              f"bac_r(cov) {bestC['bac_r_cov']:.0%} bac_p {bestC['bac_p']:.0%}")
    else:
        print("   NOT ACHIEVABLE at any band (recall over covered cases).")


if __name__ == "__main__":
    main()
