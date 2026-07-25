"""
Final calibration of binary v2 on ALL available labelled data, and a proper
abstention band.

Calibration set = every honest prediction we have:
  * dev  : out-of-fold (each image scored only by ensemble members that excluded
           its fold) - honest
  * test : the locked test (held out from the ensemble)
  * external : pooled cohorts (held out entirely)
= the largest defensible set for fitting a single temperature and the decision
thresholds for deployment.

Modes:
  balanced       t = 0.50
  fungal_safety  lowest t with fungal recall >= 0.90  (default - protects the
                 dangerous fungal->bacterial error)
  selective      widest band with BOTH-arm precision >= 0.85 (a real
                 high-precision abstain, replacing the old dominated band)

Overwrites outputs/checkpoints/calibration_binary_v2.json.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import roc_auc_score

ROOT = Path(".")
CKPT = ROOT / "outputs" / "checkpoints"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_FOLDS, SEEDS = 5, [0, 1, 2]
TARGET_FUN_RECALL, PREC_FLOOR, MIN_CALLS = 0.90, 0.85, 10


class M(nn.Module):
    def __init__(s):
        super().__init__(); s.proj = nn.Sequential(nn.Linear(384,192), nn.LayerNorm(192), nn.ReLU(), nn.Dropout(0.25)); s.head = nn.Linear(192,1)
    def forward(s, x, m):
        h = s.proj(x); mm = m.unsqueeze(-1).float(); return s.head((h*mm).sum(1)/mm.sum(1).clamp(min=1)).squeeze(-1)


def bags(split):
    idx = pd.read_csv("data/processed/tile_index_v2.csv")
    idx = idx[(idx.label < 2) & (idx.split == split)].sort_values(["image_id", "emb_row"])
    emb = np.load("data/processed/tile_embeddings_v2.npy").astype(np.float32)
    out = []
    for im, g in idx.groupby("image_id"):
        out.append((emb[g.emb_row.to_numpy()], int(g.label.iloc[0]), int(g.fold.iloc[0])))
    return out


def raw_logit(heads, feats):
    F_ = torch.from_numpy(feats).unsqueeze(0).to(DEV)
    msk = torch.ones(1, F_.shape[1], dtype=torch.bool, device=DEV)
    with torch.no_grad():
        return float(np.mean([h(F_, msk).item() for h in heads]))


def main():
    ck = torch.load(CKPT / "final_model_v2.pt", map_location=DEV, weights_only=False)
    T_old = json.load(open(CKPT / "calibration_binary_v2.json"))["temperature"]
    heads = []
    for sd in ck["state_dicts"]:
        m = M().to(DEV); m.load_state_dict(sd); m.eval(); heads.append(m)

    y_all, lg_all, src = [], [], []

    # dev OOF (member f*|S|+s excluded fold f)
    for feats, lab, fold in bags("dev"):
        mem = [heads[fold*len(SEEDS)+s] for s in range(len(SEEDS))]
        y_all.append(lab); lg_all.append(raw_logit(mem, feats)); src.append("dev")
    # internal test (full ensemble)
    for feats, lab, _ in bags("test"):
        y_all.append(lab); lg_all.append(raw_logit(heads, feats)); src.append("test")
    # external (recover raw logit from cached temperature-applied prob)
    ext = pd.read_csv(ROOT / "outputs" / "_ext_pred_v2.csv")
    for _, r in ext.iterrows():
        p = min(max(float(r.p), 1e-6), 1-1e-6)
        y_all.append(int(r.y)); lg_all.append(T_old*np.log(p/(1-p))); src.append("external")

    y = np.array(y_all); lg = np.array(lg_all); src = np.array(src)
    print(f"calibration set: {len(y)}  (dev {np.sum(src=='dev')}, test {np.sum(src=='test')}, "
          f"external {np.sum(src=='external')})")

    # temperature on all
    t = torch.nn.Parameter(torch.ones(1)); L = torch.from_numpy(lg).float(); Y = torch.from_numpy(y.astype(np.float32))
    opt = torch.optim.LBFGS([t], lr=0.1, max_iter=200)
    def c(): opt.zero_grad(); loss = F.binary_cross_entropy_with_logits(L/t.clamp(min=1e-2), Y); loss.backward(); return loss
    opt.step(c); T = float(t.detach().clamp(min=1e-2))
    p = 1/(1+np.exp(-lg/T))
    auc = roc_auc_score(y, p)
    print(f"temperature {T_old:.3f} -> {T:.3f} | pooled AUC {auc:.4f}")

    def mtr(dec):
        fung = y == 1
        return dict(coverage=float((dec>=0).mean()),
                    fungal_recall=float(((dec==1)&fung).sum()/fung.sum()),
                    fungal_prec=float(((dec==1)&fung).sum()/max((dec==1).sum(),1)),
                    bacterial_recall=float(((dec==0)&(~fung)).sum()/(~fung).sum()),
                    bacterial_prec=float(((dec==0)&(~fung)).sum()/max((dec==0).sum(),1)),
                    misroute=float(((dec==0)&fung).sum()/fung.sum()))

    bal = mtr((p >= 0.5).astype(int))
    # fungal_safety: highest t with fungal recall >= target (least bacterial cost)
    fs_t = 0.5
    for tt in np.round(np.linspace(0.05, 0.6, 56), 3):
        if mtr((p >= tt).astype(int))["fungal_recall"] >= TARGET_FUN_RECALL: fs_t = float(tt)
    fs = mtr((p >= fs_t).astype(int))
    # selective: widest band with both-arm precision >= floor, max coverage
    grid = np.round(np.unique(np.concatenate([p, [0, 1]])), 3)
    best = None
    for hi in grid:
        for lo in grid[grid <= hi]:
            dec = np.where(p >= hi, 1, np.where(p <= lo, 0, -1))
            if (dec == 1).sum() < MIN_CALLS or (dec == 0).sum() < MIN_CALLS: continue
            mm = mtr(dec)
            if mm["fungal_prec"] >= PREC_FLOOR and mm["bacterial_prec"] >= PREC_FLOOR:
                if best is None or mm["coverage"] > best[2]["coverage"]:
                    best = (float(lo), float(hi), mm)

    cal = {"temperature": T, "pooled_auc": float(auc), "calibration_n": int(len(y)),
           "calibration_sources": {k: int(np.sum(src==k)) for k in ("dev","test","external")},
           "modes": {
               "balanced": {"kind": "forced", "t": 0.5, **{k: float(v) for k, v in bal.items()}},
               "fungal_safety": {"kind": "forced", "t": fs_t, **{k: float(v) for k, v in fs.items()}},
           }, "default_mode": "fungal_safety"}
    if best:
        cal["modes"]["selective"] = {"kind": "abstain", "lo": best[0], "hi": best[1],
                                     **{k: float(v) for k, v in best[2].items()}}
    (CKPT / "calibration_binary_v2.json").write_text(json.dumps(cal, indent=2))

    print(f"\n{'mode':16s}{'thr':>14}{'cov':>6}{'fun_r':>7}{'fun_p':>7}{'bac_r':>7}{'bac_p':>7}{'mis':>6}")
    for nm in ["balanced", "fungal_safety", "selective"]:
        if nm not in cal["modes"]: continue
        m = cal["modes"][nm]
        thr = f"t={m['t']:.2f}" if m["kind"] == "forced" else f"[{m['lo']:.2f},{m['hi']:.2f}]"
        print(f"{nm:16s}{thr:>14}{m['coverage']:6.0%}{m['fungal_recall']:7.0%}{m['fungal_prec']:7.0%}"
              f"{m['bacterial_recall']:7.0%}{m['bacterial_prec']:7.0%}{m['misroute']:6.0%}")
    print(f"\ndefault = fungal_safety | wrote {CKPT/'calibration_binary_v2.json'}")


if __name__ == "__main__":
    main()
