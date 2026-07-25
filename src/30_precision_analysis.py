"""
Precision (with recall) for the binary v2 model across modes, internal + external,
plus a precision-vs-coverage sweep to see how far abstention can push precision.

Caches external per-image probabilities so re-runs are instant.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys, json, numpy as np, pandas as pd, torch, torch.nn as nn, cv2
from pathlib import Path
from PIL import Image
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "src")
Image.MAX_IMAGE_PIXELS = None
DEV = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = Path(".")
ck = torch.load("outputs/checkpoints/final_model_v2.pt", map_location=DEV, weights_only=False)
cal = json.load(open("outputs/checkpoints/calibration_binary_v2.json")); T = cal["temperature"]
sel, fs = cal["modes"]["selective"], cal["modes"]["fungal_safety"]


class M(nn.Module):
    def __init__(s):
        super().__init__(); s.proj = nn.Sequential(nn.Linear(384,192), nn.LayerNorm(192), nn.ReLU(), nn.Dropout(0.25)); s.head = nn.Linear(192,1)
    def forward(s, x, m):
        h = s.proj(x); mm = m.unsqueeze(-1).float(); return s.head((h*mm).sum(1)/mm.sum(1).clamp(min=1)).squeeze(-1)


heads = []
for sd in ck["state_dicts"]:
    m = M().to(DEV); m.load_state_dict(sd); m.eval(); heads.append(m)


def internal_probs():
    idx = pd.read_csv("data/processed/tile_index_v2.csv"); idx = idx[(idx.label < 2) & (idx.split == "test")]
    emb = np.load("data/processed/tile_embeddings_v2.npy").astype(np.float32)
    Y, P = [], []
    with torch.no_grad():
        for im, g in idx.groupby("image_id"):
            F_ = torch.from_numpy(emb[g.emb_row.to_numpy()]).unsqueeze(0).to(DEV)
            msk = torch.ones(1, F_.shape[1], dtype=torch.bool, device=DEV)
            lg = np.mean([h(F_, msk).item() for h in heads])
            P.append(1/(1+np.exp(-lg/T))); Y.append(int(g.label.iloc[0]))
    return np.array(Y), np.array(P)


def external_probs():
    cache = ROOT / "outputs" / "_ext_pred_v2.csv"
    if cache.exists():
        d = pd.read_csv(cache); return d.y.to_numpy(), d.p.to_numpy()
    from inference_3class import Pipeline3, CROP, INPUT, IMNET_MEAN, IMNET_STD, MAX_GLARE_FRAC
    base = Pipeline3.get()
    NORM = {"bacterial":"Bacterial","fungal":"Fungal","others":"Other","other":"Other"}
    def rgt(p):
        sep = "\t" if str(p).endswith((".tsv",".txt")) else ","
        d = pd.read_csv(p, sep=sep)
        f = [c for c in d.columns if any(k in c.lower() for k in ("file","name","image"))][0]
        l = [c for c in d.columns if c != f and any(k in c.lower() for k in ("gt","label","truth","class"))][0]
        d = d[[f,l]].copy(); d.columns = ["filename","gt"]; d["g"] = d["gt"].astype(str).str.strip().str.lower().map(NORM); return d
    def pf(rgb):
        c, mask = base._segment_limbus(rgb)
        if c is None: return None
        tiles = []
        for t in base._plan(mask):
            cr = rgb[t["y"]:t["y"]+CROP, t["x"]:t["x"]+CROP]
            if cr.shape[:2] != (CROP, CROP): continue
            hsv = cv2.cvtColor(cv2.resize(cr,(256,256),interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2HSV)
            if float(((hsv[...,2]/255.>.94)&(hsv[...,1]/255.<.15)).mean()) > MAX_GLARE_FRAC: continue
            cc = cv2.resize(cr,(INPUT,INPUT),interpolation=cv2.INTER_AREA)
            tiles.append(((cc.astype(np.float32)/255.-IMNET_MEAN)/IMNET_STD).transpose(2,0,1))
        if not tiles: return None
        with torch.no_grad():
            fe = []
            for i in range(0, len(tiles), 8):
                bt = torch.from_numpy(np.stack(tiles[i:i+8])).to(DEV)
                with torch.autocast("cuda", dtype=torch.float16, enabled=(DEV=="cuda")): fe.append(base.backbone(bt).float())
            F_ = torch.cat(fe,0).unsqueeze(0); msk = torch.ones(1, F_.shape[1], dtype=torch.bool, device=DEV)
            lg = np.mean([h(F_, msk).item() for h in heads])
        return 1/(1+np.exp(-lg/T))
    rows = []
    for dir_, gt in [(r"C:\Users\satyam.tripathi\Desktop\Model2Image_ReviewR\images","outputs/review_gt.tsv"),
                     (r"C:\Users\satyam.tripathi\Desktop\Model2ImageReview\images","outputs/review_gt_cohort02.tsv")]:
        g = rgt(gt)
        for p in sorted(Path(dir_).iterdir()):
            if p.suffix.lower() not in (".jpg",".jpeg",".png"): continue
            r = g[g.filename == p.name]
            if not len(r) or r.g.iloc[0] not in ("Bacterial","Fungal"): continue
            v = pf(np.asarray(Image.open(p).convert("RGB")))
            if v is not None: rows.append({"y": 1 if r.g.iloc[0]=="Fungal" else 0, "p": v})
    d = pd.DataFrame(rows); d.to_csv(cache, index=False)
    return d.y.to_numpy(), d.p.to_numpy()


def metrics(Y, P, dec):
    fung = Y == 1
    cf, cb = (dec == 1), (dec == 0)
    return dict(
        coverage=(dec >= 0).mean(),
        fungal_recall=((dec == 1) & fung).sum()/fung.sum(),
        fungal_prec=((dec == 1) & fung).sum()/max(cf.sum(), 1),
        bacterial_recall=((dec == 0) & (~fung)).sum()/(~fung).sum(),
        bacterial_prec=((dec == 0) & (~fung)).sum()/max(cb.sum(), 1),
        misroute=((dec == 0) & fung).sum()/fung.sum())


def show(tag, Y, P):
    print(f"\n=== {tag}: n={len(Y)} ({(Y==0).sum()} bac / {(Y==1).sum()} fun) | AUC {roc_auc_score(Y,P):.4f} ===")
    print(f"{'mode':16s}{'cov':>6}{'fun_rec':>9}{'fun_prec':>10}{'bac_rec':>9}{'bac_prec':>10}{'misroute':>10}")
    modes = [("Cautious", np.where(P>=sel['hi'],1,np.where(P<=sel['lo'],0,-1))),
             ("Fungal-safety", (P>=fs['t']).astype(int)),
             ("Balanced", (P>=0.5).astype(int))]
    for nm, dec in modes:
        d = metrics(Y, P, dec)
        print(f"{nm:16s}{d['coverage']:6.0%}{d['fungal_recall']:9.0%}{d['fungal_prec']:10.0%}"
              f"{d['bacterial_recall']:9.0%}{d['bacterial_prec']:10.0%}{d['misroute']:10.0%}")
    # precision-vs-coverage sweep (symmetric abstention band around 0.5)
    print("  precision reachable by abstaining (symmetric band, external-style):")
    print(f"    {'band':>14}{'coverage':>10}{'fun_prec':>10}{'bac_prec':>10}{'acc_cov':>9}")
    for w in [0.0, 0.10, 0.20, 0.30]:
        lo, hi = 0.5 - w, 0.5 + w
        dec = np.where(P >= hi, 1, np.where(P <= lo, 0, -1))
        if (dec == 1).sum() < 3 or (dec == 0).sum() < 3: continue
        d = metrics(Y, P, dec); acc = ((dec == Y) & (dec >= 0)).sum()/max((dec >= 0).sum(), 1)
        print(f"    [{lo:.2f},{hi:.2f}]{d['coverage']:10.0%}{d['fungal_prec']:10.0%}{d['bacterial_prec']:10.0%}{acc:9.0%}")


yI, pI = internal_probs(); show("INTERNAL locked test", yI, pI)
yE, pE = external_probs(); show("EXTERNAL pooled", yE, pE)
print("\nsource-prevalence note: at ~90% fungal clinic prevalence, fungal precision rises "
      "toward ~97% and bacterial precision falls (bacterial is rare).")
