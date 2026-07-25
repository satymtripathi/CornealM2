"""
Train a lesion / sign segmenter from LabelMe polygons.  (Option A: two focused
models, each trained only on data where its classes are fully labelled.)

    python src/18_train_segmenter.py lesion    # infiltrate/cellularity/hypopyon/glare (+eye/limbus)
    python src/18_train_segmenter.py fungal     # feathery_margin / raised_edges

Design choices, made once so we do not revisit:

  * Reuse, don't replace, the limbus model. This trainer *also* learns eye/limbus
    as auxiliary classes (they stabilise training and the labels are free), but
    the validated limbus checkpoint stays the ROI source downstream.

  * Glare is a class. It is the most abundant annotation and stops specular
    highlights on the wet lesion being read as infiltrate/hypopyon.

  * Painter order matters - the polygons are nested. Later classes overwrite
    earlier where they overlap: eye < limbus < cellularity < infiltrate <
    hypopyon < glare  (dense core sits on top of its halo; glare on top of all).

  * NO vertical flip. The hypopyon is gravity-dependent and always inferior;
    flipping it to the top teaches a false invariance. HFlip + small rotation
    + photometric jitter only.

  * Patient-grouped split - the same rule as the classifier, so a patient never
    spans train and val.

Outputs
    models/<name>_seg/<name>_unetpp.pth   (state_dict + config)
    outputs/reports/18_<name>_segmenter.md
    outputs/figures/<name>_seg_qc/*.jpg
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import re
import sys
import json
import glob
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "DataM2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 512
SEED = 42

# painter order per config (index = class id; 0 = background)
CONFIGS = {
    "lesion": {
        "folder": DATA / "Infection",
        "classes": ["background", "eye", "limbus", "cellularity",
                    "infiltrate", "hypopyon", "glare"],
        "aliases": {"light": "glare"},
        "epochs": 90, "batch": 4,
        "downstream": ["infiltrate", "hypopyon", "cellularity", "glare"],
    },
    "fungal": {
        "folder": DATA / "Fungal_Infection",
        "classes": ["background", "feathery_margin", "raised_edges"],
        "aliases": {},
        "epochs": 120, "batch": 4,
        "downstream": ["feathery_margin"],
    },
}

IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD = (0.229, 0.224, 0.225)


def patient_key(stem: str) -> str:
    prefix = stem.split("__")[0] if "__" in stem else re.split(r"_[Ii]maging", stem)[0]
    m = re.match(r"^(VC-[A-Za-z]{2,4}-[A-Za-z]{0,3}\d{3,10})", prefix) \
        or re.match(r"^([A-Za-z]{2,5}-[A-Za-z]{0,3}\d{3,10})", prefix) \
        or re.match(r"^([A-Za-z]{0,3}\d{4,10})", prefix)
    return (m.group(1).upper() if m else prefix.strip().upper())


def find_pairs(folder: Path):
    pairs = []
    for jf in sorted(folder.glob("*.json")):
        img = None
        for ext in (".jpg", ".JPG", ".jpeg", ".png"):
            if (folder / (jf.stem + ext)).exists():
                img = folder / (jf.stem + ext); break
        if img:
            pairs.append((img, jf))
    return pairs


def rasterize(jf: Path, classes, aliases, out_hw):
    d = json.load(open(jf, encoding="utf-8"))
    W, H = d.get("imageWidth"), d.get("imageHeight")
    cls_id = {c: i for i, c in enumerate(classes)}
    mask = np.zeros((H, W), np.uint8)
    # paint in class order so later (higher-id) classes overwrite earlier
    by_cls = {c: [] for c in classes}
    for s in d["shapes"]:
        lab = s["label"].strip().lower()
        lab = aliases.get(lab, lab)
        if lab in cls_id and s["shape_type"] in ("polygon", "rectangle"):
            pts = np.array(s["points"], np.int32)
            if s["shape_type"] == "rectangle" and len(pts) == 2:
                (x0, y0), (x1, y1) = pts
                pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.int32)
            if len(pts) >= 3:
                by_cls[lab].append(pts)
    for c in classes[1:]:                       # skip background
        for pts in by_cls[c]:
            cv2.fillPoly(mask, [pts], cls_id[c])
    mask = cv2.resize(mask, out_hw[::-1], interpolation=cv2.INTER_NEAREST)
    return mask


def build_cache(pairs, cfg, which):
    """
    One-time: decode each 20 MP image + rasterize polygons, resize both to
    IMG_SIZE, store as a small npz. Without this, training is IO-bound (~10
    min/epoch decoding full-res); with it, epochs are GPU-bound (seconds).
    """
    cdir = ROOT / "data" / "interim" / f"seg_cache_{which}_{IMG_SIZE}"
    cdir.mkdir(parents=True, exist_ok=True)
    out = []
    todo = [(p, jf, cdir / f"{p.stem}.npz") for p, jf in pairs]
    miss = [t for t in todo if not t[2].exists()]
    if miss:
        for img_p, jf, cp in tqdm(miss, desc=f"caching {which}"):
            rgb = cv2.cvtColor(cv2.imread(str(img_p)), cv2.COLOR_BGR2RGB)
            m = rasterize(jf, cfg["classes"], cfg["aliases"], rgb.shape[:2])
            rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            m = cv2.resize(m, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
            np.savez_compressed(cp, img=rgb, mask=m)
    return [cp for _, _, cp in todo]


class SegDS(Dataset):
    def __init__(self, cache_files, train):
        self.files = cache_files
        aug = ([A.HorizontalFlip(p=0.5),                 # nasal/temporal symmetric
                A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT),
                A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
                A.RandomGamma((80, 120), p=0.3),
                A.HueSaturationValue(8, 12, 8, p=0.3)] if train else [])
        self.tf = A.Compose(aug + [A.Normalize(IMNET_MEAN, IMNET_STD), ToTensorV2()])

    def __len__(self): return len(self.files)

    def __getitem__(self, i):
        z = np.load(self.files[i])
        a = self.tf(image=z["img"], mask=z["mask"])
        return a["image"], a["mask"].long()


def dice_per_class(logits, target, n_cls, eps=1e-6):
    pred = logits.argmax(1)
    out = []
    for c in range(n_cls):
        p, t = (pred == c), (target == c)
        inter = (p & t).sum().float()
        denom = p.sum().float() + t.sum().float()
        out.append(((2 * inter + eps) / (denom + eps)).item() if denom > 0 else float("nan"))
    return out


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "lesion"
    cfg = CONFIGS[which]
    classes = cfg["classes"]; n_cls = len(classes)

    pairs = find_pairs(cfg["folder"])
    groups = np.array([patient_key(p.stem) for p, _ in pairs])
    print(f"[{which}] {len(pairs)} pairs, {len(set(groups))} patients, {n_cls} classes")

    cache = build_cache(pairs, cfg, which)          # one-time 512px cache
    tr_i, va_i = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED)
                      .split(pairs, groups=groups))
    tr = [cache[i] for i in tr_i]; va = [cache[i] for i in va_i]
    va_pairs = [pairs[i] for i in va_i]             # originals, for QC overlays
    print(f"  train {len(tr)} / val {len(va)}")

    nw = 4 if DEVICE == "cuda" else 0
    dl_tr = DataLoader(SegDS(tr, True), batch_size=cfg["batch"], shuffle=True,
                       num_workers=nw, drop_last=True, persistent_workers=(nw > 0))
    dl_va = DataLoader(SegDS(va, False), batch_size=cfg["batch"], shuffle=False,
                       num_workers=nw, persistent_workers=(nw > 0))

    model = smp.UnetPlusPlus("timm-efficientnet-b0", encoder_weights="imagenet",
                             in_channels=3, classes=n_cls).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"])
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))
    dice_loss = smp.losses.DiceLoss(mode="multiclass")
    ce = nn.CrossEntropyLoss()

    # downstream (lesion) classes are what we care about; mean their Dice for early stop
    watch = [classes.index(c) for c in cfg["downstream"]]

    best, best_state, bad = -1, None, 0
    for ep in range(cfg["epochs"]):
        model.train()
        for x, y in dl_tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                out = model(x)
                loss = 0.5 * ce(out, y) + 0.5 * dice_loss(out, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sched.step()

        model.eval(); accum = np.zeros(n_cls); cnt = np.zeros(n_cls)
        with torch.no_grad():
            for x, y in dl_va:
                x, y = x.to(DEVICE), y.to(DEVICE)
                with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                    out = model(x)
                dd = dice_per_class(out, y, n_cls)
                for c, v in enumerate(dd):
                    if not np.isnan(v): accum[c] += v; cnt[c] += 1
        dice = accum / np.maximum(cnt, 1)
        watch_dice = float(np.mean([dice[c] for c in watch]))
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  ep {ep+1:3d}  watch Dice {watch_dice:.4f}  | " +
                  " ".join(f"{classes[c][:4]}={dice[c]:.2f}" for c in range(1, n_cls)))
        if watch_dice > best:
            best, bad = watch_dice, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_dice = dice.copy()
        else:
            bad += 1
            if bad >= 20:
                print(f"  early stop @ ep {ep+1}"); break

    # ---------- save ----------
    outdir = ROOT / "models" / f"{which}_seg"
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state,
                "config": {"encoder_name": "timm-efficientnet-b0", "classes": classes,
                           "aliases": cfg["aliases"], "img_size": IMG_SIZE,
                           "downstream": cfg["downstream"]}},
               outdir / f"{which}_unetpp.pth")

    # ---------- QC overlays ----------
    qc = ROOT / "outputs" / "figures" / f"{which}_seg_qc"; qc.mkdir(parents=True, exist_ok=True)
    model.load_state_dict(best_state); model.eval()
    palette = np.array([[0, 0, 0], [0, 180, 0], [0, 255, 255], [255, 180, 0],
                        [255, 0, 0], [255, 0, 255], [255, 255, 255]], np.uint8)
    for k, (img_p, jf) in enumerate(va_pairs[:8]):
        rgb = cv2.cvtColor(cv2.imread(str(img_p)), cv2.COLOR_BGR2RGB)
        t = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), A.Normalize(IMNET_MEAN, IMNET_STD),
                       ToTensorV2()])(image=rgb)["image"].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pr = model(t).argmax(1)[0].cpu().numpy().astype(np.uint8)
        col = palette[pr]
        base = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
        vis = cv2.addWeighted(base, 0.6, col, 0.4, 0)
        cv2.imwrite(str(qc / f"{which}_{k}.jpg"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    # ---------- report ----------
    L = [f"# Lesion segmenter — `{which}`\n"]
    L.append(f"UNet++ / timm-efficientnet-b0, {IMG_SIZE}², {n_cls} classes. "
             f"{len(pairs)} images / {len(set(groups))} patients, patient-grouped 80/20.\n")
    L.append("| class | val Dice |\n|---|---|")
    for c in range(1, n_cls):
        L.append(f"| {classes[c]} | {best_dice[c]:.4f} |")
    L.append(f"\n**Watched (downstream) mean Dice: {best:.4f}**\n")
    L.append("Downstream use: " + ", ".join(cfg["downstream"]) + ".")
    L.append(f"\nQC overlays: `outputs/figures/{which}_seg_qc/`  "
             "(green=eye, cyan=limbus, orange=cellularity, red=infiltrate, "
             "magenta=hypopyon, white=glare).")
    (ROOT / "outputs" / "reports" / f"18_{which}_segmenter.md").write_text(
        "\n".join(L), encoding="utf-8")

    print(f"\n[{which}] best watch Dice {best:.4f}")
    for c in range(1, n_cls):
        print(f"    {classes[c]:14s} {best_dice[c]:.4f}")
    print(f"saved {outdir / f'{which}_unetpp.pth'}")


if __name__ == "__main__":
    main()
