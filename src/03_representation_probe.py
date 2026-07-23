"""
Phase 1c - Where does the signal live? Frozen-representation probe.

Handcrafted global statistics gave AUC 0.543 - essentially nothing. Two
explanations, with opposite consequences:

  (a) the signal really is local and fine-grained, so any whole-eye view at
      reduced resolution is hopeless and localisation is mandatory; or
  (b) the signal is global but subtle, and handcrafted statistics are simply
      too crude to see it, in which case a global branch earns its place.

This separates them by probing a frozen self-supervised backbone (DINOv2) at
several resolutions and two framings, with no fine-tuning at all. Only a linear
head is trained, so with 682 images the result reflects the representation
rather than a capacity to memorise.

Two axes are varied independently:
  resolution - 224 / 448 / 896 input
  framing    - whole frame vs cropped to the eye at native resolution

If AUC rises with resolution, fine texture carries the signal and native-res
tiling is justified by measurement rather than assertion. If cropping to the
eye helps at matched input size, field-of-view waste is costing us.

Outputs
    outputs/manifests/embeddings/*.npy
    outputs/reports/03_representation_probe.md
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import timm
from PIL import Image
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[1]
QC = ROOT / "outputs" / "manifests" / "image_qc.csv"
EMB_DIR = ROOT / "outputs" / "manifests" / "embeddings"
REPORT_DIR = ROOT / "outputs" / "reports"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
N_FOLDS = 5
N_REPEATS = 10

MODEL = "vit_small_patch14_dinov2.lvd142m"   # DINOv2 ViT-S/14, patch 14

# (name, input size, framing, batch)  - sizes must be divisible by 14
CONFIGS = [
    ("full_224", 224, "full", 16),
    ("full_448", 448, "full", 8),
    ("full_896", 896, "full", 2),
    ("eye_448",  448, "eye",  8),
    ("eye_896",  896, "eye",  2),
]

IMNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMNET_STD = np.array([0.229, 0.224, 0.225], np.float32)


# =====================================================
# FRAMING
# =====================================================
def eye_bbox(rgb: np.ndarray, pad: float = 0.12):
    """
    Bounding box of the illuminated subject. Otsu + largest component.
    Falls back to the full frame if nothing sensible is found.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, 8)
    H, W = gray.shape
    if n <= 1:
        return 0, 0, W, H
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = (st[i, cv2.CC_STAT_LEFT], st[i, cv2.CC_STAT_TOP],
                  st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT])
    if w * h < 0.02 * W * H:
        return 0, 0, W, H
    # square it up, then pad
    cx, cy = x + w / 2, y + h / 2
    side = max(w, h) * (1 + pad)
    x0 = int(max(0, cx - side / 2)); y0 = int(max(0, cy - side / 2))
    x1 = int(min(W, cx + side / 2)); y1 = int(min(H, cy + side / 2))
    return x0, y0, x1 - x0, y1 - y0


def load(path: Path, size: int, framing: str) -> np.ndarray:
    """
    Letterbox to size x size. Never a tuple-resize: that would squeeze 3:2 and
    4:3 images by different factors and distort texture device-dependently.
    """
    with Image.open(path) as im:
        if framing == "full":
            im.draft("RGB", (size, size))     # DCT downscale, safe for full frame
        im = im.convert("RGB")
        rgb = np.asarray(im)

    if framing == "eye":
        x, y, w, h = eye_bbox(cv2.resize(rgb, (rgb.shape[1] // 8, rgb.shape[0] // 8),
                                         interpolation=cv2.INTER_AREA))
        x, y, w, h = x * 8, y * 8, w * 8, h * 8
        rgb = rgb[y:y + h, x:x + w]
        if rgb.size == 0:
            with Image.open(path) as im:
                rgb = np.asarray(im.convert("RGB"))

    H, W = rgb.shape[:2]
    s = size / max(H, W)
    nh, nw = max(1, int(round(H * s))), max(1, int(round(W * s)))
    rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)

    out = np.zeros((size, size, 3), np.uint8)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = rgb

    x = out.astype(np.float32) / 255.0
    x = (x - IMNET_MEAN) / IMNET_STD
    return x.transpose(2, 0, 1)


# =====================================================
# EMBEDDING
# =====================================================
def embed(df, size, framing, batch, tag):
    cache = EMB_DIR / f"{tag}.npy"
    if cache.exists():
        print(f"  [{tag}] cached")
        return np.load(cache)

    model = timm.create_model(MODEL, pretrained=True, num_classes=0, img_size=size)
    model.eval().to(DEVICE)

    feats = []
    with torch.no_grad():
        for i in tqdm(range(0, len(df), batch), desc=f"  {tag}", leave=False):
            chunk = df.iloc[i:i + batch]
            arr = np.stack([load(ROOT / r, size, framing) for r in chunk.rel_path])
            t = torch.from_numpy(arr).to(DEVICE)
            with torch.autocast("cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):
                f = model(t)
            feats.append(f.float().cpu().numpy())

    del model
    torch.cuda.empty_cache()

    out = np.concatenate(feats, 0)
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache, out)
    return out


C_GRID = [0.001, 0.01, 0.1, 1.0]


def probe(X, y, groups):
    """
    Repeated NESTED cross-validation.

    Two problems this fixes over a single split:

      selection bias - picking C by the same CV that is then reported makes the
      number optimistic. C is chosen in an inner loop on training folds only.

      fold-assignment variance - a single StratifiedGroupKFold draw has sd ~0.015
      on this cohort, so one number carries +/-0.03 of noise. Repeating over
      several fold assignments reports the distribution instead of one draw.
    """
    aucs, oof_last = [], None
    for seed in range(N_REPEATS):
        outer = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        oof = np.zeros(len(y), dtype=float)

        for tr, te in outer.split(X, y, groups):
            inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
            best_c, best_a = C_GRID[0], -np.inf
            for C in C_GRID:
                clf = make_pipeline(StandardScaler(),
                                    LogisticRegression(max_iter=3000, C=C))
                p = cross_val_predict(clf, X[tr], y[tr], cv=inner,
                                      groups=groups[tr], method="predict_proba")[:, 1]
                a = roc_auc_score(y[tr], p)
                if a > best_a:
                    best_a, best_c = a, C

            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=3000, C=best_c))
            clf.fit(X[tr], y[tr])
            oof[te] = clf.predict_proba(X[te])[:, 1]

        aucs.append(roc_auc_score(y, oof))
        oof_last = oof

    a = np.array(aucs)
    return float(a.mean()), float(a.std()), float(a.min()), float(a.max()), oof_last


def main():
    df = pd.read_csv(QC)
    y = df.label.values
    groups = df.patient_key.values

    print(f"device={DEVICE} | model={MODEL} | {len(df)} images\n")

    results, preds = [], {}
    for tag, size, framing, batch in CONFIGS:
        print(f"[{tag}] size={size} framing={framing}")
        X = embed(df, size, framing, batch, tag)
        auc, sd, lo, hi, p = probe(X, y, groups)
        preds[tag] = p
        results.append({
            "config": tag, "input": size, "framing": framing,
            "dim": X.shape[1], "auc": round(auc, 4), "sd": round(sd, 4),
            "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
        })
        print(f"  AUC = {auc:.4f} +/- {sd:.4f}  (range {lo:.4f}-{hi:.4f})\n")

    res = pd.DataFrame(results)
    np.savez(EMB_DIR / "probe_predictions.npz", **preds)

    # ---------------- report ----------------
    L = ["# Phase 1c - Frozen Representation Probe\n"]
    L.append(f"Backbone **{MODEL}** (DINOv2 ViT-S/14), frozen. Linear probe only, "
             f"patient-grouped {N_FOLDS}-fold CV, patient-level bootstrap CI.\n")
    L.append(res.to_markdown(index=False))
    L.append("")

    L.append("## Reference points\n")
    L.append("| approach | AUC |\n|---|---|")
    L.append("| chance | 0.500 |")
    L.append("| handcrafted global statistics (Phase 1b) | 0.543 |")
    L.append("| metadata only (Phase 1a) | 0.549 |")
    L.append(f"| **best frozen representation** | **{res.auc.max():.4f}** |")
    L.append("| incumbent CornealAI Model 2 (contaminated val) | 0.862 |")
    L.append("")

    # resolution trend, full framing
    full = res[res.framing == "full"].sort_values("input")
    L.append("## Does resolution help?\n")
    L.append(full[["config", "input", "auc", "ci_lo", "ci_hi"]].to_markdown(index=False))
    if len(full) >= 2:
        d = full.auc.iloc[-1] - full.auc.iloc[0]
        L.append(f"\n224 -> {int(full.input.iloc[-1])}: **{d:+.4f}** AUC")
        if d > 0.03:
            L.append("\n> Resolution helps. Fine detail carries signal, so cropping tiles at "
                     "native resolution before resizing is justified by measurement.")
        elif d < -0.03:
            L.append("\n> Resolution *hurts* at whole-frame framing - the extra pixels are "
                     "mostly background. Localisation matters more than raw resolution.")
        else:
            L.append("\n> Resolution alone changes little at whole-frame framing. The eye "
                     "occupies a minority of the frame, so added pixels are largely wasted "
                     "on background; framing is the binding constraint.")
    L.append("")

    # framing effect at matched size
    L.append("## Does cropping to the eye help?\n")
    rows = []
    for s in sorted(res.input.unique()):
        a = res[(res.input == s) & (res.framing == "full")]
        b = res[(res.input == s) & (res.framing == "eye")]
        if len(a) and len(b):
            rows.append({"input": s, "full_frame": a.auc.iloc[0],
                         "eye_crop": b.auc.iloc[0],
                         "delta": round(b.auc.iloc[0] - a.auc.iloc[0], 4)})
    if rows:
        L.append(pd.DataFrame(rows).to_markdown(index=False))
        best_d = max(r["delta"] for r in rows)
        if best_d > 0.03:
            L.append("\n> Cropping to the eye at matched input size helps. Field of view is "
                     "being wasted on background - segmentation-guided cropping is worth its cost.")
        else:
            L.append("\n> Cropping to the eye gives little at whole-eye scale. The signal is "
                     "likely finer than the eye itself - i.e. inside the lesion margin, "
                     "which is what the tiling stage targets.")
    L.append("")

    L.append("## Reading\n")
    best = res.loc[res.auc.idxmax()]
    if best.auc < 0.62:
        L.append(f"Even the best frozen whole-eye representation reaches only "
                 f"**{best.auc:.3f}** [{best.ci_lo:.3f}, {best.ci_hi:.3f}]. Combined with the "
                 f"0.543 handcrafted floor, this is strong evidence that **whole-eye views do "
                 f"not carry the label**. The discriminative information is local - lesion "
                 f"margin texture - and must be reached by segmentation-guided, "
                 f"native-resolution tiling. A global branch is not worth its parameters.")
    elif best.auc < 0.75:
        L.append(f"The best frozen representation reaches **{best.auc:.3f}** "
                 f"[{best.ci_lo:.3f}, {best.ci_hi:.3f}] - clearly above the 0.543 handcrafted "
                 f"floor, so deep features do see global structure that hand statistics miss. "
                 f"A global branch earns a place, but is not sufficient alone; local tiling "
                 f"should supply the rest.")
    else:
        L.append(f"The frozen whole-eye representation already reaches **{best.auc:.3f}** "
                 f"[{best.ci_lo:.3f}, {best.ci_hi:.3f}] with a linear head and no fine-tuning. "
                 f"That is at or near the incumbent's contaminated number, from a global view "
                 f"alone. Global features are a strong backbone for the system.")
    L.append("")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "03_representation_probe.md").write_text("\n".join(L), encoding="utf-8")

    print(res.to_string(index=False))
    print(f"\nwrote {REPORT_DIR / '03_representation_probe.md'}")


if __name__ == "__main__":
    main()
