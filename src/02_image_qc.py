"""
Phase 1b - Image-level QC, appearance confound audit, and the signal floor.

Three jobs:

1. QC        - per-image sharpness / exposure / glare, so unusable images are
               identified now rather than blamed on the model later.
2. Confound  - do the classes differ in ACQUISITION appearance (exposure,
               framing, focus) rather than pathology? Metadata came back clean;
               this is the same question one level down, in pixels.
3. Floor     - how much of the label is recoverable from global image
               statistics alone, with no notion of a lesion? Any real model
               must beat this, and by a margin that justifies its complexity.

Images are decoded at reduced scale via JPEG DCT scaling (PIL draft), then
letterboxed to a fixed working size so that focus and texture measures are
comparable across the three native resolutions present.

Outputs
    outputs/manifests/image_qc.csv
    outputs/reports/02_image_qc.md
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
from tqdm import tqdm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "outputs" / "manifests" / "manifest.csv"
REPORT_DIR = ROOT / "outputs" / "reports"

WORK = 1024          # working long side - all images normalised to this
SEED = 42
N_FOLDS = 5
N_WORKERS = 6


# =====================================================
# LOADING
# =====================================================
def load_standard(path: Path, work: int = WORK) -> np.ndarray:
    """
    Decode at reduced scale, then letterbox to work x work.

    Letterbox (not stretch): a tuple-resize would squeeze a 3:2 image by 1.5x
    and a 4:3 image by 1.33x, distorting every shape and texture measure by a
    device-dependent factor. Padding keeps geometry honest.
    """
    with Image.open(path) as im:
        im.draft("RGB", (work, work))       # JPEG DCT downscale - fast
        im = im.convert("RGB")
        rgb = np.asarray(im)

    h, w = rgb.shape[:2]
    s = work / max(h, w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)

    out = np.zeros((work, work, 3), np.uint8)
    y0, x0 = (work - nh) // 2, (work - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = rgb
    return out, (y0, x0, nh, nw)


# =====================================================
# FEATURES
# =====================================================
def colourfulness(rgb: np.ndarray) -> float:
    """Hasler-Susstrunk."""
    r, g, b = rgb[..., 0].astype(np.float32), rgb[..., 1].astype(np.float32), rgb[..., 2].astype(np.float32)
    rg, yb = r - g, 0.5 * (r + g) - b
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


def fft_high_ratio(gray: np.ndarray) -> float:
    f = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
    mag = np.abs(f)
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    tot = mag.sum()
    return float(mag[r > 0.25 * r.max()].sum() / tot) if tot > 0 else 0.0


def gray_entropy(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    p = hist / max(hist.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def eye_region(gray: np.ndarray, valid: np.ndarray):
    """
    Crude subject-region estimate: Otsu inside the non-padded area, largest
    component. Gives a framing / magnification proxy (how much of the frame the
    eye fills) which is an acquisition covariate, not a pathology one.
    """
    g = gray.copy()
    g[~valid] = 0
    _, th = cv2.threshold(g[valid].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = int(_)
    mask = ((gray > thr) & valid).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return 0.0, 0.5, 0.5
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    area = st[i, cv2.CC_STAT_AREA] / max(valid.sum(), 1)
    cy, cx = cen[i][1] / gray.shape[0], cen[i][0] / gray.shape[1]
    return float(area), float(cx), float(cy)


def extract(args):
    image_id, rel = args
    try:
        rgb, (y0, x0, nh, nw) = load_standard(ROOT / rel)
    except Exception as e:
        return {"image_id": image_id, "qc_error": f"{type(e).__name__}: {e}"}

    valid = np.zeros(rgb.shape[:2], bool)
    valid[y0:y0 + nh, x0:x0 + nw] = True

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    gv = gray[valid].astype(np.float32)
    Lc, ac, bc = lab[..., 0][valid].astype(np.float32), lab[..., 1][valid].astype(np.float32), lab[..., 2][valid].astype(np.float32)
    S, V = hsv[..., 1][valid].astype(np.float32) / 255.0, hsv[..., 2][valid].astype(np.float32) / 255.0

    crop = rgb[y0:y0 + nh, x0:x0 + nw]
    gcrop = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

    gx = cv2.Sobel(gcrop, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gcrop, cv2.CV_32F, 0, 1, ksize=3)
    gmag = np.sqrt(gx ** 2 + gy ** 2)

    r, g, b = [rgb[..., i][valid].astype(np.float32) for i in range(3)]
    tot = r + g + b + 1e-6

    area, cx, cy = eye_region(gray, valid)

    return {
        "image_id": image_id,
        # exposure
        "L_mean": float(Lc.mean()), "L_std": float(Lc.std()),
        "a_mean": float(ac.mean()), "b_mean": float(bc.mean()),
        "V_mean": float(V.mean()), "S_mean": float(S.mean()), "S_std": float(S.std()),
        "dark_frac": float((V < 0.10).mean()),
        "bright_frac": float((V > 0.90).mean()),
        # glare: bright and desaturated - specular reflection, not tissue
        "glare_frac": float(((V > 0.94) & (S < 0.15)).mean()),
        # focus / texture
        "lap_var": float(cv2.Laplacian(gcrop, cv2.CV_32F).var()),
        "tenengrad": float((gmag ** 2).mean()),
        "edge_density": float((cv2.Canny(gcrop, 50, 150) > 0).mean()),
        "fft_high_ratio": fft_high_ratio(gcrop),
        "entropy": gray_entropy(gray),
        "rms_contrast": float(gv.std() / (gv.mean() + 1e-6)),
        # colour
        "colourfulness": colourfulness(rgb),
        "red_ratio": float((r / tot).mean()),
        "green_ratio": float((g / tot).mean()),
        # framing (acquisition covariate)
        "subject_area": area, "subject_cx": cx, "subject_cy": cy,
        "qc_error": None,
    }


# =====================================================
# ANALYSIS
# =====================================================
def compare(df, cols):
    rows = []
    for c in cols:
        a = df.loc[df.label == 0, c].dropna()
        b = df.loc[df.label == 1, c].dropna()
        if len(a) < 5 or len(b) < 5:
            continue
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        auc = u / (len(a) * len(b))
        rows.append({
            "feature": c,
            "bacterial_med": round(float(np.median(a)), 4),
            "fungal_med": round(float(np.median(b)), 4),
            "auc": round(float(max(auc, 1 - auc)), 4),
            "p": float(p),
        })
    r = pd.DataFrame(rows).sort_values("auc", ascending=False)
    # Holm-Bonferroni
    r = r.sort_values("p").reset_index(drop=True)
    m = len(r)
    r["p_adj"] = [min(1.0, (m - i) * p) for i, p in enumerate(r.p)]
    r["p_adj"] = r.p_adj.cummax()
    return r.sort_values("auc", ascending=False)


def main():
    man = pd.read_csv(MANIFEST)
    print(f"extracting QC features from {len(man)} images ({N_WORKERS} workers)...")

    args = list(zip(man.image_id, man.rel_path))
    recs = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for r in tqdm(ex.map(extract, args, chunksize=4), total=len(args)):
            recs.append(r)

    qc = pd.DataFrame(recs)
    df = man.merge(qc, on="image_id", how="left")
    df.to_csv(ROOT / "outputs" / "manifests" / "image_qc.csv", index=False)

    feat_cols = [c for c in qc.columns if c not in ("image_id", "qc_error")]

    L = ["# Phase 1b - Image QC, Appearance Confound, Signal Floor\n"]
    n_err = int(df.qc_error.notna().sum())
    L.append(f"{len(df)} images | extraction errors: **{n_err}**\n")

    # ---------- QC ----------
    L.append("## Quality flags\n")
    lap_lo = df.lap_var.quantile(0.05)
    flags = pd.DataFrame({
        "blurry (lap_var < p5)": df.lap_var < lap_lo,
        "heavy glare (>5% of frame)": df.glare_frac > 0.05,
        "underexposed (L_mean < 40)": df.L_mean < 40,
        "overexposed (bright_frac > 25%)": df.bright_frac > 0.25,
        "tiny subject (<10% of frame)": df.subject_area < 0.10,
    })
    fl = pd.DataFrame({
        "n": flags.sum(),
        "bacterial": [int(((flags[c]) & (df.label == 0)).sum()) for c in flags],
        "fungal": [int(((flags[c]) & (df.label == 1)).sum()) for c in flags],
    })
    L.append(fl.to_markdown())
    L.append(f"\n- any flag: **{int(flags.any(axis=1).sum())}** images\n")

    # ---------- appearance confound ----------
    L.append("## Appearance differences by class\n")
    L.append("Mann-Whitney with Holm correction. `auc` is per-feature "
             "separability - 0.50 means the classes are indistinguishable on "
             "that measure.\n")
    cmp = compare(df, feat_cols)
    L.append(cmp.assign(p=cmp.p.map(lambda v: f"{v:.2e}"),
                        p_adj=cmp.p_adj.map(lambda v: f"{v:.2e}")).to_markdown(index=False))

    sig = cmp[(cmp.p_adj < 0.05) & (cmp.auc > 0.60)]
    L.append(f"\n**{len(sig)}** features separate the classes at AUC > 0.60 after correction.\n")

    # ---------- signal floor ----------
    L.append("## Signal floor - global statistics only\n")
    L.append("No lesion, no segmentation, no tiles. Patient-grouped 5-fold CV.\n")

    X = df[feat_cols].fillna(df[feat_cols].median())
    y = df.label.values
    groups = df.patient_key.values
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    clf = HistGradientBoostingClassifier(max_depth=3, max_iter=300, random_state=SEED)
    p = cross_val_predict(clf, X, y, cv=cv, groups=groups, method="predict_proba")[:, 1]
    auc_floor = roc_auc_score(y, p)

    # split acquisition-only vs appearance features to see what carries it
    acq = ["L_mean", "L_std", "V_mean", "dark_frac", "bright_frac", "glare_frac",
           "lap_var", "tenengrad", "subject_area", "subject_cx", "subject_cy"]
    acq = [c for c in acq if c in X.columns]
    p_acq = cross_val_predict(clf, X[acq], y, cv=cv, groups=groups, method="predict_proba")[:, 1]
    auc_acq = roc_auc_score(y, p_acq)

    L.append(f"| features | CV AUC |\n|---|---|")
    L.append(f"| all global statistics ({len(feat_cols)}) | **{auc_floor:.4f}** |")
    L.append(f"| acquisition only ({len(acq)}: exposure, focus, framing) | **{auc_acq:.4f}** |")
    L.append("")
    L.append(f"- **{auc_floor:.3f}** is the floor a lesion-aware model must clear.")
    if auc_acq >= 0.60:
        L.append(f"- ⚠ acquisition-only reaches **{auc_acq:.3f}** - part of the separability is "
                 f"capture conditions, not pathology. Report this alongside any headline number.")
    else:
        L.append(f"- acquisition-only is **{auc_acq:.3f}**, so the global signal is "
                 f"largely colour/texture of the eye itself rather than how it was photographed.")
    L.append("")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "02_image_qc.md").write_text("\n".join(L), encoding="utf-8")

    print(f"\nsignal floor (all global stats) AUC = {auc_floor:.4f}")
    print(f"acquisition-only            AUC = {auc_acq:.4f}")
    print(f"wrote {REPORT_DIR / '02_image_qc.md'}")


if __name__ == "__main__":
    main()
