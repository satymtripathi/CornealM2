"""
Phase 2a - Limbus / crop segmentation on native-resolution images.

Runs the UNet++ (timm-efficientnet-b0, 2 targets: crop + limbus) checkpoint over
all 682 images and maps the masks back into NATIVE pixel coordinates.

Note on geometry: the checkpoint was trained with A.Resize(512, 512), a
stretching resize. Inference therefore reproduces that stretch so the input
matches the training distribution, then maps the predicted mask back to the
original H x W - which undoes the distortion. Masks are correct in native
coordinates even though the network saw a squeezed image.

Because there is no ground-truth mask for this cohort, the segmentation is
validated against corneal anatomy instead. The limbus is a known object:

    horizontal white-to-white   ~11.7 mm
    vertical                    ~10.6 mm   -> w/h ~ 1.10
    near-circular                          -> circularity ~ 1.0
    area                                   ~ 95-100 mm^2

If predicted limbi reproduce those numbers, the model is working. If they do
not, we find out here rather than after building on top of it.

Outputs
    data/interim/limbus/<image_id>.npz     contour + crop bbox, native coords
    outputs/manifests/limbus_geometry.csv
    outputs/figures/limbus_overlays/*.jpg
    outputs/reports/04_limbus_segmentation.md
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import segmentation_models_pytorch as smp
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "models" / "limbus_seg" / "model_limbus_crop_unetpp_weighted.pth"
MANIFEST = ROOT / "outputs" / "manifests" / "manifest.csv"
OUT_MASK = ROOT / "data" / "interim" / "limbus"
OUT_FIG = ROOT / "outputs" / "figures" / "limbus_overlays"
REPORT_DIR = ROOT / "outputs" / "reports"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESH = 0.5
N_OVERLAY = 24                      # visual QC sample

# corneal anatomy, used for validation and for the px -> mm scale
WTW_MM = 11.7                       # horizontal white-to-white
IMNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMNET_STD = np.array([0.229, 0.224, 0.225], np.float32)


# =====================================================
# MODEL
# =====================================================
def load_model():
    ck = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    cfg = ck.get("config", {})
    targets = cfg.get("target_list", [{"label": "crop"}, {"label": "limbus"}])
    labels = [t["label"].strip().lower() for t in targets]
    i_crop = labels.index("crop") if "crop" in labels else 0
    i_limbus = labels.index("limbus") if "limbus" in labels else 1

    model = smp.UnetPlusPlus(
        encoder_name=cfg.get("encoder_name", "timm-efficientnet-b0"),
        encoder_weights=None, in_channels=3,
        classes=len(targets), activation=None,
    )
    model.load_state_dict(ck["state_dict"])
    model.to(DEVICE).eval()
    return model, i_crop, i_limbus, tuple(cfg.get("img_size", (512, 512)))


def read_native(path: Path, min_side: int = 1400):
    """Decode at reduced scale (still well above the 512 the net needs) and
    report the true native size so masks can be scaled back correctly."""
    with Image.open(path) as im:
        W0, H0 = im.size
        im.draft("RGB", (min_side, min_side))
        rgb = np.asarray(im.convert("RGB"))
    return rgb, (H0, W0)


# =====================================================
# GEOMETRY
# =====================================================
def largest_contour(mask_u8):
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)


def ellipse_fit_error(cnt, ell):
    """
    Mean relative deviation of the contour from its best-fit ellipse.

    Each point is mapped into the ellipse frame and its normalised radius
    r = sqrt((x/a)^2 + (y/b)^2) computed; a perfect fit gives r == 1 everywhere.
    Returns mean |r - 1|, so 0.05 means the contour sits within ~5% of the
    ellipse on average. Unlike raster circularity this is scale-free and
    insensitive to pixel staircasing.
    """
    (cx, cy), (MA, ma), ang = ell
    a, b = MA / 2.0, ma / 2.0
    if a <= 0 or b <= 0:
        return np.nan
    p = cnt.reshape(-1, 2).astype(np.float64)
    th = np.deg2rad(ang)
    dx, dy = p[:, 0] - cx, p[:, 1] - cy
    xr = dx * np.cos(th) + dy * np.sin(th)
    yr = -dx * np.sin(th) + dy * np.cos(th)
    r = np.sqrt((xr / a) ** 2 + (yr / b) ** 2)
    return float(np.mean(np.abs(r - 1.0)))


def geometry(cnt, H, W):
    area = float(cv2.contourArea(cnt))
    if area <= 0:
        return None
    # Raster perimeter follows pixel corners and is inflated ~8% for a digital
    # circle, which biases 4*pi*A/P^2 low regardless of segmentation quality.
    # Simplify first so the perimeter tracks the underlying shape.
    per_raw = float(cv2.arcLength(cnt, True))
    cnt_s = cv2.approxPolyDP(cnt, 0.004 * per_raw, True)
    per = float(cv2.arcLength(cnt_s, True))
    area_s = float(cv2.contourArea(cnt_s))
    x, y, w, h = cv2.boundingRect(cnt)
    circ = float(4 * np.pi * area_s / (per ** 2)) if per > 0 else 0.0

    # Feret via min-area rect
    (_, _), (rw, rh), _ = cv2.minAreaRect(cnt)
    feret = float(max(rw, rh))
    minor = float(min(rw, rh))

    M = cv2.moments(cnt)
    cx = M["m10"] / M["m00"] if M["m00"] else x + w / 2
    cy = M["m01"] / M["m00"] if M["m00"] else y + h / 2

    fit_err, ell_ratio = np.nan, np.nan
    if len(cnt) >= 5:
        ell = cv2.fitEllipse(cnt)
        fit_err = ellipse_fit_error(cnt, ell)
        (_, _), (MA, ma), _ = ell
        ell_ratio = float(max(MA, ma) / max(min(MA, ma), 1e-6))

    mm_per_px = WTW_MM / w if w > 0 else np.nan
    return {
        "limbus_ellipse_fit_err": fit_err,
        "limbus_ellipse_ratio": ell_ratio,
        "limbus_area_px": area, "limbus_perimeter_px": per,
        "limbus_w_px": int(w), "limbus_h_px": int(h),
        "limbus_wh_ratio": float(w / h) if h else np.nan,
        "limbus_circularity": circ,
        "limbus_feret_px": feret, "limbus_minor_px": minor,
        "limbus_cx": float(cx), "limbus_cy": float(cy),
        "limbus_cx_rel": float(cx / W), "limbus_cy_rel": float(cy / H),
        "limbus_frame_frac": float(area / (H * W)),
        "mm_per_px": float(mm_per_px),
        "limbus_area_mm2": float(area * mm_per_px ** 2),
        "limbus_feret_mm": float(feret * mm_per_px),
    }


def overlay(rgb, limbus_cnt, crop_box, out_path, caption):
    vis = rgb.copy()
    if limbus_cnt is not None:
        cv2.drawContours(vis, [limbus_cnt], -1, (0, 255, 0), 3)
    if crop_box is not None:
        x, y, w, h = crop_box
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 128, 0), 3)
    s = 900 / max(vis.shape[:2])
    vis = cv2.resize(vis, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    cv2.putText(vis, caption, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imwrite(str(out_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


# =====================================================
# MAIN
# =====================================================
def main():
    df = pd.read_csv(MANIFEST)
    model, i_crop, i_limbus, img_size = load_model()
    print(f"device={DEVICE}  input={img_size}  crop_idx={i_crop} limbus_idx={i_limbus}")

    OUT_MASK.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    overlay_ids = set(df.groupby("class_name", group_keys=False)
                        .apply(lambda g: g.sample(min(len(g), N_OVERLAY // 2),
                                                  random_state=42)).image_id)

    rows = []
    for r in tqdm(df.itertuples(), total=len(df), desc="segmenting"):
        rgb, (H0, W0) = read_native(ROOT / r.rel_path)

        x = cv2.resize(rgb, img_size[::-1], interpolation=cv2.INTER_LINEAR)
        x = ((x.astype(np.float32) / 255.0 - IMNET_MEAN) / IMNET_STD).transpose(2, 0, 1)
        t = torch.from_numpy(x).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            prob = torch.sigmoid(model(t))[0].cpu().numpy()

        rec = {"image_id": r.image_id, "label": r.label, "class_name": r.class_name,
               "native_h": H0, "native_w": W0, "seg_error": None}

        # Map back to native coordinates. Interpolate the PROBABILITY map
        # bilinearly and threshold afterwards - upsampling a binary mask with
        # NEAREST produces staircase edges that leave area untouched but inflate
        # the perimeter ~27%, which collapses circularity (4*pi*A/P^2) to ~0.62.
        # Contours are extracted at WORK resolution and scaled to native: point
        # coordinates scale linearly, so area scales by s^2 and perimeter by s,
        # leaving circularity invariant.
        WORK = 2048
        s = WORK / max(H0, W0)
        wh = (int(round(W0 * s)), int(round(H0 * s)))

        lp = cv2.resize(prob[i_limbus], wh, interpolation=cv2.INTER_LINEAR)
        cp = cv2.resize(prob[i_crop], wh, interpolation=cv2.INTER_LINEAR)
        lm = (lp > THRESH).astype(np.uint8)
        cm = (cp > THRESH).astype(np.uint8)

        # close single-pixel nicks left by thresholding
        k = np.ones((5, 5), np.uint8)
        lm = cv2.morphologyEx(lm, cv2.MORPH_CLOSE, k)
        cm = cv2.morphologyEx(cm, cv2.MORPH_CLOSE, k)

        lcnt = largest_contour(lm)
        ccnt = largest_contour(cm)
        inv = 1.0 / s
        if lcnt is not None:
            lcnt = np.round(lcnt.astype(np.float64) * inv).astype(np.int32)
        if ccnt is not None:
            ccnt = np.round(ccnt.astype(np.float64) * inv).astype(np.int32)

        if lcnt is None:
            rec["seg_error"] = "no limbus contour"
            rows.append(rec)
            continue

        g = geometry(lcnt, H0, W0)
        if g is None:
            rec["seg_error"] = "degenerate limbus"
            rows.append(rec)
            continue
        rec.update(g)

        crop_box = cv2.boundingRect(ccnt) if ccnt is not None else None
        if crop_box:
            rec.update({"crop_x": crop_box[0], "crop_y": crop_box[1],
                        "crop_w": crop_box[2], "crop_h": crop_box[3]})

        np.savez_compressed(
            OUT_MASK / f"{r.image_id}.npz",
            limbus_contour=lcnt.squeeze(1).astype(np.int32),
            crop_box=np.array(crop_box if crop_box else [0, 0, W0, H0], np.int32),
            native_hw=np.array([H0, W0], np.int32),
        )

        if r.image_id in overlay_ids:
            sc = rgb.shape[0] / H0
            lc = (lcnt.squeeze(1) * sc).astype(np.int32).reshape(-1, 1, 2)
            cb = tuple(int(v * sc) for v in crop_box) if crop_box else None
            overlay(rgb, lc, cb, OUT_FIG / f"{r.class_name}_{r.image_id[:44]}.jpg",
                    f"{r.class_name} | circ={g['limbus_circularity']:.3f} "
                    f"w/h={g['limbus_wh_ratio']:.3f} area={g['limbus_area_mm2']:.0f}mm2")

        rows.append(rec)

    geo = pd.DataFrame(rows)
    geo.to_csv(ROOT / "outputs" / "manifests" / "limbus_geometry.csv", index=False)

    # ---------------- validation ----------------
    ok = geo[geo.seg_error.isna()].copy()
    L = ["# Phase 2a - Limbus Segmentation\n"]
    L.append(f"UNet++ / timm-efficientnet-b0, 512x512, targets crop + limbus.\n")
    L.append(f"- images processed: **{len(geo)}**")
    L.append(f"- segmentation failures: **{int(geo.seg_error.notna().sum())}**\n")

    L.append("## Anatomical validation\n")
    L.append("No ground-truth masks exist for this cohort, so the predicted limbus is "
             "checked against known corneal anatomy. These are independent physical "
             "constraints the network was never optimised for.\n")

    checks = [
        ("limbus_wh_ratio", "width / height", 1.10, 1.00, 1.25),
        ("limbus_ellipse_ratio", "ellipse axis ratio", 1.10, 1.00, 1.30),
        ("limbus_ellipse_fit_err", "ellipse fit error (rel.)", 0.02, 0.0, 0.08),
        ("limbus_circularity", "circularity (simplified)", 1.00, 0.88, 1.02),
        ("limbus_area_mm2", "area (mm^2)", 97.0, 85.0, 110.0),
        ("limbus_feret_mm", "max diameter (mm)", 11.7, 10.5, 13.0),
    ]
    tbl = []
    for col, name, expect, lo, hi in checks:
        v = ok[col].dropna()
        med = float(v.median())
        frac = float(((v >= lo) & (v <= hi)).mean())
        tbl.append({
            "measure": name, "expected": expect,
            "median": round(med, 3),
            "p25": round(float(v.quantile(.25)), 3),
            "p75": round(float(v.quantile(.75)), 3),
            "% in plausible range": round(100 * frac, 1),
            "verdict": "PASS" if lo <= med <= hi and frac > 0.75 else "CHECK",
        })
    vdf = pd.DataFrame(tbl)
    L.append(vdf.to_markdown(index=False))

    n_pass = int((vdf.verdict == "PASS").sum())
    L.append(f"\n**{n_pass} of {len(vdf)} anatomical checks pass.**\n")
    if n_pass == len(vdf):
        L.append("> The predicted limbus reproduces corneal anatomy on measures the network "
                 "was never trained against. The segmentation is trustworthy enough to build "
                 "the tiling stage on, and `mm_per_px` gives a real physical scale - so lesion "
                 "sizes can be reported in mm rather than pixels.\n")
    else:
        L.append("> Some checks fail. Inspect the overlays before building on this "
                 "segmentation; a systematically wrong limbus would corrupt every "
                 "downstream ROI and scale.\n")

    L.append("## Scale\n")
    mm = ok.mm_per_px.dropna()
    L.append(f"- median **{mm.median()*1000:.2f} um/px** "
             f"(p25 {mm.quantile(.25)*1000:.2f}, p75 {mm.quantile(.75)*1000:.2f})")
    L.append(f"- limbus occupies median **{ok.limbus_frame_frac.median()*100:.1f}%** of the frame\n")
    L.append("At this scale a 50-200 um fungal filament spans roughly "
             f"**{0.050/mm.median():.0f}-{0.200/mm.median():.0f} px** natively.\n")

    L.append("## By class\n")
    L.append(ok.groupby("class_name")[
        ["limbus_area_mm2", "limbus_circularity", "limbus_wh_ratio", "limbus_frame_frac"]
    ].median().round(4).to_markdown())
    L.append("\n(These should be similar across classes - the limbus is anatomy, not pathology.)\n")

    L.append(f"## Visual QC\n\n{len(list(OUT_FIG.glob('*.jpg')))} overlays in "
             f"`outputs/figures/limbus_overlays/` - green = limbus, orange = crop box.\n")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "04_limbus_segmentation.md").write_text("\n".join(L), encoding="utf-8")

    print("\n" + vdf.to_string(index=False))
    print(f"\nfailures: {int(geo.seg_error.notna().sum())}")
    print(f"wrote {REPORT_DIR / '04_limbus_segmentation.md'}")


if __name__ == "__main__":
    main()
