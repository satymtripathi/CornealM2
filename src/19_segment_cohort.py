"""
Run the trained lesion segmenter over all 682 cohort images and cache the masks.

These masks (infiltrate / hypopyon / cellularity / glare) are the input to
lesion-guided tiling and to the biomarker/concept features. Stored as a small
argmax label map at the segmenter's native 512, plus the image's true H,W so it
can be mapped back to full resolution when tiles are cut.

Outputs
    data/interim/lesion_masks/<image_id>.npz   (label512 uint8, native_hw)
    outputs/reports/19_cohort_segmentation.md
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
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "models" / "lesion_seg" / "lesion_unetpp.pth"
MANIFEST = ROOT / "outputs" / "manifests" / "manifest.csv"
OUT = ROOT / "data" / "interim" / "lesion_masks"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD = (0.229, 0.224, 0.225)


def main():
    ck = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    cfg = ck["config"]
    classes = cfg["classes"]; size = cfg["img_size"]
    model = smp.UnetPlusPlus(cfg["encoder_name"], encoder_weights=None,
                             in_channels=3, classes=len(classes)).to(DEVICE)
    model.load_state_dict(ck["state_dict"]); model.eval()
    tf = A.Compose([A.Resize(size, size), A.Normalize(IMNET_MEAN, IMNET_STD), ToTensorV2()])
    print(f"classes: {classes}")

    df = pd.read_csv(MANIFEST)
    OUT.mkdir(parents=True, exist_ok=True)
    cid = {c: i for i, c in enumerate(classes)}

    area = {c: [] for c in classes}
    hypo_present = 0
    for r in tqdm(list(df.itertuples()), desc="segmenting cohort"):
        with Image.open(ROOT / r.rel_path) as im:
            rgb = np.asarray(im.convert("RGB"))
        H, W = rgb.shape[:2]
        x = tf(image=rgb)["image"].unsqueeze(0).to(DEVICE)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            lab = model(x).argmax(1)[0].cpu().numpy().astype(np.uint8)
        np.savez_compressed(OUT / f"{r.image_id}.npz",
                            label512=lab, native_hw=np.array([H, W], np.int32))
        tot = lab.size
        for c, i in cid.items():
            area[c].append(float((lab == i).mean()))
        if (lab == cid["hypopyon"]).mean() > 0.001:
            hypo_present += 1

    L = ["# Phase - Cohort lesion segmentation\n"]
    L.append(f"Segmented **{len(df)}** images with the lesion UNet++ "
             f"(infiltrate Dice 0.88, hypopyon 0.86).\n")
    L.append("Median area (% of frame) predicted per class:\n")
    L.append("| class | median % | present (>0.1%) |\n|---|---|---|")
    for c in classes[1:]:
        a = np.array(area[c])
        L.append(f"| {c} | {np.median(a)*100:.3f} | {int((a > 0.001).sum())} |")
    L.append(f"\n- hypopyon detected in **{hypo_present}** of {len(df)} images "
             f"({hypo_present/len(df):.0%})")
    L.append("\nMasks cached to `data/interim/lesion_masks/`. Next: lesion-guided tiling.")
    (ROOT / "outputs" / "reports" / "19_cohort_segmentation.md").write_text(
        "\n".join(L), encoding="utf-8")

    print(f"\nhypopyon present in {hypo_present}/{len(df)} images")
    print(f"infiltrate median area {np.median(area['infiltrate'])*100:.2f}%")
    print(f"cached masks -> {OUT}")


if __name__ == "__main__":
    main()
