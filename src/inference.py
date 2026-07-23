"""
End-to-end inference: raw slit-lamp image -> calibrated probability + decision.

Mirrors the training pipeline exactly. Any divergence here silently invalidates
the reported metrics, so the geometry constants are read from the checkpoint
rather than restated.

    image -> limbus segmentation -> 896px native tiles -> fed at 448
          -> frozen DINOv2 ViT-S/14 -> 15-model ensemble -> temperature -> decision

Per-tile attribution comes free with mean pooling. The head is a single Linear
layer, so it commutes with the mean:

    bag_logit = head(mean_i h_i) = mean_i head(h_i)

Each tile's contribution to the decision is therefore exactly head(h_i)/n - an
exact decomposition, not an approximation like Grad-CAM or attention weights.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import timm
import segmentation_models_pytorch as smp
from PIL import Image

warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[1]
LIMBUS_CKPT = ROOT / "models" / "limbus_seg" / "model_limbus_crop_unetpp_weighted.pth"
MODEL_CKPT = ROOT / "outputs" / "checkpoints" / "final_model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMNET_STD = np.array([0.229, 0.224, 0.225], np.float32)

MIN_LIMBUS_FRAC = 0.50
MAX_GLARE_FRAC = 0.60
MAX_TILES = 24
WTW_MM = 11.7


class MILMean(nn.Module):
    def __init__(self, d_in=384, d_hid=192, dropout=0.25):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d_in, d_hid), nn.LayerNorm(d_hid),
                                  nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Linear(d_hid, 1)

    def tile_logits(self, x):
        return self.head(self.proj(x)).squeeze(-1)     # (T,) exact per-tile contribution


class Pipeline:
    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        ck = torch.load(MODEL_CKPT, map_location=DEVICE, weights_only=False)
        self.cfg = ck["config"]
        self.temperature = ck["temperature"]
        self.op = ck.get("operating_point")
        self.test_auc = ck.get("test_auc")
        self.dev_auc = ck.get("dev_auc")

        # External calibration supersedes the internal one when available. The
        # internal band was fitted on a 1:1 curated dev set and did not hold up
        # on external cohorts (coverage 81.5%->76.8%, accuracy on covered
        # 88.7%->77.6%), so 164 genuinely external cases are the better basis.
        self.cal = None
        cal_p = MODEL_CKPT.parent / "calibration_external.json"
        if cal_p.exists():
            import json
            self.cal = json.loads(cal_p.read_text())
            self.temperature = self.cal["temperature_external"]

        self.heads = []
        for sd in ck["state_dicts"]:
            m = MILMean(self.cfg["d_in"], self.cfg["d_hid"], self.cfg["dropout"])
            m.load_state_dict(sd)
            m.eval().to(DEVICE)
            self.heads.append(m)

        lc = torch.load(LIMBUS_CKPT, map_location=DEVICE, weights_only=False)
        lcfg = lc.get("config", {})
        targets = lcfg.get("target_list", [{"label": "crop"}, {"label": "limbus"}])
        labels = [t["label"].strip().lower() for t in targets]
        self.i_limbus = labels.index("limbus") if "limbus" in labels else 1
        self.seg_size = tuple(lcfg.get("img_size", (512, 512)))
        self.seg = smp.UnetPlusPlus(
            encoder_name=lcfg.get("encoder_name", "timm-efficientnet-b0"),
            encoder_weights=None, in_channels=3, classes=len(targets), activation=None)
        self.seg.load_state_dict(lc["state_dict"])
        self.seg.eval().to(DEVICE)

        self.backbone = timm.create_model(self.cfg["backbone"], pretrained=True,
                                          num_classes=0, img_size=self.cfg["input_px"])
        self.backbone.eval().to(DEVICE)

    # ---------------- decision modes ----------------
    def modes(self):
        """
        Operating points, externally calibrated where available.

        Thresholds come from 164 external cases across two independent cohorts,
        not from the internal dev set - the internal band was fitted at 1:1
        prevalence and did not transfer (coverage 81.5% -> 76.8%, accuracy on
        covered 88.7% -> 77.6%).
        """
        m = {"balanced": {"kind": "forced", "t": 0.50,
                          "desc": "Balanced - highest overall accuracy (77% external)"},
             "fungal_safety": {"kind": "forced", "t": 0.25,
                               "desc": "Maximise fungal recall (97%) and minimise the "
                                       "dangerous fungal->bacterial error (3%), at the "
                                       "cost of bacterial recall (29%)"}}
        if self.cal and self.cal.get("selective_band"):
            lo, hi = self.cal["selective_band"]
            m["selective"] = {"kind": "abstain", "lo": lo, "hi": hi,
                              "desc": "May answer 'Indeterminate' - 81% accurate on the 78% of cases it answers"}
        elif self.op:
            m["selective"] = {"kind": "abstain", "lo": self.op["lo"], "hi": self.op["hi"],
                              "desc": "May answer 'Indeterminate' (internal calibration)"}
        return m

    @staticmethod
    def apply_mode(p, mode):
        if mode["kind"] == "forced":
            return "Fungal" if p >= mode["t"] else "Bacterial"
        return ("Fungal" if p >= mode["hi"]
                else "Bacterial" if p <= mode["lo"] else "Indeterminate")

    # ---------------- stages ----------------
    def segment_limbus(self, rgb):
        H, W = rgb.shape[:2]
        x = cv2.resize(rgb, self.seg_size[::-1], interpolation=cv2.INTER_LINEAR)
        x = ((x.astype(np.float32) / 255.0 - IMNET_MEAN) / IMNET_STD).transpose(2, 0, 1)
        with torch.no_grad():
            prob = torch.sigmoid(self.seg(torch.from_numpy(x).unsqueeze(0).to(DEVICE)))[0]
        prob = prob[self.i_limbus].cpu().numpy()

        # interpolate probability then threshold - a NEAREST-upsampled binary mask
        # gives staircase edges that inflate perimeter ~27%
        work = 2048
        s = work / max(H, W)
        pm = cv2.resize(prob, (int(round(W * s)), int(round(H * s))),
                        interpolation=cv2.INTER_LINEAR)
        m = cv2.morphologyEx((pm > 0.5).astype(np.uint8), cv2.MORPH_CLOSE,
                             np.ones((5, 5), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None, None
        c = max(cnts, key=cv2.contourArea)
        c = np.round(c.astype(np.float64) / s).astype(np.int32)

        mask = np.zeros((H, W), np.uint8)
        cv2.fillPoly(mask, [c.reshape(-1, 2)], 1)
        return c.reshape(-1, 2), mask

    def plan_tiles(self, mask):
        crop = self.cfg["crop_px"]
        stride = crop // 2
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return []
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        ii = cv2.integral(mask)
        H, W = mask.shape
        area = float(crop * crop)
        out = []
        for y in range(y0, max(y0 + 1, y1 - crop + 2), stride):
            for x in range(x0, max(x0 + 1, x1 - crop + 2), stride):
                if y + crop > H or x + crop > W:
                    continue
                s = ii[y + crop, x + crop] - ii[y, x + crop] - ii[y + crop, x] + ii[y, x]
                f = s / area
                if f >= MIN_LIMBUS_FRAC:
                    out.append({"x": x, "y": y, "limbus_frac": float(f)})
        out.sort(key=lambda t: -t["limbus_frac"])
        return out[:MAX_TILES]

    def predict(self, rgb):
        contour, mask = self.segment_limbus(rgb)
        if contour is None:
            return {"error": "limbus not detected"}

        crop, inp = self.cfg["crop_px"], self.cfg["input_px"]
        plan = self.plan_tiles(mask)
        tiles, kept = [], []
        for t in plan:
            c = rgb[t["y"]:t["y"] + crop, t["x"]:t["x"] + crop]
            if c.shape[:2] != (crop, crop):
                continue
            small = cv2.resize(c, (256, 256), interpolation=cv2.INTER_AREA)
            hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
            g = float(((hsv[..., 2] / 255. > .94) & (hsv[..., 1] / 255. < .15)).mean())
            if g > MAX_GLARE_FRAC:
                continue
            t["glare_frac"] = g
            kept.append(t)
            c = cv2.resize(c, (inp, inp), interpolation=cv2.INTER_AREA)
            x = c.astype(np.float32) / 255.0
            tiles.append(((x - IMNET_MEAN) / IMNET_STD).transpose(2, 0, 1))

        if not tiles:
            return {"error": "no usable tiles (image may be too small or glare-obscured)"}

        with torch.no_grad():
            feats = []
            for i in range(0, len(tiles), 8):
                b = torch.from_numpy(np.stack(tiles[i:i + 8])).to(DEVICE)
                with torch.autocast("cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):
                    feats.append(self.backbone(b).float())
            F_ = torch.cat(feats, 0)

            per_tile = torch.stack([m.tile_logits(F_) for m in self.heads]).mean(0)
        per_tile = per_tile.cpu().numpy()
        bag_logit = float(per_tile.mean())          # exact: linear head commutes with mean

        p = 1.0 / (1.0 + np.exp(-bag_logit / self.temperature))

        modes = self.modes()
        labels = {k: self.apply_mode(p, v) for k, v in modes.items()}
        label = labels["balanced"]

        # px -> mm via limbus width
        x_, y_, w_, h_ = cv2.boundingRect(contour)
        mm_per_px = WTW_MM / w_ if w_ > 0 else np.nan

        return {
            "p_fungal": float(p), "logit": bag_logit, "label": label,
            "labels_by_mode": labels, "modes": modes,
            "contour": contour, "mask": mask,
            "tiles": kept, "tile_logits": per_tile,
            "n_tiles": len(kept), "mm_per_px": float(mm_per_px),
            "limbus_w_px": int(w_), "tile_mm": crop * float(mm_per_px),
        }


def overlay_limbus(rgb, contour, max_side=900):
    vis = rgb.copy()
    cv2.drawContours(vis, [contour.reshape(-1, 1, 2)], -1, (0, 255, 0), max(2, rgb.shape[1] // 700))
    s = max_side / max(vis.shape[:2])
    return cv2.resize(vis, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def evidence_map(rgb, res, max_side=900):
    """
    Per-tile contribution to the decision. Red pushes fungal, blue bacterial.
    These are exact contributions, not saliency estimates.
    """
    H, W = rgb.shape[:2]
    heat = np.zeros((H, W), np.float32)
    cnt = np.zeros((H, W), np.float32)
    crop = res["tiles"][0] and Pipeline.get().cfg["crop_px"]
    for t, lg in zip(res["tiles"], res["tile_logits"]):
        heat[t["y"]:t["y"] + crop, t["x"]:t["x"] + crop] += lg
        cnt[t["y"]:t["y"] + crop, t["x"]:t["x"] + crop] += 1
    heat = np.divide(heat, np.maximum(cnt, 1))

    v = np.abs(heat).max() or 1.0
    norm = np.clip(heat / v, -1, 1)
    col = np.zeros((H, W, 3), np.uint8)
    col[..., 0] = (np.clip(norm, 0, 1) * 255).astype(np.uint8)      # R = fungal
    col[..., 2] = (np.clip(-norm, 0, 1) * 255).astype(np.uint8)     # B = bacterial

    m = (cnt > 0) & (res["mask"] > 0)
    vis = rgb.copy()
    vis[m] = (0.55 * vis[m] + 0.45 * col[m]).astype(np.uint8)
    cv2.drawContours(vis, [res["contour"].reshape(-1, 1, 2)], -1, (0, 255, 0),
                     max(2, W // 700))
    s = max_side / max(vis.shape[:2])
    return cv2.resize(vis, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
