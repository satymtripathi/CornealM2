"""
Phase 3b - Does tile SCALE explain the Phase 3 null result?

Phase 3: 224 px tiles (0.92 mm) gave 0.746 mean-pooled against a 0.747 whole-eye
baseline - no gain. The hypothesis under test here is that the tiles were too
small to contain the diagnostic structure: `feathery_margin` spans a median
3.44 mm, so a 0.92 mm tile holds about a quarter of its width.

Scales compared, all with the same evaluation protocol as Phase 1c and 3:

    s224   0.92 mm   quarter of the pattern      (Phase 2b)
    s448   1.83 mm   half the pattern
    s896   3.67 mm   contains the whole pattern
    combinations, to test whether scales are complementary

Prediction if the scale explanation is right: s896 > s448 > s224, and s896
should clear the 0.747 baseline. If every scale still lands at ~0.745, then the
frozen representation is the ceiling and tile geometry is not the issue.

Outputs
    outputs/reports/09_scale_comparison.md
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "outputs" / "reports"

spec = importlib.util.spec_from_file_location("mil7", ROOT / "src" / "07_mil.py")
mil7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mil7)

N_FOLDS = 5
N_REPEATS = 8
POOLINGS = ["mean", "max", "gated"]
BASE, BASE_SD = 0.747, 0.013

SUBSETS = [
    ("s224", ["s224"]),
    ("s448", ["s448"]),
    ("s896", ["s896"]),
    ("s448+s896", ["s448", "s896"]),
    ("all", ["s224", "s448", "s896"]),
]


def load_all():
    """Merge the Phase 2b (224px) and Phase 2c (448/896px) tile sets."""
    frames, embs, offset = [], [], 0

    p224 = PROC / "tile_index.csv"
    if p224.exists():
        a = pd.read_csv(p224)
        a["scale"] = "s224"
        e = np.load(PROC / "tile_embeddings.npy").astype(np.float32)
        a["row"] = a.emb_row + offset
        frames.append(a)
        embs.append(e)
        offset += len(e)

    pms = PROC / "tile_index_ms.csv"
    if pms.exists():
        b = pd.read_csv(pms)
        e = np.load(PROC / "tile_embeddings_ms.npy").astype(np.float32)
        b["row"] = b.emb_row + offset
        frames.append(b)
        embs.append(e)
        offset += len(e)

    return pd.concat(frames, ignore_index=True), np.concatenate(embs, 0)


def build_bags(idx, emb, scales):
    sub = idx[idx.scale.isin(scales)].sort_values(["image_id", "scale", "tile_i"])
    images = sub.image_id.unique()
    max_t = int(sub.groupby("image_id").size().max())
    d = emb.shape[1]

    X = np.zeros((len(images), max_t, d), np.float32)
    M = np.zeros((len(images), max_t), bool)
    meta = []
    for i, im in enumerate(images):
        g = sub[sub.image_id == im]
        rows = g.row.to_numpy()
        X[i, :len(rows)] = emb[rows]
        M[i, :len(rows)] = True
        meta.append({"image_id": im, "label": g.label.iloc[0],
                     "patient_key": g.patient_key.iloc[0]})
    return X, M, pd.DataFrame(meta), max_t, d


def evaluate(X, M, y, groups, pooling, d):
    aucs = []
    for seed in range(N_REPEATS):
        cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        oof = np.zeros(len(y))
        for tr, te in cv.split(X, y, groups):
            oof[te] = mil7.fit_predict(X[tr], M[tr], y[tr], groups[tr],
                                       X[te], M[te], pooling, seed, d)
        aucs.append(roc_auc_score(y, oof))
    a = np.array(aucs)
    return float(a.mean()), float(a.std())


def main():
    idx, emb = load_all()
    print(f"{len(idx):,} tiles | scales {sorted(idx.scale.unique())}")

    rows = []
    for name, scales in SUBSETS:
        if not set(scales).issubset(set(idx.scale.unique())):
            print(f"  [{name}] missing scales - skipped")
            continue
        X, M, meta, max_t, d = build_bags(idx, emb, scales)
        y = meta.label.to_numpy()
        g = meta.patient_key.to_numpy()
        for pooling in POOLINGS:
            auc, sd = evaluate(X, M, y, g, pooling, d)
            rows.append({"subset": name, "pooling": pooling,
                         "max_tiles": max_t, "auc": round(auc, 4), "sd": round(sd, 4)})
            print(f"  {name:10s} {pooling:6s} AUC = {auc:.4f} +/- {sd:.4f}")

    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "outputs" / "manifests" / "scale_comparison.csv", index=False)

    piv = res.pivot(index="subset", columns="pooling", values="auc")
    best = res.loc[res.auc.idxmax()]

    L = ["# Phase 3b - Tile Scale Comparison\n"]
    L.append(f"Patient-grouped {N_FOLDS}-fold CV over {N_REPEATS} fold assignments, "
             f"early stopping on an inner split of training folds only.\n")
    L.append("| scale | field of view | vs 3.44 mm pattern |\n|---|---|---|")
    L.append("| s224 | 0.92 mm | quarter of its width |")
    L.append("| s448 | 1.83 mm | half |")
    L.append("| s896 | 3.67 mm | **contains it** |")
    L.append("")
    L.append(res.to_markdown(index=False))
    L.append("\n### AUC by subset and pooling\n")
    L.append(piv.round(4).to_markdown())
    L.append(f"\n**Whole-eye baseline (Phase 1c): {BASE:.3f} +/- {BASE_SD:.3f}**\n")

    L.append("## Reading\n")
    if best.auc > BASE + BASE_SD:
        L.append(f"Best configuration is **{best.subset} / {best.pooling} = {best.auc:.3f} "
                 f"+/- {best.sd:.3f}**, clearing the {BASE:.3f} whole-eye baseline. "
                 f"Tile scale was the limiting factor in Phase 3, not the tiling idea.")
    else:
        L.append(f"Best configuration reaches **{best.auc:.3f} +/- {best.sd:.3f}**, which does "
                 f"not clear the {BASE:.3f} whole-eye baseline. Tile scale does **not** explain "
                 f"the Phase 3 null result.\n")
        L.append("With four independent approaches - handcrafted statistics, whole-eye frozen "
                 "features at several resolutions and framings, native-resolution tiles, and "
                 "now tiles at three scales - all converging on ~0.75, the conclusion is that "
                 "**the frozen DINOv2 representation is the ceiling**, not the geometry. "
                 "The next lever is the representation itself (fine-tuning, or a "
                 "domain-matched backbone), or information outside the pixels.\n")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "09_scale_comparison.md").write_text("\n".join(L), encoding="utf-8")
    print(f"\n{piv.round(4).to_string()}")
    print(f"wrote {REPORT_DIR / '09_scale_comparison.md'}")


if __name__ == "__main__":
    main()
