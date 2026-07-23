"""
External validation harness - score a labelled cohort, archive it, accumulate.

    python src/14_external_eval.py <images_dir> <gt_file> <cohort_name>

The ground-truth file needs a filename column and a label column; both are
auto-detected, and CSV/TSV are both accepted. Labels outside
{bacterial, fungal} are kept and reported separately as out-of-scope - the
model is binary and cannot express anything else, so folding them into either
class would misrepresent what it does.

Every cohort is archived under outputs/external_validation/<cohort>/ and a row
is appended to registry.csv, so cohorts accumulate into one comparable table
rather than being re-derived each time.

Three things are checked automatically because each has already caught a real
problem:
  - content-hash overlap with the training manifest (leakage)
  - label disagreement on any overlapping image (reference-standard conflict)
  - tile field of view vs the 3.67 mm the model was tuned for (domain shift)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None

from inference import Pipeline

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "outputs" / "external_validation"
MANIFEST = ROOT / "outputs" / "manifests" / "manifest.csv"
EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
POS, NEG = "fungal", "bacterial"


def read_gt(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() in (".tsv", ".txt") else ","
    df = pd.read_csv(path, sep=sep)
    fcol = next((c for c in df.columns
                 if any(k in c.lower() for k in ("file", "name", "image", "id"))), df.columns[0])
    lcol = next((c for c in df.columns
                 if c != fcol and any(k in c.lower()
                                      for k in ("gt", "label", "truth", "class", "diag"))),
                df.columns[-1])
    out = df[[fcol, lcol]].copy()
    out.columns = ["filename", "gt"]
    out["gt"] = out["gt"].astype(str).str.strip()
    out["gt_norm"] = out["gt"].str.lower()
    return out


def boot_ci(y, p, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) > 1:
            out.append(roc_auc_score(y[i], p[i]))
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) if out else (np.nan, np.nan)


def binary_metrics(y, dec):
    """dec: 1 fungal, 0 bacterial, -1 abstain."""
    m = {}
    for cls, name in [(1, POS), (0, NEG)]:
        tp = int(((dec == cls) & (y == cls)).sum())
        called = int((dec == cls).sum())
        tot = int((y == cls).sum())
        prec = tp / called if called else np.nan
        rec = tp / tot if tot else np.nan
        m[f"{name}_precision"] = prec
        m[f"{name}_recall"] = rec
        m[f"{name}_n"] = tot
    cov = float((dec >= 0).mean())
    m["coverage"] = cov
    m["accuracy_on_covered"] = float(((dec == y) & (dec >= 0)).sum() / max((dec >= 0).sum(), 1))
    # the error that actually harms: fungal routed to bacterial
    m["fungal_misroute"] = float(((dec == 0) & (y == 1)).sum() / max((y == 1).sum(), 1))
    return m


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)
    img_dir, gt_path, cohort = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]

    gt = read_gt(gt_path)
    pipe = Pipeline.get()
    modes = pipe.modes()
    sel = modes.get("selective")
    lo, hi = (sel["lo"], sel["hi"]) if sel else (0.5, 0.5)
    man = pd.read_csv(MANIFEST)
    train_lut = dict(zip(man.md5, zip(man.image_id, man.class_name, man.split)))

    files = sorted(p for p in img_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in EXTS)
    print(f"cohort '{cohort}': {len(files)} images, {len(gt)} labels")

    rows = []
    for p in tqdm(files, desc="predicting"):
        rec = {"filename": p.name, "md5": hashlib.md5(p.read_bytes()).hexdigest()}
        try:
            rgb = np.asarray(Image.open(p).convert("RGB"))
            res = pipe.predict(rgb)
            if "error" in res:
                rec["status"] = res["error"]
            else:
                rec.update({"status": "ok", "p_fungal": round(res["p_fungal"], 4),
                            "prediction": res["label"],
                            "forced_choice": "Fungal" if res["p_fungal"] >= .5 else "Bacterial",
                            "n_tiles": res["n_tiles"], "tile_mm": round(res["tile_mm"], 2)})
        except Exception as e:
            rec["status"] = f"{type(e).__name__}: {e}"
        rows.append(rec)

    d = pd.DataFrame(rows).merge(gt, on="filename", how="left")
    seen = d.md5.map(lambda h: train_lut.get(h))
    d["train_overlap"] = seen.map(lambda v: v[2] if v else None)
    d["train_label"] = seen.map(lambda v: v[1] if v else None)

    ok = d[(d.status == "ok") & d.gt_norm.notna()]
    inscope = ok[ok.gt_norm.isin([POS, NEG])].copy()
    oos = ok[~ok.gt_norm.isin([POS, NEG])].copy()

    y = (inscope.gt_norm == POS).astype(int).to_numpy()
    pr = inscope.p_fungal.to_numpy()
    auc = roc_auc_score(y, pr) if len(np.unique(y)) > 1 else np.nan
    ci = boot_ci(y, pr)

    fc = (pr >= 0.5).astype(int)
    dec = np.where(pr >= hi, 1, np.where(pr <= lo, 0, -1))
    m_fc, m_ab = binary_metrics(y, fc), binary_metrics(y, dec)

    conflicts = d[(d.train_label.notna()) &
                  (d.train_label.str.lower() != d.gt_norm)]

    res = {
        "cohort": cohort, "images": len(files),
        "in_scope_n": int(len(inscope)),
        "n_bacterial": int((y == 0).sum()), "n_fungal": int((y == 1).sum()),
        "out_of_scope_n": int(len(oos)),
        "out_of_scope_labels": oos["gt"].value_counts().to_dict(),
        "auc": round(float(auc), 4), "auc_ci_lo": round(ci[0], 4), "auc_ci_hi": round(ci[1], 4),
        "forced_choice": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m_fc.items()},
        "with_abstention": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m_ab.items()},
        "forced_accuracy": round(float((fc == y).mean()), 4),
        "train_overlap_n": int(d.train_overlap.notna().sum()),
        "label_conflicts_n": int(len(conflicts)),
        "tile_mm_median": float(ok.tile_mm.median()),
        "failures": int((d.status != "ok").sum()),
        "model_test_auc": pipe.test_auc, "decision_band": [lo, hi],
        "temperature": pipe.temperature,
    }
    if len(oos):
        res["out_of_scope_predictions"] = oos.prediction.value_counts().to_dict()

    outdir = EXT / cohort
    outdir.mkdir(parents=True, exist_ok=True)
    d.to_csv(outdir / "predictions_scored.csv", index=False)
    with open(outdir / "metrics.json", "w") as f:
        json.dump(res, f, indent=2)
    if len(conflicts):
        conflicts[["filename", "train_label", "gt", "train_overlap", "p_fungal"]] \
            .to_csv(outdir / "label_conflicts.csv", index=False)

    # ---------- registry ----------
    reg_p = EXT / "registry.csv"
    row = {"cohort": cohort, "n_total": len(files), "n_in_scope": res["in_scope_n"],
           "n_bacterial": res["n_bacterial"], "n_fungal": res["n_fungal"],
           "n_out_of_scope": res["out_of_scope_n"], "auc": res["auc"],
           "ci_lo": res["auc_ci_lo"], "ci_hi": res["auc_ci_hi"],
           "forced_accuracy": res["forced_accuracy"],
           "fungal_recall_fc": round(m_fc["fungal_recall"], 4),
           "fungal_precision_fc": round(m_fc["fungal_precision"], 4),
           "bacterial_recall_fc": round(m_fc["bacterial_recall"], 4),
           "bacterial_precision_fc": round(m_fc["bacterial_precision"], 4),
           "fungal_misroute_fc": round(m_fc["fungal_misroute"], 4),
           "coverage_ab": round(m_ab["coverage"], 4),
           "accuracy_covered_ab": round(m_ab["accuracy_on_covered"], 4),
           "train_overlap": res["train_overlap_n"], "label_conflicts": res["label_conflicts_n"],
           "tile_mm_median": round(res["tile_mm_median"], 2)}
    reg = pd.read_csv(reg_p) if reg_p.exists() else pd.DataFrame()
    reg = reg[reg.cohort != cohort] if len(reg) else reg
    reg = pd.concat([reg, pd.DataFrame([row])], ignore_index=True)
    reg.to_csv(reg_p, index=False)

    # ---------- console ----------
    print(f"\n{'='*60}\n{cohort}\n{'='*60}")
    print(f"in scope {res['in_scope_n']} ({res['n_bacterial']} bact / {res['n_fungal']} fung)"
          f" | out of scope {res['out_of_scope_n']}")
    print(f"AUC {res['auc']:.4f}  [{ci[0]:.3f}, {ci[1]:.3f}]")
    print(f"\nforced choice: accuracy {res['forced_accuracy']:.1%}")
    for n in (NEG, POS):
        print(f"  {n:10s} precision {m_fc[n+'_precision']:6.1%}  recall {m_fc[n+'_recall']:6.1%}")
    print(f"  fungal misrouted to bacterial: {m_fc['fungal_misroute']:.1%}")
    print(f"\nwith abstention: coverage {m_ab['coverage']:.1%}, "
          f"accuracy on covered {m_ab['accuracy_on_covered']:.1%}")
    if res["train_overlap_n"]:
        print(f"\n!! {res['train_overlap_n']} images overlap training data "
              f"({res['label_conflicts_n']} with conflicting labels)")
    if len(oos):
        print(f"\nout-of-scope ({res['out_of_scope_n']}) forced into a binary call: "
              f"{res.get('out_of_scope_predictions')}")
    print(f"\nsaved -> {outdir}")
    print(f"registry -> {reg_p}  ({len(reg)} cohort(s))")


if __name__ == "__main__":
    main()
