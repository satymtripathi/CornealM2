"""
Batch inference over a folder of unseen images.

Blind by construction: no labels are read, and nothing here is fitted. The model,
the calibration temperature and the decision band all come from the checkpoint
exactly as trained.

    python src/13_batch_predict.py <folder> [out.csv]
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None

from inference import Pipeline

EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def main():
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else \
        Path(__file__).resolve().parents[1] / "outputs" / "batch_predictions.csv"

    files = sorted(p for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in EXTS)
    print(f"{len(files)} images in {folder}")

    pipe = Pipeline.get()
    lo, hi = pipe.op["lo"], pipe.op["hi"]
    print(f"decision band: bacterial <= {lo:.3f} | indeterminate | fungal >= {hi:.3f}")
    print(f"model: test AUC {pipe.test_auc:.4f}, temperature {pipe.temperature:.3f}\n")

    rows = []
    for p in tqdm(files, desc="predicting"):
        rec = {"filename": p.name}
        try:
            rgb = np.asarray(Image.open(p).convert("RGB"))
            rec["width"], rec["height"] = rgb.shape[1], rgb.shape[0]
            res = pipe.predict(rgb)
            if "error" in res:
                rec["status"] = res["error"]
            else:
                pf = res["p_fungal"]
                rec.update({
                    "status": "ok",
                    "p_fungal": round(pf, 4),
                    # deployed setting - may abstain
                    "prediction": res["label"],
                    # forced choice, for comparison against papers that never abstain
                    "forced_choice": "Fungal" if pf >= 0.5 else "Bacterial",
                    "confidence": round(abs(pf - 0.5) * 2, 3),
                    "n_tiles": res["n_tiles"],
                    "tile_mm": round(res["tile_mm"], 2),
                    "mm_per_px": round(res["mm_per_px"], 5),
                    "limbus_w_px": res["limbus_w_px"],
                })
        except Exception as e:
            rec["status"] = f"{type(e).__name__}: {e}"
        rows.append(rec)

    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    ok = df[df.status == "ok"]
    print(f"\nprocessed {len(ok)}/{len(df)}")
    if len(ok) < len(df):
        print("\nfailures:")
        print(df[df.status != "ok"][["filename", "status"]].to_string(index=False))

    print("\n--- with abstention (deployed setting) ---")
    print(ok.prediction.value_counts().to_string())
    print("\n--- forced choice ---")
    print(ok.forced_choice.value_counts().to_string())

    print("\n--- tile geometry sanity ---")
    print(f"  tile mm: median {ok.tile_mm.median():.2f}, "
          f"p05 {ok.tile_mm.quantile(.05):.2f}, p95 {ok.tile_mm.quantile(.95):.2f}")
    off = int((ok.tile_mm > 5.0).sum())
    if off:
        print(f"  {off} images have tiles > 5 mm - above the 3.67 mm optimum, so those "
              f"predictions are read off-scale and are less reliable")

    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
