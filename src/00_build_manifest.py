"""
Phase 0 - Data spine for LVP Model 2 (Bacterial vs Fungal).

Builds the single source of truth every later stage reads:
  - one row per image
  - patient identity parsed from the EMR filename
  - image integrity / resolution / device probe
  - exact-duplicate detection (content hash)
  - patient-grouped locked test set + 5-fold CV assignment

Nothing downstream may re-split. If a split is wrong, it is wrong here.

Outputs
    outputs/manifests/manifest.csv
    outputs/manifests/manifest_summary.json
    outputs/reports/00_manifest_report.md
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import re
import json
import hashlib
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image, ExifTags
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit

# =====================================================
# CONFIG
# =====================================================
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
MANIFEST_DIR = ROOT / "outputs" / "manifests"
REPORT_DIR = ROOT / "outputs" / "reports"

CLASS_TO_LABEL = {"Bacterial": 0, "Fungal": 1}

TEST_FRACTION = 0.20
N_FOLDS = 5
SEED = 42

# Source-population prevalence (bacterial : fungal) measured on the full
# EMR extraction, not on this curated 1:1 cohort. Carried through the manifest
# so every later metric can be reported at deployment prevalence.
SOURCE_PREVALENCE_BAC_FUN = (2562, 27111)

Image.MAX_IMAGE_PIXELS = None


# =====================================================
# FILENAME PARSING
# =====================================================
# Observed EMR export shapes:
#   ADL-P67624_Jadi Posham__28_9_22 6_19 PM_003.JPG
#   ADL-PN83891_Sorthe Laxmi__Imaging_1_7_24 9_34 AM_003.JPG
#   1104450_MUD-P79267__Imaging_27_12_22 11_09 AM_003.JPG
#   BEL-P30937_Perumal_Bhupalan_Imaging_2021_09_27_16_18_58_002.JPG
#   ADL-P76049_Piple Ravindar__Imaging_2023.08.12 13_25_48_003.JPG
#
# Identity lives in the leading [CENTRE-]MRN token. The patient *name* follows
# and is unreliable (spelling drifts between visits), so it is used only as a
# fallback key and is never emitted in shareable outputs.

# "VC-" marks a rural Vision Centre capture (teleophthalmology screening arm)
# rather than a tertiary-centre slit lamp. Kept as a covariate: it is the most
# likely source of acquisition/device shift in this cohort.
RE_VC = re.compile(r"^VC-([A-Za-z]{2,5})-([A-Za-z]{0,3}\d{3,10})", re.I)
RE_CENTRE_MRN = re.compile(r"^([A-Za-z]{2,5})-([A-Za-z]{0,3}\d{3,10})")
RE_BARE_MRN = re.compile(r"^([A-Za-z]{0,3}\d{3,10})")
RE_EMBEDDED_CENTRE_MRN = re.compile(r"([A-Za-z]{2,5})-([A-Za-z]{0,3}\d{3,10})")
RE_SERIES = re.compile(r"_(\d{1,3})$")

# Date forms seen: 27_12_22 / 1_7_24 / 2023.08.12 / 2021_09_27
# NOTE: \b is useless here - "_" is a word character, so "__28_9_22" has no
# boundary before the day. Digit lookarounds are what actually anchor these,
# and they also stop the pattern matching inside a long MRN.
RE_DATE = re.compile(
    r"(?<!\d)("
    r"\d{4}[._]\d{1,2}[._]\d{1,2}"             # 2023.08.12 | 2021_09_27
    r"|\d{1,2}_\d{1,2}_\d{2,4}"                # 27_12_22 | 1_7_24
    r")(?!\d)"
)


def split_prefix(stem: str) -> str:
    """Everything before the acquisition timestamp block."""
    if "__" in stem:
        return stem.split("__", 1)[0]
    for marker in ("_Imaging_", "_imaging_"):
        if marker in stem:
            return stem.split(marker, 1)[0]
    m = RE_DATE.search(stem)
    if m:
        return stem[: m.start()].rstrip("_ ")
    return stem


def parse_identity(stem: str) -> dict:
    """
    Returns centre, mrn, and two candidate patient keys.

    mrn_key    - identity token only; robust to name spelling drift
    prefix_key - token + name; over-splits if a name is typed differently
    """
    prefix = split_prefix(stem)

    centre, mrn, is_vc = None, None, False

    m = RE_VC.match(prefix)                      # VC-BEL-N1067992
    if m:
        centre, mrn, is_vc = m.group(1).upper(), m.group(2).upper(), True
    else:
        m = RE_CENTRE_MRN.match(prefix)          # ADL-P67624
        if m:
            centre, mrn = m.group(1).upper(), m.group(2).upper()
        else:
            m = RE_BARE_MRN.match(prefix)        # 1104450_MUD-P79267
            if m:
                mrn = m.group(1).upper()
                m2 = RE_EMBEDDED_CENTRE_MRN.search(prefix)
                if m2:
                    centre = m2.group(1).upper()
                    mrn = f"{mrn}+{m2.group(2).upper()}"
            else:
                m3 = RE_EMBEDDED_CENTRE_MRN.search(prefix)   # Laxmaiah Bhukya_PCH-P124660
                if m3:
                    centre, mrn = m3.group(1).upper(), m3.group(2).upper()

    mrn_key = f"{centre or 'NA'}-{mrn}" if mrn else None
    prefix_key = re.sub(r"\s+", " ", prefix).strip().upper()

    series = None
    ms = RE_SERIES.search(stem)
    if ms:
        series = int(ms.group(1))

    md = RE_DATE.search(stem)
    visit_raw = md.group(0) if md else None

    return {
        "centre": centre,
        "mrn": mrn,
        "mrn_key": mrn_key,
        "prefix_key": prefix_key,
        "is_vision_centre": is_vc,
        "series_no": series,
        "visit_raw": visit_raw,
    }


# =====================================================
# IMAGE PROBE
# =====================================================
def probe_image(path: Path) -> dict:
    out = {
        "readable": False, "width": None, "height": None,
        "megapixels": None, "aspect": None, "mode": None,
        "fmt": None, "camera": None, "error": None,
    }
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            w, h = im.size
            out.update(
                readable=True, width=w, height=h,
                megapixels=round(w * h / 1e6, 2),
                aspect=round(w / h, 4) if h else None,
                mode=im.mode, fmt=im.format,
            )
            try:
                exif = im.getexif()
                if exif:
                    tags = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
                    out["camera"] = str(tags.get("Model", "")).strip() or None
            except Exception:
                pass
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def content_hash(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# =====================================================
# BUILD
# =====================================================
def build_rows() -> pd.DataFrame:
    rows = []
    for cls, label in CLASS_TO_LABEL.items():
        cls_dir = RAW_DIR / cls
        if not cls_dir.exists():
            raise RuntimeError(f"Missing class directory: {cls_dir}")

        files = sorted(p for p in cls_dir.iterdir()
                       if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff"))
        print(f"  {cls:10s} {len(files)} images")

        for p in files:
            rec = {
                "image_id": p.stem,
                "filename": p.name,
                "rel_path": str(p.relative_to(ROOT)).replace("\\", "/"),
                "class_name": cls,
                "label": label,
                "bytes": p.stat().st_size,
            }
            rec.update(parse_identity(p.stem))
            rec.update(probe_image(p))
            rec["md5"] = content_hash(p)
            rows.append(rec)

    return pd.DataFrame(rows)


def choose_patient_key(df: pd.DataFrame) -> str:
    """
    Pick the grouping key. mrn_key is preferred (immune to name drift) but is
    only usable if it parsed for every row.
    """
    n_missing = df["mrn_key"].isna().sum()
    n_mrn = df["mrn_key"].nunique(dropna=True)
    n_prefix = df["prefix_key"].nunique()
    print(f"\n  mrn_key    -> {n_mrn} unique ({n_missing} unparsed)")
    print(f"  prefix_key -> {n_prefix} unique")

    if n_missing == 0:
        print("  using mrn_key for patient grouping")
        return "mrn_key"
    print("  mrn_key incomplete -> falling back to prefix_key")
    return "prefix_key"


def assign_splits(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """
    Locked test set by patient, then StratifiedGroupKFold over the remainder.
    A patient never appears in two partitions.
    """
    # Patient-level stratum. If a patient carries both labels, stratify on the
    # majority and flag the conflict - it is a labelling event, not noise.
    pat = (df.groupby(key)["label"]
             .agg(["mean", "size"])
             .rename(columns={"mean": "label_mean", "size": "n_images"}))
    pat["stratum"] = (pat["label_mean"] >= 0.5).astype(int)
    pat["label_conflict"] = (pat["label_mean"] > 0) & (pat["label_mean"] < 1)

    n_conflict = int(pat["label_conflict"].sum())
    if n_conflict:
        print(f"  [!] {n_conflict} patients carry BOTH labels "
              f"({int(df[key].isin(pat.index[pat['label_conflict']]).sum())} images)")

    patients = pat.index.to_numpy()
    strata = pat["stratum"].to_numpy()

    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_FRACTION, random_state=SEED)
    dev_idx, test_idx = next(sss.split(patients, strata))
    test_patients = set(patients[test_idx])

    df["split"] = np.where(df[key].isin(test_patients), "test", "dev")
    df["fold"] = -1

    dev = df[df["split"] == "dev"]
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (_, va) in enumerate(sgkf.split(dev, dev["label"], groups=dev[key])):
        df.loc[dev.index[va], "fold"] = fold

    df["patient_key"] = df[key]
    return df


def verify_splits(df: pd.DataFrame, key: str) -> list:
    """Hard assertions - a silent split bug is the whole reason we are here."""
    problems = []

    overlap = (set(df.loc[df.split == "test", key]) &
               set(df.loc[df.split == "dev", key]))
    if overlap:
        problems.append(f"{len(overlap)} patients span test and dev")

    dev = df[df.split == "dev"]
    for f in range(N_FOLDS):
        a = set(dev.loc[dev.fold == f, key])
        b = set(dev.loc[dev.fold != f, key])
        if a & b:
            problems.append(f"fold {f}: {len(a & b)} patients leak across folds")

    if (dev.fold < 0).any():
        problems.append(f"{int((dev.fold < 0).sum())} dev rows unassigned to a fold")

    dup = df[df.duplicated("md5", keep=False)]
    if len(dup):
        cross = dup.groupby("md5")["class_name"].nunique()
        n_cross = int((cross > 1).sum())
        problems.append(
            f"{len(dup)} rows share {dup['md5'].nunique()} content hashes"
            + (f"; {n_cross} hash groups span BOTH classes" if n_cross else "")
        )

    bad = df[~df.readable]
    if len(bad):
        problems.append(f"{len(bad)} unreadable images")

    return problems


# =====================================================
# REPORT
# =====================================================
def write_report(df: pd.DataFrame, key: str, problems: list) -> dict:
    n_pat = df[key].nunique()
    bac, fun = SOURCE_PREVALENCE_BAC_FUN

    summary = {
        "n_images": len(df),
        "n_patients": int(n_pat),
        "images_per_patient_mean": round(len(df) / n_pat, 3),
        "class_counts": df["class_name"].value_counts().to_dict(),
        "patient_key_used": key,
        "n_distinct_content": int(df["md5"].nunique()),
        "n_duplicate_rows": int(len(df) - df["md5"].nunique()),
        "resolutions": {f"{w}x{h}": int(c) for (w, h), c in
                        df.groupby(["width", "height"]).size().sort_values(ascending=False).items()},
        "cameras": {str(k): int(v) for k, v in df["camera"].fillna("none").value_counts().items()},
        "centres": {str(k): int(v) for k, v in df["centre"].fillna("none").value_counts().items()},
        "vision_centre_counts": df.groupby(["is_vision_centre", "class_name"]).size().unstack(fill_value=0).to_dict(),
        "split_counts": df.groupby(["split", "class_name"]).size().unstack(fill_value=0).to_dict(),
        "fold_counts": df[df.split == "dev"].groupby(["fold", "class_name"]).size().unstack(fill_value=0).to_dict(),
        "cohort_prevalence_fungal": round(float((df.label == 1).mean()), 4),
        "source_prevalence_fungal": round(fun / (bac + fun), 4),
        "problems": problems,
        "seed": SEED,
        "n_folds": N_FOLDS,
        "test_fraction": TEST_FRACTION,
    }

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_DIR / "manifest_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    L = []
    L.append("# Phase 0 - Manifest Report\n")
    L.append(f"- images **{summary['n_images']}** | patients **{summary['n_patients']}** "
             f"| {summary['images_per_patient_mean']} images/patient")
    L.append(f"- classes: {summary['class_counts']}")
    L.append(f"- patient key: `{key}`")
    L.append(f"- distinct content hashes: {summary['n_distinct_content']} "
             f"(duplicate rows: {summary['n_duplicate_rows']})\n")

    L.append("## Cohort vs deployment prevalence\n")
    L.append(f"- this cohort is **{summary['cohort_prevalence_fungal']:.1%}** fungal (curated ~1:1)")
    L.append(f"- source population is **{summary['source_prevalence_fungal']:.1%}** fungal "
             f"({bac} bacterial : {fun} fungal)")
    L.append("- every headline metric must also be reported reweighted to source prevalence\n")

    L.append("## Resolutions\n")
    for k, v in list(summary["resolutions"].items())[:10]:
        L.append(f"- {k}: {v}")
    L.append("\n## Acquisition device\n")
    for k, v in list(summary["cameras"].items())[:10]:
        L.append(f"- {k}: {v}")

    L.append("\n## Vision Centre vs tertiary capture\n")
    L.append("`VC-` filenames are rural Vision Centre screening captures - the most likely "
             "source of device/acquisition shift, and the arm the teleophthalmology workflow "
             "actually deploys into.\n")
    L.append(df.groupby(["is_vision_centre", "class_name"]).size().unstack(fill_value=0).to_markdown())

    L.append("\n## Splits (patient-disjoint)\n")
    piv = df.groupby(["split", "class_name"]).size().unstack(fill_value=0)
    L.append(piv.to_markdown())
    L.append("\n### Dev folds\n")
    piv2 = df[df.split == "dev"].groupby(["fold", "class_name"]).size().unstack(fill_value=0)
    L.append(piv2.to_markdown())

    L.append("\n## Integrity checks\n")
    if problems:
        for p in problems:
            L.append(f"- ⚠ {p}")
    else:
        L.append("- ✅ no patient leakage, no duplicates, all images readable")

    L.append("\n## Known open item\n")
    L.append("- **Reference standard is unrecorded.** Labels derive from the folder the image "
             "was filed in. No culture / KOH / smear / confocal result is linked to any image. "
             "This bounds every claim the model can make and must be resolved with the "
             "clinical team before external reporting.\n")

    (REPORT_DIR / "00_manifest_report.md").write_text("\n".join(L), encoding="utf-8")
    return summary


def main():
    print("=" * 60)
    print("Phase 0 - building manifest")
    print("=" * 60)

    df = build_rows()
    key = choose_patient_key(df)
    df = assign_splits(df, key)
    problems = verify_splits(df, key)

    cols = ["image_id", "filename", "rel_path", "class_name", "label",
            "patient_key", "centre", "mrn", "is_vision_centre", "visit_raw",
            "series_no", "split", "fold", "width", "height", "megapixels",
            "aspect", "mode", "fmt", "camera", "bytes", "md5", "readable", "error"]
    df = df[[c for c in cols if c in df.columns]]

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(MANIFEST_DIR / "manifest.csv", index=False)

    summary = write_report(df, "patient_key", problems)

    print("\n" + "=" * 60)
    print(f"images   {summary['n_images']}")
    print(f"patients {summary['n_patients']}  ({summary['images_per_patient_mean']} img/patient)")
    print(f"test     {int((df.split == 'test').sum())} images")
    print(f"dev      {int((df.split == 'dev').sum())} images across {N_FOLDS} folds")
    if problems:
        print("\nISSUES:")
        for p in problems:
            print(f"  ! {p}")
    else:
        print("\nno integrity issues")
    print(f"\nwrote {MANIFEST_DIR / 'manifest.csv'}")
    print(f"wrote {REPORT_DIR / '00_manifest_report.md'}")


if __name__ == "__main__":
    main()
