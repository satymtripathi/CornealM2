"""
Build the combined 3-class dataset (bacterial / fungal / other) from the old
682 cohort + the new DataM2_01 delivery, de-identifying the new images.

Handled explicitly:
  * de-identify new images (pseudonymise filename, lossless EXIF strip)
  * global patient grouping - a patient in both old and new keeps ONE pseudo id
    and ONE split, so nothing leaks across the old/new boundary
  * locked test preserved - old test patients stay test; a stratified slice of
    new-only patients is added so the test set covers all 3 classes
  * dedup: the 5 Nocardia images that were filed "Bacterial" in the old cohort
    are relabelled "other" in place (new label wins), no duplicate bytes
  * "other" = Acanthamoeba + Nocardia

Outputs
    data/raw/{Bacterial,Fungal,Other}/*.jpg      (new de-identified copies added)
    outputs/manifests/manifest_v2.csv
    .phi/crosswalk_v2.csv                          (PHI - new mappings)
    outputs/reports/22_combined_manifest.md
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import re
import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit

warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MAN = ROOT / "outputs" / "manifests"
PHI = ROOT / ".phi"
NEW = ROOT / "DataM2_01"
SEED, N_FOLDS, TEST_FRAC = 42, 5, 0.20

NEW_FOLDERS = {
    "bacteria": ("Bacterial", 0), "Fungal": ("Fungal", 1),
    "Other Organisms/Acanthamoeba_new": ("Other", 2), "Other Organisms/Nocardia": ("Other", 2),
}
LABEL3 = {"Bacterial": 0, "Fungal": 1, "Other": 2}
DROP_MARKERS = {0xE1, 0xFE, 0xED} | set(range(0xE3, 0xF0))


def strip_jpeg(src: Path, dst: Path):
    data = src.read_bytes()
    if data[:2] != b"\xff\xd8":
        dst.write_bytes(data); return
    out = bytearray(b"\xff\xd8"); i = 2
    while i < len(data) - 1:
        if data[i] != 0xFF:
            out += data[i:]; break
        m = data[i + 1]
        if m == 0xD9:
            out += data[i:]; break
        if m == 0x01 or 0xD0 <= m <= 0xD7:
            out += data[i:i + 2]; i += 2; continue
        ln = int.from_bytes(data[i + 2:i + 4], "big")
        if m == 0xDA:
            out += data[i:]; break
        if m not in DROP_MARKERS:
            out += data[i:i + 2 + ln]
        i += 2 + ln
    dst.write_bytes(bytes(out))


def pid(stem):
    pre = stem.split("__")[0] if "__" in stem else re.split(r"_[Ii]maging", stem)[0]
    m = (re.match(r"^(VC-[A-Za-z]{2,4}-[A-Za-z]{0,3}\d{3,10})", pre) or
         re.match(r"^([A-Za-z]{2,5}-[A-Za-z]{0,3}\d{3,10})", pre) or
         re.match(r"^([A-Za-z]{0,3}\d{4,10})", pre))
    return (m.group(1).upper() if m else pre.strip().upper())


def main():
    old = pd.read_csv(MAN / "manifest.csv")
    cw = pd.read_csv(PHI / "crosswalk.csv")
    # old pseudo-patient -> original patient key
    pseudo2orig = dict(zip(cw.pseudo_patient_id, cw.patient_key))
    old["orig_pat"] = old.patient_key.map(pseudo2orig).fillna(old.patient_key)

    # ---- scan new data ----
    new_rows = []
    for sub, (cls, _) in NEW_FOLDERS.items():
        for p in (NEW / sub).iterdir():
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
                new_rows.append({"src": p, "class_name": cls,
                                 "orig_pat": pid(p.stem),
                                 "md5": hashlib.md5(p.read_bytes()).hexdigest()})
    new = pd.DataFrame(new_rows)
    print(f"old {len(old)} | new {len(new)}")

    # ---- the 5 (or more) exact dups: relabel old row to new label, skip new copy ----
    new_by_md5 = dict(zip(new.md5, new.class_name))
    relabelled = 0
    for idx, r in old.iterrows():
        if r.md5 in new_by_md5 and new_by_md5[r.md5] != r.class_name:
            old.at[idx, "class_name"] = new_by_md5[r.md5]
            relabelled += 1
    new = new[~new.md5.isin(set(old.md5))].reset_index(drop=True)   # drop exact dups from new
    print(f"relabelled {relabelled} old rows to new label; new after dedup {len(new)}")

    # ---- global pseudo patient ids ----
    orig2pseudo = dict(zip(old.orig_pat, old.patient_key))   # existing
    next_n = max(int(re.sub(r"\D", "", p) or 0) for p in old.patient_key) + 1
    for op in sorted(new.orig_pat.unique()):
        if op not in orig2pseudo:
            orig2pseudo[op] = f"P{next_n:04d}"; next_n += 1
    new["patient_key"] = new.orig_pat.map(orig2pseudo)

    # ---- de-identify + copy new images ----
    seq = old.patient_key.map(lambda k: 0).to_dict()          # per-patient counter
    counter = {}
    # seed counter from existing image_ids (P####_n)
    for iid in old.image_id:
        m = re.match(r"(P\d+)_(\d+)", str(iid))
        if m:
            counter[m.group(1)] = max(counter.get(m.group(1), 0), int(m.group(2)))
    added = []
    for r in new.itertuples():
        pk = r.patient_key
        counter[pk] = counter.get(pk, 0) + 1
        iid = f"{pk}_{counter[pk]}"
        dst = RAW / r.class_name / f"{iid}.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        strip_jpeg(r.src, dst)
        with Image.open(dst) as im:
            w, h = im.size
        added.append({"image_id": iid, "class_name": r.class_name,
                      "patient_key": pk, "orig_pat": r.orig_pat, "md5": r.md5,
                      "rel_path": str(dst.relative_to(ROOT)).replace("\\", "/"),
                      "width": w, "height": h, "src_name": r.src.name})
    addf = pd.DataFrame(added)

    # ---- crosswalk append (PHI) ----
    cw_new = addf[["image_id", "patient_key", "orig_pat", "class_name", "src_name"]].copy()
    cw_new.to_csv(PHI / "crosswalk_v2.csv", index=False)

    # ---- combined manifest ----
    comb = pd.concat([
        old[["image_id", "class_name", "patient_key", "orig_pat", "md5", "rel_path", "width", "height", "split"]],
        addf.assign(split=np.nan)[["image_id", "class_name", "patient_key", "orig_pat", "md5", "rel_path", "width", "height", "split"]],
    ], ignore_index=True)
    comb["label"] = comb.class_name.map(LABEL3)

    # ---- splits: preserve old locked test; add new-only test slice ----
    # patient-level table
    pat = comb.groupby("patient_key").agg(
        label=("label", lambda s: s.value_counts().index[0]),
        any_old_test=("split", lambda s: (s == "test").any()),
        has_old=("split", lambda s: s.notna().any())).reset_index()

    # patients already assigned test in old stay test
    test_pat = set(pat.loc[pat.any_old_test, "patient_key"])
    # new-only patients (no old split) - stratified sample into test
    newonly = pat[~pat.has_old]
    if len(newonly):
        sss = StratifiedShuffleSplit(1, test_size=TEST_FRAC, random_state=SEED)
        _, te = next(sss.split(newonly.patient_key, newonly.label))
        test_pat |= set(newonly.patient_key.iloc[te])

    comb["split"] = np.where(comb.patient_key.isin(test_pat), "test", "dev")
    comb["fold"] = -1
    dev = comb[comb.split == "dev"]
    sgkf = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED)
    for f, (_, va) in enumerate(sgkf.split(dev, dev.label, groups=dev.patient_key)):
        comb.loc[dev.index[va], "fold"] = f

    # ---- verify ----
    problems = []
    ov = set(comb.loc[comb.split == "test", "patient_key"]) & set(comb.loc[comb.split == "dev", "patient_key"])
    if ov: problems.append(f"{len(ov)} patients span test/dev")
    if comb.md5.duplicated().any(): problems.append(f"{comb.md5.duplicated().sum()} duplicate hashes remain")

    comb.to_csv(MAN / "manifest_v2.csv", index=False)

    # ---- report ----
    L = ["# Combined 3-class manifest\n"]
    L.append(f"Old cohort {len(old)} + new {len(addf)} de-identified = **{len(comb)}** images, "
             f"**{comb.patient_key.nunique()}** patients. {relabelled} Nocardia relabelled "
             f"Bacterial→Other.\n")
    L.append("## Class distribution\n")
    L.append(comb.groupby("class_name").agg(images=("image_id", "size"),
             patients=("patient_key", "nunique")).to_markdown())
    L.append("\n## Splits (patient-disjoint)\n")
    L.append(comb.groupby(["split", "class_name"]).size().unstack(fill_value=0).to_markdown())
    L.append("\n### Dev folds\n")
    L.append(comb[comb.split == "dev"].groupby(["fold", "class_name"]).size().unstack(fill_value=0).to_markdown())
    L.append("\n## Integrity\n")
    L.append("\n".join(f"- ⚠ {p}" for p in problems) if problems else
             "- ✅ no patient leakage, no duplicate hashes")
    L.append("\n\n**PHI:** new mappings in `.phi/crosswalk_v2.csv` (git-ignored). "
             "New images de-identified (pseudonymised + EXIF stripped).")
    (ROOT / "outputs" / "reports" / "22_combined_manifest.md").write_text("\n".join(L), encoding="utf-8")

    print("\n=== combined ===")
    print(comb.groupby(["split", "class_name"]).size().unstack(fill_value=0).to_string())
    print("problems:", problems or "none")
    print(f"wrote {MAN/'manifest_v2.csv'}")


if __name__ == "__main__":
    main()
