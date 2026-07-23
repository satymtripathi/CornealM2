"""
Phase 1d - De-identification.

Every filename in this cohort is a raw EMR export carrying the patient's name
and MRN, e.g.

    ADL-PN83891_Sorthe Laxmi__Imaging_1_7_24 9_34 AM_003.JPG

and every JPEG carries EXIF with the exact capture timestamp, camera identity
and an embedded thumbnail. None of that may leave this machine.

What this does
    1. assigns stable pseudonymous patient / image IDs
    2. renames every image, mask, overlay and reference file
    3. strips EXIF / XMP / IPTC / comment segments from every JPEG, LOSSLESSLY
    4. rewrites every derived CSV to pseudonymous IDs, dropping name / MRN and
       replacing exact dates with a shifted day index
    5. writes ONE crosswalk, under .phi/, which is the only PHI left
    6. verifies no original name or MRN survives anywhere in the project

Losslessness matters here. Re-saving a JPEG through an image library would
re-encode it and destroy exactly the fine margin texture the whole project
depends on (a fungal filament is 12-49 px at native scale). So EXIF removal is
done by JPEG marker surgery on the byte stream - the entropy-coded image data is
copied verbatim and pixels are provably unchanged.

Pixel redaction is deliberately NOT performed: border regions were inspected and
contain only eyelashes and specular highlights, no burned-in identifiers.

Outputs
    .phi/crosswalk.csv          <- PHI. never share, never commit.
    outputs/reports/05_deidentification.md
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import re
import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[1]
MAN_DIR = ROOT / "outputs" / "manifests"
PHI_DIR = ROOT / ".phi"
REPORT_DIR = ROOT / "outputs" / "reports"
SEED = 42

# JPEG markers to drop. APP0 (JFIF) and APP2 (ICC colour profile) are kept:
# neither carries identity and ICC affects colour interpretation.
DROP_MARKERS = {0xE1, 0xFE, 0xED} | set(range(0xE3, 0xF0))   # APP1(EXIF/XMP), COM, APP13(IPTC), APP3-15


# =====================================================
# LOSSLESS JPEG METADATA STRIP
# =====================================================
def strip_jpeg_metadata(src: Path, dst: Path) -> tuple:
    """
    Remove metadata segments without touching compressed image data.
    Returns (bytes_before, bytes_after). Raises if the file is not a JPEG.
    """
    data = src.read_bytes()
    if data[:2] != b"\xff\xd8":
        raise ValueError(f"not a JPEG: {src.name}")

    out = bytearray(b"\xff\xd8")
    i = 2
    while i < len(data) - 1:
        if data[i] != 0xFF:
            out += data[i:]
            break
        m = data[i + 1]

        if m == 0xD9:                                  # EOI
            out += data[i:]
            break
        if m == 0x01 or 0xD0 <= m <= 0xD7:             # standalone, no length
            out += data[i:i + 2]
            i += 2
            continue

        ln = int.from_bytes(data[i + 2:i + 4], "big")
        if m == 0xDA:                                  # SOS - rest is entropy data
            out += data[i:]
            break

        if m not in DROP_MARKERS:
            out += data[i:i + 2 + ln]
        i += 2 + ln

    dst.write_bytes(bytes(out))
    return len(data), len(out)


def pixels_identical(a: Path, b: Path) -> bool:
    with Image.open(a) as ia, Image.open(b) as ib:
        if ia.size != ib.size:
            return False
        return np.array_equal(np.asarray(ia.convert("RGB")),
                              np.asarray(ib.convert("RGB")))


# =====================================================
# ID ASSIGNMENT
# =====================================================
def build_maps(df: pd.DataFrame):
    """Shuffled patient order so IDs leak no alphabetical or temporal ordering."""
    rng = np.random.default_rng(SEED)
    pats = df.patient_key.unique()
    rng.shuffle(pats)
    pmap = {p: f"P{i + 1:04d}" for i, p in enumerate(pats)}

    imap, counter = {}, {}
    for r in df.sort_values(["patient_key", "image_id"]).itertuples():
        pid = pmap[r.patient_key]
        counter[pid] = counter.get(pid, 0) + 1
        imap[r.image_id] = f"{pid}_{counter[pid]}"
    return pmap, imap


# Not identity: EMR export artefacts and words that legitimately appear in
# documentation. "View Details" is UI text that leaked into some filenames.
NON_IDENTIFYING = {
    "imaging", "image", "images", "jpg", "jpeg", "view", "details", "copy",
    "final", "test", "left", "right", "eye", "photo", "scan", "case", "data",
    "more", "less", "none", "null", "type", "name", "site", "time", "year",
}


def phi_tokens(df: pd.DataFrame) -> set:
    """
    Name and MRN fragments that must not survive anywhere.

    Excluded deliberately:
      - centre codes  - a facility is not a person, and they are kept as a covariate
      - bare years    - permitted, and retained for confound analysis
      - EMR UI artefacts and ordinary English words, which collide with prose
    """
    centres = {str(c).lower() for c in df.get("centre", pd.Series(dtype=str)).dropna().unique()}

    toks = set()
    for r in df.itertuples():
        stem = r.image_id
        prefix = stem.split("__")[0] if "__" in stem else stem.split("_Imaging")[0]
        for t in re.split(r"[_\-\s]+", prefix):
            t = t.strip().lower()
            if len(t) < 4:
                continue
            if t.isdigit():                       # years and short numerics
                continue
            if t in centres or t in NON_IDENTIFYING:
                continue
            toks.add(t)
        if isinstance(r.mrn, str) and r.mrn.strip():
            toks.add(r.mrn.lower())               # MRNs always count
    return toks


# =====================================================
# MAIN
# =====================================================
def main():
    man = pd.read_csv(MAN_DIR / "manifest.csv")
    images_done = bool(man.image_id.str.match(r"^P\d{4}_\d+$").all())

    if images_done:
        # Re-run for the content passes only. Rebuild the maps from the
        # crosswalk, since the manifest no longer holds the original names.
        cwp = PHI_DIR / "crosswalk.csv"
        if not cwp.exists():
            print("images already de-identified and no crosswalk - nothing to do")
            return
        print("images already de-identified - running content scrub + verification only")
        cw0 = pd.read_csv(cwp)
        man = cw0.rename(columns={"image_id": "image_id", "patient_key": "patient_key"})
        pmap = dict(zip(cw0.patient_key, cw0.pseudo_patient_id))
        imap = dict(zip(cw0.image_id, cw0.pseudo_image_id))
    else:
        print(f"de-identifying {len(man)} images / {man.patient_key.nunique()} patients")
        pmap, imap = build_maps(man)
    tokens = phi_tokens(man)
    print(f"tracking {len(tokens)} PHI tokens for verification")

    PHI_DIR.mkdir(exist_ok=True)
    (PHI_DIR / ".gitignore").write_text("*\n", encoding="utf-8")

    stats = {"before": 0, "after": 0, "verified": 0}
    n_lm = n_ov = n_ref = 0

    if not images_done:
        # ---------- crosswalk FIRST, so nothing is unrecoverable ----------
        cw = man[["image_id", "filename", "rel_path", "patient_key", "mrn",
                  "centre", "visit_raw", "class_name"]].copy()
        cw["pseudo_image_id"] = cw.image_id.map(imap)
        cw["pseudo_patient_id"] = cw.patient_key.map(pmap)
        cw.to_csv(PHI_DIR / "crosswalk.csv", index=False)
        print(f"wrote crosswalk -> {PHI_DIR / 'crosswalk.csv'}")

        # ---------- images: rename + strip ----------
        check_idx = set(np.random.default_rng(SEED).choice(len(man), 12, replace=False))

        for n, r in enumerate(tqdm(list(man.itertuples()), desc="images")):
            src = ROOT / r.rel_path
            dst = src.parent / f"{imap[r.image_id]}.jpg"
            if not src.exists():
                continue
            tmp = src.parent / f".tmp_{imap[r.image_id]}.jpg"
            b, a = strip_jpeg_metadata(src, tmp)
            stats["before"] += b
            stats["after"] += a
            if n in check_idx:
                if not pixels_identical(src, tmp):
                    raise RuntimeError(f"LOSSY strip detected on {src.name} - aborting")
                stats["verified"] += 1
            src.unlink()
            tmp.rename(dst)

        # ---------- limbus masks ----------
        lm_dir = ROOT / "data" / "interim" / "limbus"
        for old, new in imap.items():
            p = lm_dir / f"{old}.npz"
            if p.exists():
                p.rename(lm_dir / f"{new}.npz")
                n_lm += 1

    # ---------- overlays (captions were class+metrics only; names are in filenames) ----------
    ov_dir = ROOT / "outputs" / "figures" / "limbus_overlays"
    if ov_dir.exists() and not images_done:
        for f in list(ov_dir.glob("*.jpg")):
            hit = next((new for old, new in imap.items()
                        if old[:44] in f.stem), None)
            if hit:
                cls = f.stem.split("_")[0]
                f.rename(ov_dir / f"{cls}_{hit}.jpg")
                n_ov += 1
            else:
                f.unlink()

    # ---------- reference folders ----------
    for sub in (["finaldataset", "globalonly", "precomputed_tiles"] if not images_done else []):
        base = ROOT / "data" / "reference" / sub
        if not base.exists():
            continue
        for cls_dir in base.iterdir():
            if not cls_dir.is_dir():
                continue
            for item in list(cls_dir.iterdir()):
                stem = item.stem
                new = imap.get(stem)
                if new is None:                       # e.g. "<id>_polar_128"
                    new = next((v for k, v in imap.items() if stem.startswith(k)), None)
                    if new and stem != new:
                        new = new + stem[len(next(k for k in imap if stem.startswith(k))):]
                if new is None:
                    shutil.rmtree(item) if item.is_dir() else item.unlink()
                    continue
                item.rename(cls_dir / (new + item.suffix))
                n_ref += 1

    # ---------- derived tables ----------
    def deid_table(path: Path):
        if not path.exists():
            return False
        d = pd.read_csv(path)
        if "image_id" in d:
            d["image_id"] = d.image_id.map(imap).fillna(d.image_id)
        if "patient_key" in d:
            d["patient_key"] = d.patient_key.map(pmap).fillna(d.patient_key)
        if "filename" in d:
            d["filename"] = d.image_id.astype(str) + ".jpg"
        if "rel_path" in d and "class_name" in d:
            d["rel_path"] = "data/raw/" + d.class_name + "/" + d.image_id.astype(str) + ".jpg"
        # dates: keep year, replace calendar date with a shifted day index
        if "visit_raw" in d:
            d = d.drop(columns=["visit_raw"])
        if "visit_date" in d:
            vd = pd.to_datetime(d.visit_date, errors="coerce")
            d["year"] = vd.dt.year
            d["day_index"] = (vd - vd.min()).dt.days
            d = d.drop(columns=["visit_date"])
            if "year_month" in d:
                d = d.drop(columns=["year_month"])
        for c in ["mrn", "mrn_key", "prefix_key"]:
            if c in d:
                d = d.drop(columns=[c])
        d.to_csv(path, index=False)
        return True

    tables = ["manifest.csv", "manifest_dated.csv", "image_qc.csv", "limbus_geometry.csv"]
    done = [t for t in tables if deid_table(MAN_DIR / t)]

    # ---------- scrub PHI inside file CONTENTS ----------
    # The old pipeline's tile manifests embed the original patient-named path in
    # a "file" field, so renaming the .json was not enough.
    n_json = 0
    ordered = sorted(imap.items(), key=lambda kv: -len(kv[0]))   # longest first
    for jp in (ROOT / "data" / "reference").rglob("*.json"):
        try:
            txt = jp.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        new_txt = txt
        for old, new in ordered:
            if old in new_txt:
                new_txt = new_txt.replace(old, new)
        if new_txt != txt:
            jp.write_text(new_txt, encoding="utf-8")
            n_json += 1

    # ---------- verification ----------
    # Token boundaries: plain substring matching produces false positives
    # ("devi" inside "device"). Underscore must count as a separator, so \b is
    # unusable - "n478171" in "N478171_Pullanna" has no \b before the "_".
    # Require that a match is not flanked by another alphanumeric.
    tok_re = re.compile(
        r"(?<![a-z0-9])(" + "|".join(sorted(map(re.escape, tokens), key=len, reverse=True))
        + r")(?![a-z0-9])"
    ) if tokens else None

    # This report lists the tokens it found, so scanning it matches itself.
    SELF = REPORT_DIR / "05_deidentification.md"

    def scan():
        name_hits, content_hits = [], []
        for p in ROOT.rglob("*"):
            if PHI_DIR in p.parents or p == PHI_DIR or p == SELF:
                continue
            m = tok_re.search(p.name.lower())
            if m:
                name_hits.append((str(p.relative_to(ROOT)), m.group(1)))
            if p.is_file() and p.suffix.lower() in (".csv", ".md", ".json", ".txt", ".yaml"):
                try:
                    txt = p.read_text(encoding="utf-8", errors="ignore").lower()
                except Exception:
                    continue
                m = tok_re.search(txt)
                if m:
                    content_hits.append((str(p.relative_to(ROOT)), m.group(1)))
        return name_hits, content_hits

    name_hits, content_hits = scan()

    # ---------- report ----------
    saved = (stats["before"] - stats["after"]) / 1e6
    L = ["# Phase 1d - De-identification\n"]
    L.append(f"- images processed: **{len(man)}**")
    L.append(f"- patients pseudonymised: **{man.patient_key.nunique()}** (`P0001`..)")
    L.append(f"- limbus masks renamed: {n_lm} | overlays: {n_ov} | reference files: {n_ref} | json scrubbed: {n_json}")
    L.append(f"- derived tables rewritten: {', '.join(done)}\n")

    L.append("## What was removed\n")
    L.append("| item | action |\n|---|---|")
    L.append("| patient name + MRN in filename | replaced with `P####_n` |")
    L.append("| EXIF (capture timestamp, camera identity, embedded thumbnail) | stripped |")
    L.append("| XMP (APP1) and IPTC (APP13) | stripped |")
    L.append("| exact visit date | replaced with `year` + shifted `day_index` |")
    L.append("| MRN / centre-MRN columns | dropped from all tables |")
    L.append(f"\nMetadata removed: **{saved:.1f} MB** across {len(man)} files.\n")

    L.append("## What was deliberately kept\n")
    L.append("- **APP0 (JFIF)** and **APP2 (ICC colour profile)** - no identity, and ICC "
             "affects colour interpretation")
    L.append("- **centre code** - a facility, not a person, and a useful covariate")
    L.append("- **year + shifted day index** - preserves temporal ordering for confound "
             "checks without exposing a calendar date")
    L.append("- **pixels, untouched** - border regions were inspected and contain only "
             "eyelashes and specular highlights; no burned-in identifiers\n")

    L.append("## Losslessness\n")
    L.append(f"Metadata was removed by JPEG marker surgery on the byte stream; entropy-coded "
             f"image data is copied verbatim. **{stats['verified']} files** were decoded "
             f"before and after and compared pixel-by-pixel - all identical.\n")
    L.append("> This matters: re-encoding through an image library would have degraded the "
             "fine margin texture the classifier depends on (a fungal filament is 12-49 px "
             "at 4.09 um/px).\n")

    L.append("## Verification\n")
    L.append(f"Scanned every path and every CSV/MD/JSON in the project against "
             f"**{len(tokens)}** name and MRN tokens taken from the original filenames.\n")
    L.append(f"- filename hits: **{len(name_hits)}**")
    L.append(f"- file-content hits: **{len(content_hits)}**\n")
    if name_hits or content_hits:
        L.append("### Remaining hits\n")
        for p, t in (name_hits + content_hits)[:40]:
            L.append(f"- `{p}` -> `{t}`")
        L.append("")
    else:
        L.append("> No patient name or MRN survives anywhere outside `.phi/`.\n")

    L.append("## The crosswalk\n")
    L.append("`.phi/crosswalk.csv` maps pseudonymous IDs back to the original filenames "
             "and MRNs. It is the **only** PHI-bearing artefact left in the project.\n")
    L.append("- it is git-ignored (`.phi/.gitignore`)")
    L.append("- it must never be shared, published, or copied off this machine")
    L.append("- the original PHI-bearing images still exist at "
             "`Desktop\\Model2LVP\\DatasetModel2` and should be handled under the same rule\n")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "05_deidentification.md").write_text("\n".join(L), encoding="utf-8")

    print(f"\nmetadata removed: {saved:.1f} MB")
    print(f"lossless verified on {stats['verified']} files")
    print(f"PHI hits - names: {len(name_hits)}  contents: {len(content_hits)}")
    print(f"wrote {REPORT_DIR / '05_deidentification.md'}")


if __name__ == "__main__":
    main()
