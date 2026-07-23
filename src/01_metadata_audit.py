"""
Phase 1a - Confound audit on metadata alone.

The question this answers: can the class be predicted WITHOUT looking at the
cornea? If acquisition date, centre, resolution or file size carry the label,
then any image model trained on this cohort is partly learning provenance
rather than pathology, and its reported accuracy is inflated.

The decisive test is the metadata-only classifier at the bottom. Under a
patient-grouped split it should sit at AUC ~0.50. Anything materially above
that is a shortcut the image model will also find - and exploit.

Outputs
    outputs/reports/01_metadata_audit.md
    outputs/manifests/manifest_dated.csv
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import re
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "outputs" / "manifests" / "manifest.csv"
REPORT_DIR = ROOT / "outputs" / "reports"
SEED = 42
N_FOLDS = 5
N_REPEATS = 20

L = []


def say(s=""):
    print(s)
    L.append(s)


# =====================================================
# DATE PARSING
# =====================================================
def parse_visit_date(raw):
    """
    Observed forms:
        27_12_22      D_M_YY
        1_7_24        D_M_YY
        2023.08.12    YYYY.MM.DD
        2021_09_27    YYYY_MM_DD
    """
    if not isinstance(raw, str) or not raw.strip():
        return pd.NaT
    s = raw.strip()

    m = re.match(r"^(\d{4})[._](\d{1,2})[._](\d{1,2})$", s)
    if m:
        y, mo, d = map(int, m.groups())
    else:
        m = re.match(r"^(\d{1,2})_(\d{1,2})_(\d{2,4})$", s)
        if not m:
            return pd.NaT
        d, mo, y = map(int, m.groups())
        if y < 100:
            y += 2000

    try:
        return pd.Timestamp(year=y, month=mo, day=d)
    except ValueError:
        return pd.NaT


# =====================================================
# HELPERS
# =====================================================
def chi2_table(df, col, label_col="class_name", min_n=1):
    ct = pd.crosstab(df[col].fillna("none"), df[label_col])
    ct = ct[ct.sum(axis=1) >= min_n]
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return ct, None, None
    chi2, p, _, _ = stats.chi2_contingency(ct)
    return ct, chi2, p


def numeric_compare(df, col):
    a = df.loc[df.label == 0, col].dropna()
    b = df.loc[df.label == 1, col].dropna()
    if len(a) < 5 or len(b) < 5:
        return None
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    # rank-biserial effect size: 0 = no separation, 1 = complete
    auc = u / (len(a) * len(b))
    return {
        "bacterial_median": float(np.median(a)),
        "fungal_median": float(np.median(b)),
        "auc": float(max(auc, 1 - auc)),
        "p": float(p),
    }


# =====================================================
# MAIN
# =====================================================
def main():
    df = pd.read_csv(MANIFEST)
    say("# Phase 1a - Metadata Confound Audit\n")
    say(f"{len(df)} images | {df.patient_key.nunique()} patients\n")

    # ---------- effective sample size ----------
    say("## Effective sample size (patient-bounded)\n")
    pat = df.groupby(["class_name"])["patient_key"].nunique()
    imgs = df.groupby("class_name").size()
    tbl = pd.DataFrame({"images": imgs, "patients": pat})
    tbl["images_per_patient"] = (tbl.images / tbl.patients).round(3)
    say(tbl.to_markdown())
    say("")

    # ---------- dates ----------
    # After de-identification the exact calendar date is gone; a shifted
    # day_index and the year survive. Both paths are supported so this audit can
    # be re-run on the de-identified manifest without leaking a date back in.
    deid = "visit_raw" not in df.columns
    if deid:
        df["visit_date"] = pd.NaT
        df["month"] = np.nan
        n_bad = 0
    else:
        df["visit_date"] = df["visit_raw"].apply(parse_visit_date)
        n_bad = int(df.visit_date.isna().sum())
        df["year"] = df.visit_date.dt.year
        df["month"] = df.visit_date.dt.month
        df["day_index"] = (df.visit_date - df.visit_date.min()).dt.days

    say("## Temporal distribution\n")
    if deid:
        say("- source: de-identified manifest (`year` + shifted `day_index`; "
            "no calendar dates)\n")
    say(f"- unparsable dates: **{n_bad}** of {len(df)}")
    if n_bad < len(df):
        if not deid:
            say(f"- range: {df.visit_date.min().date()} -> {df.visit_date.max().date()}\n")
        ct = pd.crosstab(df.year, df.class_name)
        say(ct.to_markdown())
        _, chi2, p = chi2_table(df.dropna(subset=["year"]), "year")
        if p is not None:
            flag = "**CONFOUND**" if p < 0.01 else "ok"
            say(f"\nchi2={chi2:.1f}, p={p:.3g} -> {flag}")
        # class separation on the time axis (day_index is shift-invariant)
        a = df.loc[df.label == 0, "day_index"].dropna()
        b = df.loc[df.label == 1, "day_index"].dropna()
        if len(a) and len(b):
            gap = abs(float(a.median()) - float(b.median()))
            say(f"\n- bacterial median day index: **{a.median():.0f}**")
            say(f"- fungal median day index:    **{b.median():.0f}**")
            say(f"- separation: **{gap:.0f} days**")
    say("")

    # ---------- centre ----------
    say("## Centre\n")
    ct, chi2, p = chi2_table(df, "centre", min_n=5)
    say(ct.to_markdown())
    if p is not None:
        flag = "**CONFOUND**" if p < 0.01 else "ok"
        say(f"\nchi2={chi2:.1f}, p={p:.3g} -> {flag}")
    say(f"\n- distinct centres: {df.centre.nunique()}")
    say(f"- images with no centre code: {int(df.centre.isna().sum())}")
    say("")

    # ---------- device / resolution ----------
    say("## Acquisition\n")
    df["resolution"] = df.width.astype(str) + "x" + df.height.astype(str)
    for col in ["resolution", "camera", "is_vision_centre"]:
        ct, chi2, p = chi2_table(df, col)
        say(f"### {col}\n")
        say(ct.to_markdown())
        if p is not None:
            flag = "**CONFOUND**" if p < 0.01 else "ok"
            say(f"\nchi2={chi2:.1f}, p={p:.3g} -> {flag}\n")
        else:
            say("")

    # ---------- numeric metadata ----------
    say("## Numeric metadata (Mann-Whitney, AUC = separability)\n")
    rows = []
    for col in ["bytes", "megapixels", "aspect", "series_no"]:
        r = numeric_compare(df, col)
        if r:
            rows.append({"feature": col, **r})
    if rows:
        rdf = pd.DataFrame(rows)
        rdf["flag"] = np.where((rdf.p < 0.01) & (rdf.auc > 0.60), "CONFOUND", "ok")
        say(rdf.round(4).to_markdown(index=False))
    say("")

    # ---------- THE DECISIVE TEST ----------
    say("## Metadata-only classifier\n")
    say("Predict bacterial-vs-fungal from provenance alone - no pixels. "
        "Patient-grouped 5-fold CV.\n")

    feats = pd.DataFrame(index=df.index)
    feats["year"] = df.year
    feats["month"] = df.month
    feats["days"] = df.day_index
    feats["bytes"] = df.bytes
    feats["width"] = df.width
    feats["height"] = df.height
    feats["series_no"] = df.series_no
    feats["is_vc"] = df.is_vision_centre.astype(int)
    # centre as frequency encoding (avoids leaking identity through one-hot)
    freq = df.centre.value_counts(normalize=True)
    feats["centre_freq"] = df.centre.map(freq).fillna(0)

    y = df.label.values
    groups = df.patient_key.values

    # Repeated CV, not a single split. A single StratifiedGroupKFold draw has
    # sd ~0.015 on this cohort, so any one number is +/-0.03 of noise - enough
    # to move a verdict a whole category. Report the distribution instead.
    clf = HistGradientBoostingClassifier(max_depth=3, max_iter=200, random_state=SEED)
    aucs = []
    for s in range(N_REPEATS):
        cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=s)
        p = cross_val_predict(clf, feats, y, cv=cv, groups=groups, method="predict_proba")[:, 1]
        aucs.append(roc_auc_score(y, p))
    aucs = np.array(aucs)
    auc_meta = float(aucs.mean())

    say(f"### AUC = **{auc_meta:.4f}** +/- {aucs.std():.4f} "
        f"(mean +/- sd over {N_REPEATS} fold assignments; "
        f"range {aucs.min():.4f}-{aucs.max():.4f})\n")
    if auc_meta >= 0.70:
        say("> **SEVERE CONFOUND.** Provenance alone predicts the label well. Any image "
            "model on this cohort will inherit a large shortcut, and its headline number "
            "will not reflect pathology. Must be controlled before modelling.")
    elif auc_meta >= 0.60:
        say("> **MATERIAL CONFOUND.** Metadata carries real signal. Image-model results "
            "must be reported alongside this number, and confound-controlled variants "
            "(e.g. date-matched or centre-stratified splits) are required.")
    elif auc_meta >= 0.55:
        say("> **MILD CONFOUND.** Small but non-zero. Worth reporting; unlikely to "
            "dominate an image model.")
    else:
        say("> **CLEAN.** Metadata carries essentially no label information. "
            "Image-model performance can be attributed to image content.")
    say("")

    # per-feature ablation to find the culprit
    if auc_meta >= 0.55:
        say("### Which field carries it?\n")
        ab = []
        for c in feats.columns:
            p1 = cross_val_predict(clf, feats[[c]], y, cv=cv, groups=groups,
                                   method="predict_proba")[:, 1]
            ab.append({"feature": c, "solo_auc": round(roc_auc_score(y, p1), 4)})
        say(pd.DataFrame(ab).sort_values("solo_auc", ascending=False).to_markdown(index=False))
        say("")

    # ---------- both-label patients ----------
    conf = (df.groupby("patient_key")["label"].nunique() > 1)
    conf_pat = conf[conf].index.tolist()
    if conf_pat:
        say("## Patients with both culture results\n")
        sub = df[df.patient_key.isin(conf_pat)][
            ["patient_key", "class_name", "year", "day_index", "image_id"]
        ].sort_values(["patient_key", "day_index"])
        say(sub.to_markdown(index=False))
        say("\nWith culture-proven labels these are substantive: mixed infection, "
            "a revised culture, or a second episode. Needs a clinical read.\n")

    df.to_csv(ROOT / "outputs" / "manifests" / "manifest_dated.csv", index=False)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "01_metadata_audit.md").write_text("\n".join(L), encoding="utf-8")
    print(f"\nwrote {REPORT_DIR / '01_metadata_audit.md'}")


if __name__ == "__main__":
    main()
