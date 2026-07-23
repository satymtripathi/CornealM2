"""
LVP Model 2 - Bacterial vs Fungal Keratitis

    streamlit run app.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "src"))
Image.MAX_IMAGE_PIXELS = None

st.set_page_config(page_title="LVP Model 2 - Bacterial vs Fungal", layout="wide")

# Selective first = default. It halves the dangerous fungal->bacterial error
# versus Balanced (2 vs 7 per 100 patients) and is the most accurate on the
# cases it answers (81.2%).
MODE_LABELS = {
    "selective": "Selective (recommended)",
    "fungal_safety": "Fungal safety",
    "balanced": "Balanced",
}


def _src_version():
    """
    Cache key. Without this, st.cache_resource keeps handing back a Pipeline
    built from an older inference.py, and attribute errors appear after edits.
    """
    import hashlib
    h = hashlib.md5()
    for f in ["src/inference.py", "outputs/checkpoints/calibration_external.json",
              "outputs/checkpoints/final_model.pt"]:
        fp = Path(__file__).parent / f
        h.update(str(fp.stat().st_mtime if fp.exists() else 0).encode())
    return h.hexdigest()[:12]


@st.cache_resource(show_spinner="Loading models (first run downloads DINOv2)...")
def load_pipeline(version: str):
    import importlib, inference
    importlib.reload(inference)
    inference.Pipeline._instance = None
    return inference.Pipeline.get()


def main():
    st.title("Bacterial vs Fungal Keratitis")
    st.caption("Frozen DINOv2 ViT-S/14 · 3.67 mm native tiles · mean-pooled MIL · "
               "15-model ensemble · temperature calibrated")

    try:
        pipe = load_pipeline(_src_version())
    except Exception as e:
        st.error(f"Could not load models: {e}")
        st.stop()

    from inference import overlay_limbus, evidence_map

    # ---------------- sidebar ----------------
    with st.sidebar:
        st.header("Model")
        st.metric("Locked test AUC", f"{pipe.test_auc:.3f}")
        st.caption("131 images, patient-disjoint, used once")

        cal = getattr(pipe, "cal", None)
        if cal:
            st.metric("Pooled external AUC", f"{cal['pooled_auc']:.3f}")
            st.caption(f"{cal['n_calibration']} cases, "
                       f"{len(cal['cohorts'])} independent cohorts")
        st.caption(f"Calibration temperature {pipe.temperature:.3f}")

        st.divider()
        st.subheader("Reading a result")
        st.markdown(
            "Fungal keratitis is far more common in clinic than bacterial. At real "
            "prevalence a **fungal** call is highly reliable, but a **bacterial** "
            "call is not — bacterial recall was 51–63% across validation cohorts.\n\n"
            "**Treat a bacterial call as a prompt to confirm, never as a conclusion.**"
        )

    st.warning(
        "**Scope:** for images already confirmed as bacterial or fungal keratitis. "
        "Any other condition — normal, viral, scar, non-infectious — will still "
        "receive a confident-looking call. In a real review cohort 35% of images "
        "were out of scope and 27 of 35 were confidently mislabelled. Use behind an "
        "infection detector, not on unfiltered images."
    )

    # ---------------- predict ----------------
    up = st.file_uploader("Slit-lamp image",
                          type=["jpg", "jpeg", "png", "tif", "tiff"])
    if up is None:
        st.info("Upload a slit-lamp photograph of an infected cornea. "
                "Full-resolution originals work best — the model reads a 3.67 mm "
                "window and needs the native pixels.")
        return

    mode_key = st.radio(
        "Decision mode", list(MODE_LABELS), horizontal=True,
        format_func=lambda k: MODE_LABELS[k],
        help="Selective may answer 'Indeterminate' and is the most accurate on the "
             "cases it does answer. Fungal safety maximises fungal recall. Balanced "
             "always answers but makes the dangerous error 3.5x more often.")

    rgb = np.asarray(Image.open(up).convert("RGB"))
    st.caption(f"{rgb.shape[1]} × {rgb.shape[0]} px")

    with st.spinner("Segmenting limbus, tiling, embedding..."):
        res = pipe.predict(rgb)

    if "error" in res:
        st.error(res["error"])
        return

    p = res["p_fungal"]
    mode = res["modes"][mode_key]
    verdict = res["labels_by_mode"].get(mode_key, res["label"])
    st.caption(mode["desc"])

    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        if verdict == "Fungal":
            st.success(f"### {verdict}")
        elif verdict == "Bacterial":
            st.info(f"### {verdict}")
            st.caption("Least reliable output — confirm before acting.")
        else:
            st.warning(f"### {verdict}")
            st.caption("Below the confidence required for a call — "
                       "recommend smear / culture / confocal.")
    with c2:
        st.metric("P(fungal)", f"{p:.3f}")
        st.progress(float(p))
    with c3:
        st.metric("Tiles analysed", res["n_tiles"])
        st.caption(f"{res['tile_mm']:.2f} mm each · {res['mm_per_px']*1000:.2f} µm/px")

    st.divider()
    v1, v2 = st.columns(2)
    with v1:
        st.subheader("Limbus")
        st.image(overlay_limbus(rgb, res["contour"]), use_container_width=True)
        st.caption("Green = detected limbus. Tiles are sampled only inside it.")
    with v2:
        st.subheader("Evidence map")
        st.image(evidence_map(rgb, res), use_container_width=True)
        st.caption("Red pushes fungal, blue pushes bacterial. Pooling is a mean and "
                   "the head is linear, so these are **exact** contributions, not a "
                   "saliency approximation.")

    with st.expander("Per-tile detail"):
        st.dataframe(
            pd.DataFrame({
                "tile": range(len(res["tile_logits"])),
                "x": [t["x"] for t in res["tiles"]],
                "y": [t["y"] for t in res["tiles"]],
                "cornea %": [round(100 * t["limbus_frac"], 1) for t in res["tiles"]],
                "glare %": [round(100 * t["glare_frac"], 1) for t in res["tiles"]],
                "logit": np.round(res["tile_logits"], 3),
                "pushes": ["fungal" if l > 0 else "bacterial"
                           for l in res["tile_logits"]],
            }).sort_values("logit", ascending=False),
            use_container_width=True, hide_index=True)
        st.caption(f"Bag logit = mean of tile logits = {res['logit']:.3f} → "
                   f"p = {p:.3f} after temperature {pipe.temperature:.3f}")


if __name__ == "__main__":
    main()
