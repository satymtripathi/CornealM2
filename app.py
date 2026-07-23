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
    "fungal_safety": "Fungal safety (t=0.25)",
    "balanced": "Balanced (t=0.50)",
}


def _src_version():
    """Cache key. Without this, st.cache_resource keeps returning a Pipeline
    built from an older inference.py and attribute errors appear after edits."""
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
        st.metric("Dev out-of-fold AUC", f"{pipe.dev_auc:.3f}")
        st.caption(f"Calibration temperature {pipe.temperature:.3f}")

        cal = getattr(pipe, "cal", None)
        if cal:
            st.divider()
            st.subheader("External validation")
            st.metric("Pooled external AUC", f"{cal['pooled_auc']:.3f}")
            st.caption(f"{cal['n_calibration']} cases, "
                       f"{len(cal['cohorts'])} independent cohorts")
            st.caption(f"Calibrated temperature {pipe.temperature:.3f}")

        st.divider()
        st.subheader("Read the probability carefully")
        st.markdown(
            "This cohort is curated to **1:1**. In clinic fungal is far more common "
            "(~91% in the source extraction). At real prevalence a **fungal** call is "
            "~97% reliable, but a **bacterial** call is only ~29% reliable — "
            "because bacterial is rare. Treat a bacterial call as a prompt to "
            "confirm, not a conclusion."
        )

    # st.warning(
    #     "**Scope:** for images already confirmed as bacterial or fungal keratitis. "
    #     "Any other condition — including normal, viral, scar or non-infectious — "
    #     "will still receive a confident-looking call. In a real review cohort 35% "
    #     "of images were out of scope and 27 of 35 were confidently mislabelled. "
    #     "Use behind an infection detector, not on unfiltered images."
    # )

    tab_predict, tab_compare, tab_method = st.tabs(
        ["Predict", "vs CornealAI Model 2", "Method"])

    # ---------------- predict ----------------
    with tab_predict:
        up = st.file_uploader("Slit-lamp image", type=["jpg", "jpeg", "png", "tif", "tiff"])
        if up is None:
            st.info("Upload a slit-lamp photograph of an infected cornea. "
                    "Full-resolution originals work best.")
            return

        mode_key = st.radio(
            "Decision mode", list(MODE_LABELS), horizontal=True,
            format_func=lambda k: MODE_LABELS[k],
            help="Balanced maximises overall accuracy. Fungal-safety trades "
                 "bacterial recall for near-zero fungal misroute. Selective may "
                 "decline to answer.")

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
            else:
                st.warning(f"### {verdict}")
                st.caption("Below the confidence required for a call — "
                           "recommend smear / culture / confocal.")
        with c2:
            st.metric("P(fungal)", f"{p:.3f}")
            st.progress(float(p))
        with c3:
            st.metric("Tiles analysed", res["n_tiles"])
            st.caption(f"{res['tile_mm']:.2f} mm each · "
                       f"{res['mm_per_px']*1000:.2f} µm/px")

        st.divider()
        v1, v2 = st.columns(2)
        with v1:
            st.subheader("Limbus")
            st.image(overlay_limbus(rgb, res["contour"]), use_container_width=True)
            st.caption("Green = detected limbus. Tiles are sampled only inside it.")
        with v2:
            st.subheader("Evidence map")
            st.image(evidence_map(rgb, res), use_container_width=True)
            st.caption("Red pushes fungal, blue pushes bacterial. Because pooling is "
                       "a mean and the head is linear, these are **exact** "
                       "contributions — not a saliency approximation.")

        with st.expander("Per-tile detail"):
            df = pd.DataFrame({
                "tile": range(len(res["tile_logits"])),
                "x": [t["x"] for t in res["tiles"]],
                "y": [t["y"] for t in res["tiles"]],
                "cornea %": [round(100 * t["limbus_frac"], 1) for t in res["tiles"]],
                "glare %": [round(100 * t["glare_frac"], 1) for t in res["tiles"]],
                "logit": np.round(res["tile_logits"], 3),
                "pushes": ["fungal" if l > 0 else "bacterial" for l in res["tile_logits"]],
            }).sort_values("logit", ascending=False)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.caption(f"Bag logit = mean of tile logits = {res['logit']:.3f} "
                       f"→ p = {p:.3f} after temperature {pipe.temperature:.3f}")

    # ---------------- comparison ----------------
    with tab_compare:
        st.subheader("Comparison with CornealAI Model 2")
        st.dataframe(pd.DataFrame([
            {"": "Reported AUC", "CornealAI Model 2": "0.949",
             "This model": f"{pipe.test_auc:.3f}"},
            {"": "What that number measures",
             "CornealAI Model 2": "all 686 images — 548 of them its own training data",
             "This model": "131 held-out images, never seen"},
            {"": "Best honest figure available",
             "CornealAI Model 2": "0.862 (138-image val split)",
             "This model": f"{pipe.test_auc:.3f}"},
            {"": "Independent test set", "CornealAI Model 2": "none",
             "This model": "131 images, used once"},
            {"": "Patient-level splitting", "CornealAI Model 2": "no",
             "This model": "yes"},
            {"": "Bag construction",
             "CornealAI Model 2": "label-dependent — extra lesion tile if bacterial, "
                                  "extra hypopyon tile if fungal, in train AND val",
             "This model": "label-free"},
            {"": "Calibration", "CornealAI Model 2": "temperature, fitted on contaminated data",
             "This model": f"temperature {pipe.temperature:.3f} on out-of-fold dev"},
            {"": "Abstention", "CornealAI Model 2": "no", "This model": "yes"},
        ]), use_container_width=True, hide_index=True)

        st.warning(
            "**These numbers are not directly comparable, and it would be wrong to "
            "claim otherwise.** CornealAI's 0.949 includes its own training data. Its "
            "0.862 validation figure is also contaminated, because the tile planner "
            "adds a class-specific tile using the label — in training *and* validation. "
            "Neither can be set against a clean held-out figure. The defensible claim "
            "is that this is the first uncontaminated measurement of the task, not "
            "that it scores higher."
        )
        st.info(
            "Both models were built on the same 686-image cohort (Dataset9 + Dataset14), "
            "so a genuine head-to-head is possible — it would need CornealAI Model 2 "
            "re-run, with the tile-planner leak removed, on this same locked test set."
        )

    # ---------------- method ----------------
    with tab_method:
        st.subheader("How the configuration was chosen")
        st.markdown(
            "Tile field of view was swept; the curve is unimodal and peaks at **3.67 mm** "
            "— which matches the **3.44 mm** median span of the `feathery_margin` sign "
            "in expert annotations. Those two figures were derived independently."
        )
        st.dataframe(pd.DataFrame([
            {"field of view": "0.92 mm", "AUC": 0.745},
            {"field of view": "1.83 mm", "AUC": 0.759},
            {"field of view": "3.67 mm  ←", "AUC": 0.791},
            {"field of view": "5.50 mm", "AUC": 0.775},
            {"field of view": "7.33 mm", "AUC": 0.722},
            {"field of view": "11.7 mm (whole eye)", "AUC": 0.747},
        ]), use_container_width=True, hide_index=True)

        st.markdown(
            "**What did not work:** attention pooling (overfits at this sample size — "
            "0.689 vs 0.746 for plain averaging), full pixel fidelity (−0.009), and "
            "combining multiple scales. Plain mean pooling over 3.67 mm tiles won.\n\n"
            "**Confounds checked and cleared:** metadata alone gives AUC 0.577, global "
            "image statistics 0.543, acquisition-only 0.531. No feature separates the "
            "classes at AUC > 0.60 after correction, so performance is attributable to "
            "the cornea rather than the camera.\n\n"
            "**Known limits:** labels are culture-proven but the cohort is "
            "culture-*positive* only, so culture-negative presentations are unseen. "
            "The test set is 131 images, giving a wide confidence interval. Only 11 "
            "images come from rural Vision Centres, which is the deployment target."
        )


if __name__ == "__main__":
    main()
