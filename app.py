"""
LVP Model 2 — Bacterial vs Fungal Keratitis

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

st.set_page_config(page_title="Keratitis AI — Bacterial vs Fungal",
                   page_icon="👁", layout="wide")

# Modes are read from the model's calibration. Cautious is the recommended
# default: it commits only when confident and refers the rest for culture.
MODE_NAMES = {"selective": "Cautious (recommended)",
              "fungal_safety": "Fungal-safety", "balanced": "Balanced"}
MODE_HELP = {
    "selective": "Commits only when confident and marks the rest ‘Not Sure’ — the safe default for unclear cases.",
    "fungal_safety": "Maximises fungal detection (≈92% recall); more bacterial cases go to antifungals pending culture.",
    "balanced": "Highest overall accuracy; makes the dangerous fungal→bacterial error more often.",
}
MODE_ORDER = ["selective", "fungal_safety", "balanced"]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stRadio, .stButton, div, p, span, label,
h1, h2, h3, h4 { font-family: 'Inter', -apple-system, "Segoe UI", Roboto, sans-serif; }
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 1.6rem; max-width: 1160px;}
h1, h2, h3 {color: #6a1b2d; letter-spacing:-.2px;}
.brandbar {background:linear-gradient(100deg,#4a1220,#6a1b2d 58%,#9c3450);
  color:#fff;padding:18px 24px;border-radius:12px;margin-bottom:4px;
  box-shadow:0 2px 10px rgba(74,18,32,.18);}
.brandbar h1 {color:#fff;margin:0;font-size:24px;font-weight:800;letter-spacing:-.4px;}
.brandbar p {margin:3px 0 0;opacity:.88;font-size:12.5px;font-weight:400;}
.verdict {border-radius:14px;padding:22px 24px;text-align:center;}
.verdict .lab {font-size:32px;font-weight:800;line-height:1;letter-spacing:-.5px;}
.verdict .sub {font-size:13px;margin-top:7px;opacity:.9;font-weight:500;}
.v-fung {background:#eaf5ec;border:1px solid #b6ddc0;color:#1e6b34;}
.v-bact {background:#eef2fb;border:1px solid #c2d0ef;color:#274b8f;}
.v-uns  {background:#fbf3e6;border:1px solid #e6d4a8;color:#8a6414;}
.chip {display:inline-block;background:#f3e9eb;border:1px solid #e3d5d8;color:#4a1220;
  border-radius:16px;padding:6px 14px;margin:3px 4px 0 0;font-size:13px;font-weight:600;}
.stImage img {border-radius:10px;}
div[role="radiogroup"] label {padding:3px 0;font-size:14px;}
div[role="radiogroup"] label p {font-weight:500;}
section[data-testid="stSidebar"] {background:#faf6f7;}
section[data-testid="stSidebar"] h3 {font-size:13px;text-transform:uppercase;
  letter-spacing:.6px;color:#8a6b72;}
</style>
"""


def _srcver():
    import hashlib
    h = hashlib.md5()
    for f in ["src/inference.py", "outputs/checkpoints/calibration_external.json",
              "outputs/checkpoints/final_model.pt",
              "models/lesion_seg/lesion_unetpp.pth"]:
        fp = Path(__file__).parent / f
        h.update(str(fp.stat().st_mtime if fp.exists() else 0).encode())
    return h.hexdigest()[:12]


@st.cache_resource(show_spinner="Loading models (first run downloads DINOv2)…")
def load_pipeline(version: str):
    import importlib, inference
    importlib.reload(inference)
    inference.Pipeline._instance = None
    return inference.Pipeline.get()


def main():
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="brandbar"><h1>Keratitis AI — Bacterial vs Fungal</h1>'
        '<p>Slit-lamp decision support · frozen DINOv2 · lesion-aware · calibrated & abstaining</p></div>',
        unsafe_allow_html=True)
    st.caption("Note: Decision aid only - this tool can make mistakes. "
               "Please confirm the result clinically before acting.")

    try:
        pipe = load_pipeline(_srcver())
    except Exception as e:
        st.error(f"Could not load models: {e}")
        st.stop()
    from inference import lesion_overlay, evidence_map

    # ---- sidebar: performance + mode + guidance ----
    with st.sidebar:
        st.subheader("Performance")
        a, b = st.columns(2)
        a.metric("Test AUC", f"{pipe.test_auc:.2f}")
        b.metric("External AUC", "0.84")
        st.caption("Retrained on 1,484 images (v2).")

        st.subheader("Decision mode")
        avail = [k for k in MODE_ORDER if k in pipe.modes()]
        mode_key = st.radio("mode", avail, format_func=lambda k: MODE_NAMES.get(k, k),
                            label_visibility="collapsed")
        st.caption(MODE_HELP.get(mode_key, ""))

        st.divider()
        st.caption("For images already confirmed as bacterial or fungal keratitis.")

    up = st.file_uploader("Upload a slit-lamp image",
                          type=["jpg", "jpeg", "png", "tif", "tiff"])
    if up is None:
        st.info("Upload a full-resolution slit-lamp photograph of an infected cornea "
                "to begin.")
        return

    rgb = np.asarray(Image.open(up).convert("RGB"))
    with st.spinner("Analysing…"):
        res = pipe.predict(rgb)
    if "error" in res:
        st.error(res["error"]); return

    p = res["p_fungal"]
    verdict = res["labels_by_mode"].get(mode_key, res["label"])

    # ---- verdict + probability ----
    left, right = st.columns([1, 1.15])
    with left:
        cls = {"Fungal": "v-fung", "Bacterial": "v-bact"}.get(verdict, "v-uns")
        sub = {"Fungal": "Suggestive of fungal keratitis",
               "Bacterial": "Suggestive of bacterial keratitis",
               "Not Sure": "Evidence unclear — smear / culture / confocal advised"}[verdict]
        st.markdown(f'<div class="verdict {cls}"><div class="lab">{verdict}</div>'
                    f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)
        st.write("")
        prob_df = (pd.DataFrame({"Class": ["Bacterial", "Fungal"],
                                 "Probability": [1.0 - p, p]})
                   .sort_values("Probability", ascending=False).reset_index(drop=True))
        st.dataframe(prob_df, hide_index=True, use_container_width=True,
                     column_config={
                         "Class": st.column_config.TextColumn(width="small"),
                         "Probability": st.column_config.ProgressColumn(
                             "Probability", min_value=0.0, max_value=1.0, format="%.2f")})
        st.caption(f"{res['n_tiles']} tiles analysed · {res['tile_mm']:.2f} mm each")
    with right:
        bm = res.get("biomarkers")
        if bm:
            st.markdown("**Clinical signs** &nbsp;<span style='color:#888;font-size:11px'>"
                        "(shown for review — not used in the decision)</span>",
                        unsafe_allow_html=True)
            chips = []
            chips.append(f"Hypopyon: {'present' if bm['hypopyon_present'] else 'absent'}")
            if bm["hypopyon_present"]:
                chips.append(f"Flatness {bm['hypopyon_flatness']:.1f}")
            chips.append(f"Infiltrate foci: {int(bm['infiltrate_n_comp'])}")
            chips.append(f"Margin solidity {bm['infiltrate_solidity']:.2f}")
            st.markdown("".join(f'<span class="chip">{c}</span>' for c in chips),
                        unsafe_allow_html=True)
            st.caption("Flat crescent & compact margin lean bacterial; multifocal & "
                       "spiky margin lean fungal.")

    st.write("")
    v1, v2 = st.columns(2)
    with v1:
        st.markdown("**Detected anatomy**")
        st.image(lesion_overlay(rgb, res), use_container_width=True)
        st.caption("🟢 limbus · 🔴 infiltrate · 🟣 hypopyon")
    with v2:
        st.markdown("**Evidence map**")
        st.image(evidence_map(rgb, res), use_container_width=True)
        st.caption("Red pushes fungal · blue pushes bacterial (exact per-tile contribution)")

    with st.expander("Technical detail"):
        st.caption(f"{rgb.shape[1]}×{rgb.shape[0]} px · {res['tile_mm']:.2f} mm/tile · "
                   f"{res['mm_per_px']*1000:.1f} µm/px · bag logit {res['logit']:.3f} · "
                   f"temperature {pipe.temperature:.2f}")
        st.dataframe(
            pd.DataFrame({
                "tile": range(len(res["tile_logits"])),
                "cornea %": [round(100 * t["limbus_frac"], 1) for t in res["tiles"]],
                "logit": np.round(res["tile_logits"], 3),
                "pushes": ["fungal" if l > 0 else "bacterial" for l in res["tile_logits"]],
            }).sort_values("logit", ascending=False),
            use_container_width=True, hide_index=True, height=240)


if __name__ == "__main__":
    main()
