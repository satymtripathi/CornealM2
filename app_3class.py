"""
Keratitis AI — 3-class (Bacterial / Fungal / Other)

    streamlit run app_3class.py

Shipped separately from the binary app (app.py).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "src"))
Image.MAX_IMAGE_PIXELS = None

st.set_page_config(page_title="Keratitis AI — 3-class", page_icon="👁", layout="wide")

MODES = {"balanced": "Balanced", "cautious": "Cautious (high precision)"}
CSS = """
<style>
#MainMenu, footer, header {visibility:hidden;}
.block-container{padding-top:2rem;max-width:1150px;}
h1,h2,h3{color:#6a1b2d;}
.brandbar{background:linear-gradient(100deg,#4a1220,#6a1b2d 60%,#9c3450);color:#fff;
  padding:16px 22px;border-radius:10px;margin-bottom:6px;}
.brandbar h1{color:#fff;margin:0;font-size:23px;} .brandbar p{margin:2px 0 0;opacity:.85;font-size:12.5px;}
.verdict{border-radius:12px;padding:18px 22px;text-align:center;}
.verdict .lab{font-size:28px;font-weight:800;} .verdict .sub{font-size:12px;margin-top:5px;opacity:.9;}
.v-fung{background:#eaf5ec;border:1px solid #b6ddc0;color:#1e6b34;}
.v-bact{background:#eef2fb;border:1px solid #c2d0ef;color:#274b8f;}
.v-oth {background:#f0ecf6;border:1px solid #cfc2e6;color:#4a2f7a;}
.v-uns {background:#fbf3e6;border:1px solid #e6d4a8;color:#8a6414;}
.chip{display:inline-block;background:#f3e9eb;border:1px solid #e3d5d8;color:#4a1220;
  border-radius:16px;padding:6px 14px;margin:3px 4px 0 0;font-size:13px;font-weight:600;}
.stImage img{border-radius:10px;}
</style>"""


def _ver():
    import hashlib
    h = hashlib.md5()
    for f in ["src/inference_3class.py", "outputs/checkpoints/model_3class.pt",
              "outputs/checkpoints/calibration_3class.json"]:
        fp = Path(__file__).parent / f
        h.update(str(fp.stat().st_mtime if fp.exists() else 0).encode())
    return h.hexdigest()[:12]


@st.cache_resource(show_spinner="Loading models (first run downloads DINOv2)…")
def load(version):
    import importlib, inference_3class
    importlib.reload(inference_3class)
    inference_3class.Pipeline3._instance = None
    return inference_3class.Pipeline3.get()


def main():
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown('<div class="brandbar"><h1>Keratitis AI — Bacterial / Fungal / Other</h1>'
                '<p>3-class slit-lamp decision support · lesion-aware · calibrated & abstaining</p></div>',
                unsafe_allow_html=True)
    st.caption("⚠️ Decision aid only — this tool can make mistakes. "
               "Please confirm the result clinically before acting.")
    try:
        pipe = load(_ver())
    except Exception as e:
        st.error(f"Could not load models: {e}"); st.stop()
    from inference_3class import lesion_overlay  # self-contained

    with st.sidebar:
        st.subheader("Performance")
        st.metric("Dev macro AUC", f"{pipe.dev_macro:.2f}")
        if pipe.test_macro:
            st.caption(f"Test macro AUC {pipe.test_macro:.2f} (locked, once)")
        st.subheader("Decision mode")
        mode = st.radio("m", list(MODES), format_func=lambda k: MODES[k], label_visibility="collapsed")
        mm = pipe.modes()[mode]
        st.caption(f"Answers ~{mm['coverage']:.0%} of cases at {mm['acc_covered']:.0%} accuracy "
                   f"(dev). Below confidence → 'Not Sure'.")
        st.divider()
        st.caption("**Other** = Acanthamoeba / Nocardia and other non-bacterial-non-fungal "
                   "organisms — a small, heterogeneous class; treat its calls as provisional. "
                   "For images already confirmed as microbial keratitis.")

    up = st.file_uploader("Upload a slit-lamp image", type=["jpg", "jpeg", "png", "tif", "tiff"])
    if up is None:
        st.info("Upload a full-resolution slit-lamp photograph of an infected cornea."); return
    rgb = np.asarray(Image.open(up).convert("RGB"))
    with st.spinner("Analysing…"):
        res = pipe.predict(rgb, mode=mode)
    if "error" in res:
        st.error(res["error"]); return

    v = res["verdict"]
    cls = {"Fungal": "v-fung", "Bacterial": "v-bact", "Other": "v-oth"}.get(v, "v-uns")
    sub = {"Fungal": "Suggestive of fungal keratitis",
           "Bacterial": "Suggestive of bacterial keratitis",
           "Other": "Suggestive of other organism (Acanthamoeba / Nocardia)",
           "Not Sure": "Evidence unclear — smear / culture / confocal advised"}[v]
    left, right = st.columns([1, 1.15])
    with left:
        st.markdown(f'<div class="verdict {cls}"><div class="lab">{v}</div>'
                    f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)
        st.write("")
        prob_df = (pd.DataFrame({"Class": pipe.classes,
                                 "Probability": [res["probs"][k] for k in pipe.classes]})
                   .sort_values("Probability", ascending=False).reset_index(drop=True))
        st.dataframe(prob_df, hide_index=True, use_container_width=True,
                     column_config={
                         "Class": st.column_config.TextColumn(width="small"),
                         "Probability": st.column_config.ProgressColumn(
                             "Probability", min_value=0.0, max_value=1.0, format="%.2f")})
    with right:
        bm = res.get("biomarkers")
        if bm:
            st.markdown("**Clinical signs** &nbsp;<span style='color:#888;font-size:11px'>"
                        "(review only — not used in the decision)</span>", unsafe_allow_html=True)
            chips = [f"Hypopyon: {'present' if bm['hypopyon_present'] else 'absent'}"]
            if bm["hypopyon_present"]:
                chips.append(f"Flatness {bm['hypopyon_flatness']:.1f}")
            chips += [f"Infiltrate foci: {int(bm['infiltrate_n_comp'])}",
                      f"Margin solidity {bm['infiltrate_solidity']:.2f}"]
            st.markdown("".join(f'<span class="chip">{c}</span>' for c in chips), unsafe_allow_html=True)

    st.write("")
    st.markdown("**Detected anatomy**")
    st.image(lesion_overlay(rgb, res), use_container_width=False, width=520)
    st.caption("🟢 limbus · 🔴 infiltrate · 🟣 hypopyon · "
               f"{res['n_tiles']} tiles · {res['tile_mm']:.2f} mm each")


if __name__ == "__main__":
    main()
