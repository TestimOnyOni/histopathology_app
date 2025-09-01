import streamlit as st
import torch

from backend.model_loader import load_model_and_threshold, get_device
from backend.preprocessing import handle_uploaded_file
from backend.inference import run_inference_on_patches
from config import MODEL_PATH, THR_PATH, MODEL_URL

st.set_page_config(page_title="Histopathology App", layout="wide")

# ----------------- Cached model loader -----------------
@st.cache_resource
def _load_model_cached(model_path, thr_path, url=None):
    return load_model_and_threshold(
        model_path=model_path,
        thr_path=thr_path,
        num_classes=2,
        map_location=get_device(),
        url=url,
    )

st.title("🧬 Histopathology Classification")
st.write("Upload histopathology image(s) and get predictions.")

# ----------------- Load model -----------------
with st.spinner("Loading model..."):
    model, device, best_thr = _load_model_cached(MODEL_PATH, THR_PATH, url=MODEL_URL)

# ----------------- Uploads -----------------
uploaded_file = st.file_uploader(
    "Upload a single image or a ZIP of images", type=["png", "jpg", "jpeg", "tif", "zip"]
)

if uploaded_file:
    with st.spinner("Processing uploaded file..."):
        patches = handle_uploaded_file(uploaded_file)

    if not patches:
        st.error("No valid images found in upload.")
    else:
        # ----------------- Inference -----------------
        st.info("Running inference on all patches... Please wait ⏳")
        slide_pred, agg_prob, patch_probs = run_inference_on_patches(
            model, device, patches, threshold=best_thr
        )

        # ----------------- Results -----------------
        st.subheader("📊 Slide-level Result")
        st.write(f"**Prediction:** {'Positive' if slide_pred else 'Negative'}")
        st.write(f"**Aggregated probability:** {agg_prob:.4f} (threshold = {best_thr:.2f})")

        st.subheader("📈 Patch-level Summary")
        st.write(f"Number of patches processed: {len(patch_probs)}")
        st.bar_chart(patch_probs)
