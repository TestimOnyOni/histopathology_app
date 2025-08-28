import sys
import os
import streamlit as st
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# Bootstrap sys.path so "backend" is always importable
# -------------------------------------------------------------------
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

# Now imports will work on both local + Streamlit Cloud
from backend.config import MODEL_PATH, THR_PATH, MODEL_URL, DEFAULT_NUM_CLASSES
from backend.model_loader import load_model_and_threshold
from backend.preprocessing import handle_uploaded_file
from backend.inference import run_inference_on_patches
from backend.model_loader import load_model_and_threshold, get_device
# from backend.config import MODEL_PATH, THR_PATH, DEFAULT_NUM_CLASSES, MODEL_URL

# from backend.utils import get_device
# -------------------------------------------------------------------
# Cache model + threshold loading
# -------------------------------------------------------------------
@st.cache_resource
def _load_model_cached(model_path, thr_path=None):
    return load_model_and_threshold(
        model_path=model_path,
        thr_path=thr_path,
        num_classes=2,
        map_location=get_device(),
        url=MODEL_URL  # 👈 provide fallback
    )

# @st.cache_resource
# def _load_model_cached(model_path, thr_path, url=None):
#     return load_model_and_threshold(
#         model_path=model_path,
#         thr_path=thr_path,
#         num_classes=2,
#         map_location=get_device(),
#         url=url
#     )


# Load model once at startup
model, device, best_thr = _load_model_cached(MODEL_PATH, THR_PATH, url=MODEL_URL)

# -------------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------------
st.set_page_config(page_title="Histopathology AI", layout="wide")
st.title("🧬 Histopathology Slide Classifier")

# Upload section
uploaded_file = st.file_uploader("Upload an image or ZIP of patches", type=["png", "jpg", "jpeg", "zip"])

if uploaded_file is not None:
    # Load model
    model, device, best_thr = _load_model_cached(MODEL_PATH, THR_PATH)

    # Extract patches
    st.info("📂 Processing uploaded file...")
    patches = handle_uploaded_file(uploaded_file)

    if not patches:
        st.error("⚠️ No patches were processed from the upload.")
    else:
        # Run inference
        st.info("Running inference on all patches... Please wait ⏳")
        slide_pred, agg_prob, patch_probs = run_inference_on_patches(
            model, device, patches, threshold=best_thr
        )

        # --- Slide-level result ---
        st.subheader("📊 Slide-level Result")
        if slide_pred == 1:
            st.error(f"🔴 Malignant (Aggregated Probability = {agg_prob:.2f}, Threshold = {best_thr:.2f})")
        else:
            st.success(f"🟢 Benign (Aggregated Probability = {agg_prob:.2f}, Threshold = {best_thr:.2f})")

        # --- Patch-level summary ---
        st.subheader("📈 Patch-level Summary")
        st.write(f"Number of patches processed: {len(patch_probs)}")
        st.write(
            f"Patch probabilities → min: {min(patch_probs):.2f}, "
            f"mean: {sum(patch_probs)/len(patch_probs):.2f}, "
            f"max: {max(patch_probs):.2f}"
        )

        # Histogram
        fig, ax = plt.subplots()
        ax.hist(patch_probs, bins=10, color="skyblue", edgecolor="black")
        ax.set_title("Patch Probability Distribution")
        ax.set_xlabel("Probability of Malignant")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)