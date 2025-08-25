import sys
import os

# Ensure backend/ is on sys.path no matter where Streamlit runs from
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from backend.config import (
    DEFAULT_MODEL_PATH, DEFAULT_THRESHOLD_PATH,
    DEFAULT_NUM_CLASSES, DEFAULT_AGG_METHOD, DEFAULT_PERCENTILE,
    DEFAULT_THRESHOLD
)
from backend.model_loader import load_model_and_threshold, get_device
from backend.preprocessing import (
    preprocess_single_image, load_image_from_bytes, preprocess_zip_to_tensor_list
)
from backend.inference import run_single_image, run_zip_patches
from backend.file_io import secure_filename, human_readable_prob
from backend.utils import set_seed

# ------------- Page config (first Streamlit call) -------------
st.set_page_config(page_title="BreakHis Classifier", page_icon="🧬", layout="centered")

# ------------- Title / Intro -------------
st.title("🧬 BreakHis Slide Classifier")
st.caption("ResNet50 • Percentile aggregation • Tunable threshold")

# ------------- Seed for reproducibility -------------
set_seed(1337)

# ------------- Sidebar: model + params -------------
with st.sidebar:
    st.header("⚙️ Settings")
    model_path = st.text_input("Model path", value=str(DEFAULT_MODEL_PATH))
    thr_path   = st.text_input("Threshold JSON path", value=str(DEFAULT_THRESHOLD_PATH))

    agg_method = st.selectbox("Aggregation method", options=["percentile", "mean", "max"], index=0)
    percentile = st.slider("Percentile (if used)", 50, 100, DEFAULT_PERCENTILE)
    override_threshold = st.checkbox("Override threshold?", value=False)
    manual_threshold = st.slider("Manual threshold", 0.0, 1.0, DEFAULT_THRESHOLD, 0.01)

    st.markdown("---")
    st.write("Upload either a **single image** or a **ZIP of patches**:")

# ------------- Cache model loading -------------
@st.cache_resource(show_spinner=True)
def _load_model_cached(model_path, thr_path):
    model, device, best_thr = load_model_and_threshold(
        model_path, thr_path, num_classes=DEFAULT_NUM_CLASSES, map_location=get_device()
    )
    return model, device, best_thr

with st.spinner("Loading model..."):
    model, device, best_thr = _load_model_cached(model_path, thr_path)

threshold = manual_threshold if override_threshold else best_thr
st.info(f"Model loaded. Device: `{device}` | Aggregation: `{agg_method}` "
        f"| Percentile: `{percentile}` | Threshold: `{threshold:.2f}`")

# ------------- File uploader -------------
uploaded_file = st.file_uploader(
    "Upload a single image (.png/.jpg/...) or a ZIP of patches", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp", "zip"]
)

if not uploaded_file:
    st.stop()

# ------------- Branch: ZIP vs single image -------------
is_zip = uploaded_file.name.lower().endswith(".zip")

if is_zip:
    # ZIP of patches
    with st.spinner("Processing ZIP…"):
        tensors, names = preprocess_zip_to_tensor_list(uploaded_file.read())

    if len(tensors) == 0:
        st.error("⚠️ No valid image files were found in the ZIP. Ensure it contains image files, not empty folders.")
        with st.expander("Debug info"):
            st.write("We accept: .png, .jpg, .jpeg, .tif, .tiff, .bmp")
            st.write("Nested folders are supported.")
        st.stop()

    with st.spinner("Running inference on patches…"):
        agg_prob, details = run_zip_patches(
            model, device, tensors,
            agg_method=agg_method, percentile=percentile, batch_size=32
        )

    # Results
    pred_label = "Malignant" if agg_prob >= threshold else "Benign"
    color = "🔴" if pred_label == "Malignant" else "🟢"
    st.subheader("📊 Slide-level Result")
    st.write(f"{color} **{pred_label}** (Aggregated Probability = {agg_prob:.2f}, Threshold = {threshold:.2f})")

    st.subheader("📈 Patch-level Summary")
    st.write(f"Number of patches processed: **{details['n_patches']}**")
    st.write(f"Patch probabilities → min: {details['min']:.2f}, mean: {details['mean']:.2f}, max: {details['max']:.2f}")

    # Histogram
    fig, ax = plt.subplots()
    ax.hist(details["patch_probs"], bins=20)
    ax.set_xlabel("Malignant probability")
    ax.set_ylabel("Count")
    ax.set_title("Patch Probability Distribution")
    st.pyplot(fig, use_container_width=True)

else:
    # Single image
    try:
        pil = load_image_from_bytes(uploaded_file.read())
    except Exception:
        st.error("Could not read the image. Please upload a valid image file.")
        st.stop()

    st.image(pil, caption=secure_filename(uploaded_file.name), use_container_width=True)

    tensor_1x = preprocess_single_image(pil)
    with st.spinner("Running inference…"):
        prob = run_single_image(model, device, tensor_1x)

    pred_label = "Malignant" if prob >= threshold else "Benign"
    color = "🔴" if pred_label == "Malignant" else "🟢"
    st.subheader("🖼️ Image Result")
    st.write(f"{color} **{pred_label}** (Probability = {prob:.2f}, Threshold = {threshold:.2f})")

    # Simple bar for single-prob visual
    fig, ax = plt.subplots()
    ax.bar([0], [prob])
    ax.set_ylim(0, 1)
    ax.set_xticks([0])
    ax.set_xticklabels(["Malignant prob"])
    ax.set_ylabel("Probability")
    st.pyplot(fig, use_container_width=True)

    # Debugger
from backend.config import MODEL_PATH, THR_PATH

st.write("MODEL PATH:", MODEL_PATH, "Exists?", MODEL_PATH.exists())
st.write("THR PATH:", THR_PATH, "Exists?", THR_PATH.exists())
