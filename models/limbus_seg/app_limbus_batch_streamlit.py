import streamlit as st
import os
import cv2
import numpy as np
import torch
from PIL import Image
import io
import zipfile

# Import your custom utils (must be in the same folder)
# We wrap this in a try-except block to prevent app crashing if utils are missing during setup
try:
    from inference_utils import load_model, predict_masks
except ImportError:
    st.error("Could not import 'inference_utils'. Please ensure 'inference_utils.py' is in the same directory.")

# ---------------- CONFIG ----------------
DEFAULT_MODEL_PATH = "model_limbus_crop_unetpp_weighted.pth"
# ----------------------------------------

def crop_limbus(image_bgr, mask):
    """
    1. Applies the mask to the image (making background black).
    2. Crops the image to the bounding rectangle of the limbus.
    """
    # Ensure mask is binary (0 or 255) and type uint8
    mask = (mask * 255).astype(np.uint8) if mask.max() <= 1 else mask.astype(np.uint8)
    
    # 1. Apply mask to remove background
    result = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)
    
    # 2. Find bounding box to crop empty space
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # If no limbus detected, return the original (or a black image)
        return result 
    
    # Find the largest contour (assuming it's the limbus)
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    
    # Crop
    cropped_result = result[y:y+h, x:x+w]
    return cropped_result

@st.cache_resource
def get_model(model_path, device):
    """Load model only once using Streamlit caching."""
    if not os.path.exists(model_path):
        return None, None, None, None
    return load_model(model_path, device)

def main():
    st.set_page_config(page_title="Limbus Cropper Tool", layout="wide")
    st.title("👁️ Limbus Extraction & Cropping Tool")

    # --- Sidebar Configuration ---
    st.sidebar.header("Configuration")
    model_path = st.sidebar.text_input("Model Checkpoint Path", value=DEFAULT_MODEL_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.sidebar.write(f"**Running on:** `{device}`")

    # --- Load Model ---
    try:
        model, idx_crop, idx_limbus, img_size = get_model(model_path, device)
        if model is None:
            st.error(f"Model not found at `{model_path}`. Please check the path.")
            st.stop()
        else:
            st.sidebar.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    # --- File Upload ---
    st.write("---")
    uploaded_files = st.file_uploader(
        "Upload Eye Images (Single or Batch)", 
        type=["jpg", "jpeg", "png", "bmp", "tif"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        st.write(f"Processing **{len(uploaded_files)}** images...")
        
        # Container for results
        processed_images = [] # List of (filename, image_array)
        
        progress_bar = st.progress(0)
        
        # Create columns for grid display
        cols = st.columns(3) 

        for i, uploaded_file in enumerate(uploaded_files):
            # 1. Read Image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            bgr_image = cv2.imdecode(file_bytes, 1)
            
            # 2. Predict
            # Note: We assume predict_masks returns a numpy array of masks
            masks = predict_masks(model, bgr_image, img_size, device)
            
            # 3. Process Limbus
            limbus_mask = masks[idx_limbus] # Extract specific channel
            
            # Convert probability map to binary mask if necessary
            if limbus_mask.dtype != np.uint8:
                limbus_mask = (limbus_mask > 0.5).astype(np.uint8)

            # Crop logic
            cropped_limbus = crop_limbus(bgr_image, limbus_mask)
            
            # Convert BGR to RGB for Streamlit/PIL
            cropped_rgb = cv2.cvtColor(cropped_limbus, cv2.COLOR_BGR2RGB)
            
            # Store result
            filename = os.path.splitext(uploaded_file.name)[0] + "_limbus.png"
            processed_images.append((filename, cropped_rgb))

            # Display in grid (limit to first 9 to save memory/space in UI)
            if i < 9:
                with cols[i % 3]:
                    st.image(cropped_rgb, caption=uploaded_file.name, use_container_width=True)
            
            progress_bar.progress((i + 1) / len(uploaded_files))

        st.success("Processing Complete!")

        # --- ZIP Download ---
        if processed_images:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zf:
                for fname, img_array in processed_images:
                    # Convert numpy array back to bytes for zip
                    pil_img = Image.fromarray(img_array)
                    img_byte_arr = io.BytesIO()
                    pil_img.save(img_byte_arr, format='PNG')
                    zf.writestr(fname, img_byte_arr.getvalue())
            
            st.download_button(
                label="📥 Download All Cropped Images (ZIP)",
                data=zip_buffer.getvalue(),
                file_name="limbus_crops.zip",
                mime="application/zip"
            )

if __name__ == "__main__":
    main()