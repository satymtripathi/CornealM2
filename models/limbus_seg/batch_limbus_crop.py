import argparse
import os
import cv2
import torch
from tqdm import tqdm
from inference_utils import load_model, predict_masks, get_cropped_roi

def main():
    parser = argparse.ArgumentParser(description="Batch Limbus Crop Segmentation")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input image directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save cropped ROIs")
    parser.add_argument("--model_path", type=str, default="model_limbus_crop_unetpp_weighted.pth", help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run inference on")
    
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load Model
    print(f"Loading model from {args.model_path} on {args.device}...")
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        return

    try:
        model, idx_crop, idx_limbus, img_size = load_model(args.model_path, args.device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Get valid images
    valid_exts = {".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".tif"}
    images = [f for f in os.listdir(args.input_dir) if os.path.splitext(f)[1].lower() in valid_exts]
    
    if not images:
        print(f"No valid images found in {args.input_dir}")
        return

    print(f"Found {len(images)} images in {args.input_dir}")

    # Process Loop
    processed_count = 0
    with torch.no_grad():
        for img_name in tqdm(images, desc="Processing"):
            img_path = os.path.join(args.input_dir, img_name)
            
            # Read Image
            bgr = cv2.imread(img_path)
            if bgr is None:
                # print(f"Warning: Could not read {img_path}")
                continue

            # Predict
            try:
                masks = predict_masks(model, bgr, img_size, args.device)
                crop_mask = masks[idx_crop]
                
                # Extract ROI
                roi = get_cropped_roi(bgr, crop_mask)
                
                if roi is not None:
                    out_path = os.path.join(args.output_dir, img_name)
                    cv2.imwrite(out_path, roi)
                    processed_count += 1
                else:
                    # ROI extraction failed (no mask or empty bbox)
                    pass
                    
            except Exception as e:
                print(f"Error processing {img_name}: {e}")

    print(f"Batch processing complete. {processed_count}/{len(images)} images successfully cropped and saved to {args.output_dir}")

if __name__ == "__main__":
    main()
