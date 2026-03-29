
"""
install necessary libraries ->
!pip install einops addict future lmdb yapf lpips thop timm basicsr


next cell -> 
import os

# 1. Define the path where Colab installs basicsr
# In Colab, it's usually in /usr/local/lib/python3.12/dist-packages/
path = '/usr/local/lib/python3.12/dist-packages/basicsr/data/degradations.py'

# 2. Check if file exists and patch it directly
if os.path.exists(path):
    with open(path, 'r') as f:
        content = f.read()
    
    # Replace the broken import with the correct one
    fixed_content = content.replace(
        'from torchvision.transforms.functional_tensor import rgb_to_grayscale', 
        'from torchvision.transforms.functional import rgb_to_grayscale'
    )
    
    with open(path, 'w') as f:
        f.write(fixed_content)
    print("✅ Surgery Successful! basicsr patched without importing it.")
else:
    print("❌ File not found at that path. Run '!pip show basicsr' to check the Location.")
"""

import os
import sys
import importlib.util
import torch
import cv2
import numpy as np
from basicsr.utils.img_util import img2tensor, tensor2img

# --- 1. CONFIGURATION ---
BASE_DIR = '/content/Retinexformer'
DATA_DIR = os.path.join(BASE_DIR, 'data')
TEST_DIR = os.path.join(DATA_DIR, 'test')
RESULT_BASE = os.path.join(DATA_DIR, 'result')
WEIGHTS_DIR = os.path.join(BASE_DIR, 'pretrained_weights')
ARCH_PATH = os.path.join(BASE_DIR, 'Enhancement', 'models', 'RetinexFormer_arch.py')

# Add paths to sys
sys.path.insert(0, BASE_DIR)
if os.path.exists(os.path.join(BASE_DIR, 'basicsr')):
    sys.path.insert(0, os.path.join(BASE_DIR, 'basicsr'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 2. LOAD ARCHITECTURE ---
spec = importlib.util.spec_from_file_location("RetinexFormer_arch", ARCH_PATH)
arch_module = importlib.util.module_from_spec(spec)
sys.modules["models.RetinexFormer_arch"] = arch_module
spec.loader.exec_module(arch_module)
RetinexFormer = arch_module.RetinexFormer

def get_model():
    model = RetinexFormer(in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1, 2, 2])
    model.to(device).eval()
    return model

# --- 3. BATCH PROCESSING LOGIC ---
if __name__ == "__main__":
    shared_model = get_model()
    
    # List of all models to run
    model_tasks = [
        'SID', 'SDSD_indoor', 'SDSD_outdoor', 'LOL_v1', 
        'LOL_v2_real', 'LOL_v2_synthetic', 'SMID', 'FiveK'
    ]

    # Get all images from test folder
    valid_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.PNG')
    image_files = [f for f in os.listdir(TEST_DIR) if f.endswith(valid_extensions)]
    
    if not image_files:
        print(f"❌ No images found in {TEST_DIR}")
        sys.exit()

    print(f"📸 Found {len(image_files)} images. Starting enhancement...")

    for image_name in image_files:
        img_path = os.path.join(TEST_DIR, image_name)
        img_base_name = os.path.splitext(image_name)[0] # e.g., 'test1'
        
        # Load and Prepare Image once for all 8 models
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img_rgb.shape
        
        pad_h, pad_w = (8 - h % 8) % 8, (8 - w % 8) % 8
        img_in = np.pad(img_rgb.astype(np.float32) / 255., ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        input_tensor = img2tensor(img_in, bgr2rgb=False, float32=True).unsqueeze(0).to(device)

        print(f"\n🖼️ Processing Image: {image_name}")

        for model_name in model_tasks:
            weights_path = os.path.join(WEIGHTS_DIR, f"{model_name}.pth")
            
            if not os.path.exists(weights_path):
                print(f"  ⚠️ Skipping {model_name}: Weights not found.")
                continue

            # Create specific result folder: data/result/LOL_v1/
            current_result_dir = os.path.join(RESULT_BASE, model_name)
            os.makedirs(current_result_dir, exist_ok=True)

            # Load weights into the shell
            checkpoint = torch.load(weights_path, map_location=device)
            state_dict = checkpoint['params'] if 'params' in checkpoint else checkpoint
            shared_model.load_state_dict(state_dict)

            # Inference
            with torch.no_grad():
                output = shared_model(input_tensor)

            # Save as test1_enhanced.png
            res = tensor2img(output, rgb2bgr=True, out_type=np.uint8)[:h, :w, :]
            save_filename = f"{img_base_name}_enhanced.png"
            cv2.imwrite(os.path.join(current_result_dir, save_filename), res)
            
            print(f"  ✅ {model_name} done.")

    print(f"\n🚀 All processing complete! Results are in {RESULT_BASE}")


    