import os
import sys
import importlib.util
import torch
import cv2
import numpy as np
from basicsr.utils.img_util import img2tensor, tensor2img

# --- 1. CONFIGURATION & PATHS ---
# Based on your structure: Retinexformer -> data -> (test/result)
BASE_DIR = '/content/Retinexformer'
DATA_DIR = os.path.join(BASE_DIR, 'data')
TEST_DIR = os.path.join(DATA_DIR, 'test')
RESULT_DIR = os.path.join(DATA_DIR, 'result')
WEIGHTS_DIR = os.path.join(BASE_DIR, 'pretrained_weights')

# Ensure the result folder exists
os.makedirs(RESULT_DIR, exist_ok=True)

# Architecture path (matching your screenshot casing)
ARCH_PATH = os.path.join(BASE_DIR, 'Enhancement', 'models', 'RetinexFormer_arch.py')

sys.path.insert(0, BASE_DIR)
if os.path.exists(os.path.join(BASE_DIR, 'basicsr')):
    sys.path.insert(0, os.path.join(BASE_DIR, 'basicsr'))

# --- 2. GPU INITIALIZATION ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"📡 Device: {device}")

# --- 3. LOAD ARCHITECTURE ONCE ---
spec = importlib.util.spec_from_file_location("RetinexFormer_arch", ARCH_PATH)
arch_module = importlib.util.module_from_spec(spec)
sys.modules["models.RetinexFormer_arch"] = arch_module
spec.loader.exec_module(arch_module)
RetinexFormer = arch_module.RetinexFormer

def get_model():
    """Builds the shell of the model."""
    model = RetinexFormer(in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1, 2, 2])
    model.to(device)
    model.eval()
    return model

def process_batch(model, input_tensor, h, w):
    """Iterates through all 8 weights using the same input tensor."""
    tasks = [
        ('SID.pth', 'SID.png'),
        ('SDSD_indoor.pth', 'SDSD_indoor.png'),
        ('SDSD_outdoor.pth', 'SDSD_outdoor.png'),
        ('LOL_v1.pth', 'LOL_v1.png'),
        ('LOL_v2_real.pth', 'LOL_v2_real.png'),
        ('LOL_v2_synthetic.pth', 'LOL_v2_synthetic.png'),
        ('SMID.pth', 'SMID.png'),
        ('FiveK.pth', 'FiveK.png')
    ]

    for w_file, out_name in tasks:
        w_path = os.path.join(WEIGHTS_DIR, w_file)
        if not os.path.exists(w_path):
            print(f"⚠️ Skipping: {w_file} (Not found in weights folder)")
            continue

        print(f"🔄 Enhancing with {w_file}...")
        
        # Load weights into existing model
        checkpoint = torch.load(w_path, map_location=device)
        state_dict = checkpoint['params'] if 'params' in checkpoint else checkpoint
        model.load_state_dict(state_dict)

        with torch.no_grad():
            output = model(input_tensor)

        # Post-process and save to RESULT_DIR
        res = tensor2img(output, rgb2bgr=True, out_type=np.uint8)[:h, :w, :]
        save_path = os.path.join(RESULT_DIR, out_name)
        cv2.imwrite(save_path, res)
        print(f"✅ Saved to: {save_path}")

# --- 4. EXECUTION ---
if __name__ == "__main__":
    test_image_path = os.path.join(TEST_DIR, 'test.jpg')
    
    if not os.path.exists(test_image_path):
        print(f"❌ Error: Could not find 'test.jpg' in {TEST_DIR}")
    else:
        # Load and Prepare Image ONCE
        img = cv2.imread(test_image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img_rgb.shape
        
        # Padding
        pad_h, pad_w = (8 - h % 8) % 8, (8 - w % 8) % 8
        img_in = np.pad(img_rgb.astype(np.float32) / 255., ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        input_tensor = img2tensor(img_in, bgr2rgb=False, float32=True).unsqueeze(0).to(device)

        # Run model
        shared_model = get_model()
        process_batch(shared_model, input_tensor, h, w)
        
        print("\n🚀 Batch enhancement complete! Check the 'data/result' folder.")