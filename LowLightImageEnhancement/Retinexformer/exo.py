import os
import sys
import importlib.util
import torch
import cv2
import numpy as np
from basicsr.utils.img_util import img2tensor, tensor2img

# --- 1. SETUP PATHS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
arch_path = os.path.join(current_dir, 'Enhancement', 'models', 'retinexformer_arch.py')

sys.path.insert(0, current_dir)
if os.path.exists(os.path.join(current_dir, 'basicsr')):
    sys.path.insert(0, os.path.join(current_dir, 'basicsr'))

# --- 2. SINGLE-TIME MODEL LOAD ---
spec = importlib.util.spec_from_file_location("retinexformer_arch", arch_path)
arch_module = importlib.util.module_from_spec(spec)
sys.modules["models.retinexformer_arch"] = arch_module
spec.loader.exec_module(arch_module)
RetinexFormer = arch_module.RetinexFormer

# Force CPU Optimization
device = torch.device('cpu')
torch.set_num_threads(os.cpu_count()) # Use all CPU cores

# Initialize model structure ONCE
# (Using the parameters you confirmed: n_feat=40, stage=1, num_blocks=[1, 2, 2])
GLOBAL_MODEL = RetinexFormer(in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1, 2, 2])
GLOBAL_MODEL.to(device)
GLOBAL_MODEL.eval()

def run_inference(model, input_tensor, weights_path, h, w):
    """Swaps weights and runs inference on a pre-processed tensor."""
    print(f"🔄 Loading weights: {os.path.basename(weights_path)}...")
    
    # Load weights into the existing model structure
    checkpoint = torch.load(weights_path, map_location='cpu')
    state_dict = checkpoint['params'] if 'params' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    # Convert back to image format
    return tensor2img(output, rgb2bgr=True, out_type=np.uint8)[:h, :w, :]

def main():
    input_file = 'test.jpg'
    weights_dir = './pretrained_weights'
    
    # 1. Load and Pre-process Image ONCE
    img = cv2.imread(input_file)
    if img is None:
        print(f"❌ Error: {input_file} not found!")
        return

    # --- CPU SPEED HACK: RESIZE ---
    # If the image is huge, this reduces O(N^2) complexity of the Transformer
    if img.shape[1] > 1000:
        scale = 1000 / img.shape[1]
        img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
        print(f"📉 Resized to {img.shape[1]}x{img.shape[0]} for faster CPU processing.")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape

    # Padding for Transformer (Multiple of 8)
    pad_h, pad_w = (8 - h % 8) % 8, (8 - w % 8) % 8
    img_in = np.pad(img_rgb.astype(np.float32) / 255., ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    input_tensor = img2tensor(img_in, bgr2rgb=False, float32=True).unsqueeze(0).to(device)

    # 2. Define your test cases
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

    # 3. Batch Process
    for w_file, out_file in tasks:
        w_path = os.path.join(weights_dir, w_file)
        if os.path.exists(w_path):
            result = run_inference(GLOBAL_MODEL, input_tensor, w_path, h, w)
            cv2.imwrite(out_file, result)
            print(f"✅ Saved: {out_file}")
        else:
            print(f"⚠️ Skipped: {w_file} (File not found)")

if __name__ == "__main__":
    main()