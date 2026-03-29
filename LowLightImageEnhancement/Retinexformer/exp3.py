
# Entry point:
#   - Add your low light image as test.jpg in Retinexformer
#   - Run this file
#   - Get enhanced_local.png in Retinexformer



import os
import sys
import importlib.util

# 1. Setup Absolute Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
arch_path = os.path.join(current_dir, 'Enhancement', 'models', 'retinexformer_arch.py')

# 2. Add the project root and basicsr to sys.path
sys.path.insert(0, current_dir)
if os.path.exists(os.path.join(current_dir, 'basicsr')):
    sys.path.insert(0, os.path.join(current_dir, 'basicsr'))

# 3. MANUALLY LOAD THE ARCHITECTURE FILE (Bypassing __init__.py)
spec = importlib.util.spec_from_file_location("retinexformer_arch", arch_path)
arch_module = importlib.util.module_from_spec(spec)
sys.modules["models.retinexformer_arch"] = arch_module
spec.loader.exec_module(arch_module)

# 4. Extract the class
RetinexFormer = arch_module.RetinexFormer




# Continue with your model loading...

import torch
import cv2
import numpy as np
from basicsr.utils.img_util import img2tensor, tensor2img

# 1. Path Setup
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'Enhancement'))

# 2. Import Model
#from models.retinexformer_arch import RetinexFormer

def enhance_image(input_path, output_path, weights_path):
    # FORCE CPU
    device = torch.device('cpu')
    print(f"🚀 Processing on {device}... please wait.")

    # Load Model Structure (LOL-v1 settings)
    model = RetinexFormer(in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1, 2, 2])
    
    # Load Weights for CPU
    checkpoint = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(checkpoint['params'] if 'params' in checkpoint else checkpoint)
    model.to(device)
    model.eval()

    # Load and Preprocess Image
    img = cv2.imread(input_path)
    if img is None: 
        print("Error: Image not found!"); return
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    # Padding to multiple of 8 (Mandatory for Transformers)
    pad_h, pad_w = (8 - h % 8) % 8, (8 - w % 8) % 8
    img_in = np.pad(img.astype(np.float32) / 255., ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    input_tensor = img2tensor(img_in, bgr2rgb=False, float32=True).unsqueeze(0).to(device)

    # Inference (No Grad saves massive RAM on CPU)
    with torch.no_grad():
        output = model(input_tensor)

    # Post-process
    res = tensor2img(output, rgb2bgr=True, out_type=np.uint8)[:h, :w, :]
    cv2.imwrite(output_path, res)
    print(f"✅ Success! Enhanced image saved as: {output_path}")

if __name__ == "__main__":
    # Make sure these files exist in your folder!

    # SID.pth
    enhance_image(
        input_path='test.jpg', 
        output_path='SID.png', 
        weights_path='./pretrained_weights/SID.pth'
    )
    # SDSD_indoor.pth
    enhance_image(
        input_path='test.jpg', 
        output_path='SDSD_indoor.png', 
        weights_path='./pretrained_weights/SDSD_indoor.pth'
    )
    # SDSD_outdoor.pth
    enhance_image(
        input_path='test.jpg', 
        output_path='SDSD_outdoor.png', 
        weights_path='./pretrained_weights/SDSD_outdoor.pth'
    )
    # LOL_v1.pth
    enhance_image(
        input_path='test.jpg', 
        output_path='LOL_v1.png', 
        weights_path='./pretrained_weights/LOL_v1.pth'
    ) # LOL_v2_real.pth
    enhance_image(
        input_path='test.jpg', 
        output_path='LOL_v2_real.png', 
        weights_path='./pretrained_weights/LOL_v2_real.pth'
    )
    # LOL_v2_synthetic.pth
    enhance_image(
        input_path='test.jpg', 
        output_path='LOL_v2_synthetic.png', 
        weights_path='./pretrained_weights/LOL_v2_synthetic.pth'
    )
    # SMID.pth
    enhance_image(
        input_path='test.jpg', 
        output_path='SMID.png', 
        weights_path='./pretrained_weights/SMID.pth'
    )
    # FiveK.pth
    enhance_image(
        input_path='test.jpg', 
        output_path='FiveK.png', 
        weights_path='./pretrained_weights/FiveK.pth'
    )
