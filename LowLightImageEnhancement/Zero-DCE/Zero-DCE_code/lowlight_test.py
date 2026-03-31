import torch
import torch.nn as nn
import torchvision
import os
import model
import numpy as np
from PIL import Image
import glob
import time

def lowlight(image_path, DCE_net, device):
    # Load and Preprocess
    data_lowlight = Image.open(image_path)
    data_lowlight = (np.asarray(data_lowlight) / 255.0)
    
    # Convert to tensor and move to device (CPU/GPU)
    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2, 0, 1).unsqueeze(0).to(device)

    # Inference
    start = time.time()
    _, enhanced_image, _ = DCE_net(data_lowlight)
    end_time = (time.time() - start)
    
    # Path Logic
    # Replacing 'test_data' with 'result' as per your original logic
    result_path = image_path.replace('test_data', 'result')
    result_dir = os.path.dirname(result_path)
    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    torchvision.utils.save_image(enhanced_image, result_path)
    print(f"✅ {image_path} -> {end_time:.4f}s")

if __name__ == '__main__':
    # 1. Dynamic Device Selection (Uses GPU if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📡 Running on: {device}")

    # 2. Load Model ONCE outside the loop
    # Ensure 'snapshots/Epoch99.pth' exists!
    DCE_net = model.enhance_net_nopool().to(device)
    
    # map_location ensures it loads on CPU even if saved on GPU
    checkpoint = torch.load('snapshots/Epoch99.pth', map_location=device)
    DCE_net.load_state_dict(checkpoint)
    DCE_net.eval()

    with torch.no_grad():
        filePath = 'data/test_data/'
        # Use glob to find all images in subfolders
        test_list = glob.glob(os.path.join(filePath, '**/*.*'), recursive=True)

        for image_path in test_list:
            # Basic check to ensure we only process images
            if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                lowlight(image_path, DCE_net, device)