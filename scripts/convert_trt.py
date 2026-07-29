"""
convert_trt.py — Convert LightDehaze_Net PyTorch model to TensorRT (FP16).

Loads the trained PyTorch weights, runs torch2trt conversion with FP16 mode,
and saves the TensorRT engine weights.

Usage (inside Docker container):
    python scripts/convert_trt.py

Output:
    weights/ldnet_trt.pth
"""

import torch
from torch2trt import torch2trt

from src.inference import load_model

OUTPUT_PATH = "weights/ldnet_trt.pth"
INPUT_SIZE = (1, 3, 256, 256)  # Fixed input shape required by TRT

print("Loading PyTorch model...")
model = load_model()

print(f"Converting to TensorRT FP16 (input shape: {INPUT_SIZE})...")
dummy_input = torch.randn(*INPUT_SIZE).cuda()
model_trt = torch2trt(model, [dummy_input], fp16_mode=True)

torch.save(model_trt.state_dict(), OUTPUT_PATH)
print(f"TensorRT conversion complete — saved to: {OUTPUT_PATH}")
