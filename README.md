# LightDehazeNet on Jetson Nano

[![Python 3.7+](https://img.shields.io/badge/Python-3.7%2B-3572A5?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 1.9](https://img.shields.io/badge/PyTorch-1.9-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TensorRT FP16](https://img.shields.io/badge/TensorRT-FP16-76b900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/tensorrt)
[![Jetson Nano](https://img.shields.io/badge/NVIDIA-Jetson%20Nano-76b900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/embedded/jetson-nano)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

End-to-end deployment of **LightDehazeNet** on **NVIDIA Jetson Nano** — trained with a custom Hybrid MS-SSIM + L₁ loss, accelerated via TensorRT FP16 engine conversion, and served as a real-time MJPEG dehazing web stream.

---

## Overview

| | |
|---|---|
| **Custom Loss** | Hybrid MS-SSIM + L₁ (α = 0.84) — 93% faster convergence vs. L₁ baseline |
| **Benchmark** | FoggyCityscapes: PSNR **22.99 dB** / SSIM **0.9214**  ·  SmokeBench: PSNR **19.32 dB** / SSIM **0.7614** |
| **TensorRT FP16** | PyTorch `.pth` → TensorRT engine via `torch2trt`, fixed 256×256 input |
| **Edge Streaming** | Real-time side-by-side dehazing served as MJPEG over Flask on Jetson Nano |

---

## Training — Custom Loss & Ablation Study

Standard L₁ and MSE-trained dehazing models exhibit an **Identity Trap**: the network learns to pass hazy inputs through unchanged for up to 70 epochs before achieving any meaningful dehazing. We resolved this with a perceptual-structural loss formulation:

$$\mathcal{L} = 0.84 \cdot \mathcal{L}_{\text{MS-SSIM}} + 0.16 \cdot \mathcal{L}_{L1}$$

The MS-SSIM term prioritizes multi-scale structural and edge recovery; the L₁ term stabilizes global luminance and prevents color drift.

### Alpha Ablation

| α | PSNR (dB) | SSIM | Notes |
|:---:|:---:|:---:|:---|
| 0.00 (L₁ baseline) | 22.54 | 0.9260 | Identity Trap — ~70 epochs before dehazing initiates |
| 0.50 | 22.09 | 0.9369 | Slow structural convergence |
| 0.60 | 20.86 | **0.9439** | Peak SSIM, but high startup latency |
| 0.70 | 22.10 | 0.9139 | Moderate stability |
| 0.80 | 21.22 | 0.9140 | Improved convergence |
| **0.84 (proposed)** | **20.78** | **0.9194** | Breakthrough at Epoch 5 — 93% latency reduction |
| 0.90 | 8.43 | 0.7711 | Optimization collapse |

| PSNR & SSIM vs. Alpha | Convergence — Escaping the Identity Trap |
|:---:|:---:|
| ![Metrics Plot](assets/metrics_plot.png) | ![Convergence Elbow](assets/convergence_elbow.png) |

**Key finding:** α = 0.84 acts as an optimization catalyst. The model achieves functional dehazing at Epoch 5 (compared to ~70 for L₁), while α ≥ 0.90 leads to stochastic collapse.

---

## Benchmark Results

The table below compares the base LightDehazeNet (trained with standard L₁ loss) against our model trained with the custom Hybrid MS-SSIM + L₁ loss (α = 0.84). Both trained on FoggyCityscapes + SmokeBench.

| Test Set | Base Model (L₁) | | Custom Loss (α = 0.84) | |
|:---|:---:|:---:|:---:|:---:|
| | PSNR | SSIM | PSNR | SSIM |
| FoggyCityscapes (dense fog) | 18.87 dB | 0.9077 | **22.99 dB** | **0.9214** |
| SmokeBench | 18.02 dB | 0.7608 | **19.32 dB** | **0.7614** |
| **Average** | 18.45 dB | 0.8343 | **21.15 dB** | **0.8414** |

### FoggyCityscapes — Qualitative Output
![FoggyCityscapes Results](assets/results_foggy_cityscapes.png)

### SmokeBench — Qualitative Output
![SmokeBench Results](assets/results_smokebench.jpg)

---

## Zero-Shot Real-World Video Inference

The TensorRT FP16 engine was tested on `bikers.mp4` — an unseen internet video the model was never trained on. Output is streamed side-by-side (hazy | dehazed) at real-time frame rates via Flask on Jetson Nano.

| Frame 1 | Frame 2 |
|:---:|:---:|
| ![](assets/screenshots/bikers_video_inference_1.png) | ![](assets/screenshots/bikers_video_inference_2.png) |

| Frame 3 | Frame 4 |
|:---:|:---:|
| ![](assets/screenshots/bikers_video_inference_3.png) | ![](assets/screenshots/bikers_video_inference_4.png) |

---

## Deployment Pipeline

```
PyTorch checkpoint   →   torch2trt (FP16)   →   TensorRT engine   →   Flask MJPEG stream
trained_LDNet.pth                               ldnet_trt.pth         http://<jetson>:5000
```

---

## Repository Structure

```
.
├── src/
│   ├── model.py            # LightDehaze_Net architecture (~0.21M params)
│   ├── inference.py        # Model loader and preprocessing helpers
│   ├── cvr.py              # Color Visibility Restoration (CLAHE post-processing)
│   └── model_info.py       # Layer-wise parameter counter
│
├── scripts/
│   ├── convert_trt.py          # PyTorch → TensorRT FP16 conversion
│   ├── infer_image.py          # Single image dehazing
│   ├── infer_batch.py          # Batch image dehazing
│   ├── stream_live_camera.py   # Live Jetson camera MJPEG stream (TRT)
│   ├── stream_video_file.py    # Video file MJPEG stream (TRT)
│   └── camera_raw_stream.py    # Raw camera passthrough (debug)
│
├── notebooks/
│   └── evaluation_plots.ipynb  # PSNR/SSIM ablation and metric visualizations
│
├── assets/                 # Architecture diagram, ablation plots, result images
│   └── screenshots/        # Zero-shot bikers.mp4 inference frames
│
├── weights/
│   ├── trained_LDNet.pth   # Trained PyTorch checkpoint
│   ├── ldnet_trt.pth       # TensorRT FP16 engine
│   └── README.md           # Weights download guide
│
├── samples/                # Test images (indoor/outdoor, synthetic/natural haze)
├── docker/SETUP.md         # Container load and run guide
├── Dockerfile
├── run_container.sh
└── requirements.txt
```

---

## Quickstart

```bash
# Load and launch the Docker container on Jetson
docker load -i my_jetson_env.tar
bash run_container.sh

# Convert PyTorch checkpoint to TensorRT FP16
python scripts/convert_trt.py

# Live camera dehazing stream
python scripts/stream_live_camera.py
# Open http://<jetson-ip>:5000

# Video file dehazing stream (side-by-side)
python scripts/stream_video_file.py
# Open http://<jetson-ip>:5000

# Single image
python scripts/infer_image.py -i samples/outdoor_synthetic/soh(1).jpg

# Batch inference on a directory
python scripts/infer_batch.py -td samples/outdoor_natural/
```

See [docker/SETUP.md](docker/SETUP.md) for detailed container setup and GPU/camera verification.

