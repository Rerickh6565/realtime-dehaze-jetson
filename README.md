# LightDehazeNet on Jetson Nano: Custom Loss Optimization, TensorRT FP16 & Edge Deployment

[![Python 3.7+](https://img.shields.io/badge/Python-3.7%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![TensorRT](https://img.shields.io/badge/TensorRT-FP16-76b900?logo=nvidia)](https://developer.nvidia.com/tensorrt)
[![Jetson Nano](https://img.shields.io/badge/Platform-NVIDIA%20Jetson%20Nano-76b900?logo=nvidia)](https://developer.nvidia.com/embedded/jetson-nano)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)](https://www.docker.com/)
[![Paper](https://img.shields.io/badge/IEEE%20Tencon-2026-blue)](https://ieeexplore.ieee.org/abstract/document/9562276)

> End-to-end edge AI project deploying an optimized **LightDehazeNet (LD-Net)** model on **NVIDIA Jetson Nano**. Includes custom **Hybrid MS-SSIM + L1 loss training innovations**, **TensorRT FP16 engine conversion**, and **real-time Flask web streaming**.

---

## 📌 Executive Summary & Key Highlights

This repository showcases the complete training enhancements, TensorRT optimization pipeline, and Jetson Nano edge deployment for **LightDehazeNet**:

1. **🏋️ Custom Loss Function Innovation**: Formulated a hybrid loss ($\alpha \cdot \text{MS-SSIM} + (1-\alpha) \cdot L_1$) with $\alpha = 0.84$, achieving a **93% reduction in convergence latency** (breakthrough at Epoch 5) and eliminating the **"Identity Trap"** inherent to standard $L_1$/MSE loss training.
2. **📊 Robust Evaluation Metrics**: Benchmarked on **FoggyCityscapes** (PSNR: **22.99 dB**, SSIM: **0.9214**) and **SmokeBench** (PSNR: **19.32 dB**, SSIM: **0.7614**).
3. **⚡ TensorRT FP16 Acceleration**: Converted the PyTorch `.pth` checkpoint using `torch2trt` to a TensorRT engine, enabling low-latency FP16 execution on Jetson GPU hardware.
4. **🎥 Real-World Video & Edge Deployment**: Deployed inside an isolated Docker container with real-time camera passthrough and zero-shot real-world video dehazing (`bikers.mp4`) served via MJPEG web streaming (Flask + OpenCV).

---

## 🔬 Training Innovations & Loss Function Ablation Study

Standard image dehazing models trained purely on pixel-level loss ($L_1$ or MSE) suffer from blurry outputs and severe convergence delay, often stuck in an **"Identity Trap"** for up to 70 epochs before learning meaningful structural dehazing.

### 1. Modified Hybrid Loss Function

We introduced a weighted structural-perceptual loss:

$$\mathcal{L}_{\text{custom}} = 0.84 \cdot \mathcal{L}_{\text{MS-SSIM}} + 0.16 \cdot \mathcal{L}_{L1}$$

* **Perceptual Dominance (84% MS-SSIM)**: Prioritizes high-frequency edge details, multi-scale texture recovery, and structural consistency across scales.
* **Balanced Convergence (16% L1)**: Stabilizes global color distribution and guards against luminance drift.

### 2. Loss Weighting ($\alpha$) Ablation Study

We conducted an extensive ablation study across $\alpha \in [0.0, 0.9]$ to identify the trade-off between numerical fidelity (PSNR/SSIM) and optimization convergence latency:

| Experiment | Structural Weight ($\alpha$) | Mean PSNR (dB) | Mean SSIM | Convergence Latency / Notes |
|:---|:---:|:---:|:---:|:---|
| **$L_1$ Baseline** | `0.00` | 22.54 | 0.9260 | Severe 'Identity Trap' (~70 epochs to initiate dehazing) |
| **Alpha 0.50** | `0.50` | 22.09 | 0.9369 | Slow structural convergence |
| **Alpha 0.60** | `0.60` | 20.86 | **0.9439** | Peak numerical SSIM, but high startup latency |
| **Alpha 0.70** | `0.70` | 22.10 | 0.9139 | Moderate stability |
| **Alpha 0.80** | `0.80` | 21.22 | 0.9140 | Improved convergence |
| **Proposed ($\alpha=0.84$)** | **`0.84`** | **20.78** | **0.9194** | ⚡ **Engineering Optimum** (93% latency reduction, breakthrough at Epoch 5) |
| **Alpha 0.90** | `0.90` | 8.43 | 0.7711 | Optimization collapse / instability |

### 3. Ablation Visualization

| Metrics vs. Loss Weighting | Escaping the Identity Trap |
|:---:|:---:|
| ![Metrics vs Alpha](assets/metrics_plot.png) | ![Convergence Elbow](assets/convergence_elbow.png) |

> **Key Finding**: Setting $\alpha = 0.84$ acts as an optimization catalyst. It achieves functional dehazing breakthrough at **Epoch 5** (compared to 70 epochs for $L_1$), providing structural pressure while avoiding the stochastic collapse observed at $\alpha \ge 0.90$.

---

## 📈 Quantitative Benchmark Results

The model was evaluated on standard benchmark datasets representing synthetic dense fog and real smoke conditions:

### Model Performance Summary

| Model Setup / Training Data | Test Dataset | Mean PSNR (dB) | Mean SSIM |
|:---|:---|:---:|:---:|
| **LightDehazeNet**<br>*(Trained on FoggyCityscapes + SmokeBench)* | FoggyCityscapes (Dense Fog) | **22.99 dB** | **0.9214** |
| | SmokeBench Test Set | **19.32 dB** | **0.7614** |
| | **Averaged Overall** | **21.15 dB** | **0.8414** |
| **LightDehazeNet**<br>*(Trained on FoggyCityscapes Only)* | FoggyCityscapes (Dense Fog) | **22.37 dB** | **0.9137** |

---

## 🖼️ Qualitative Results

### 1. Benchmark Dataset Restoration (FoggyCityscapes & SmokeBench)

The stitched evaluation outputs below illustrate qualitative restoration performance on benchmark test sets:

#### FoggyCityscapes Evaluation Result
![FoggyCityscapes Results](assets/results_foggy_cityscapes.png)

#### SmokeBench Evaluation Result
![SmokeBench Results](assets/results_smokebench.jpg)

---

### 2. Zero-Shot Real-World Video Inference (`bikers.mp4`)

To test real-world generalization, the TensorRT FP16 optimized engine was evaluated on an un-seen internet video (`bikers.mp4`). The model was **not** trained on this video, demonstrating zero-shot dehazing under dynamic real-world conditions:

| Real-World Video Inference Sample 1 | Real-World Video Inference Sample 2 |
|:---:|:---:|
| ![Bikers Video Inference 1](assets/screenshots/bikers_video_inference_1.png) | ![Bikers Video Inference 2](assets/screenshots/bikers_video_inference_2.png) |

| Real-World Video Inference Sample 3 | Real-World Video Inference Sample 4 |
|:---:|:---:|
| ![Bikers Video Inference 3](assets/screenshots/bikers_video_inference_3.png) | ![Bikers Video Inference 4](assets/screenshots/bikers_video_inference_4.png) |

---

## 🏗️ Architecture & Reformulated Scattering Model

![LD-Net Architecture](assets/framework.png)

LightDehazeNet utilizes a compact 8-layer convolutional neural network (~0.21M parameters) with concatenated skip connections (`Concat 1`, `Concat 2`, `Concat 3`).

Instead of estimating atmospheric light $A$ and transmission map $t(x)$ separately, LD-Net reformulates the Atmospheric Scattering Model (ASM) as a single direct transformation:

$$J(x) = K(x) \cdot I(x) - K(x) + b$$

where $K(x)$ is the learned feature output from Conv8, $I(x)$ is the hazy input image, and $b = 1$.

---

## ⚙️ TensorRT FP16 Optimization & Jetson Nano Deployment

To achieve real-time throughput on Jetson Nano edge hardware:

1. **PyTorch Checkpoint**: Trained model saved as `weights/trained_LDNet.pth` (~125 KB).
2. **TensorRT FP16 Engine**: Converted using `torch2trt` with fixed $(1, 3, 256, 256)$ input shape.
3. **Execution**: Saved as `weights/ldnet_trt.pth` (~2.2 MB), running with FP16 precision for GPU execution.

```
+-------------------+      torch2trt      +-----------------------+      Flask + OpenCV      +------------------------+
| PyTorch (.pth)    |  ---------------->  | TensorRT Engine (FP16)|  --------------------->  | MJPEG Web Stream       |
| LightDehazeNet    |   FP16 Conversion   | ldnet_trt.pth         |   Edge Video Feed        | http://<jetson-ip>:5000|
+-------------------+                     +-----------------------+                          +------------------------+
```

---

## 📁 Repository Structure

```
.
├── src/                        # Core model architecture & inference logic
│   ├── model.py                # LightDehaze_Net neural network (~0.21M params)
│   ├── inference.py            # PyTorch inference & tensor preprocessing helpers
│   ├── cvr.py                  # Color Visibility Restoration (CLAHE post-processing)
│   └── utils/
│       └── model_info.py       # Layer-by-layer parameter counter
│
├── scripts/                    # Runnable entry-point scripts
│   ├── convert_trt.py          # Convert PyTorch .pth → TensorRT FP16 engine
│   ├── infer_image.py          # Run dehazing on a single image file
│   ├── infer_batch.py          # Run batch dehazing on a directory of images
│   ├── stream_live_camera.py   # Live Jetson camera dehazing MJPEG web stream (TRT)
│   ├── stream_video_file.py    # Video file dehazing MJPEG web stream (TRT)
│   └── camera_raw_stream.py    # Raw camera feed verification stream
│
├── docker/
│   └── SETUP.md                # Docker container load & execution guide
│
├── assets/                     # Architecture diagrams, ablation charts & result samples
│   ├── metrics_plot.png        # PSNR & SSIM vs Alpha ablation plot
│   ├── convergence_elbow.png   # Identity trap convergence latency plot
│   ├── framework.png           # Network architecture diagram
│   ├── results_foggy_cityscapes.png  # FoggyCityscapes evaluation strip
│   ├── results_smokebench.jpg        # SmokeBench evaluation strip
│   ├── bikers.mp4              # Test video for zero-shot inference
│   └── screenshots/            # Internet video (bikers.mp4) inference screenshots
│
├── weights/                    # Pre-trained model checkpoints & TensorRT engines
│   ├── trained_LDNet.pth       # PyTorch trained weights
│   ├── ldnet_trt.pth           # TensorRT FP16 engine weights
│   └── README.md               # Weights documentation
│
├── results/                    # Output directory for infer_batch / infer_image scripts
├── Dockerfile                  # Container definition for Jetson environment
├── run_container.sh            # One-command container launcher script
└── requirements.txt            # Python dependencies
```

---

## 🚀 Quickstart & Usage

### 1. Launch Docker Container on Jetson

```bash
# Load Docker image
docker load -i my_jetson_env.tar

# Launch interactive container with GPU & camera passthrough
bash run_container.sh
```

### 2. Convert PyTorch Model to TensorRT FP16

Inside the container:

```bash
python scripts/convert_trt.py
# Saves TensorRT engine to weights/ldnet_trt.pth
```

### 3. Single Image Dehazing

```bash
python scripts/infer_image.py -i assets/hazy_sample.png
# Saved output to results/
```

### 4. Batch Image Dehazing

```bash
python scripts/infer_batch.py -td query_hazy_images/outdoor_natural/
# Saves outputs to results/ directory
```

### 5. Real-Time Camera Stream (TensorRT FP16)

```bash
python scripts/stream_live_camera.py
# Open http://<jetson-ip>:5000 in your browser
```

### 6. Video Dehazing Stream (Side-by-Side Comparison)

```bash
python scripts/stream_video_file.py
# Streams side-by-side (Hazy | Dehazed) feed at http://<jetson-ip>:5000
```

---

## 📜 Citation

If you use this work or reference our custom loss training & TensorRT deployment on Jetson, please cite:

```bibtex
@article{ullah2021light,
  title={Light-DehazeNet: A Novel Lightweight CNN Architecture for Single Image Dehazing},
  author={Ullah, Hayat and Muhammad, Khan and Irfan, Muhammad and Anwar, Saeed and
          Sajjad, Muhammad and Imran, Ali Shariq and De Albuquerque, Victor Hugo C},
  journal={IEEE Transactions on Image Processing},
  year={2021},
  publisher={IEEE}
}
```

---

## 🤝 Acknowledgements

- **Light-DehazeNet**: Original architecture by Hayat Ullah et al. (IEEE TIP 2021).
- **NVIDIA `torch2trt`**: TensorRT PyTorch converter by NVIDIA AI-IOT.
