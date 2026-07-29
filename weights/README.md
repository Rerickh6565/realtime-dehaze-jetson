# Trained Weights

Pre-trained weights for LightDehazeNet and the converted TensorRT FP16 engine are available as assets in the [v0.1.0-alpha Release](https://github.com/sid-stack001/realtime-dehaze-jetson/releases/tag/v0.1.0-alpha).

## Download Links

| File | Description | Size | Download Link |
|---|---|---|---|
| `trained_LDNet.pth` | PyTorch weights (full precision) | ~125 KB | [Download](https://github.com/sid-stack001/realtime-dehaze-jetson/releases/download/v0.1.0-alpha/trained_LDNet.pth) |
| `ldnet_trt.pth` | TensorRT FP16 engine weights | ~2.2 MB | [Download](https://github.com/sid-stack001/realtime-dehaze-jetson/releases/download/v0.1.0-alpha/ldnet_trt.pth) |

### Quick Download via Command Line

Run from the repository root:

```bash
# Download PyTorch weights
wget https://github.com/sid-stack001/realtime-dehaze-jetson/releases/download/v0.1.0-alpha/trained_LDNet.pth -P weights/

# Download TensorRT FP16 engine
wget https://github.com/sid-stack001/realtime-dehaze-jetson/releases/download/v0.1.0-alpha/ldnet_trt.pth -P weights/
```

## Placement

After downloading, ensure the files are located in the `weights/` directory:

```
weights/
├── trained_LDNet.pth   # PyTorch model checkpoint
└── ldnet_trt.pth       # TensorRT FP16 engine weights
```

## Generating TensorRT Weights Manually

If you have downloaded `trained_LDNet.pth` and wish to regenerate the TensorRT FP16 engine on your Jetson device:

```bash
python scripts/convert_trt.py
```

This script reads `weights/trained_LDNet.pth` and outputs `weights/ldnet_trt.pth`.
