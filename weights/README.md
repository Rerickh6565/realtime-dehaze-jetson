# Trained Weights

Pre-trained LD-Net weights are **not committed to this repository** due to file size constraints.

## Download

| File | Description | Size |
|---|---|---|
| `trained_LDNet.pth` | PyTorch weights (full precision) | ~125 KB |
| `ldnet_trt.pth` | TensorRT FP16 converted weights | ~2.2 MB |

> Weights will be available on the [GitHub Releases](../../releases) page or via the project's Google Drive link.

## Placement

After downloading, place the files in this directory:

```
weights/
├── trained_LDNet.pth   ← PyTorch model
└── ldnet_trt.pth       ← TensorRT model (after running scripts/convert_trt.py)
```

## Generating TRT Weights Yourself

If you have the PyTorch weights, convert them to TensorRT on your Jetson:

```bash
python scripts/convert_trt.py
```

This reads `weights/trained_LDNet.pth` and outputs `weights/ldnet_trt.pth`.
