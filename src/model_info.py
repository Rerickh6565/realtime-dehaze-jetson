"""
model_info.py — Print LightDehaze_Net parameter counts (total and per-layer).

Usage:
    python src/model_info.py
"""

import torch
from src.model import LightDehaze_Net

WEIGHTS_PATH = "weights/trained_LDNet.pth"


def count_parameters(model: torch.nn.Module):
    """Return (total_params, trainable_params) for a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    model = LightDehaze_Net()
    state_dict = torch.load(WEIGHTS_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    total, trainable = count_parameters(model)

    print("\n===== MODEL PARAMS =====")
    print(f"Total Params      : {total:,}")
    print(f"Trainable Params  : {trainable:,}")
    print(f"In Millions       : {total / 1e6:.6f} M")

    print("\n===== PER LAYER =====")
    for name, param in model.named_parameters():
        print(f"{name:45s} {param.numel():>10,}")


if __name__ == "__main__":
    main()
