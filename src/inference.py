"""
inference.py — Model loading and image preprocessing utilities.

Provides a singleton model loader (loaded once, reused across calls)
and helper functions for preprocessing PIL images and running dehazing inference.
"""

import torch
import numpy as np
from src.model import LightDehaze_Net

# Singleton model instance — loaded once per process lifetime.
_model = None
_WEIGHTS_PATH = "weights/trained_LDNet.pth"


def load_model(weights_path: str = _WEIGHTS_PATH) -> LightDehaze_Net:
    """
    Load and return the LightDehaze_Net model.

    On the first call the model is loaded from disk and cached. Subsequent
    calls return the cached instance without re-reading the file.

    Args:
        weights_path: Path to the trained PyTorch weights (.pth file).

    Returns:
        LightDehaze_Net model in eval mode on CUDA.
    """
    global _model
    if _model is None:
        _model = LightDehaze_Net().cuda()
        _model.load_state_dict(torch.load(weights_path))
        _model.eval()
    return _model


def preprocess(input_image) -> torch.Tensor:
    """
    Convert a PIL image to a normalised CUDA tensor ready for inference.

    Args:
        input_image: PIL.Image in RGB mode.

    Returns:
        Float32 CUDA tensor of shape (1, 3, H, W) in [0, 1].
    """
    img = np.asarray(input_image) / 255.0
    img_tensor = torch.from_numpy(img).float().permute(2, 0, 1)
    return img_tensor.unsqueeze(0).cuda()


def image_haze_removal(input_image) -> torch.Tensor:
    """
    Run dehazing inference on a single PIL image.

    Args:
        input_image: PIL.Image in RGB mode.

    Returns:
        Output tensor of shape (1, 3, H, W) with dehazed pixel values.
    """
    model = load_model()
    inp = preprocess(input_image)
    with torch.no_grad():
        out = model(inp)
    return out
