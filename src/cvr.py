"""
cvr.py — Color Visibility Restoration (CVR) post-processing module.

Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to the
L channel of the LAB color space to improve perceptual contrast of a
dehazed image without over-saturating colors.

Reference:
    Ullah et al., Light-DehazeNet (IEEE TIP 2021) — CVR module.
    Original MATLAB implementation ported to Python/OpenCV.
"""

import cv2
import numpy as np


def cvr(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_size: tuple = (8, 8),
) -> np.ndarray:
    """
    Apply Color Visibility Restoration to a BGR image.

    Converts to LAB color space, applies CLAHE to the lightness (L) channel,
    then converts back to BGR.

    Args:
        image:      Input BGR image as a uint8 numpy array.
        clip_limit: CLAHE contrast limit threshold (default: 2.0).
        tile_size:  Grid size for CLAHE tiles (default: (8, 8)).

    Returns:
        Contrast-enhanced BGR image as a uint8 numpy array.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    enhanced_l = clahe.apply(l)

    enhanced_lab = cv2.merge([enhanced_l, a, b])
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
