"""
infer_image.py — Dehaze a single image using the LightDehaze_Net model.

Usage:
    python scripts/infer_image.py -i assets/hazy_sample.png

Output:
    Saves dehazed result to dehaze.jpg in the current directory.
"""

import argparse

import torchvision
from PIL import Image

from src.inference import image_haze_removal

OUTPUT_PATH = "dehaze.jpg"


def dehaze_single_image(image_path: str) -> None:
    """
    Load an image, run dehazing, and save the result.

    Args:
        image_path: Path to the input hazy image.
    """
    print(f"Input : {image_path}")
    hazy_img = Image.open(image_path).convert("RGB")
    dehazed = image_haze_removal(hazy_img)
    torchvision.utils.save_image(dehazed, OUTPUT_PATH)
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Dehaze a single image")
    ap.add_argument("-i", "--image", required=True, help="Path to input hazy image")
    args = vars(ap.parse_args())
    dehaze_single_image(args["image"])
