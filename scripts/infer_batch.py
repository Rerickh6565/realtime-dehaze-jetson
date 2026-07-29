"""
infer_batch.py — Dehaze a directory of images using the LightDehaze_Net model.

Usage:
    python scripts/infer_batch.py -td query_hazy_images/outdoor_natural/

Output:
    Saves dehazed results to results/ in the current directory.
"""

import argparse
import os

import torchvision
from PIL import Image

from src.inference import image_haze_removal

OUTPUT_DIR = "results"


def dehaze_batch(directory: str) -> None:
    """
    Dehaze all images in a directory and save results.

    Args:
        directory: Path to a directory of hazy input images.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image_files = [
        f for f in os.listdir(directory)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]

    if not image_files:
        print(f"No image files found in: {directory}")
        return

    print(f"Found {len(image_files)} images in: {directory}")

    for idx, filename in enumerate(image_files, start=1):
        img_path = os.path.join(directory, filename)
        try:
            img = Image.open(img_path).convert("RGB")
            dehazed = image_haze_removal(img)
            out_path = os.path.join(OUTPUT_DIR, f"dehazed_{idx:04d}.jpg")
            torchvision.utils.save_image(dehazed, out_path)
            print(f"  [{idx}/{len(image_files)}] {filename} → {out_path}")
        except Exception as e:
            print(f"  [{idx}] Skipped {filename}: {e}")

    print(f"\nDone. Results saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Dehaze a directory of images")
    ap.add_argument("-td", "--test_directory", required=True,
                    help="Path to directory of hazy images")
    args = vars(ap.parse_args())
    dehaze_batch(args["test_directory"])
