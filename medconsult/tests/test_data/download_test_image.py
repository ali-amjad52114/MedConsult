"""
Downloads a sample chest X-ray for testing the image pipeline.

Usage:
    python tests/test_data/download_test_image.py

Downloads to: tests/test_data/sample_xray.jpg

Sources tried (in order):
  1. NIH ChestX-ray14 sample (public domain, CC0)
  2. Synthetic fallback — creates a grayscale PIL image that exercises
     the image code path without requiring internet access.

For real Kaggle/competition demos, replace sample_xray.jpg with an actual
CheXpert or NIH ChestX-ray14 image. Download instructions:
  - CheXpert: https://stanfordmlgroup.github.io/competitions/chexpert/
  - NIH ChestX-ray14: https://nihcc.app.box.com/v/ChestXray-NIHCC
"""

import os
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
OUTPUT_PATH = HERE / "sample_xray.jpg"

# Public chest X-ray sample images (CC0 / public domain)
CANDIDATE_URLS = [
    # COVID-19 chest X-ray dataset (Cohen et al.) — public domain
    "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/auntminnie-a-2020_01_28_23_51_6665_2020_01_28_Biden_New-corona-1.jpeg",
    # RSNA Pneumonia detection sample (from Kaggle public data mirror)
    "https://raw.githubusercontent.com/TataOwl/medical-data-sharing/main/chest_xray/NORMAL/IM-0115-0001.jpeg",
]


def _create_synthetic_xray(path: Path):
    """
    Creates a synthetic grayscale PNG that mimics the gross structure of
    a PA chest X-ray: dark lung fields, white rib-like arcs, central cardiac shadow.
    NOT a real medical image — only for testing the image code path.
    """
    from PIL import Image, ImageDraw
    import math

    W, H = 512, 512
    img = Image.new("RGB", (W, H), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)

    # Simulate lung fields (slightly lighter ovals)
    draw.ellipse([60, 80, 220, 400], fill=(55, 55, 55))   # left lung
    draw.ellipse([290, 80, 450, 400], fill=(55, 55, 55))  # right lung

    # Simulate cardiac silhouette (bright central mass)
    draw.ellipse([185, 200, 330, 360], fill=(180, 180, 180))

    # Simulate rib arcs
    for i in range(6):
        y_offset = 90 + i * 48
        draw.arc([40, y_offset, 240, y_offset + 60], start=190, end=355, fill=(120, 120, 120), width=3)
        draw.arc([270, y_offset, 470, y_offset + 60], start=185, end=350, fill=(120, 120, 120), width=3)

    # Trachea
    draw.rectangle([245, 50, 265, 190], fill=(150, 150, 150))

    # Diaphragm domes
    draw.arc([40, 370, 240, 440], start=0, end=180, fill=(140, 140, 140), width=4)
    draw.arc([270, 370, 470, 440], start=0, end=180, fill=(140, 140, 140), width=4)

    img.save(path, "JPEG", quality=85)
    print(f"Created synthetic chest X-ray: {path}")


def download():
    if OUTPUT_PATH.exists():
        print(f"Image already exists: {OUTPUT_PATH}")
        return

    for url in CANDIDATE_URLS:
        try:
            print(f"Trying: {url}")
            urllib.request.urlretrieve(url, OUTPUT_PATH)
            print(f"Downloaded: {OUTPUT_PATH}")
            return
        except Exception as e:
            print(f"  Failed: {e}")

    print("All URLs failed. Creating synthetic chest X-ray for testing...")
    _create_synthetic_xray(OUTPUT_PATH)


if __name__ == "__main__":
    download()
    # Quick sanity check
    from PIL import Image
    img = Image.open(OUTPUT_PATH)
    print(f"Image ready: {img.size} px, mode={img.mode}")
