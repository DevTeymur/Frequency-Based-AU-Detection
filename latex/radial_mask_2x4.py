"""Generate a 2x4 radial mask visualization.

Columns: keep 100%, 10%, 5%, 1%.
Row 1: original image (RGB)
Row 2: corresponding masked frequency log-magnitude (centered)

Usage:
    python3 latex/radial_mask_2x4.py --mode high-pass --image_index 0
"""

import argparse
import json
import sys
from pathlib import Path
import importlib.util

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "BP4D" / "BP4D_cropped"
JSON_FILE = PROJECT_ROOT / "BP4D" / "BP4D_test1.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "radial_mask_output"
OUTPUT_DIR.mkdir(exist_ok=True)

IMG_SIZE = 224


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["low-pass", "high-pass"], default="high-pass")
    p.add_argument("--image_index", type=int, default=0)
    p.add_argument("--out", type=str, default=str(OUTPUT_DIR / "radial_2x4.png"))
    return p.parse_args()


"""
Load the demo functions directly so masking is identical.
"""
demo_path = PROJECT_ROOT / "latex" / "radial_mask_demo.py"
spec = importlib.util.spec_from_file_location("radial_mask_demo", str(demo_path))
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)

# Expose demo helpers locally
load_first_image = demo.load_first_image
radial_distance_map = demo.radial_distance_map
generate_radial_mask = demo.generate_radial_mask
masked_spectrum_map = demo.masked_spectrum_map
tensor_to_image = demo.tensor_to_image
apply_frequency_mask = demo.apply_frequency_mask


def main():
    args = parse_args()
    keep_list = [100.0, 10.0, 5.0, 1.0]
    # demo.load_first_image returns only the tensor (not the path), so obtain
    # the path separately from the demo JSON file for provenance printing.
    img_tensor = load_first_image(args.image_index)
    original_img = tensor_to_image(img_tensor)
    # read image path from demo JSON
    try:
        with open(demo.JSON_FILE, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        sample = json.loads(lines[args.image_index % len(lines)])
        img_path = sample.get("img_path", "")
    except Exception:
        img_path = ""
    dist = radial_distance_map(IMG_SIZE)

    masked_imgs = []
    masked_specs = []
    keep_pcts = []
    radii = []
    for k in keep_list:
        mask, radius = generate_radial_mask(k, mode=args.mode, dist=dist)
        # compute masked image (reconstructed spatial domain) using demo's function
        masked_img = apply_frequency_mask(img_tensor, mask)
        masked_img_np = tensor_to_image(masked_img, normalize=True)
        # compute masked spectrum (log-magnitude, centered)
        masked_spec = masked_spectrum_map(img_tensor, mask)
        masked_imgs.append(masked_img_np)
        masked_specs.append(masked_spec)
        keep_pcts.append(k)
        radii.append(radius)

    # Plot 2x4: top row images, bottom row spectra (tight spacing to match demo)
    cols = len(keep_list)
    fig, axes = plt.subplots(2, cols, figsize=(4 * cols, 4.5))
    for c in range(cols):
        axes[0, c].imshow(original_img)
        axes[0, c].set_title(f"Keep {keep_pcts[c]:.0f}%", fontsize=10)
        axes[0, c].axis("off")

        axes[1, c].imshow(masked_specs[c], cmap="magma")
        # overlay radius circle on spectrum (visual guide)
        circle = plt.Circle((IMG_SIZE // 2, IMG_SIZE // 2), radii[c], color="cyan", fill=False, linewidth=1.0)
        axes[1, c].add_patch(circle)
        axes[1, c].axis("off")

    # Tighten spacing to approximate 0.1 mm gaps (use very small fractions)
    # Note: wspace/hspace are fractions of average axis width/height; values ~0.001 approximate ~0.1mm for our figure size.
    # Remove spaces between columns/rows
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.0, hspace=0.0)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path} (image_index={args.image_index} | src={img_path})")


if __name__ == "__main__":
    main()
