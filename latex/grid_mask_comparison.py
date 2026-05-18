"""
Create a simple stage comparison figure from saved HFSS PKL masks.
Top-level output: hfss/figures/grid_mask_comparison.png
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hfss_search_au import load_masks_from_pkl  # reuse existing code


def get_mask_array(mask_obj):
    if hasattr(mask_obj, "mask"):
        return np.asarray(mask_obj.mask, dtype=np.float32)
    return np.asarray(mask_obj, dtype=np.float32)


def main():
    stage1_pkl = PROJECT_ROOT / "hfss" / "DFM" / "Fold 1" / "FMAE_stage1_DFMs.pkl"
    stage3_pkl = PROJECT_ROOT / "hfss" / "DFM" / "Fold 1" / "FMAE_stage3_DFMs.pkl"

    stage1_masks = load_masks_from_pkl(stage1_pkl)
    stage3_masks = load_masks_from_pkl(stage3_pkl)

    # Use first saved mask from each stage for an example comparison.
    m1 = get_mask_array(stage1_masks[0])
    m3 = get_mask_array(stage3_masks[0])

    out_path = PROJECT_ROOT / "hfss" / "figures" / "grid_mask_comparison.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

    axes[0].imshow(m1, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Stage 1 mask (coarse)")
    axes[0].axis("off")

    axes[1].imshow(m3, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Stage 3 mask (finer)")
    axes[1].axis("off")

    fig.suptitle("HFSS Grid Mask Comparison (224x224)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
