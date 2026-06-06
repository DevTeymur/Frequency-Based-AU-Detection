"""
Generate a per-AU F1 drop heatmap across hierarchical search stages (FMAE Fold 1).

Color scale:
  - Negative values (blue): improvement
  - Positive values (red): performance drop
  - Centered at 0

Output: hfss/figures/per_au_stage_heatmap.png
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from pathlib import Path

# Data per AU and stage (F1 drop in percentage points)
aus = ["AU01", "AU02", "AU04", "AU06", "AU07", "AU10", "AU12", "AU14", "AU15", "AU17", "AU23", "AU24"]

stage_1 = [-1.03, -0.85, -0.76, -0.20, 0.00, -2.39, -0.08, -0.75, -2.19, 0.00, -0.06, -6.05]
stage_2 = [-0.36, -1.04, -1.43, -0.65, 0.01, -3.35, -0.08, -0.54, -1.78, -0.71, 0.72, -7.87]
stage_3 = [4.57, 3.17, 0.24, -0.33, 0.55, -3.92, 0.83, -0.78, 2.22, -0.11, 0.74, -5.34]

stages = ["Stage 1", "Stage 2", "Stage 3"]

# Construct data matrix: rows are AUs, columns are stages
data_matrix = np.array([stage_1, stage_2, stage_3]).T

# Create figure
fig, ax = plt.subplots(figsize=(8, 5))

# Create diverging colormap centered at 0
norm = mcolors.CenteredNorm(vcenter=0)
im = ax.imshow(data_matrix, cmap="RdBu_r", norm=norm, aspect="auto")

# Set ticks and labels
ax.set_xticks(np.arange(len(stages)))
ax.set_yticks(np.arange(len(aus)))
ax.set_xticklabels(stages)
ax.set_yticklabels(aus)

# Add text annotations
for i in range(len(aus)):
    for j in range(len(stages)):
        val = data_matrix[i, j]
        text_str = f"+{val:.2f}%" if val >= 0 else f"{val:.2f}%"
        ax.text(j, i, text_str, ha="center", va="center", color="black", fontsize=9)

# Labels and title
ax.set_xlabel("Search Stage", fontsize=12)
ax.set_ylabel("Action Unit", fontsize=12)
ax.set_title("Per-AU F1 Drop at Best Mask - Per-AU Search (FMAE Fold 1)", fontsize=13, pad=15)

# Colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("F1 Drop (%)", fontsize=11)

# Layout
plt.tight_layout()

# Ensure output directory exists
output_path = Path(__file__).resolve().parent.parent / "hfss" / "figures" / "per_au_stage_heatmap.png"
output_path.parent.mkdir(parents=True, exist_ok=True)

# Save as PNG
plt.savefig(str(output_path), dpi=100, bbox_inches="tight")
print(f"Saved heatmap to {output_path}")
plt.close()
