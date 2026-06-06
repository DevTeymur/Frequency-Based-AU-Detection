"""Generate a per-AU F1 heatmap across radial keep ratios for FMAE fold 1.

The color of each cell is centered on that AU's 100% keep baseline, so values
above baseline appear blue and values below baseline appear red.
"""

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def main():
    keep_ratios = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 8, 4]

    data = {
        "AU01": [57.8, 57.8, 57.8, 57.8, 58.2, 58.4, 58.8, 58.7, 58.8, 56.1, 56.5, 54.5],
        "AU02": [60.1, 60.0, 60.1, 60.0, 60.3, 60.3, 60.9, 61.1, 61.1, 58.8, 59.0, 55.3],
        "AU04": [56.0, 56.0, 56.0, 55.7, 55.7, 55.6, 55.4, 55.4, 55.6, 56.5, 56.7, 57.6],
        "AU06": [76.8, 76.8, 76.8, 76.7, 76.7, 76.6, 76.6, 76.5, 76.8, 76.9, 76.8, 77.0],
        "AU07": [81.1, 81.1, 81.1, 81.0, 80.9, 80.8, 80.8, 80.8, 81.2, 81.5, 81.7, 79.2],
        "AU10": [77.0, 77.0, 77.0, 77.0, 77.0, 77.0, 77.1, 77.2, 77.5, 78.7, 79.3, 78.4],
        "AU12": [89.0, 89.0, 89.0, 89.0, 88.9, 88.9, 88.9, 88.9, 88.9, 89.4, 89.3, 89.0],
        "AU14": [62.5, 62.5, 62.5, 62.6, 62.7, 62.7, 62.9, 63.0, 63.3, 63.1, 62.6, 62.6],
        "AU15": [35.9, 35.9, 35.8, 35.5, 35.8, 35.9, 35.7, 35.9, 35.8, 33.3, 30.9, 21.6],
        "AU17": [61.9, 61.9, 62.0, 61.8, 61.5, 61.7, 61.0, 60.9, 60.5, 60.0, 59.9, 59.1],
        "AU23": [49.7, 49.7, 49.6, 49.5, 49.4, 49.2, 49.2, 49.1, 49.0, 45.8, 45.1, 40.6],
        "AU24": [54.2, 54.2, 54.2, 54.2, 53.8, 54.1, 53.6, 54.4, 54.6, 52.2, 50.5, 45.1],
    }

    aus = list(data.keys())
    values = np.array([data[au] for au in aus], dtype=float)
    baseline = values[:, [0]]
    delta = values - baseline

    # Center each row on its 100% keep baseline so positive deltas are blue and negative deltas are red.
    max_abs = float(np.max(np.abs(delta)))
    norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(14, 7))

    heatmap = sns.heatmap(
        delta,
        ax=ax,
        cmap="RdBu_r",
        norm=norm,
        annot=values,
        fmt=".1f",
        annot_kws={"size": 8, "color": "black"},
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "F1 difference vs 100% keep baseline"},
    )

    ax.set_xticklabels([str(r) for r in keep_ratios], rotation=0)
    ax.set_yticklabels(aus, rotation=0)
    ax.set_xlabel("Keep Ratio (%)")
    ax.set_ylabel("Action Unit")
    ax.set_title("Per-AU F1 Scores Across Radial Keep Ratios (FMAE Fold 1)")

    # Separator between the 10% and 8% columns.
    # With seaborn heatmap, column boundaries sit on integer x positions.
    ax.axvline(10, color="black", linestyle="--", linewidth=1.2)

    fig.tight_layout()

    output_path = Path(__file__).resolve().parent / "radial_per_au_heatmap.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved heatmap to {output_path}")


if __name__ == "__main__":
    main()
