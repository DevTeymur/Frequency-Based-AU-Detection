from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_PATH = Path(__file__).resolve().parent / "retraining_curves.png"


def annotate(ax, x, y, text, color, dy=8, ha="center"):
    ax.annotate(
        text,
        (x, y),
        xytext=(0, dy),
        textcoords="offset points",
        ha=ha,
        va="bottom" if dy >= 0 else "top",
        fontsize=8,
        color=color,
    )


def plot_fmae(ax):
    epochs = np.arange(1, 31)
    clean = np.array([
        61.38, 62.43, 63.10, 59.98, 61.03, 59.87, 61.55, 61.00, 59.88,
        60.65, 60.07, 60.69, 60.02, 60.38, 60.50, 60.15, 60.27, 60.00,
        60.51, 60.43, 59.81, 59.67, 60.19, 60.43, 60.60, 60.27, 60.15,
        60.48, 60.43, 60.47,
    ])
    masked = np.array([
        57.60, 58.91, 60.14, 57.77, 59.36, 57.89, 59.65, 59.23, 57.41,
        58.34, 57.29, 58.01, 57.68, 58.75, 58.30, 58.20, 58.05, 57.81,
        58.60, 58.36, 57.70, 57.34, 58.07, 58.22, 58.39, 58.06, 57.91,
        58.15, 58.10, 58.14,
    ])

    blue = "#1f77b4"
    grey = "#7a7a7a"

    ax.plot(epochs, clean, color=blue, linewidth=2.2, label="Clean test F1")
    ax.plot(epochs, masked, color=blue, linestyle="--", linewidth=2.2, label="Masked test F1")
    ax.axhline(63.51, color=grey, linestyle="--", linewidth=1.4, label="Baseline (63.51%)")
    ax.axvline(3, color=grey, linestyle=":", linewidth=1.4)

    annotate(ax, 3, clean[2], "Best epoch (63.10%)", grey, dy=10)

    ax.set_title("FMAE — Low-Frequency Retraining (Fold 1)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AU Macro F1 (%)")
    ax.set_xlim(1, 30)
    ax.set_ylim(56, 65)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.grid(True, color="#cccccc", alpha=0.3, linewidth=0.8)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95)


def plot_iat(ax):
    epochs = np.arange(1, 31)
    clean = np.array([
        63.16, 63.37, 61.01, 62.12, 60.43, 62.10, 61.88, 59.30, 60.79,
        61.31, 61.16, 60.93, 61.38, 60.49, 60.89, 60.47, 61.03, 59.44,
        60.71, 60.47, 60.30, 60.39, 60.57, 60.28, 59.94, 59.94, 59.73,
        59.68, 59.76, 59.75,
    ])

    green = "#2ca02c"
    grey = "#7a7a7a"

    ax.plot(epochs, clean, color=green, linewidth=2.2, label="Clean test F1")
    ax.axhline(65.88, color=grey, linestyle="--", linewidth=1.4, label="Baseline (65.88%)")
    ax.axvline(2, color=grey, linestyle=":", linewidth=1.4)

    annotate(ax, 2, clean[1], "Best epoch (63.37%)", grey, dy=10)

    ax.set_title("FMAE-IAT — Low-Frequency Retraining (Fold 1)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AU Macro F1 (%)")
    ax.set_xlim(1, 30)
    ax.set_ylim(56, 67)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.grid(True, color="#cccccc", alpha=0.3, linewidth=0.8)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95)


def main():
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 9,
        "font.family": "DejaVu Sans",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_fmae(axes[0])
    plot_iat(axes[1])
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
