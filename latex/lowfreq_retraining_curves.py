from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "hfss" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _annotate_point(ax, x, y, text, color, dy=8):
    ax.annotate(
        text,
        (x, y),
        xytext=(0, dy),
        textcoords="offset points",
        ha="center",
        va="bottom" if dy >= 0 else "top",
        fontsize=8,
        color=color,
    )


def _plot_fmae_subplot(ax):
    epochs = np.arange(1, 31, dtype=float)
    clean_test = np.array([
        61.38, 62.43, 63.10, 59.98, 61.03, 59.87, 61.55, 61.00, 59.88, 60.65,
        60.07, 60.69, 60.02, 60.38, 60.50, 60.15, 60.27, 60.00, 60.51, 60.43,
        59.81, 59.67, 60.19, 60.43, 60.60, 60.27, 60.15, 60.48, 60.43, 60.47,
    ], dtype=float)
    masked_test = np.array([
        57.60, 58.91, 60.14, 57.77, 59.36, 57.89, 59.65, 59.23, 57.41, 58.34,
        57.29, 58.01, 57.68, 58.75, 58.30, 58.20, 58.05, 57.81, 58.60, 58.36,
        57.70, 57.34, 58.07, 58.22, 58.39, 58.06, 57.91, 58.15, 58.10, 58.14,
    ], dtype=float)

    blue = "#1f77b4"
    red = "#d62728"

    ax.plot(epochs, clean_test, color=blue, linestyle="-", linewidth=2.4, label="AU F1 (clean test)")
    ax.plot(epochs, masked_test, color=blue, linestyle="--", linewidth=2.0, label="AU F1 (masked test)")

    ax.axhline(63.51, color="grey", linestyle="--", linewidth=1.6, alpha=0.9, label="Baseline (63.51%)")

    ax.set_title("FMAE — Low-Frequency Retraining (Fold 1)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AU Macro F1 (%)")
    ax.set_xlim(1, 30)
    ax.set_ylim(55, 66)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.grid(True, alpha=0.22)

    ax.plot(3, clean_test[2], marker="o", color=red, markersize=10, linestyle="None", zorder=5)
    _annotate_point(ax, 3, clean_test[2], f"Best: {clean_test[2]:.2f}%", red, dy=10)

    ax.legend(loc="upper right", fontsize=9)

    return ax


def _plot_iat_subplot(ax):
    epochs = np.arange(1, 31, dtype=float)
    clean_test = np.array([
        63.16, 63.37, 61.01, 62.12, 60.43, 62.10, 61.88, 59.30, 60.79, 61.31,
        61.16, 60.93, 61.38, 60.49, 60.89, 60.47, 61.03, 59.44, 60.71, 60.47,
        60.30, 60.39, 60.57, 60.28, 59.94, 59.94, 59.73, 59.68, 59.76, 59.75,
    ], dtype=float)

    blue = "#ff7f0e"
    red = "#d62728"

    ax.plot(epochs, clean_test, color=blue, linestyle="-", linewidth=2.4, label="AU F1 (clean test)")
    ax.axhline(65.88, color="grey", linestyle="--", linewidth=1.6, alpha=0.9, label="Baseline (65.88%)")

    ax.set_title("FMAE-IAT — Low-Frequency Retraining (Fold 1)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AU Macro F1 (%)")
    ax.set_xlim(1, 30)
    ax.set_ylim(55, 68)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.grid(True, alpha=0.22)

    ax.plot(2, clean_test[1], marker="o", color=red, markersize=10, linestyle="None", zorder=5)
    _annotate_point(ax, 2, clean_test[1], f"Best: {clean_test[1]:.2f}%", red, dy=10)

    ax.legend(loc="upper right", fontsize=9)

    return ax


def main():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    _plot_fmae_subplot(axes[0])
    _plot_iat_subplot(axes[1])

    fig.tight_layout()

    output_path = FIGURES_DIR / "lowfreq_retraining_curves.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()