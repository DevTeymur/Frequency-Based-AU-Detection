"""
Generate the radial macro-F1 comparison figures requested for the thesis.

Outputs:
  - hfss/figures/radial_macro_f1_curve.png
  - hfss/figures/radial_fmae_vs_iat_curve.png

The figures are sized at 10x5 inches, 150 dpi with extended low-frequency data.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "hfss" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _setup_figure():
    fig = plt.figure(figsize=(10, 5), dpi=150)
    ax = fig.add_subplot(111)
    return fig, ax


def plot_macro_f1_curve():
    # FMAE Fold 1: extended to very low keep ratios.
    fold1_keep = np.array([100, 90, 80, 70, 60, 50, 40, 30, 20, 14.2, 10, 8, 6.3, 4, 1.6], dtype=float)
    fold1_f1 = np.array([63.51, 63.50, 63.50, 63.48, 63.45, 63.45, 63.40, 63.49, 63.58, 63.38, 62.69, 62.35, 61.62, 59.99, 55.99], dtype=float)

    # FMAE Fold 2: extended with low-frequency points.
    fold2_keep = np.array([100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 8, 4], dtype=float)
    fold2_f1 = np.array([68.37, 68.37, 68.37, 68.35, 68.46, 68.55, 68.64, 68.64, 68.73, 68.31, 67.83, 65.44], dtype=float)

    fig, ax = _setup_figure()
    color_fold1 = "#1f77b4"
    color_fold2 = "#ff7f0e"

    # Plot lines with markers
    ax.plot(fold1_keep, fold1_f1, marker="o", linewidth=2.5, markersize=7, color=color_fold1, label="FMAE Fold 1")
    ax.plot(fold2_keep, fold2_f1, marker="s", linewidth=2.5, markersize=7, color=color_fold2, label="FMAE Fold 2")

    # Baseline horizontal dashed lines
    ax.axhline(63.51, linestyle="--", linewidth=1.8, color=color_fold1, alpha=0.8, label="Fold 1 baseline")
    ax.axhline(68.37, linestyle="--", linewidth=1.8, color=color_fold2, alpha=0.8, label="Fold 2 baseline")

    # Vertical dashed line at 10% keep
    ax.axvline(10, linestyle="--", linewidth=1.5, color="grey", alpha=0.6, label="10% keep")

    ax.set_xlabel("Frequency Keep Ratio (%)", fontsize=12)
    ax.set_ylabel("Macro F1 (%)", fontsize=12)
    ax.set_xlim(100, 0)
    ax.set_ylim(54, 72)
    ax.set_xticks([100, 80, 60, 40, 20, 10, 8, 4])
    ax.grid(True, alpha=0.25)
    ax.set_title("Macro F1 vs Frequency Keep Ratio — Radial Ablation (FMAE)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)

    output_path = FIGURES_DIR / "radial_macro_f1_curve.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_fmae_vs_iat_curve():
    keep = np.array([100, 90, 80, 70, 60, 50, 40, 30, 20, 10], dtype=float)

    # FMAE Fold 1
    fmae = np.array([63.51, 63.50, 63.50, 63.48, 63.45, 63.45, 63.40, 63.49, 63.58, 62.69], dtype=float)

    # FMAE-IAT Fold 1: computed from AU averages
    # Step 1 (100%): avg([58.4, 63.0, 61.0, 76.3, 82.6, 80.1, 89.6, 66.0, 50.5, 61.4, 47.8, 53.9]) = 65.88
    # Step 2 (90%):  avg([58.5, 63.0, 61.0, 76.3, 82.6, 80.1, 89.6, 66.0, 50.5, 61.4, 47.8, 53.9]) = 65.89
    # Step 3 (80%):  avg([58.5, 63.1, 61.0, 76.3, 82.6, 80.1, 89.6, 66.0, 50.5, 61.4, 47.8, 53.9]) = 65.90
    # Step 4 (70%):  avg([58.4, 63.2, 61.2, 76.3, 82.6, 80.0, 89.6, 65.8, 50.6, 61.3, 47.6, 53.9]) = 65.88
    # Step 5 (60%):  avg([58.3, 63.3, 61.3, 76.3, 82.5, 79.9, 89.5, 65.6, 50.8, 61.5, 47.4, 53.8]) = 65.85
    # Step 6 (50%):  avg([58.2, 63.3, 61.3, 76.3, 82.4, 80.1, 89.5, 65.5, 50.7, 61.3, 47.4, 53.7]) = 65.81
    # Step 7 (40%):  avg([58.7, 63.6, 61.4, 76.3, 82.3, 80.2, 89.5, 65.6, 50.7, 60.7, 47.3, 53.4]) = 65.81
    # Step 8 (30%):  avg([59.2, 63.4, 61.8, 76.3, 82.2, 80.4, 89.5, 65.6, 50.4, 60.7, 47.1, 53.5]) = 65.84
    # Step 9 (20%):  avg([59.6, 63.4, 61.8, 76.4, 82.2, 80.5, 89.4, 65.8, 50.0, 60.7, 46.7, 53.7]) = 65.85
    # Step 10 (10%): avg([58.4, 64.4, 62.2, 76.4, 82.0, 81.5, 89.3, 65.5, 48.8, 59.5, 45.4, 52.9]) = 65.53
    iat = np.array([65.88, 65.89, 65.90, 65.88, 65.85, 65.81, 65.81, 65.84, 65.85, 65.53], dtype=float)

    fig, ax = _setup_figure()
    color_fmae = "#1f77b4"
    color_iat = "#ff7f0e"

    # Plot lines with markers
    ax.plot(keep, fmae, marker="o", linewidth=2.5, markersize=7, color=color_fmae, label="FMAE")
    ax.plot(keep, iat, marker="s", linewidth=2.5, markersize=7, color=color_iat, label="FMAE-IAT")

    # Baseline horizontal dashed lines
    ax.axhline(63.51, linestyle="--", linewidth=1.8, color=color_fmae, alpha=0.8, label="FMAE baseline")
    ax.axhline(65.88, linestyle="--", linewidth=1.8, color=color_iat, alpha=0.8, label="IAT baseline")

    ax.set_xlabel("Frequency Keep Ratio (%)", fontsize=12)
    ax.set_ylabel("Macro F1 (%)", fontsize=12)
    ax.set_xlim(100, 0)
    ax.set_ylim(60, 70)
    ax.set_xticks([100, 80, 60, 40, 20, 10])
    ax.grid(True, alpha=0.25)
    ax.set_title("FMAE vs FMAE-IAT Macro F1 under Radial Masking (Fold 1)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10)

    output_path = FIGURES_DIR / "radial_fmae_vs_iat_curve.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    plot_macro_f1_curve()
    plot_fmae_vs_iat_curve()


if __name__ == "__main__":
    main()