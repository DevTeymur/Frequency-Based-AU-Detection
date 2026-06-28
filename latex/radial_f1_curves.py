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
    # FMAE Fold 1: 100% down to 4%
    fold1_keep = np.array([100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 8, 4], dtype=float)
    fold1_f1   = np.array([63.51, 63.50, 63.50, 63.48, 63.45, 63.45, 63.40, 63.49, 63.58, 62.69, 62.35, 59.99], dtype=float)

    # FMAE Fold 2: 100% down to 4%
    fold2_keep = np.array([100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 8, 4], dtype=float)
    fold2_f1   = np.array([68.37, 68.37, 68.37, 68.35, 68.46, 68.55, 68.64, 68.64, 68.73, 68.31, 67.83, 65.44], dtype=float)

    # FMAE Fold 3: 100% down to 4%
    fold3_keep = np.array([100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 8, 4], dtype=float)
    fold3_f1   = np.array([64.52, 64.52, 64.54, 64.54, 64.78, 64.86, 64.89, 64.93, 64.89, 64.89, 64.79, 63.96], dtype=float)

    fig, ax = _setup_figure()
    color_fold1 = "#1f77b4"
    color_fold2 = "#ff7f0e"
    color_fold3 = "#2ca02c"

    ax.plot(fold1_keep, fold1_f1, marker="o", linewidth=2.5, markersize=7, color=color_fold1, label="FMAE Fold 1")
    ax.plot(fold2_keep, fold2_f1, marker="s", linewidth=2.5, markersize=7, color=color_fold2, label="FMAE Fold 2")
    ax.plot(fold3_keep, fold3_f1, marker="^", linewidth=2.5, markersize=7, color=color_fold3, label="FMAE Fold 3")

    ax.axhline(63.51, linestyle="--", linewidth=1.8, color=color_fold1, alpha=0.8, label="Fold 1 baseline")
    ax.axhline(68.37, linestyle="--", linewidth=1.8, color=color_fold2, alpha=0.8, label="Fold 2 baseline")
    ax.axhline(64.52, linestyle="--", linewidth=1.8, color=color_fold3, alpha=0.8, label="Fold 3 baseline")

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

    # FMAE-IAT Fold 1
    iat_fold1 = np.array([65.88, 65.89, 65.90, 65.88, 65.85, 65.81, 65.81, 65.84, 65.85, 65.53], dtype=float)

    # FMAE-IAT Fold 2
    iat_fold2 = np.array([69.03, 69.03, 69.02, 69.05, 69.15, 69.19, 69.27, 69.32, 69.16, 68.31], dtype=float)

    # FMAE-IAT Fold 3
    iat_fold3 = np.array([64.27, 64.27, 64.27, 64.29, 64.39, 64.38, 64.36, 64.46, 64.33, 64.02], dtype=float)

    fig, ax = _setup_figure()
    color_fmae     = "#1f77b4"
    color_iat_f1   = "#ff7f0e"
    color_iat_f2   = "#2ca02c"
    color_iat_f3   = "#d62728"

    ax.plot(keep, fmae,      marker="o", linewidth=2.5, markersize=7, color=color_fmae,   label="FMAE Fold 1")
    ax.plot(keep, iat_fold1, marker="s", linewidth=2.5, markersize=7, color=color_iat_f1, label="FMAE-IAT Fold 1")
    ax.plot(keep, iat_fold2, marker="^", linewidth=2.5, markersize=7, color=color_iat_f2, label="FMAE-IAT Fold 2")
    ax.plot(keep, iat_fold3, marker="D", linewidth=2.5, markersize=7, color=color_iat_f3, label="FMAE-IAT Fold 3")

    ax.axhline(63.51, linestyle="--", linewidth=1.8, color=color_fmae,   alpha=0.8, label="FMAE Fold 1 baseline")
    ax.axhline(65.88, linestyle="--", linewidth=1.8, color=color_iat_f1, alpha=0.8, label="IAT Fold 1 baseline")
    ax.axhline(69.03, linestyle="--", linewidth=1.8, color=color_iat_f2, alpha=0.8, label="IAT Fold 2 baseline")
    ax.axhline(64.27, linestyle="--", linewidth=1.8, color=color_iat_f3, alpha=0.8, label="IAT Fold 3 baseline")

    ax.set_xlabel("Frequency Keep Ratio (%)", fontsize=12)
    ax.set_ylabel("Macro F1 (%)", fontsize=12)
    ax.set_xlim(100, 0)
    ax.set_ylim(60, 73)
    ax.set_xticks([100, 80, 60, 40, 20, 10])
    ax.grid(True, alpha=0.25)
    ax.set_title("FMAE vs FMAE-IAT Macro F1 under Radial Masking (All Folds)", fontsize=13, fontweight="bold")
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