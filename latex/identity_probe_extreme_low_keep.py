from pathlib import Path

import matplotlib.pyplot as plt


def main():
    keep_ratios = [8, 4, 2, 1]
    fmae_id_acc = [92.44, 91.30, 88.21, 78.37]
    fmae_au_f1 = [62.35, 59.99, 56.83, 53.11]
    x_positions = [8.0, 4.45, 2.45, 0.95]

    fmae_id_acc_10pct = 92.52
    fmae_au_f1_10pct = 62.69

    degradation_labels = [
        "Blurry",
        "Near featureless",
        "Only global shape",
        "DC component only",
    ]
    x_labels = [f"{keep}%\n{label}" for keep, label in zip(keep_ratios, degradation_labels)]

    output_path = Path(__file__).resolve().parent / "identity_probe_extreme_low_keep.png"

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 11,
            "font.family": "DejaVu Sans",
        }
    )

    fig, ax_left = plt.subplots(figsize=(9, 5))
    ax_right = ax_left.twinx()

    left_line = ax_left.plot(
        x_positions,
        fmae_id_acc,
        color="#c62828",
        marker="o",
        markersize=7,
        linewidth=2.2,
        label="Identity Accuracy",
    )[0]
    right_line = ax_right.plot(
        x_positions,
        fmae_au_f1,
        color="#1565c0",
        marker="s",
        markersize=7,
        linewidth=2.2,
        linestyle="--",
        label="AU Macro F1",
    )[0]

    ax_left.axhline(
        2.4,
        color="#7a7a7a",
        linestyle=":",
        linewidth=1.2,
        label="Chance (2.4%)",
    )

    ax_left.axhline(
        fmae_id_acc_10pct,
        color="#c62828",
        linestyle="--",
        linewidth=1.0,
        alpha=0.55,
    )
    ax_right.axhline(
        fmae_au_f1_10pct,
        color="#1565c0",
        linestyle="--",
        linewidth=1.0,
        alpha=0.55,
    )

    for x, y in zip(x_positions, fmae_id_acc):
        ax_left.annotate(
            f"{y:.2f}%",
            (x, y),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            va="bottom",
            fontsize=9,
            color="#8e1b1b",
        )

    for x, y in zip(x_positions, fmae_au_f1):
        ax_right.annotate(
            f"{y:.2f}%",
            (x, y),
            textcoords="offset points",
            xytext=(0, -14),
            ha="center",
            va="top",
            fontsize=9,
            color="#0d47a1",
        )

    ax_left.set_xlabel("Keep Ratio (%)")
    ax_left.set_ylabel("Identity Accuracy (%)")
    ax_right.set_ylabel("AU Macro F1 (%)")
    ax_left.set_title("Identity Encoding at Extreme Low Frequency Keep Ratios (FMAE)")

    ax_left.set_xticks(x_positions)
    ax_left.set_xticklabels(x_labels)
    ax_left.set_xlim(8.5, 0.55)

    ax_left.set_ylim(76, 94.5)
    ax_right.set_ylim(52, 63.8)

    ax_left.grid(True, axis="y", linestyle=":", linewidth=0.7, alpha=0.35)
    ax_left.set_axisbelow(True)

    ax_left.text(
        0.03,
        0.13,
        "Face visually unrecognizable below ~4% keep",
        transform=ax_left.transAxes,
        fontsize=10,
        color="#333333",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cfcfcf", "alpha": 0.95},
    )

    # Reference labels for the 10% keep step.
    ax_left.annotate(
        "10% keep reference",
        (7.45, fmae_id_acc_10pct),
        textcoords="offset points",
        xytext=(0, 8),
        ha="left",
        va="bottom",
        color="#c62828",
        fontsize=9,
    )
    ax_right.annotate(
        "10% keep reference",
        (7.45, fmae_au_f1_10pct),
        textcoords="offset points",
        xytext=(0, -14),
        ha="left",
        va="top",
        color="#1565c0",
        fontsize=9,
    )

    handles_left, labels_left = ax_left.get_legend_handles_labels()
    handles_right, labels_right = ax_right.get_legend_handles_labels()
    ax_left.legend(
        handles_left + handles_right,
        labels_left + labels_right,
        loc="upper right",
        frameon=False,
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.26)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
