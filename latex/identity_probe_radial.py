from pathlib import Path

import matplotlib.pyplot as plt


def main():
    stages = ["Stage 1", "Stage 2", "Stage 3", "Stage 4"]
    best_f1 = [63.53, 63.28, 61.26, 45.07]
    baseline = 63.51

    output_path = Path(__file__).resolve().parent / "hfss_macro_f1_stages.png"

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "font.family": "DejaVu Sans",
    })

    fig, ax = plt.subplots(figsize=(8, 5))

    x = range(len(stages))
    ax.plot(
        x,
        best_f1,
        color="#1f4e79",
        marker="o",
        markersize=7,
        linewidth=2.2,
        markerfacecolor="#2f6fa5",
        markeredgecolor="white",
        markeredgewidth=1.0,
    )

    ax.axhline(
        baseline,
        color="#7a7a7a",
        linestyle="--",
        linewidth=1.6,
        label=f"Baseline ({baseline:.2f}%)",
    )

    for xi, yi in zip(x, best_f1):
        ax.annotate(
            f"{yi:.2f}%",
            (xi, yi),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            va="bottom",
            fontsize=10,
            color="#222222",
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(stages)
    ax.set_xlabel("Search Stage")
    ax.set_ylabel("Best Macro F1 (%)")
    ax.set_title("Macro F1 at Best Mask Across HFSS Stages")
    ax.set_ylim(40, 66)

    ax.grid(True, axis="y", linestyle=":", linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()