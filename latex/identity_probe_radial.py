import matplotlib.pyplot as plt
import numpy as np

def main():
    # Data
    keep = np.array([100,90,80,70,60,50,40,30,20,10])

    id_fmae = np.array([93.25,93.25,93.17,93.25,93.09,93.17,93.17,92.76,92.76,92.52])
    f1_fmae = np.array([63.51,63.50,63.50,63.41,63.42,63.45,63.41,63.49,63.58,62.69])

    id_iat = np.array([69.02,69.19,69.27,69.27,69.27,68.21,69.19,69.43,69.19,67.56])
    f1_iat = np.array([65.88,65.89,65.89,65.87,65.86,65.80,65.79,65.84,65.87,65.54])

    fig, ax = plt.subplots(figsize=(10,5), dpi=150)

    ax2 = ax.twinx()

    # Left y-axis: Identity Accuracy
    ax.set_ylim(60,96)
    ax.set_ylabel('Identity Accuracy (%)')

    # Right y-axis: AU Macro F1
    ax2.set_ylim(60,70)
    ax2.set_ylabel('AU Macro F1 (%)')

    # X-axis
    ax.set_xlabel('Frequency Keep Ratio (%)')
    ax.set_xlim(100,0)
    ax.set_xticks([100,80,60,40,20,10])

    # Plot identity accuracy lines on left axis
    ax.plot(keep, id_fmae, color='tab:blue', marker='o', linestyle='-', label='FMAE – Identity Acc')
    ax.plot(keep, id_iat, color='tab:orange', marker='s', linestyle='-', label='FMAE-IAT – Identity Acc')

    # Plot AU F1 lines on right axis (dashed)
    ax2.plot(keep, f1_fmae, color='tab:blue', marker='o', linestyle='--', label='FMAE – AU F1')
    ax2.plot(keep, f1_iat, color='tab:orange', marker='s', linestyle='--', label='FMAE-IAT – AU F1')

    # Horizontal dotted grey line at y=2.4 on left axis (chance level)
    ax.axhline(2.4, color='gray', linestyle=':', linewidth=1)
    # place label near left side
    ax.text(100, 2.4, 'Chance (2.4%)', color='gray', va='bottom', ha='left')

    # Vertical dashed grey line at x=10
    ax.axvline(10, color='gray', linestyle='--', linewidth=1)
    ax.text(10, 96, '10% keep', color='gray', rotation=90, va='bottom', ha='right')

    # Title
    plt.title('Identity Probe Accuracy and AU F1 vs Frequency Keep Ratio')

    # Combined legend from both axes, placed inside the plot (upper-right)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax.legend(lines + lines2, labels + labels2,
                       loc='upper right', fontsize=8, framealpha=0.9, ncol=4)

    out_path = 'hfss/figures/identity_probe_radial.png'
    fig.savefig(out_path)
    print(f'Saved figure to {out_path}')

if __name__ == '__main__':
    main()
