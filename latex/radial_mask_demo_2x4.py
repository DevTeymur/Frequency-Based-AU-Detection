"""
Radial frequency masking demo (2x4 comparison output).

This is a copy of radial_mask_demo.py but the comparison figure is fixed
to a 2x4 layout (columns: Keep 100%, 10%, 5%, 1%). Top row shows the
original image (RGB) and bottom row shows the masked frequency
log-magnitude (centered) with a cyan radius overlay. Individual 2x2
figures are still saved per keep ratio as in the original demo.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "fmae"))
sys.path.insert(0, str(PROJECT_ROOT / "hfss" / "hfss"))

# Constants
IMG_SIZE = 224
DATA_ROOT = PROJECT_ROOT / "BP4D" / "BP4D_cropped"
JSON_FILE = PROJECT_ROOT / "BP4D" / "BP4D_test1.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "radial_mask_output"
OUTPUT_DIR.mkdir(exist_ok=True)

FIXED_RADIAL_STEPS = 10
FIXED_RADIAL_START_KEEP_PCT = 100.0
FIXED_RADIAL_DIRECTION = "big_to_small"  # big_to_small | small_to_big


def parse_args():
    parser = argparse.ArgumentParser(description="Radial frequency masking demo (2x4)")
    parser.add_argument(
        "--mode",
        type=str,
        default="low-pass",
        choices=["low-pass", "high-pass"],
        help="low-pass keeps center, high-pass keeps outer ring",
    )
    parser.add_argument(
        "--keep_pct",
        type=float,
        default=None,
        help="Single keep percentage for one 2x2 output (e.g. 10)",
    )
    parser.add_argument(
        "--image_index",
        type=int,
        default=0,
        help="Deterministic BP4D JSON line index to load so both modes use the same image",
    )
    parser.add_argument(
        "--radial_steps",
        type=int,
        default=FIXED_RADIAL_STEPS,
        help="Number of keep-ratio steps for radial mode",
    )
    parser.add_argument(
        "--radial_start_keep_pct",
        type=float,
        default=FIXED_RADIAL_START_KEEP_PCT,
        help="Starting keep percentage for radial mode",
    )
    parser.add_argument(
        "--radial_direction",
        type=str,
        default=FIXED_RADIAL_DIRECTION,
        choices=["big_to_small", "small_to_big"],
        help="Direction of keep progression",
    )
    return parser.parse_args()


def load_first_image(image_index=0):
    """Load a deterministic image from BP4D test JSON."""
    with open(JSON_FILE, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        raise RuntimeError(f"No samples found in {JSON_FILE}")

    sample = json.loads(lines[image_index % len(lines)])
    img_path = sample["img_path"]

    full_path = DATA_ROOT / img_path
    img = Image.open(full_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))

    img_array = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)

    print(f"Loaded: {img_path} (index={image_index % len(lines)})")
    return img_tensor


def radial_distance_map(size=IMG_SIZE):
    """Return radial distance map around FFT center."""
    h, w = size, size
    cy, cx = h // 2, w // 2
    ys, xs = np.ogrid[:h, :w]
    return np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)


def generate_radial_mask(keep_pct, mode="low-pass", dist=None):
    """Generate radial low-pass/high-pass mask from target keep percentage."""
    if mode not in ("low-pass", "high-pass"):
        raise ValueError(f"Invalid mode: {mode}. Use 'low-pass' or 'high-pass'.")

    keep_pct = float(np.clip(keep_pct, 0.1, 100.0))
    if dist is None:
        dist = radial_distance_map(IMG_SIZE)

    if mode == "low-pass":
        radius = float(np.quantile(dist, keep_pct / 100.0))
        mask = (dist <= radius).astype(np.float32)
    else:
        # Keep outer ring with approximately keep_pct retained area.
        radius = float(np.quantile(dist, 1.0 - keep_pct / 100.0))
        mask = (dist >= radius).astype(np.float32)

    return mask, radius


def generate_radial_mask_candidates(
    num_steps,
    mode="low-pass",
    start_keep_pct=100.0,
    direction="big_to_small",
):
    """Deterministic radial schedule aligned with hfss_search_au.py style."""
    if num_steps < 2:
        num_steps = 2
    if direction not in ("big_to_small", "small_to_big"):
        raise ValueError(
            f"Invalid radial direction: {direction}. Use 'big_to_small' or 'small_to_big'."
        )

    start_keep_pct = float(np.clip(start_keep_pct, 0.1, 100.0))
    end_keep_pct = max(start_keep_pct / float(num_steps), 0.1)
    target_keep_pcts = np.linspace(start_keep_pct, end_keep_pct, num_steps)
    if direction == "small_to_big":
        target_keep_pcts = target_keep_pcts[::-1]

    dist = radial_distance_map(IMG_SIZE)
    items = []
    for target in target_keep_pcts:
        mask, radius = generate_radial_mask(target, mode=mode, dist=dist)
        items.append(
            {
                "mask": mask,
                "radius": float(radius),
                "target_keep_pct": float(target),
                "keep_pct": float(np.mean(mask > 0.5) * 100.0),
            }
        )
    return items


def apply_frequency_mask(image_tensor, mask):
    """Apply frequency domain mask to image."""
    mask_t = torch.as_tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # FFT -> apply mask -> inverse FFT
    freq = torch.fft.fftshift(torch.fft.fft2(image_tensor.unsqueeze(0), dim=(-2, -1)), dim=(-2, -1))
    freq = freq * mask_t
    recon = torch.fft.ifft2(torch.fft.ifftshift(freq, dim=(-2, -1)), dim=(-2, -1)).real.float()

    return recon.squeeze(0)


def masked_spectrum_map(image_tensor, mask):
    """Return centered log-magnitude spectrum after applying mask."""
    mask_t = torch.as_tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    freq = torch.fft.fftshift(torch.fft.fft2(image_tensor.unsqueeze(0), dim=(-2, -1)), dim=(-2, -1))
    freq_masked = freq * mask_t
    mag = torch.log1p(torch.abs(freq_masked).mean(dim=1).squeeze(0))
    return mag.cpu().numpy()


def original_spectrum_map(image_tensor):
    """Return centered log-magnitude spectrum of original image."""
    freq = torch.fft.fftshift(torch.fft.fft2(image_tensor.unsqueeze(0), dim=(-2, -1)), dim=(-2, -1))
    mag = torch.log1p(torch.abs(freq).mean(dim=1).squeeze(0))
    return mag.cpu().numpy()


def tensor_to_image(tensor, normalize=False):
    """Convert tensor [3, H, W] to numpy image [H, W, 3]."""
    img = tensor.permute(1, 2, 0).numpy()
    if normalize:
        vmin = float(img.min())
        vmax = float(img.max())
        if vmax > vmin:
            img = (img - vmin) / (vmax - vmin)
        else:
            img = np.zeros_like(img)
    else:
        img = np.clip(img, 0, 1)
    return img


def save_original_vs_masked_2x2(
    original_img,
    original_spec,
    masked_img,
    masked_spec,
    keep_pct,
    radius,
    mode,
):
    """Save a 2x2 figure: original row, masked row."""
    fig, axes = plt.subplots(2, 2, figsize=(9, 9))

    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title("Original Image", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(original_spec, cmap="magma")
    axes[0, 1].set_title("Original Frequency Spectrum", fontsize=12, fontweight="bold")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(masked_img)
    axes[1, 0].set_title(
        f"{mode} Masked Image (Keep {keep_pct:.1f}% )",
        fontsize=12,
        fontweight="bold",
    )
    axes[1, 0].axis("off")

    axes[1, 1].imshow(masked_spec, cmap="magma")
    circle = plt.Circle((IMG_SIZE // 2, IMG_SIZE // 2), radius, color="cyan", fill=False, linewidth=1.5)
    axes[1, 1].add_patch(circle)
    axes[1, 1].set_title(
        f"{mode} Masked Spectrum (Keep {keep_pct:.1f}% )",
        fontsize=12,
        fontweight="bold",
    )
    axes[1, 1].axis("off")

    plt.tight_layout()
    mode_slug = mode.replace("-", "_")
    output_path = OUTPUT_DIR / f"radial_mask_2x2_{mode_slug}_keep_{int(round(keep_pct))}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def main():
    args = parse_args()

    # Load image
    img_tensor = load_first_image(args.image_index)
    original_img = tensor_to_image(img_tensor)
    original_spec = original_spectrum_map(img_tensor)

    # If user supplied single keep_pct, keep prior behavior
    if args.keep_pct is not None:
        dist = radial_distance_map(IMG_SIZE)
        mask, radius = generate_radial_mask(args.keep_pct, mode=args.mode, dist=dist)
        radial_items = [
            {
                "mask": mask,
                "radius": float(radius),
                "keep_pct": float(np.mean(mask > 0.5) * 100.0),
            }
        ]
    else:
        # For the 2x4 demo we fix the comparison keep values to these four columns.
        fixed_keeps = [100.0, 10.0, 5.0, 1.0]
        dist = radial_distance_map(IMG_SIZE)
        radial_items = []
        for k in fixed_keeps:
            mask, radius = generate_radial_mask(k, mode=args.mode, dist=dist)
            radial_items.append(
                {
                    "mask": mask,
                    "radius": float(radius),
                    "target_keep_pct": float(k),
                    "keep_pct": float(np.mean(mask > 0.5) * 100.0),
                }
            )

    masked_images = []
    spectra = []
    radii = []
    keep_pcts = []

    for item in radial_items:
        mask = item["mask"]
        radius = item["radius"]
        keep_pct = item["keep_pct"]
        target_keep_pct = item["target_keep_pct"]

        masked_img = apply_frequency_mask(img_tensor, mask)
        masked_img_np = tensor_to_image(masked_img, normalize=True)
        masked_spec = masked_spectrum_map(img_tensor, mask)

        masked_images.append(masked_img_np)
        spectra.append(masked_spec)
        radii.append(radius)
        keep_pcts.append(keep_pct)

        save_original_vs_masked_2x2(
            original_img,
            original_spec,
            masked_img_np,
            masked_spec,
            keep_pct,
            radius,
            args.mode,
        )

        print(
            f"Applied {args.mode} mask: target_keep={target_keep_pct:.1f}% "
            f"| actual_keep={keep_pct:.1f}% | radius={radius:.2f}"
        )

    if args.keep_pct is not None:
        return

    # New: fixed 2x4 comparison figure: top row originals, bottom row masked spectra.
    cols = 4
    # Make each cell square: width per column = 2 inches, height per row = 2 inches
    fig, axes = plt.subplots(2, cols, figsize=(cols * 2, 2 * 2))
    for c in range(cols):
        axes[0, c].imshow(original_img, interpolation="nearest")
        axes[0, c].set_title(f"Keep {radial_items[c]['target_keep_pct']:.0f}%", fontsize=10)
        axes[0, c].set_aspect("equal")
        axes[0, c].axis("off")

        axes[1, c].imshow(spectra[c], cmap="magma", interpolation="nearest")
        circle = plt.Circle((IMG_SIZE // 2, IMG_SIZE // 2), radii[c], color="cyan", fill=False, linewidth=1.0)
        axes[1, c].add_patch(circle)
        axes[1, c].set_aspect("equal")
        axes[1, c].axis("off")

    plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, wspace=0.0, hspace=0.0)
    output_path = OUTPUT_DIR / "radial_masks_2x4.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0)
    print(f"\nSaved: {output_path}")
    plt.close(fig)

    # Create 2x4 figure using explicit keep ratios and the simple layout the user supplied.
    keep_ratios = [100, 10, 5, 1]
    masked_images_2 = []
    spectra_2 = []
    radii_2 = []

    for keep_pct in keep_ratios:
        mask, radius = generate_radial_mask(keep_pct, mode=args.mode, dist=radial_distance_map(IMG_SIZE))
        masked_img = apply_frequency_mask(img_tensor, mask)
        masked_images_2.append(tensor_to_image(masked_img, normalize=True))
        spectra_2.append(masked_spectrum_map(img_tensor, mask))
        radii_2.append(radius)
        print(f"Applied mask: keep_pct={keep_pct}%")

    # Create 2x4 figure: top row images, bottom row masked spectra
    fig_h, axes_h = plt.subplots(2, 4, figsize=(4 * 2, 2 * 2))
    for idx, (keep_pct, img_disp, spec_disp, radius) in enumerate(zip(keep_ratios, masked_images_2, spectra_2, radii_2)):
        axes_h[0, idx].imshow(img_disp, interpolation='nearest')
        axes_h[0, idx].set_title(f"Keep: {keep_pct}%", fontsize=12, fontweight='bold')
        axes_h[0, idx].set_aspect('equal')
        axes_h[0, idx].axis('off')

        axes_h[1, idx].imshow(spec_disp, cmap='magma', interpolation='nearest')
        circle = plt.Circle((IMG_SIZE // 2, IMG_SIZE // 2), radius, color='cyan', fill=False, linewidth=1.5)
        axes_h[1, idx].add_patch(circle)
        axes_h[1, idx].set_title("Masked Spectrum", fontsize=10)
        axes_h[1, idx].set_aspect('equal')
        axes_h[1, idx].axis('off')

    plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, wspace=0.0, hspace=0.0)
    out_h = OUTPUT_DIR / "radial_masks_2x4_humans_top.png"
    plt.savefig(out_h, dpi=150, bbox_inches='tight', pad_inches=0)
    print(f"Saved human-top 2x4: {out_h}")
    plt.close(fig_h)

    # New output: take the bottom-row (masked images and masked spectra),
    # rotate each 90 degrees and lay them out side-by-side in a single row.
    # For each keep value we place [masked_image_rotated, masked_spec_rotated].
    pairs = []
    for img, spec in zip(masked_images, spectra):
        # Rotate 90 degrees clockwise for display consistency
        img_r = np.rot90(img, k=-1)
        # spec is 2D; normalize to 0-1 and stack to 3 channels for consistent display
        s = spec.copy()
        s_min, s_max = float(s.min()), float(s.max())
        if s_max > s_min:
            s_norm = (s - s_min) / (s_max - s_min)
        else:
            s_norm = np.zeros_like(s)
        spec_rgb = np.stack([s_norm, s_norm, s_norm], axis=-1)
        spec_r = np.rot90(spec_rgb, k=-1)
        pairs.append((img_r, spec_r))

    total_cols = len(pairs) * 2
    fig2, axes2 = plt.subplots(1, total_cols, figsize=(total_cols * 2, 2))
    if total_cols == 1:
        axes2 = np.array([axes2])

    col = 0
    for keep, (img_r, spec_r) in zip(keep_pcts, pairs):
        axes2[col].imshow(img_r, interpolation="nearest")
        axes2[col].set_title(f"Img {int(round(keep))}%", fontsize=8)
        axes2[col].axis("off")
        col += 1

        axes2[col].imshow(spec_r, cmap=None, interpolation="nearest")
        axes2[col].set_title(f"Spec {int(round(keep))}%", fontsize=8)
        axes2[col].axis("off")
        col += 1

    plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, wspace=0.0, hspace=0.0)
    out2 = OUTPUT_DIR / "radial_masks_side_by_side.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight", pad_inches=0)
    print(f"Saved rotated side-by-side: {out2}")
    plt.close(fig2)


if __name__ == "__main__":
    main()
