"""
Minimal script: Load one image and apply radial masks at 100%, 50%, 20%, 10% keep ratios.
Saves a 4-panel comparison figure.
"""

import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
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


def load_first_image():
    """Load a random image from BP4D test JSON."""
    import random
    with open(JSON_FILE, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    random_line = random.choice(lines)
    sample = json.loads(random_line)
    img_path = sample['img_path']
    
    full_path = DATA_ROOT / img_path
    img = Image.open(full_path).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    
    img_array = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
    
    print(f"Loaded: {img_path}")
    return img_tensor


def generate_radial_mask(keep_pct):
    """Generate a radial mask that keeps the center keep_pct% of frequencies."""
    h, w = IMG_SIZE, IMG_SIZE
    cy, cx = h // 2, w // 2
    ys, xs = np.ogrid[:h, :w]
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    
    # Use empirical quantile
    radius = float(np.quantile(dist, keep_pct / 100.0))
    mask = (dist <= radius).astype(np.float32)
    
    return mask, radius


def apply_frequency_mask(image_tensor, mask):
    """Apply frequency domain mask to image."""
    mask_t = torch.as_tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # FFT -> apply mask -> inverse FFT
    freq = torch.fft.fftshift(torch.fft.fft2(image_tensor.unsqueeze(0), dim=(-2, -1)), dim=(-2, -1))
    freq = freq * mask_t
    recon = torch.fft.ifft2(torch.fft.ifftshift(freq, dim=(-2, -1)), dim=(-2, -1)).real.float()
    
    return recon.squeeze(0)


def masked_spectrum_map(image_tensor, mask):
    """Return centered log-magnitude spectrum after applying the mask."""
    mask_t = torch.as_tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    freq = torch.fft.fftshift(torch.fft.fft2(image_tensor.unsqueeze(0), dim=(-2, -1)), dim=(-2, -1))
    freq_masked = freq * mask_t
    mag = torch.log1p(torch.abs(freq_masked).mean(dim=1).squeeze(0))
    return mag.cpu().numpy()


def tensor_to_image(tensor):
    """Convert tensor [3, H, W] to numpy image [H, W, 3]."""
    img = tensor.permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    return img


def main():
    # Load image
    img_tensor = load_first_image()
    
    # Generate masks and apply
    keep_ratios = [100, 50, 20, 10]
    masked_images = []
    spectra = []
    radii = []
    
    for keep_pct in keep_ratios:
        mask, radius = generate_radial_mask(keep_pct)
        masked_img = apply_frequency_mask(img_tensor, mask)
        masked_images.append(tensor_to_image(masked_img))
        spectra.append(masked_spectrum_map(img_tensor, mask))
        radii.append(radius)
        print(f"Applied mask: keep_pct={keep_pct}%")
    
    # Create 2x4 figure: top row images, bottom row masked spectra
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for idx, (keep_pct, img, spec, radius) in enumerate(zip(keep_ratios, masked_images, spectra, radii)):
        axes[0, idx].imshow(img)
        axes[0, idx].set_title(f"Keep: {keep_pct}%", fontsize=12, fontweight='bold')
        axes[0, idx].axis('off')

        axes[1, idx].imshow(spec, cmap='magma')
        circle = plt.Circle((IMG_SIZE // 2, IMG_SIZE // 2), radius, color='cyan', fill=False, linewidth=1.5)
        axes[1, idx].add_patch(circle)
        axes[1, idx].set_title("Masked Spectrum", fontsize=10)
        axes[1, idx].axis('off')
    
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / "radial_masks_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
