"""
HFSS - Frequency Filtering Test
Test frequency domain masking on facial images
"""

import torch
import torch.fft as fft
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
IMG_SIZE = 224
DATA_ROOT = Path("BP4D/BP4D_cropped")
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

print("🔍 HFSS - Frequency Filtering Test")
print("="*70)

# ============================================================================
# FREQUENCY FILTERING FUNCTIONS
# ============================================================================

def load_image(img_path):
    """Load and resize image"""
    img = Image.open(img_path).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    # Convert to tensor [3, H, W] and normalize to [0, 1]
    img_array = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
    return img_tensor

def apply_frequency_mask(image_tensor, mask):
    """
    Apply frequency domain mask to image
    
    Args:
        image_tensor: [3, H, W] image tensor in range [0, 1]
        mask: [H, W] binary mask (1=keep, 0=remove)
    
    Returns:
        Filtered image tensor [3, H, W]
    """
    filtered_channels = []
    
    for c in range(3):  # Process each RGB channel
        # Convert to frequency domain
        freq = fft.fft2(image_tensor[c])
        freq_shifted = fft.fftshift(freq)
        
        # Apply mask
        freq_masked = freq_shifted * mask
        
        # Convert back to spatial domain
        freq_unshifted = fft.ifftshift(freq_masked)
        spatial = fft.ifft2(freq_unshifted)
        
        # Take real part and clamp to [0, 1]
        spatial_real = torch.real(spatial)
        spatial_clamped = torch.clamp(spatial_real, 0, 1)
        
        filtered_channels.append(spatial_clamped)
    
    return torch.stack(filtered_channels)

def create_low_freq_mask(size, cutoff=0.3):
    """
    Create circular mask that keeps only low frequencies (center)
    
    Args:
        size: Image size
        cutoff: Fraction of frequencies to keep (0.3 = keep center 30%)
    
    Returns:
        Binary mask [size, size]
    """
    mask = torch.zeros((size, size))
    center = size // 2
    radius = int(size * cutoff / 2)
    
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
    dist = torch.sqrt((x - center)**2 + (y - center)**2)
    mask[dist <= radius] = 1
    
    return mask

def create_high_freq_mask(size, cutoff=0.3):
    """Create mask that keeps only high frequencies (edges)"""
    return 1 - create_low_freq_mask(size, cutoff)

def create_band_mask(size, inner_cutoff=0.2, outer_cutoff=0.5):
    """Create mask that keeps mid-frequency band"""
    center = size // 2
    inner_radius = int(size * inner_cutoff / 2)
    outer_radius = int(size * outer_cutoff / 2)
    
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
    dist = torch.sqrt((x - center)**2 + (y - center)**2)
    
    mask = torch.zeros((size, size))
    mask[(dist > inner_radius) & (dist <= outer_radius)] = 1
    
    return mask

def create_random_mask(size, proportion=0.5, seed=None):
    """
    Create random frequency mask (HFSS method)
    
    Args:
        size: Image size
        proportion: Proportion of frequencies to keep (0.2 = keep 20% random freqs)
        seed: Random seed for reproducibility
    
    Returns:
        Random binary mask [size, size]
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    mask = torch.zeros((size, size))
    
    # Only select from upper half + DC (due to FFT symmetry)
    # This matches the HFSS paper approach
    half_size = size // 2 + 1
    total_freqs = half_size * size
    num_keep = int(total_freqs * proportion)
    
    # Randomly select frequencies to keep
    indices = torch.randperm(total_freqs)[:num_keep]
    
    # Create mask for upper half
    temp_mask = torch.zeros(half_size * size)
    temp_mask[indices] = 1
    temp_mask = temp_mask.reshape(half_size, size)
    
    # Fill upper half
    mask[:half_size, :] = temp_mask
    
    # Mirror to lower half (maintain FFT symmetry)
    for h_index in range(1, half_size):
        for w_index in range(size):
            h_mirror = size - h_index
            w_mirror = (size - w_index) % size
            mask[h_mirror, w_mirror] = mask[h_index, w_index]
    
    return mask

def create_random_masks_batch(size, proportion, num_masks=10):
    """
    Create multiple random masks for averaging (HFSS robustness test)
    
    Args:
        size: Image size
        proportion: Proportion of frequencies to keep
        num_masks: Number of random masks to generate
    
    Returns:
        List of random masks
    """
    masks = []
    for i in range(num_masks):
        mask = create_random_mask(size, proportion, seed=i)
        masks.append(mask)
    return masks

# ============================================================================
# VISUALIZATION
# ============================================================================

def tensor_to_image(tensor):
    """Convert tensor [3, H, W] to numpy array [H, W, 3] for display"""
    img = tensor.permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    return img

def visualize_frequency_spectrum(image_tensor):
    """Visualize the frequency spectrum of an image"""
    # Take first channel for visualization
    channel = image_tensor[0]
    
    # Compute FFT
    freq = fft.fft2(channel)
    freq_shifted = fft.fftshift(freq)
    
    # Compute magnitude spectrum (log scale for better visualization)
    magnitude = torch.abs(freq_shifted)
    magnitude_log = torch.log(magnitude + 1)
    
    return magnitude_log.numpy()

def test_frequency_filtering(img_path, output_name="test"):
    """
    Test frequency filtering on a single image
    
    Args:
        img_path: Path to test image
        output_name: Name for output files
    """
    print(f"\n📸 Processing: {img_path}")
    
    # Load image
    img = load_image(img_path)
    print(f"   Image shape: {img.shape}")
    
    # Create different masks
    masks = {
        'Original': None,
        'Low Freq (30%)': create_low_freq_mask(IMG_SIZE, cutoff=0.3),
        'Mid Freq Band': create_band_mask(IMG_SIZE, inner_cutoff=0.2, outer_cutoff=0.5),
        'High Freq': create_high_freq_mask(IMG_SIZE, cutoff=0.3),
        'Random (50%)': create_random_mask(IMG_SIZE, proportion=0.5)
    }
    
    # Apply filters
    results = {}
    for name, mask in masks.items():
        if mask is None:
            results[name] = img
        else:
            filtered = apply_frequency_mask(img, mask)
            results[name] = filtered
            print(f"   ✓ Applied {name}")
    
    # Visualize results
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    
    # Row 1: Filtered images
    for idx, (name, filtered_img) in enumerate(results.items()):
        ax = axes[0, idx]
        ax.imshow(tensor_to_image(filtered_img))
        ax.set_title(name)
        ax.axis('off')
    
    # Row 2: Frequency spectrums
    for idx, (name, filtered_img) in enumerate(results.items()):
        ax = axes[1, idx]
        spectrum = visualize_frequency_spectrum(filtered_img)
        ax.imshow(spectrum, cmap='viridis')
        ax.set_title(f"{name}\nFrequency Spectrum")
        ax.axis('off')
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / f'{output_name}_frequency_analysis.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved visualization to {output_file}")
    plt.close()
    
    # Also visualize the masks
    visualize_masks(masks, output_name)

def visualize_masks(masks, output_name="masks"):
    """Visualize the frequency masks themselves"""
    fig, axes = plt.subplots(1, len(masks)-1, figsize=(15, 3))
    
    idx = 0
    for name, mask in masks.items():
        if mask is None:  # Skip original
            continue
        
        ax = axes[idx]
        ax.imshow(mask.numpy(), cmap='gray')
        ax.set_title(f"{name}\nMask")
        ax.axis('off')
        idx += 1
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / f'{output_name}_masks.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved masks to {output_file}")
    plt.close()

# ============================================================================
# MAIN TEST
# ============================================================================

def main():
    # Find a test image
    test_images = list(DATA_ROOT.glob("*/T1/*.jpg"))[:3]
    
    if not test_images:
        print("❌ No images found in BP4D/BP4D_cropped/")
        print(f"   Looking in: {DATA_ROOT.absolute()}")
        return
    
    print(f"\n✓ Found {len(test_images)} test images")
    
    # Test on first few images
    for i, img_path in enumerate(test_images):
        test_frequency_filtering(img_path, output_name=f"sample_{i+1}")
    
    print("\n" + "="*70)
    print("✅ HFSS Test Complete!")
    print(f"📁 Results saved to: {OUTPUT_DIR.absolute()}")
    print("="*70)

if __name__ == "__main__":
    main()
