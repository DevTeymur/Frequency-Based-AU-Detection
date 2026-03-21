"""
HFSS Search for AU Detection
Finds frequency shortcuts in AU detection models
"""

import sys
import json
import pickle
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

from sklearn.metrics import f1_score

import torch
from torch.utils.data import DataLoader, Subset
from types import SimpleNamespace
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

# Add paths
PROJECT_ROOT = Path(__file__).resolve().parent
FMAE_PATH = PROJECT_ROOT / "fmae"
HFSS_PATH = PROJECT_ROOT / "hfss" / "hfss"
sys.path.insert(0, str(FMAE_PATH))
sys.path.insert(0, str(HFSS_PATH))

# Import from FMAE
import models_vit
from util.datasets import BP4D_AU_dataset

# Import from HFSS (in hfss/hfss/ subdirectory)
sys.path.insert(0, str(PROJECT_ROOT / "hfss" / "hfss"))
from transforms_search_space import White_Mask, gen_freqs_list, sample_frequency, generate_mask


# Config
AU_LABELS = [1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24]
IMG_SIZE = 224
PATCHES = {'stage1': 4, 'stage2': 8, 'stage3': 16, 'stage4': 28, 'stage5': 56, 'stage6': 112}


def load_model(model_path, model_type='FMAE', device='cuda'):
    """Load AU detection model"""
    grad_reverse = 1.0 if model_type == 'IAT' else 0.0
    num_subjects = 41 if model_type == 'IAT' else 0
    
    model = models_vit.vit_large_patch16(
        num_classes=12,
        num_subjects=num_subjects,
        drop_path_rate=0.0,
        global_pool=True,
        grad_reverse=grad_reverse,
    )
    
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint.get('model', checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()
    
    print(f"✓ Loaded {model_type} model")
    return model


def generate_mask_candidates(num_candidates, proportion, stage, prev_masks=None):
    """Generate random frequency mask candidates.

    Paper-aligned behavior:
    - Stage1: sample on full grid.
    - Stage>1: sample finer masks and multiply with a parent mask selected
      from previous stage top-N (cumulative masking; black stays black).
    - Also includes shifted-window variants (patch, patch+1, mixed), similar
      to original HFSS candidate generation.
    """
    candidates = []
    patch = PATCHES[stage]
    parent_masks = prev_masks if prev_masks else None

    def _make_symmetric(mask_arr):
        max_n_h = IMG_SIZE // 2
        max_n_w = IMG_SIZE // 2
        for h_index in range(-max_n_h, 1):
            for w_index in range(-max_n_w, max_n_w):
                h_matrix_index = IMG_SIZE // 2 + h_index
                w_matrix_index = IMG_SIZE // 2 + w_index
                if h_index != 0:
                    mask_arr[IMG_SIZE - h_matrix_index - 1, IMG_SIZE - w_matrix_index - 1] = mask_arr[h_matrix_index, w_matrix_index]
        return mask_arr

    while len(candidates) < num_candidates:
        parent_mask = None
        if parent_masks:
            parent_mask = parent_masks[np.random.randint(len(parent_masks))]

        # Variant A: patch grid
        freqs = gen_freqs_list(patch, patch)
        sample_frqs = sample_frequency(proportion, freqs)
        mask_int = generate_mask(sample_frqs, patch, patch)
        mask_a = np.kron(mask_int, np.ones((IMG_SIZE // patch, IMG_SIZE // patch)))

        if parent_mask is not None:
            mask_a = parent_mask * mask_a
        mask_a = _make_symmetric(mask_a)
        candidates.append(White_Mask(mask_a))
        if len(candidates) >= num_candidates:
            break

        # Variant B: shifted window (patch+1), if possible
        if patch < IMG_SIZE:
            patch_b = patch + 1
            freqs_b = gen_freqs_list(patch_b, patch_b)
            sample_frqs_b = sample_frequency(proportion, freqs_b)
            mask_int_b = generate_mask(sample_frqs_b, patch_b, patch_b)
            step = patch_b - 1
            mask_b = np.kron(mask_int_b, np.ones((IMG_SIZE // step, IMG_SIZE // step)))
            crop = int(IMG_SIZE // step / 2)
            if crop > 0:
                mask_b = mask_b[crop:-crop, crop:-crop]

            if parent_mask is not None:
                mask_b = parent_mask * mask_b
            mask_b = _make_symmetric(mask_b)
            candidates.append(White_Mask(mask_b))
            if len(candidates) >= num_candidates:
                break

            # Variant C: mixed patch and shifted patch
            freqs_c1 = gen_freqs_list(patch, patch)
            sample_c1 = sample_frequency(proportion / 2.0, freqs_c1)
            mask_c1 = generate_mask(sample_c1, patch, patch)
            mask_c1 = np.kron(mask_c1, np.ones((IMG_SIZE // patch, IMG_SIZE // patch)))

            freqs_c2 = gen_freqs_list(patch_b, patch_b)
            sample_c2 = sample_frequency(proportion / 2.0, freqs_c2)
            mask_c2 = generate_mask(sample_c2, patch_b, patch_b)
            mask_c2 = np.kron(mask_c2, np.ones((IMG_SIZE // step, IMG_SIZE // step)))
            if crop > 0:
                mask_c2 = mask_c2[crop:-crop, crop:-crop]

            mask_c = np.clip(mask_c1 + mask_c2, 0, 1)
            if parent_mask is not None:
                mask_c = parent_mask * mask_c
            mask_c = _make_symmetric(mask_c)
            candidates.append(White_Mask(mask_c))

    return candidates[:num_candidates]


@torch.no_grad()
def evaluate_mask(model, dataloader, mask_transform, device):
    """Evaluate AU performance with frequency mask applied (mean F1 over AUs).

    This accumulates predictions across all batches then computes the macro
    average F1 across AU labels (equivalent to mean per-AU F1).
    """
    all_preds = []
    all_labels = []

    for images, (au_labels, _) in dataloader:
        # Apply frequency mask
        images = torch.stack([mask_transform(img) for img in images])
        images = images.to(device)
        au_labels = au_labels.to(device)

        # Forward pass
        outputs = model(images)
        if isinstance(outputs, tuple):
            outputs = outputs[0]  # Take AU head only

        # Binary predictions
        preds = (torch.sigmoid(outputs) >= 0.5).float()

        all_preds.append(preds.cpu().numpy())
        all_labels.append(au_labels.cpu().numpy())

    if not all_preds:
        return 0.0

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Compute macro (per-AU mean) F1 using sklearn on 2D arrays
    f1 = f1_score(all_labels, all_preds, average='macro')
    return float(f1)
    # return accuracy


# ============================================================
# OPTIONAL FUNCTION 1: Per-AU F1 evaluation
# Enable with --per_au_eval flag
# ============================================================
@torch.no_grad()
def evaluate_mask_per_au(model, dataloader, mask_transform, baseline_per_au, device):
    """Evaluate per-AU F1 and show drop vs baseline for each AU.
    Returns dict: {AU_label: {'f1': float, 'drop': float}}
    """
    all_preds = []
    all_labels = []

    for images, (au_labels, _) in dataloader:
        images = torch.stack([mask_transform(img) for img in images])
        images = images.to(device)
        au_labels = au_labels.to(device)
        outputs = model(images)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        all_preds.append(preds.cpu().numpy())
        all_labels.append(au_labels.cpu().numpy())

    if not all_preds:
        return {}

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    per_au = {}
    for i, au in enumerate(AU_LABELS):
        au_f1 = f1_score(all_labels[:, i], all_preds[:, i], average='binary', zero_division=0)
        baseline_entry = baseline_per_au.get(au, 0.0) if baseline_per_au else 0.0
        baseline = baseline_entry.get('f1', 0.0) if isinstance(baseline_entry, dict) else baseline_entry
        drop = baseline - au_f1
        per_au[au] = {'f1': au_f1, 'drop': drop}
    return per_au


def print_per_au_results(per_au, stage, mask_id):
    """Pretty-print per-AU F1 drops for a given mask."""
    print(f"   Per-AU breakdown — {stage} mask{mask_id}:")
    for au, vals in per_au.items():
        bar = '▓' * int(abs(vals['drop']) * 100 / 5)  # 1 block per 5% drop
        print(f"     AU{au:02d}: F1={vals['f1']*100:.1f}%  drop={vals['drop']*100:+.1f}%  {bar}")


def save_per_au_grouped_bar(baseline_per_au, masked_per_au, stage, model_type, save_dir):
    """Save grouped bar chart: baseline vs masked per-AU F1 for one stage."""
    aus = AU_LABELS
    baseline_vals = []
    masked_vals = []

    for au in aus:
        b = baseline_per_au.get(au, 0.0) if baseline_per_au else 0.0
        b = b.get('f1', 0.0) if isinstance(b, dict) else b
        m = masked_per_au.get(au, {}) if masked_per_au else {}
        m = m.get('f1', 0.0) if isinstance(m, dict) else m
        baseline_vals.append(float(b))
        masked_vals.append(float(m))

    x = np.arange(len(aus))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, baseline_vals, width=width, label='Baseline F1', color='#4C78A8')
    ax.bar(x + width / 2, masked_vals, width=width, label='Masked F1', color='#F58518')

    ax.set_title(f"Per-AU F1 | {model_type} | {stage} (best mask)")
    ax.set_xlabel('Action Unit (AU)')
    ax.set_ylabel('F1 score')
    ax.set_xticks(x)
    ax.set_xticklabels([f"AU{au}" for au in aus], rotation=0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis='y', alpha=0.2)
    ax.legend(loc='best')

    plt.tight_layout()
    save_path = save_dir / f"{model_type}_{stage}_per_au_grouped_bar.png"
    plt.savefig(save_path, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"   📊 Saved per-AU grouped bar to {save_path}")


def save_per_au_stage_heatmap(per_au_drop_by_stage, model_type, save_dir):
    """Save AU x stage heatmap where value is per-AU F1 drop (baseline - masked)."""
    if not per_au_drop_by_stage:
        return

    stages = list(per_au_drop_by_stage.keys())
    aus = AU_LABELS

    heat = np.array([
        [float(per_au_drop_by_stage.get(stage, {}).get(au, 0.0)) for stage in stages]
        for au in aus
    ])

    vmax = np.max(np.abs(heat)) if heat.size > 0 else 1.0
    vmax = max(vmax, 1e-6)

    fig, ax = plt.subplots(figsize=(1.8 * len(stages) + 4, 6))
    im = ax.imshow(heat, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')

    ax.set_title(f"Per-AU F1 Drop Heatmap | {model_type}")
    ax.set_xlabel('Stage')
    ax.set_ylabel('Action Unit (AU)')
    ax.set_xticks(np.arange(len(stages)))
    ax.set_xticklabels(stages)
    ax.set_yticks(np.arange(len(aus)))
    ax.set_yticklabels([f"AU{au}" for au in aus])

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('F1 drop (baseline - masked)')

    plt.tight_layout()
    save_path = save_dir / f"{model_type}_per_au_drop_heatmap.png"
    plt.savefig(save_path, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"   🌡️ Saved per-AU heatmap to {save_path}")


# ============================================================
# OPTIONAL FUNCTION 2: Advanced frequency mask visualizations
# Enable with --visualize_masks flag
# Shows: raw mask, ring overlay, frequency band, masked FFT
# ============================================================
def visualize_mask_advanced(mask_array, save_path, stage, mask_id, sample_image=None):
    """Visualize a frequency mask in 4 views:
    1. Raw mask (white=keep, black=remove)
    2. Ring/radial distance overlay
    3. Frequency band annotation (low/mid/high)
    4. Masked FFT magnitude (if sample_image provided)
    """
    size = mask_array.shape[0]
    center = size // 2

    # --- Build radial distance map ---
    ys, xs = np.ogrid[:size, :size]
    radial = np.sqrt((xs - center) ** 2 + (ys - center) ** 2)
    max_r = np.sqrt(2) * center

    # --- Frequency band labels (low < 20%, mid 20-50%, high > 50%) ---
    band = np.zeros((size, size), dtype=int)  # 0=low, 1=mid, 2=high
    band[radial > max_r * 0.5] = 2
    band[(radial >= max_r * 0.2) & (radial <= max_r * 0.5)] = 1

    fig, axes = plt.subplots(1, 4 if sample_image is not None else 3, figsize=(16, 4))
    fig.suptitle(f"{stage} | mask{mask_id}", fontsize=12)

    # Panel 1: Raw mask
    axes[0].imshow(mask_array, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Mask\n(white=keep)')
    axes[0].axis('off')

    # Panel 2: Ring overlay
    ring_vis = plt.cm.RdYlGn(mask_array)  # green=keep, red=remove
    ring_contour = axes[1].imshow(ring_vis)
    for r_frac in [0.2, 0.5]:
        circle = plt.Circle((center, center), max_r * r_frac,
                             color='white', fill=False, linewidth=1.5, linestyle='--')
        axes[1].add_patch(circle)
    axes[1].set_title('Ring overlay\n(dashed=band boundaries)')
    axes[1].axis('off')

    # Panel 3: Frequency band annotation
    cmap_band = plt.cm.get_cmap('coolwarm', 3)
    masked_band = np.ma.masked_where(mask_array == 0, band.astype(float))
    axes[2].imshow(band, cmap='coolwarm', alpha=0.3, vmin=0, vmax=2)
    axes[2].imshow(masked_band, cmap='coolwarm', vmin=0, vmax=2)
    for label, y_frac in [('LOW', 0.85), ('MID', 0.55), ('HIGH', 0.1)]:
        axes[2].text(size * 0.02, size * y_frac, label, color='white',
                     fontsize=9, fontweight='bold')
    axes[2].set_title('Freq bands\n(low/mid/high)')
    axes[2].axis('off')

    # Panel 4: Masked FFT of sample image (optional)
    if sample_image is not None:
        import torch.fft as fft_module
        img_t = torch.tensor(sample_image, dtype=torch.float32)
        if img_t.ndim == 3:
            img_t = img_t.mean(0)  # grayscale for FFT vis
        freq = torch.fft.fftshift(torch.fft.fft2(img_t))
        freq_masked = freq * torch.tensor(mask_array, dtype=torch.complex64)
        magnitude = torch.log(torch.abs(freq_masked) + 1).numpy()
        axes[3].imshow(magnitude, cmap='inferno')
        axes[3].set_title('Masked FFT\n(log magnitude)')
        axes[3].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()


def search_stage(model, dataloader, stage, num_candidates, proportion, prev_masks, device, top_n=10):
    """Search for frequency shortcuts in one stage"""
    print(f"\n🔍 Stage {stage}: Testing {num_candidates} candidates (P={proportion})")
    
    # Generate candidates
    candidates = generate_mask_candidates(num_candidates, proportion, stage, prev_masks)
    
    # Test baseline (no mask)
    baseline_f1 = evaluate_mask(model, dataloader, lambda x: x, device)
    print(f"   Baseline mean F1: {baseline_f1*100:.2f}% (mean per-AU)")
    
    # Test each candidate (compute mean F1 per candidate)
    results = []
    for i, mask_transform in enumerate(tqdm(candidates, desc=f"   Testing masks")):
        f1 = evaluate_mask(model, dataloader, mask_transform, device)
        drop = baseline_f1 - f1
        results.append((i, f1, drop, mask_transform))
    
    # Sort all candidates by masked F1 (largest first) to align with HFSS paper
    results.sort(key=lambda x: x[1], reverse=True)
    top_k = results[:top_n]

    # Print ALL mask drops for this stage (one line each)
    print(f"\n   {stage} — all masks (sorted by masked F1, high→low):")
    for r in results:
        mask_id, f1_val, drop_val, _ = r
        marker = " ◀ best" if mask_id == results[0][0] else ""
        print(f"     mask{mask_id:>3}: F1={f1_val*100:.2f}%  |  drop={drop_val*100:+.2f}%{marker}")

    # Return top masks for next stage refinement
    return [r[3] for r in top_k], results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='FMAE', choices=['FMAE', 'IAT'])
    parser.add_argument('--model_path', default='models/FMAE_BP4D_fold1.pth')
    parser.add_argument('--test_json', default='BP4D/BP4D_test1.json')
    parser.add_argument('--data_root', default='BP4D/BP4D_cropped/')
    parser.add_argument('--num_candidates', type=int, default=200)
    parser.add_argument('--top_n', type=int, default=10,
                        help='Number of top masks to keep and propagate to next stage')
    parser.add_argument('--proportion', type=float, default=0.8)
    parser.add_argument('--num_samples', type=int, default=500, help='Number of training samples to use')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output_dir', default='hfss/DFM')
    parser.add_argument('--stages', nargs='+', default=['stage1', 'stage2', 'stage3'])
    # --- Optional feature flags (disable by omitting the flag) ---
    parser.add_argument('--per_au_eval', action='store_true',
                        help='After search, evaluate per-AU F1 drops for the best mask per stage')
    parser.add_argument('--visualize_masks', action='store_true',
                        help='Save advanced mask visualizations (ring, freq band, masked FFT) for top masks')
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir.parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print(f"HFSS Search for AU Detection - {args.model_type}")
    print(f"Model: {args.model_path} | Device: {args.device}")
    print("="*70)
    
    # Load model
    model = load_model(args.model_path, args.model_type, args.device)
    
    # Load TEST data (not training)
    dataset_args = SimpleNamespace(
        root_path=args.data_root,
        input_size=IMG_SIZE,
        color_jitter=None,
        aa='rand-m9-mstd0.5-inc1',
        reprob=0.25,
        remode='pixel',
        recount=1
    )
    dataset = BP4D_AU_dataset(args.test_json, is_train=False, args=dataset_args)
    
    # Sample subset
    if args.num_samples and args.num_samples < len(dataset):
        indices = np.random.choice(len(dataset), args.num_samples, replace=False)
        dataset = Subset(dataset, indices)
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"✓ Loaded {len(dataset)} TEST samples (evaluating on unseen data)")
    
    # Run hierarchical search
    prev_masks = None
    all_results = {}
    per_au_drop_by_stage = {}
    
    for stage in args.stages:
        top_masks, stage_results = search_stage(
            model, dataloader, stage, 
            args.num_candidates, args.proportion, 
            prev_masks, args.device, args.top_n
        )
        
        # Save stage results
        output_file = output_dir / f"{args.model_type}_{stage}_DFMs.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(top_masks, f)
        print(f"   ✓ Saved DFMs to {output_file}")

        best_mask = top_masks[0]
        best_mask_array = best_mask.mask if hasattr(best_mask, 'mask') else None

        # --- OPTIONAL: per-AU F1 eval for the best mask ---
        if args.per_au_eval and best_mask_array is not None:
            print(f"\n   🔬 Per-AU eval for best mask ({stage})...")
            # (Re-use evaluate_mask_per_au with identity transform for baseline)
            baseline_per_au_full = evaluate_mask_per_au(
                model, dataloader, lambda x: x, {}, args.device)
            best_per_au = evaluate_mask_per_au(
                model, dataloader, best_mask, baseline_per_au_full, args.device)
            print_per_au_results(best_per_au, stage, 'best')

            save_per_au_grouped_bar(
                baseline_per_au=baseline_per_au_full,
                masked_per_au=best_per_au,
                stage=stage,
                model_type=args.model_type,
                save_dir=figures_dir,
            )
            per_au_drop_by_stage[stage] = {
                au: vals.get('drop', 0.0) for au, vals in best_per_au.items()
            }

        # --- OPTIONAL: advanced mask visualization for the best mask ---
        if args.visualize_masks and best_mask_array is not None:
            vis_path = figures_dir / f"{args.model_type}_{stage}_best_mask_advanced.png"
            visualize_mask_advanced(best_mask_array, vis_path, stage, 'best')
            print(f"   🖼  Advanced mask visualization saved to {vis_path}")

        # Use top-N masks for next stage (hierarchical refinement)
        prev_masks = [m.mask for m in top_masks if hasattr(m, 'mask')]
        all_results[stage] = stage_results

    if args.per_au_eval and per_au_drop_by_stage:
        save_per_au_stage_heatmap(
            per_au_drop_by_stage=per_au_drop_by_stage,
            model_type=args.model_type,
            save_dir=figures_dir,
        )
    
    print("\n" + "="*70)
    print("✅ HFSS Search Complete!")
    print("="*70)


if __name__ == "__main__":
    main()


# python hfss_search_au.py --model_type FMAE --model_path models/FMAE_BP4D_fold1.pth --num_candidates 200 --stages stage1 stage2 stage3

'''
# All features on
python hfss_search_au.py --model_path models/FMAE_BP4D_fold1.pth \
  --test_json BP4D/BP4D_test1.json --num_samples 500 \
  --per_au_eval --visualize_masks --stages stage1 stage2


python hfss_search_au.py --model_path models/FMAE_BP4D_fold1.pth \
  --test_json BP4D/BP4D_test1.json --num_samples 50 \
  --num_candidates 30 --top_n 10 --stages stage1 stage2 stage3 \
  --per_au_eval --visualize_masks

For running all test set, 30 masks take top 10 for fold 1

python hfss_search_au.py --model_path models/FMAE_BP4D_fold1.pth \
  --test_json BP4D/BP4D_test1.json --num_samples 2000 \
  --num_candidates 30 --top_n 10 --stages stage1 stage2 stage3 \
  --per_au_eval --visualize_masks
'''