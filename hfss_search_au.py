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


def generate_mask_candidates(num_candidates, proportion, stage, prev_mask=None):
    """Generate random frequency mask candidates"""
    candidates = []
    patch = PATCHES[stage]
    
    for _ in range(num_candidates):
        # Generate random frequency mask
        freqs = gen_freqs_list(patch, patch)
        sample_frqs = sample_frequency(proportion, freqs)
        mask_int = generate_mask(sample_frqs, patch, patch)
        mask = np.kron(mask_int, np.ones((IMG_SIZE // patch, IMG_SIZE // patch)))
        
        # Apply previous stage mask if exists
        if prev_mask is not None:
            mask = prev_mask * mask
        
        # Make symmetric
        max_n_h = IMG_SIZE // 2
        max_n_w = IMG_SIZE // 2
        for h_index in range(-max_n_h, 1):
            for w_index in range(-max_n_w, max_n_w):
                h_matrix_index = IMG_SIZE // 2 + h_index
                w_matrix_index = IMG_SIZE // 2 + w_index
                if h_index != 0:
                    mask[IMG_SIZE - h_matrix_index - 1, IMG_SIZE - w_matrix_index - 1] = mask[h_matrix_index, w_matrix_index]
        
        candidates.append(White_Mask(mask))
    
    return candidates


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


def search_stage(model, dataloader, stage, num_candidates, proportion, prev_mask, device):
    """Search for frequency shortcuts in one stage"""
    print(f"\n🔍 Stage {stage}: Testing {num_candidates} candidates (P={proportion})")
    
    # Generate candidates
    candidates = generate_mask_candidates(num_candidates, proportion, stage, prev_mask)
    
    # Test baseline (no mask)
    baseline_f1 = evaluate_mask(model, dataloader, lambda x: x, device)
    print(f"   Baseline mean F1: {baseline_f1*100:.2f}% (mean per-AU)")
    
    # Test each candidate (compute mean F1 per candidate)
    results = []
    for i, mask_transform in enumerate(tqdm(candidates, desc=f"   Testing masks")):
        f1 = evaluate_mask(model, dataloader, mask_transform, device)
        drop = baseline_f1 - f1
        results.append((i, f1, drop, mask_transform))
    
    # Find top masks with biggest performance drop
    results.sort(key=lambda x: x[2], reverse=True)
    top_10 = results[:10]
    
    print(f"   Top mean-F1 drops (baseline - mask): {[f'{r[2]*100:.1f}%' for r in top_10[:5]]}")
    
    # Return top masks
    return [r[3] for r in top_10], results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='FMAE', choices=['FMAE', 'IAT'])
    parser.add_argument('--model_path', default='models/FMAE_BP4D_fold1.pth')
    parser.add_argument('--test_json', default='BP4D/BP4D_test1.json')
    parser.add_argument('--data_root', default='BP4D/BP4D_cropped/')
    parser.add_argument('--num_candidates', type=int, default=200)
    parser.add_argument('--proportion', type=float, default=0.8)
    parser.add_argument('--num_samples', type=int, default=500, help='Number of training samples to use')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output_dir', default='hfss/DFM')
    parser.add_argument('--stages', nargs='+', default=['stage1', 'stage2', 'stage3'])
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
    prev_mask = None
    all_results = {}
    
    for stage in args.stages:
        top_masks, stage_results = search_stage(
            model, dataloader, stage, 
            args.num_candidates, args.proportion, 
            prev_mask, args.device
        )
        
        # Save stage results
        output_file = output_dir / f"{args.model_type}_{stage}_DFMs.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(top_masks, f)
        print(f"   ✓ Saved to {output_file}")
        
        # Use best mask for next stage
        prev_mask = top_masks[0].mask if hasattr(top_masks[0], 'mask') else None
        all_results[stage] = stage_results
    
    print("\n" + "="*70)
    print("✅ HFSS Search Complete!")
    print("="*70)


if __name__ == "__main__":
    main()


# python hfss_search_au.py --model_type FMAE --model_path models/FMAE_BP4D_fold1.pth --num_candidates 200 --stages stage1 stage2 stage3