"""
HFSS DFM Evaluation for AU Detection
Evaluates model robustness by applying discovered frequency masks
"""

import sys
import pickle
import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add paths
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "fmae"))
sys.path.insert(0, str(PROJECT_ROOT / "hfss" / "hfss"))

# Imports
import models_vit
from util.datasets import BP4D_AU_dataset
from engine_finetune import AU_evaluate

# Config
AU_LABELS = [1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24]
IMG_SIZE = 224


def load_model(model_path, model_type='FMAE', device='cuda'):
    """Load AU detection model"""
    grad_reverse = 1.0 if model_type == 'IAT' else 0.0
    num_subjects = 41 if model_type == 'IAT' else 0
    
    model = models_vit.vit_large_patch16(
        num_classes=12, num_subjects=num_subjects,
        drop_path_rate=0.0, global_pool=True,
        grad_reverse=grad_reverse,
    )
    
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint.get('model', checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()
    
    return model


def visualize_mask(mask_array, save_path):
    """Simple visualization of frequency mask"""
    plt.figure(figsize=(8, 8))
    plt.imshow(mask_array, cmap='gray', interpolation='nearest')
    plt.title('Frequency Mask (white=keep, black=remove)')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


def load_dfms(dfm_dir, model_type, fold, stages):
    """Load DFM masks from pickle files"""
    dfms = {}
    dfm_path = Path(dfm_dir)
    
    for stage in stages:
        pkl_file = dfm_path / f"{model_type}_{stage}_DFMs.pkl"
        if pkl_file.exists():
            with open(pkl_file, 'rb') as f:
                masks = pickle.load(f)
                # Take the best mask (index 0)
                dfms[stage] = masks[0] if masks else None
                print(f"✓ Loaded {stage} DFM (top mask)")
        else:
            print(f"⚠ {pkl_file} not found, skipping {stage}")
    
    return dfms


class FilteredDataLoader:
    """Wrapper that applies frequency mask to images"""
    def __init__(self, dataloader, mask_transform=None):
        self.dataloader = dataloader
        self.mask_transform = mask_transform
    
    def __iter__(self):
        for images, (au_labels, subject_labels) in self.dataloader:
            if self.mask_transform is not None:
                images = torch.stack([self.mask_transform(img) for img in images])
            yield images, (au_labels, subject_labels)
    
    def __len__(self):
        return len(self.dataloader)


@torch.no_grad()
def evaluate_with_dfm(model, dataset, mask_transform, device, desc="Evaluating"):
    """Evaluate model with DFM mask applied"""
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    filtered_loader = FilteredDataLoader(dataloader, mask_transform)
    
    stats, f1_mean, auc_mean = AU_evaluate(filtered_loader, model, device)
    return stats, f1_mean, auc_mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='FMAE', choices=['FMAE', 'IAT'])
    parser.add_argument('--fold', type=int, default=1, choices=[1, 2, 3])
    parser.add_argument('--dfm_dir', default='hfss/DFM')
    parser.add_argument('--stages', nargs='+', default=['stage1', 'stage2', 'stage3'])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output_csv', default='hfss/dfm_results.csv')
    args = parser.parse_args()
    
    # Setup paths
    model_path = f"models/{args.model_type}_BP4D_fold{args.fold}.pth"
    test_json = f"BP4D/BP4D_test{args.fold}.json"
    
    print("="*70)
    print(f"DFM Evaluation - {args.model_type} Fold {args.fold}")
    print("="*70)
    
    # Load model
    model = load_model(model_path, args.model_type, args.device)
    print(f"✓ Loaded model: {model_path}")
    
    # Load test dataset
    dataset_args = SimpleNamespace(
        root_path='BP4D/BP4D_cropped/',
        input_size=IMG_SIZE,
        color_jitter=None, aa='rand-m9-mstd0.5-inc1',
        reprob=0.25, remode='pixel', recount=1
    )
    test_dataset = BP4D_AU_dataset(test_json, is_train=False, args=dataset_args)
    print(f"✓ Loaded test set: {len(test_dataset)} samples\n")
    
    # Load DFMs
    dfms = load_dfms(args.dfm_dir, args.model_type, args.fold, args.stages)
    print()
    
    # Evaluate baseline (no mask)
    print("🔍 Baseline (Original)...")
    baseline_stats, baseline_f1, baseline_auc = evaluate_with_dfm(
        model, test_dataset, None, args.device
    )
    print(f"   F1: {baseline_f1*100:.2f}%  |  AUC: {baseline_auc*100:.2f}%")
    
    # Evaluate with each DFM
    results = []
    per_au_results = []  # New: store per-AU results
    
    # Baseline per-AU results
    baseline_per_au_row = {'model_type': args.model_type, 'fold': args.fold, 'filter': 'original'}
    for i, au in enumerate(AU_LABELS):
        baseline_per_au_row[f'AU{au:02d}_f1'] = baseline_stats[str(i)]['f1'] * 100
        baseline_per_au_row[f'AU{au:02d}_auc'] = baseline_stats[str(i)]['auc'] * 100
    per_au_results.append(baseline_per_au_row)
    
    results.append({
        'model_type': args.model_type,
        'fold': args.fold,
        'filter': 'original',
        'f1_mean': baseline_f1 * 100,
        'auc_mean': baseline_auc * 100,
        'f1_drop': 0.0,
        'f1_drop_pct': 0.0,
    })
    
    # Visualize masks
    vis_dir = Path(args.output_csv).parent / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    
    for stage, dfm_mask in dfms.items():
        if dfm_mask is None:
            continue
        
        print(f"\n🔍 Evaluating with {stage} DFM...")
        stats, f1, auc = evaluate_with_dfm(
            model, test_dataset, dfm_mask, args.device
        )
        
        f1_drop = baseline_f1 - f1
        f1_drop_pct = (f1_drop / baseline_f1 * 100) if baseline_f1 > 0 else 0
        
        print(f"   F1: {f1*100:.2f}%  |  AUC: {auc*100:.2f}%")
        print(f"   F1 Drop: {f1_drop*100:.2f}% ({f1_drop_pct:.1f}%)")
        
        results.append({
            'model_type': args.model_type,
            'fold': args.fold,
            'filter': stage,
            'f1_mean': f1 * 100,
            'auc_mean': auc * 100,
            'f1_drop': f1_drop * 100,
            'f1_drop_pct': f1_drop_pct,
        })
        
        # Per-AU drops and save to CSV
        per_au_row = {'model_type': args.model_type, 'fold': args.fold, 'filter': stage}
        print(f"   Per-AU F1 scores and drops:")
        for i, au in enumerate(AU_LABELS):
            # Use index i (0-11) as key, not AU number
            baseline_au_f1 = baseline_stats[str(i)]['f1']
            dfm_au_f1 = stats[str(i)]['f1']
            au_drop = baseline_au_f1 - dfm_au_f1
            
            # Save to per-AU results
            per_au_row[f'AU{au:02d}_f1'] = dfm_au_f1 * 100
            per_au_row[f'AU{au:02d}_auc'] = stats[str(i)]['auc'] * 100
            per_au_row[f'AU{au:02d}_drop'] = au_drop * 100
            
            # Print all AU results (not just significant drops)
            print(f"      AU{au:02d}: F1={dfm_au_f1*100:.1f}% (drop: {au_drop*100:.1f}%)")
        
        per_au_results.append(per_au_row)
        
        # Visualize the mask
        if hasattr(dfm_mask, 'mask'):
            vis_path = vis_dir / f"{args.model_type}_fold{args.fold}_{stage}_mask.png"
            visualize_mask(dfm_mask.mask, vis_path)
            print(f"   ✓ Saved mask visualization to {vis_path}")
    
    # Save summary results
    df = pd.DataFrame(results)
    output_file = Path(args.output_csv)
    
    if output_file.exists():
        df_existing = pd.read_csv(output_file)
        df = pd.concat([df_existing, df], ignore_index=True)
    
    df.to_csv(output_file, index=False)
    print(f"\n✅ Summary results saved to {output_file}")
    
    # Save per-AU results
    per_au_file = output_file.parent / f"{output_file.stem}_per_au.csv"
    df_per_au = pd.DataFrame(per_au_results)
    
    if per_au_file.exists():
        df_per_au_existing = pd.read_csv(per_au_file)
        df_per_au = pd.concat([df_per_au_existing, df_per_au], ignore_index=True)
    
    df_per_au.to_csv(per_au_file, index=False)
    print(f"✅ Per-AU results saved to {per_au_file}")
    print(f"✅ Mask visualizations saved to {vis_dir}")
    
    # Summary
    print("\n" + "="*70)
    print("Summary:")
    print("="*70)
    print(df[df['fold'] == args.fold].to_string(index=False))
    print("="*70)


if __name__ == "__main__":
    main()


# # Evaluate FMAE fold 1 with DFMs
# python hfss_eval_dfm.py --model_type FMAE --fold 1

# # Evaluate IAT fold 2 
# python hfss_eval_dfm.py --model_type IAT --fold 2

# # All folds
# python hfss_eval_dfm.py --model_type FMAE --fold 1
# python hfss_eval_dfm.py --model_type FMAE --fold 2
# python hfss_eval_dfm.py --model_type FMAE --fold 3