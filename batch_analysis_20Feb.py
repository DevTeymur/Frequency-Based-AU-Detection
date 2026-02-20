"""
Batch Analysis: Process multiple images with HFSS and save results
Configurable to run on small sample or entire dataset
"""

import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import timm
from sklearn.metrics import roc_auc_score
from warnings import filterwarnings
filterwarnings("ignore")

# Import from our modules
from hfss import apply_frequency_mask, create_low_freq_mask, create_high_freq_mask, create_random_mask

# ============================================================================
# CONFIGURATION - ADJUST THESE PARAMETERS
# ============================================================================
MODEL_TYPE = 'FMAE'  # Choose: 'FMAE' or 'IAT'

NUM_SAMPLES = None  # Set to None to process ALL images, or a number like 50 for testing

TEST_SETS = ['test1']  # Which test sets to process

# Filter types:
# - 'original': No filtering
# - 'low': Low frequency only (circular mask, 30% center)
# - 'high': High frequency only (edges, outer 70%)
# - 'random_20', 'random_40', 'random_60', 'random_80': Random X% of frequencies (HFSS)
FILTERS = ['original'] #, 'random_20', 'random_40', 'random_60', 'random_80']

IMG_SIZE = 224
DATA_ROOT = Path("BP4D/BP4D_cropped")

# Map each test set to its corresponding model (trained on matching train set)
MODEL_PATHS = {
    'test1': Path("models/FMAE_BP4D_fold1.pth"),  # fold1 trained on train1
    'test2': Path("models/FMAE_BP4D_fold2.pth"),  # fold2 trained on train2
    'test3': Path("models/FMAE_BP4D_fold3.pth"),  # fold3 trained on train3
}

IAT_MODEL_PATHS = {
    'test1': Path("models/FMAE_IAT_BP4D_fold1.pth"),  # fold1 trained on train1
    'test2': Path("models/FMAE_IAT_BP4D_fold2.pth"),  # fold2 trained on train2
    'test3': Path("models/FMAE_IAT_BP4D_fold3.pth"),  # fold3 trained on train3
}

# Reference results (from supervisor's training - 33 epochs)
REFERENCE_RESULTS = {
    'FMAE': {
        'test1': {'F1': 67.108, 'AUC': 82.407},
        'test2': {'F1': 65.317, 'AUC': 82.143},
        'test3': {'F1': 67.705, 'AUC': 82.768},
    },
    'IAT': {
        'test1': {'F1': 66.209, 'AUC': None},
        'test2': {'F1': 69.566, 'AUC': None},
        'test3': {'F1': 65.385, 'AUC': None},
    }
}

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Auto-detect device (CUDA GPU > MPS > CPU)
# Note: If CUDA error occurs, force CPU by setting FORCE_CPU=True
FORCE_CPU = False  # Set to True if CUDA compatibility issues

if FORCE_CPU:
    DEVICE = 'cpu'
elif torch.cuda.is_available():
    DEVICE = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'

# Subject IDs for identity prediction (FMAE-IAT only)
SUBJECT_IDS = ['F001', 'F002', 'F003', 'F004', 'F005', 'F006', 'F007', 'F008', 'F009', 'F010',
               'F011', 'F012', 'F013', 'F014', 'F015', 'F016', 'F017', 'F018', 'F019', 'F020',
               'F021', 'F022', 'F023', 'M001', 'M002', 'M003', 'M004', 'M005', 'M006', 'M007',
               'M008', 'M009', 'M010', 'M011', 'M012', 'M013', 'M014', 'M015', 'M016', 'M017', 'M018']

# Action Units
AU_LABELS = [1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24]
AU_NAMES = {
    1: 'Inner Brow Raiser', 2: 'Outer Brow Raiser', 4: 'Brow Lowerer',
    6: 'Cheek Raiser', 7: 'Lid Tightener', 10: 'Upper Lip Raiser',
    12: 'Lip Corner Puller', 14: 'Dimpler', 15: 'Lip Corner Depressor',
    17: 'Chin Raiser', 23: 'Lip Tightener', 24: 'Lip Pressor'
}

print("📊 Batch Analysis with HFSS")
print("="*70)
print(f"Device: {DEVICE}")
print(f"Processing: {NUM_SAMPLES if NUM_SAMPLES else 'ALL'} samples per test set")
print(f"Test sets: {TEST_SETS}")
print(f"Filters: {FILTERS}")
print("="*70)

# ============================================================================
# LOAD DATA
# ============================================================================
def load_dataset(test_set, num_samples=None):
    """Load dataset for a test set"""
    json_file = Path(f"BP4D/BP4D_{test_set}.json")
    
    data = []
    with open(json_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    # Limit samples if specified
    if num_samples:
        data = data[:num_samples]
    
    return data

def load_and_preprocess_image(img_path):
    """Load and preprocess image for model"""
    full_path = DATA_ROOT / img_path
    img = Image.open(full_path).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))

    # Convert to tensor and normalize
    img_array = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
    
    # Normalize with ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_normalized = (img_tensor - mean) / std
    
    return img_normalized

# ============================================================================
# LOAD MODEL
# ============================================================================
def load_model(model_path):
    """Load pretrained model"""
    print(f"\n🤖 Loading model from {model_path}...")
    
    model = timm.create_model(
        'vit_large_patch16_224',
        pretrained=False,
        num_classes=12
    )
    
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    
    model = model.to(DEVICE)
    model.eval()
    
    print(f"   ✓ Model ready on {DEVICE}")
    return model

# ============================================================================
# PREDICT
# ============================================================================
@torch.no_grad()
def predict_batch(model, images, predict_identity=False):
    """
    Predict AU labels for a batch of images
    
    Args:
        model: Loaded model
        images: Batch of images [B, 3, 224, 224]
        predict_identity: If True, return identity predictions (FMAE-IAT only)
    
    Returns:
        probabilities: [B, 12] AU probabilities
        predictions: [B, 12] binary predictions
        identity_probs: [B, 41] identity probabilities (if predict_identity=True)
    """
    images = images.to(DEVICE)
    outputs = model(images)
    
    # IAT models return tuple (au_output, id_output), FMAE returns single tensor
    if predict_identity and isinstance(outputs, tuple):
        au_outputs, id_outputs = outputs
        au_probs = torch.sigmoid(au_outputs).cpu()
        au_preds = (au_probs >= 0.5).float()
        id_probs = torch.softmax(id_outputs, dim=1).cpu()
        return au_probs, au_preds, id_probs
    else:
        # FMAE or not predicting identity
        if isinstance(outputs, tuple):
            outputs = outputs[0]  # Take AU head only
        au_probs = torch.sigmoid(outputs).cpu()
        au_preds = (au_probs >= 0.5).float()
        return au_probs, au_preds, None

# ============================================================================
# BATCH PROCESSING
# ============================================================================
def process_dataset(model, dataset, test_set_name, filter_type='original', batch_size=16, predict_identity=False):
    """
    Process entire dataset with specified filter
    
    Args:
        model: Loaded model
        dataset: List of samples
        test_set_name: Name of test set (for logging)
        filter_type: 'original', 'low', or 'high'
        batch_size: Batch size for processing
        predict_identity: If True, predict identity (FMAE-IAT only)
    
    Returns:
        List of result dictionaries
    """
    results = []
    
    # Create frequency mask if needed
    mask = None
    if filter_type == 'low':
        mask = create_low_freq_mask(IMG_SIZE, cutoff=0.3)
    elif filter_type == 'high':
        mask = create_high_freq_mask(IMG_SIZE, cutoff=0.3)
    elif filter_type.startswith('random_'):
        # Extract retention percentage (e.g., 'random_20' -> 0.20)
        retention = int(filter_type.split('_')[1]) / 100.0
        # Create ONE random mask for this batch (same mask for all images)
        mask = create_random_mask(IMG_SIZE, proportion=retention, seed=42)
    
    # Process in batches
    num_batches = (len(dataset) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc=f"{test_set_name} - {filter_type}"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(dataset))
        batch_samples = dataset[start_idx:end_idx]
        
        # Load and preprocess images
        images = []
        true_labels = []
        
        for sample in batch_samples:
            # Load image
            img = load_and_preprocess_image(sample['img_path'])
            
            # Apply filter
            if mask is not None:
                img = apply_frequency_mask(img, mask)
            
            images.append(img)
            
            # Create true label vector
            true_label = torch.zeros(12)
            for au in sample['AUs']:
                if au in AU_LABELS:
                    true_label[AU_LABELS.index(au)] = 1
            true_labels.append(true_label)
        
        # Stack and predict
        images = torch.stack(images)
        true_labels = torch.stack(true_labels)
        
        au_probs, au_preds, id_probs = predict_batch(model, images, predict_identity=predict_identity)
        
        # Store results
        for i, sample in enumerate(batch_samples):
            # Get predicted AUs
            predicted_au_indices = torch.where(au_preds[i] == 1)[0].tolist()
            predicted_aus = [AU_LABELS[idx] for idx in predicted_au_indices]
            
            # Compute accuracy
            accuracy = (au_preds[i] == true_labels[i]).float().mean().item()
            
            # Store result
            result = {
                'img_path': sample['img_path'],
                'test_set': test_set_name,
                'filter_type': filter_type,
                'true_aus': sample['AUs'],
                'predicted_aus': predicted_aus,
                'accuracy': accuracy,
            }
            
            # Add individual AU probabilities
            for au_idx, au_label in enumerate(AU_LABELS):
                result[f'AU{au_label}_prob'] = au_probs[i, au_idx].item()
                result[f'AU{au_label}_pred'] = int(au_preds[i, au_idx].item())
                result[f'AU{au_label}_true'] = int(true_labels[i, au_idx].item())
            
            # Add identity prediction if available
            if predict_identity and id_probs is not None:
                # Extract true subject ID from path (e.g., "F001/T1/1.jpg" -> "F001")
                true_subject = sample['img_path'].split('/')[0]
                predicted_subject_idx = torch.argmax(id_probs[i]).item()
                predicted_subject = SUBJECT_IDS[predicted_subject_idx]
                
                result['true_subject'] = true_subject
                result['predicted_subject'] = predicted_subject
                result['identity_correct'] = int(true_subject == predicted_subject)
                result['identity_confidence'] = id_probs[i, predicted_subject_idx].item()
            
            results.append(result)
    
    return results

# ============================================================================
# SAVE RESULTS
# ============================================================================
def save_results_to_csv(all_results, filename):
    """Save results to CSV file"""
    df = pd.DataFrame(all_results)
    output_file = OUTPUT_DIR / filename
    df.to_csv(output_file, index=False)
    print(f"\n💾 Saved results to {output_file}")
    return df

def save_metrics_to_csv(metrics, test_set_name, model_type):
    """Save metrics summary to CSV"""
    rows = []
    for filter_type, metric_data in metrics.items():
        row = {
            'test_set': test_set_name,
            'model_type': model_type,
            'filter_type': filter_type,
            'mean_accuracy': metric_data['mean_accuracy'],
            'mean_f1': metric_data['mean_f1'],
            'mean_auc': metric_data['mean_auc'],
            'optimal_threshold': metric_data['optimal_threshold']
        }
        # Add per-AU metrics
        for au, au_metrics in metric_data['au_metrics'].items():
            row[f'{au}_precision'] = au_metrics['precision']
            row[f'{au}_recall'] = au_metrics['recall']
            row[f'{au}_f1'] = au_metrics['f1']
            row[f'{au}_auc'] = au_metrics['auc']
        rows.append(row)
    
    df_metrics = pd.DataFrame(rows)
    output_file = OUTPUT_DIR / f'{test_set_name}_{model_type}_metrics.csv'
    df_metrics.to_csv(output_file, index=False)
    print(f"   💾 Saved metrics to {output_file}")

# ============================================================================
# ANALYSIS & VISUALIZATION
# ============================================================================
def compute_metrics(df):
    """Compute evaluation metrics from results dataframe using optimal threshold search"""
    from sklearn.metrics import f1_score
    
    metrics = {}
    
    for filter_type in df['filter_type'].unique():
        df_filter = df[df['filter_type'] == filter_type]
        
        # Overall metrics
        mean_acc = df_filter['accuracy'].mean()
        
        # Prepare data for threshold search
        y_true_all = []
        y_prob_all = []
        for au in AU_LABELS:
            y_true_all.append(df_filter[f'AU{au}_true'].values)
            y_prob_all.append(df_filter[f'AU{au}_prob'].values)
        
        y_true = np.array(y_true_all).T  # Shape: [samples, 12]
        y_prob = np.array(y_prob_all).T  # Shape: [samples, 12]
        
        # Search for optimal threshold (matching supervisor's method)
        f1_score_ls = []
        for i in range(1, 100):
            threshold = i * 0.01
            y_pred = np.zeros(y_prob.shape)
            y_pred[np.where(y_prob >= threshold)] = 1
            
            f1_scores = []
            for class_idx in range(y_true.shape[1]):
                f1_scores.append(f1_score(y_true[:, class_idx], y_pred[:, class_idx]))
            f1_score_ls.append(f1_scores)
        
        f1_score_arr = np.array(f1_score_ls)
        max_f1_row_index = np.argmax(np.mean(f1_score_arr, axis=1))
        max_mean_row = f1_score_arr[max_f1_row_index]
        optimal_threshold = (max_f1_row_index + 1) / 100
        
        # Compute per-AU metrics at optimal threshold
        au_metrics = {}
        y_pred_optimal = np.zeros(y_prob.shape)
        y_pred_optimal[np.where(y_prob >= optimal_threshold)] = 1
        
        for au_idx, au in enumerate(AU_LABELS):
            y_true_au = y_true[:, au_idx]
            y_pred_au = y_pred_optimal[:, au_idx]
            y_prob_au = y_prob[:, au_idx]
            
            # Compute metrics
            tp = ((y_true_au == 1) & (y_pred_au == 1)).sum()
            fp = ((y_true_au == 0) & (y_pred_au == 1)).sum()
            fn = ((y_true_au == 1) & (y_pred_au == 0)).sum()
            
            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            f1 = max_mean_row[au_idx]
            
            # Compute AUC
            try:
                auc = roc_auc_score(y_true_au, y_prob_au) if len(np.unique(y_true_au)) > 1 else 0.5
            except:
                auc = 0.5
            
            au_metrics[f'AU{au}'] = {
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'auc': auc
            }
        
        # Compute mean F1 and AUC across all AUs
        mean_f1 = max_mean_row.mean()
        mean_auc = np.mean([m['auc'] for m in au_metrics.values()])
        
        metrics[filter_type] = {
            'mean_accuracy': mean_acc,
            'mean_f1': mean_f1,
            'mean_auc': mean_auc,
            'optimal_threshold': optimal_threshold,
            'au_metrics': au_metrics
        }
    
    return metrics

def visualize_results(df, test_set_name):
    """Create comprehensive visualization with delta plots"""
    print(f"\n📊 Creating visualizations for {test_set_name}...")
    
    fig = plt.figure(figsize=(20, 12))
    
    # Compute F1 scores for all filters
    f1_by_filter = {}
    for filter_type in df['filter_type'].unique():
        df_filter = df[df['filter_type'] == filter_type]
        f1_scores = []
        for au in AU_LABELS:
            y_true = df_filter[f'AU{au}_true'].values
            y_pred = df_filter[f'AU{au}_pred'].values
            tp = ((y_true == 1) & (y_pred == 1)).sum()
            fp = ((y_true == 0) & (y_pred == 1)).sum()
            fn = ((y_true == 1) & (y_pred == 0)).sum()
            f1 = 2*tp / (2*tp + fp + fn + 1e-6)
            f1_scores.append(f1)
        f1_by_filter[filter_type] = f1_scores
    
    # Plot 1: Overall F1 Delta from Original
    ax1 = plt.subplot(2, 3, 1)
    if 'original' in f1_by_filter:
        orig_f1 = np.mean(f1_by_filter['original'])
        deltas = []
        labels = []
        for ft in ['low', 'high']:
            if ft in f1_by_filter:
                delta = (np.mean(f1_by_filter[ft]) - orig_f1) * 100
                deltas.append(delta)
                labels.append(ft)
        
        colors = ['blue' if d < 0 else 'green' for d in deltas]
        bars = ax1.bar(labels, deltas, color=colors, alpha=0.7)
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=2)
        ax1.set_ylabel('F1 Change (%)')
        ax1.set_title('Mean F1 Drop from Original')
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom' if height > 0 else 'top')
    
    # Plot 2: Per-AU F1 Delta (Low Freq)
    ax2 = plt.subplot(2, 3, 2)
    if 'original' in f1_by_filter and 'low' in f1_by_filter:
        deltas_low = [(f1_by_filter['low'][i] - f1_by_filter['original'][i]) * 100 
                      for i in range(len(AU_LABELS))]
        colors = ['blue' if d < 0 else 'green' for d in deltas_low]
        ax2.bar(range(len(AU_LABELS)), deltas_low, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax2.set_ylabel('F1 Change (%)')
        ax2.set_xlabel('Action Unit')
        ax2.set_title('Per-AU F1 Drop: Low Frequency')
        ax2.set_xticks(range(len(AU_LABELS)))
        ax2.set_xticklabels([f'AU{au}' for au in AU_LABELS], rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Per-AU F1 Delta (High Freq)
    ax3 = plt.subplot(2, 3, 3)
    if 'original' in f1_by_filter and 'high' in f1_by_filter:
        deltas_high = [(f1_by_filter['high'][i] - f1_by_filter['original'][i]) * 100 
                       for i in range(len(AU_LABELS))]
        colors = ['red' if d < 0 else 'green' for d in deltas_high]
        ax3.bar(range(len(AU_LABELS)), deltas_high, color=colors, alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax3.set_ylabel('F1 Change (%)')
        ax3.set_xlabel('Action Unit')
        ax3.set_title('Per-AU F1 Drop: High Frequency')
        ax3.set_xticks(range(len(AU_LABELS)))
        ax3.set_xticklabels([f'AU{au}' for au in AU_LABELS], rotation=45)
        ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Delta Heatmap
    ax4 = plt.subplot(2, 3, 4)
    if 'original' in f1_by_filter:
        heatmap_data = []
        for au_idx in range(len(AU_LABELS)):
            row = []
            for ft in ['low', 'high']:
                if ft in f1_by_filter:
                    delta = (f1_by_filter[ft][au_idx] - f1_by_filter['original'][au_idx]) * 100
                    row.append(delta)
                else:
                    row.append(0)
            heatmap_data.append(row)
        
        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                    xticklabels=['Low', 'High'],
                    yticklabels=[f'AU{au}' for au in AU_LABELS],
                    ax=ax4, cbar_kws={'label': 'F1 Change (%)'})
        ax4.set_title('F1 Delta Heatmap')
    
    # Plot 5: Sample Distribution
    ax5 = plt.subplot(2, 3, 5)
    filter_counts = df['filter_type'].value_counts()
    ax5.pie(filter_counts.values, labels=filter_counts.index, autopct='%1.1f%%')
    ax5.set_title(f'Processed Samples Distribution\n(Total: {len(df)})')
    
    # Plot 6: Low vs High Comparison
    ax6 = plt.subplot(2, 3, 6)
    if 'original' in f1_by_filter and 'low' in f1_by_filter and 'high' in f1_by_filter:
        deltas_low = [(f1_by_filter['low'][i] - f1_by_filter['original'][i]) * 100 
                      for i in range(len(AU_LABELS))]
        deltas_high = [(f1_by_filter['high'][i] - f1_by_filter['original'][i]) * 100 
                       for i in range(len(AU_LABELS))]
        
        x = np.arange(len(AU_LABELS))
        width = 0.35
        ax6.bar(x - width/2, deltas_low, width, label='Low', alpha=0.7, color='blue')
        ax6.bar(x + width/2, deltas_high, width, label='High', alpha=0.7, color='red')
        ax6.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax6.set_ylabel('F1 Change (%)')
        ax6.set_xlabel('Action Unit')
        ax6.set_title('Low vs High Frequency Impact')
        ax6.set_xticks(x)
        ax6.set_xticklabels([f'AU{au}' for au in AU_LABELS], rotation=45)
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')
    ax6.set_title('Accuracy Distribution by Filter')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / f'{test_set_name}_analysis.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved visualization to {output_file}")
    plt.close()

# ============================================================================
# MAIN
# ============================================================================
def main():
    # Process each test set with its corresponding model
    for test_set in TEST_SETS:
        print(f"\n{'='*70}")
        print(f"PROCESSING {test_set.upper()} - {MODEL_TYPE}")
        print(f"{'='*70}")
        
        # Load the correct model for this test set
        model_path = MODEL_PATHS[test_set] if MODEL_TYPE == 'FMAE' else IAT_MODEL_PATHS[test_set]
        print(f"Using model: {model_path}")
        model = load_model(model_path)
        
        # Load dataset
        dataset = load_dataset(test_set, num_samples=NUM_SAMPLES)
        print(f"Loaded {len(dataset)} samples")
        
        # Process with each filter
        all_results = []
        predict_identity = (MODEL_TYPE == 'IAT')  # Only predict identity for IAT models
        
        for filter_type in FILTERS:
            results = process_dataset(model, dataset, test_set, filter_type, predict_identity=predict_identity)
            all_results.extend(results)
        
        # Save results
        df = save_results_to_csv(all_results, f'{test_set}_{MODEL_TYPE}_results.csv')
        
        # Compute and display metrics
        metrics = compute_metrics(df)
        print(f"\n📈 Metrics Summary:")
        
        # Get reference results for comparison
        ref = REFERENCE_RESULTS.get(MODEL_TYPE, {}).get(test_set, {})
        ref_f1 = ref.get('F1')
        ref_auc = ref.get('AUC')
        
        for filter_type, metric_data in metrics.items():
            f1 = metric_data['mean_f1'] * 100  # Convert to percentage
            auc = metric_data['mean_auc'] * 100
            threshold = metric_data['optimal_threshold']
            
            # Print our results
            result_line = f"   {filter_type:10s}: F1={f1:.3f}% | AUC={auc:.3f}% | Threshold={threshold:.2f}"
            
            # Add comparison with reference (only for original images)
            if filter_type == 'original' and ref_f1 is not None:
                f1_diff = f1 - ref_f1
                diff_str = f" | ΔF1={f1_diff:+.3f}%"
                if ref_auc is not None:
                    auc_diff = auc - ref_auc
                    diff_str += f" | ΔAUC={auc_diff:+.3f}%"
                result_line += diff_str
            
            print(result_line)
        
        # Show reference results
        if ref_f1 is not None:
            print(f"\n🎯 Reference Results (33 epochs):")
            ref_line = f"   {'original':10s}: F1={ref_f1:.3f}%"
            if ref_auc is not None:
                ref_line += f" | AUC={ref_auc:.3f}%"
            print(ref_line)
        
        # Display identity accuracy if available (IAT only)
        if predict_identity and 'identity_correct' in df.columns:
            print(f"\n🆔 Identity Prediction Accuracy:")
            for filter_type in df['filter_type'].unique():
                df_filter = df[df['filter_type'] == filter_type]
                identity_acc = df_filter['identity_correct'].mean()
                print(f"   {filter_type:10s}: {identity_acc*100:.2f}%")
        
        # Show per-AU F1 drops
        if 'original' in metrics:
            print(f"\n📉 Per-AU F1 Drops from Original:")
            orig = metrics['original']['au_metrics']
            for filter_type in ['low', 'high']:
                if filter_type in metrics:
                    print(f"   {filter_type.upper()}:")
                    for au in AU_LABELS:
                        au_key = f'AU{au}'
                        if au_key in orig and au_key in metrics[filter_type]['au_metrics']:
                            drop = (orig[au_key]['f1'] - metrics[filter_type]['au_metrics'][au_key]['f1']) * 100
                            print(f"      AU{au:2d}: {drop:+6.1f}%")
        
        # Save metrics to CSV
        save_metrics_to_csv(metrics, test_set, MODEL_TYPE)
        
        # Visualize
        visualize_results(df, test_set)
    
    print(f"\n{'='*70}")
    print("✅ Batch Analysis Complete!")
    print(f"{'='*70}")
    print(f"Results saved to: {OUTPUT_DIR.absolute()}")

if __name__ == "__main__":
    main()
