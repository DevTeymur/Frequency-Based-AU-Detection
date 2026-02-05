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
from warnings import filterwarnings
filterwarnings("ignore")

# Import from our modules
from hfss import apply_frequency_mask, create_low_freq_mask, create_high_freq_mask

# ============================================================================
# CONFIGURATION - ADJUST THESE PARAMETERS
# ============================================================================
NUM_SAMPLES = 100  # Set to None to process ALL images, or a number like 50 for testing
TEST_SETS = ['test1', 'test2', 'test3']  # Which test sets to process
FILTERS = ['original', 'low', 'high']  # Which filters to apply

IMG_SIZE = 224
DATA_ROOT = Path("BP4D/BP4D_cropped")
MODEL_PATH = Path("models/FMAE_ViT_base.pth")
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    print(f"\n🤖 Loading model...")
    
    model = timm.create_model(
        'vit_base_patch16_224',
        pretrained=False,
        num_classes=12,
        global_pool='avg'
    )
    
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    model = model.to(DEVICE)
    model.eval()
    
    print(f"   ✓ Model ready")
    return model

# ============================================================================
# PREDICT
# ============================================================================
@torch.no_grad()
def predict_batch(model, images):
    """
    Predict AU labels for a batch of images
    
    Args:
        model: Loaded model
        images: Batch of images [B, 3, 224, 224]
    
    Returns:
        probabilities: [B, 12] AU probabilities
        predictions: [B, 12] binary predictions
    """
    images = images.to(DEVICE)
    outputs = model(images)
    probs = torch.sigmoid(outputs).cpu()
    preds = (probs >= 0.5).float()
    
    return probs, preds

# ============================================================================
# BATCH PROCESSING
# ============================================================================
def process_dataset(model, dataset, test_set_name, filter_type='original', batch_size=16):
    """
    Process entire dataset with specified filter
    
    Args:
        model: Loaded model
        dataset: List of samples
        test_set_name: Name of test set (for logging)
        filter_type: 'original', 'low', or 'high'
        batch_size: Batch size for processing
    
    Returns:
        List of result dictionaries
    """
    results = []
    
    # Create frequency mask if needed
    if filter_type == 'low':
        mask = create_low_freq_mask(IMG_SIZE, cutoff=0.3)
    elif filter_type == 'high':
        mask = create_high_freq_mask(IMG_SIZE, cutoff=0.3)
    else:
        mask = None
    
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
        
        probs, preds = predict_batch(model, images)
        
        # Store results
        for i, sample in enumerate(batch_samples):
            # Get predicted AUs
            predicted_au_indices = torch.where(preds[i] == 1)[0].tolist()
            predicted_aus = [AU_LABELS[idx] for idx in predicted_au_indices]
            
            # Compute accuracy
            accuracy = (preds[i] == true_labels[i]).float().mean().item()
            
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
                result[f'AU{au_label}_prob'] = probs[i, au_idx].item()
                result[f'AU{au_label}_pred'] = int(preds[i, au_idx].item())
                result[f'AU{au_label}_true'] = int(true_labels[i, au_idx].item())
            
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

# ============================================================================
# ANALYSIS & VISUALIZATION
# ============================================================================
def compute_metrics(df):
    """Compute evaluation metrics from results dataframe"""
    metrics = {}
    
    for filter_type in df['filter_type'].unique():
        df_filter = df[df['filter_type'] == filter_type]
        
        # Overall metrics
        mean_acc = df_filter['accuracy'].mean()
        
        # Per-AU metrics
        au_metrics = {}
        for au in AU_LABELS:
            true_col = f'AU{au}_true'
            pred_col = f'AU{au}_pred'
            
            if true_col in df_filter.columns:
                y_true = df_filter[true_col].values
                y_pred = df_filter[pred_col].values
                
                # Compute F1, Precision, Recall
                tp = ((y_true == 1) & (y_pred == 1)).sum()
                fp = ((y_true == 0) & (y_pred == 1)).sum()
                fn = ((y_true == 1) & (y_pred == 0)).sum()
                
                precision = tp / (tp + fp + 1e-6)
                recall = tp / (tp + fn + 1e-6)
                f1 = 2 * precision * recall / (precision + recall + 1e-6)
                
                au_metrics[f'AU{au}'] = {
                    'f1': f1,
                    'precision': precision,
                    'recall': recall
                }
        
        metrics[filter_type] = {
            'mean_accuracy': mean_acc,
            'au_metrics': au_metrics
        }
    
    return metrics

def visualize_results(df, test_set_name):
    """Create comprehensive visualization"""
    print(f"\n📊 Creating visualizations for {test_set_name}...")
    
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 1: Overall Accuracy Comparison
    ax1 = plt.subplot(2, 3, 1)
    acc_by_filter = df.groupby('filter_type')['accuracy'].mean()
    bars = ax1.bar(acc_by_filter.index, acc_by_filter.values)
    ax1.set_ylabel('Mean Accuracy')
    ax1.set_title('Overall Performance by Filter Type')
    ax1.set_ylim([0, 1])
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    
    # Plot 2: F1-Score per AU
    ax2 = plt.subplot(2, 3, 2)
    f1_data = []
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
        f1_data.append(f1_scores)
    
    x = np.arange(len(AU_LABELS))
    width = 0.25
    for i, (filter_type, f1_scores) in enumerate(zip(df['filter_type'].unique(), f1_data)):
        ax2.bar(x + i*width, f1_scores, width, label=filter_type, alpha=0.8)
    
    ax2.set_ylabel('F1-Score')
    ax2.set_xlabel('Action Unit')
    ax2.set_title('F1-Score per AU by Filter Type')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels([f'AU{au}' for au in AU_LABELS], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Performance Degradation
    ax3 = plt.subplot(2, 3, 3)
    if 'original' in df['filter_type'].unique():
        original_acc = df[df['filter_type'] == 'original']['accuracy'].mean()
        degradation = []
        labels = []
        for filter_type in ['low', 'high']:
            if filter_type in df['filter_type'].unique():
                filter_acc = df[df['filter_type'] == filter_type]['accuracy'].mean()
                deg = (original_acc - filter_acc) / (original_acc + 1e-6) * 100
                degradation.append(deg)
                labels.append(filter_type)
        
        bars = ax3.bar(labels, degradation, color=['blue', 'red'], alpha=0.7)
        ax3.set_ylabel('Performance Drop (%)')
        ax3.set_title('Performance Degradation vs Original')
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom' if height > 0 else 'top')
    
    # Plot 4: AU Frequency Heatmap
    ax4 = plt.subplot(2, 3, 4)
    heatmap_data = []
    for au in AU_LABELS:
        row = []
        for filter_type in ['original', 'low', 'high']:
            if filter_type in df['filter_type'].unique():
                df_filter = df[df['filter_type'] == filter_type]
                y_true = df_filter[f'AU{au}_true'].values
                y_pred = df_filter[f'AU{au}_pred'].values
                tp = ((y_true == 1) & (y_pred == 1)).sum()
                fp = ((y_true == 0) & (y_pred == 1)).sum()
                fn = ((y_true == 1) & (y_pred == 0)).sum()
                f1 = 2*tp / (2*tp + fp + fn + 1e-6)
                row.append(f1)
        heatmap_data.append(row)
    
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=['Original', 'Low', 'High'],
                yticklabels=[f'AU{au}' for au in AU_LABELS],
                ax=ax4, cbar_kws={'label': 'F1-Score'})
    ax4.set_title('AU Performance Heatmap')
    
    # Plot 5: Sample Distribution
    ax5 = plt.subplot(2, 3, 5)
    filter_counts = df['filter_type'].value_counts()
    ax5.pie(filter_counts.values, labels=filter_counts.index, autopct='%1.1f%%')
    ax5.set_title(f'Processed Samples Distribution\n(Total: {len(df)})')
    
    # Plot 6: Accuracy Distribution
    ax6 = plt.subplot(2, 3, 6)
    for filter_type in df['filter_type'].unique():
        df_filter = df[df['filter_type'] == filter_type]
        ax6.hist(df_filter['accuracy'], bins=20, alpha=0.5, label=filter_type)
    ax6.set_xlabel('Accuracy')
    ax6.set_ylabel('Frequency')
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
    # Load model once
    model = load_model(MODEL_PATH)
    
    # Process each test set
    for test_set in TEST_SETS:
        print(f"\n{'='*70}")
        print(f"PROCESSING {test_set.upper()}")
        print(f"{'='*70}")
        
        # Load dataset
        dataset = load_dataset(test_set, num_samples=NUM_SAMPLES)
        print(f"Loaded {len(dataset)} samples")
        
        # Process with each filter
        all_results = []
        for filter_type in FILTERS:
            results = process_dataset(model, dataset, test_set, filter_type)
            all_results.extend(results)
        
        # Save results
        df = save_results_to_csv(all_results, f'{test_set}_results.csv')
        
        # Compute and display metrics
        metrics = compute_metrics(df)
        print(f"\n📈 Metrics Summary:")
        for filter_type, metric_data in metrics.items():
            print(f"   {filter_type:10s}: Accuracy = {metric_data['mean_accuracy']:.4f}")
        
        # Visualize
        visualize_results(df, test_set)
    
    print(f"\n{'='*70}")
    print("✅ Batch Analysis Complete!")
    print(f"{'='*70}")
    print(f"Results saved to: {OUTPUT_DIR.absolute()}")

if __name__ == "__main__":
    main()
