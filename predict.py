"""
Predict AU labels with HFSS frequency filtering
Load image → Apply HFSS → Predict with model → Compare results
"""

import json
import torch
import torch.fft as fft
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import timm
from warnings import filterwarnings
filterwarnings("ignore")

# Import frequency filtering functions from hfss.py
from hfss import apply_frequency_mask, create_low_freq_mask, create_high_freq_mask

# ============================================================================
# CONFIGURATION
# ============================================================================
IMG_SIZE = 224
DATA_ROOT = Path("BP4D/BP4D_cropped")
JSON_FILE = Path("BP4D/BP4D_test1.json")
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

print("🎯 AU Prediction with HFSS")
print("="*70)
print(f"Device: {DEVICE}")

# ============================================================================
# 1. LOAD DATA
# ============================================================================
def load_dataset(json_file):
    """Load image paths and labels from JSON"""
    data = []
    with open(json_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def load_and_preprocess_image(img_path):
    """
    Load image and preprocess for model
    Returns: tensor [3, 224, 224] normalized for model
    """
    full_path = DATA_ROOT / img_path
    img = Image.open(full_path).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    
    # Convert to tensor and normalize (ImageNet stats)
    img_array = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
    
    # Normalize with ImageNet mean/std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_normalized = (img_tensor - mean) / std
    
    return img_normalized

# ============================================================================
# 3. LOAD MODEL
# ============================================================================
def load_model(model_path):
    """Load pretrained FMAE ViT model using timm"""
    print(f"\n🤖 Loading model from {model_path}...")
    
    # Create ViT-Base model using timm (same architecture as FMAE)
    model = timm.create_model(
        'vit_base_patch16_224',
        pretrained=False,  # We'll load our own weights
        num_classes=12,    # 12 AU classes
        global_pool='avg'
    )
    
    # Load pretrained weights
    checkpoint = torch.load(model_path, map_location='cpu')
    
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Load state dict (ignore head if size mismatch)
    msg = model.load_state_dict(state_dict, strict=False)
    # print(f"   Loaded: {msg}")
    
    model = model.to(DEVICE)
    model.eval()
    
    print(f"   ✓ Model ready")
    return model

# ============================================================================
# 4. PREDICT
# ============================================================================
@torch.no_grad()
def predict_au(model, image_tensor):
    """
    Predict AU labels for one image
    
    Args:
        model: Loaded model
        image_tensor: [3, 224, 224] preprocessed image
    
    Returns:
        probabilities: [12] AU probabilities
        predictions: [12] binary predictions (0 or 1)
    """
    # Add batch dimension [1, 3, 224, 224]
    image_batch = image_tensor.unsqueeze(0).to(DEVICE)
    
    # Forward pass
    outputs = model(image_batch)
    
    # Convert to probabilities
    probs = torch.sigmoid(outputs).squeeze(0).cpu()
    
    # Binary predictions (threshold = 0.5)
    preds = (probs >= 0.5).float()
    
    return probs, preds

# ============================================================================
# 5. EVALUATE ONE IMAGE
# ============================================================================
def predict_single_image(model, sample, filter_type='none'):
    """
    Predict AUs for a single image
    
    Args:
        model: Loaded model
        sample: Dictionary with 'img_path' and 'AUs'
        filter_type: 'none', 'low', or 'high'
    
    Returns:
        Dictionary with results
    """
    img_path = sample['img_path']
    true_aus = sample['AUs']
    
    print(f"\n📸 Image: {img_path}")
    print(f"   True AUs: {true_aus}")
    
    # Load image
    img_tensor = load_and_preprocess_image(img_path)
    
    # Apply frequency filter
    if filter_type == 'low':
        mask = create_low_freq_mask(IMG_SIZE, cutoff=0.3)
        img_tensor = apply_frequency_mask(img_tensor, mask)
        print(f"   Applied: Low frequency filter")
    elif filter_type == 'high':
        mask = create_high_freq_mask(IMG_SIZE, cutoff=0.3)
        img_tensor = apply_frequency_mask(img_tensor, mask)
        print(f"   Applied: High frequency filter")
    else:
        print(f"   Applied: No filter (original)")
    
    # Predict
    probs, preds = predict_au(model, img_tensor)
    
    # Get predicted AUs
    predicted_au_indices = torch.where(preds == 1)[0].tolist()
    predicted_aus = [AU_LABELS[i] for i in predicted_au_indices]
    
    print(f"   Predicted AUs: {predicted_aus}")
    
    # Create true label vector
    true_label = torch.zeros(12)
    for au in true_aus:
        if au in AU_LABELS:
            true_label[AU_LABELS.index(au)] = 1
    
    # Compute accuracy for this image
    correct = (preds == true_label).float()
    accuracy = correct.mean().item()
    
    print(f"   Accuracy: {accuracy*100:.1f}%")
    
    return {
        'img_path': img_path,
        'true_aus': true_aus,
        'predicted_aus': predicted_aus,
        'probabilities': probs.numpy(),
        'predictions': preds.numpy(),
        'true_labels': true_label.numpy(),
        'accuracy': accuracy,
        'filter_type': filter_type
    }

# ============================================================================
# 6. VISUALIZE RESULTS
# ============================================================================
def visualize_prediction(sample, results_dict):
    """
    Visualize predictions with different filters
    
    Args:
        sample: Original sample data
        results_dict: Dict of {filter_name: results}
    """
    print(f"\n📊 Creating visualization...")
    
    num_filters = len(results_dict)
    fig, axes = plt.subplots(2, num_filters, figsize=(5*num_filters, 8))
    
    if num_filters == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, (filter_name, result) in enumerate(results_dict.items()):
        # Load and display image
        img_path = DATA_ROOT / result['img_path']
        img = Image.open(img_path)
        
        # Top: Image
        axes[0, idx].imshow(img)
        axes[0, idx].set_title(f"{filter_name}\nImage", fontsize=10)
        axes[0, idx].axis('off')
        
        # Bottom: Predictions bar chart
        ax = axes[1, idx]
        x = np.arange(12)
        
        # Plot true vs predicted
        width = 0.35
        ax.bar(x - width/2, result['true_labels'], width, label='True', alpha=0.7)
        ax.bar(x + width/2, result['predictions'], width, label='Predicted', alpha=0.7)
        
        ax.set_ylabel('Label (0 or 1)')
        ax.set_xlabel('Action Unit')
        ax.set_title(f"{filter_name}\nAccuracy: {result['accuracy']*100:.1f}%", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([f'AU{au}' for au in AU_LABELS], rotation=45, fontsize=8)
        ax.legend()
        ax.set_ylim([0, 1.2])
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / 'prediction_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved to {output_file}")
    plt.close()

# ============================================================================
# MAIN
# ============================================================================
def main():
    # Load dataset
    print("\n📊 Loading dataset...")
    dataset = load_dataset(JSON_FILE)
    print(f"   Loaded {len(dataset)} samples")
    
    # Pick one sample to test
    sample = dataset[0]  # First image
    
    # Load model
    model = load_model(MODEL_PATH)
    
    # Test with different filters
    print("\n" + "="*70)
    print("TESTING PREDICTIONS")
    print("="*70)
    
    results = {}
    
    # Original (no filter)
    results['Original'] = predict_single_image(model, sample, filter_type='none')
    
    # Low frequency only
    results['Low Freq'] = predict_single_image(model, sample, filter_type='low')
    
    # High frequency only
    results['High Freq'] = predict_single_image(model, sample, filter_type='high')
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Image: {sample['img_path']}")
    print(f"True AUs: {sample['AUs']}")
    print()
    for filter_name, result in results.items():
        print(f"{filter_name:12s}: Predicted {result['predicted_aus']} | Accuracy: {result['accuracy']*100:.1f}%")
    
    # Visualize
    visualize_prediction(sample, results)
    
    print("\n✅ Done!")

if __name__ == "__main__":
    main()
