#!/usr/bin/env python3
"""
Continue fine-tuning existing FMAE or FMAE-IAT BP4D models
"""

import os
import sys
from pathlib import Path

# ============================================================================
# CONFIGURATION - EDIT THESE VARIABLES
# ============================================================================

# Path to the model you want to fine-tune
MODEL_PATH = "models/FMAE_BP4D_fold1.pth"

# Which fold (1, 2, or 3)
FOLD = 1

# Model type: 'FMAE' or 'IAT' (auto-detected if None)
MODEL_TYPE = None  # Will auto-detect from filename

# Device: 'auto', 'cuda', 'mps', or 'cpu'
DEVICE = 'auto'

# ============================================================================

def check_requirements(model_path, fold):
    """Check if required files exist for training"""
    errors = []
    
    # Check model checkpoint
    if not Path(model_path).exists():
        errors.append(f"❌ Model not found: {model_path}")
    
    # Check dataset
    dataset_path = Path("BP4D/BP4D_cropped")
    if not dataset_path.exists():
        errors.append(f"❌ Dataset not found: {dataset_path}")
    
    # Check JSON files for specified fold
    train_json = Path(f"BP4D/BP4D_train{fold}.json")
    test_json = Path(f"BP4D/BP4D_test{fold}.json")
    if not train_json.exists():
        errors.append(f"❌ Training JSON not found: {train_json}")
    if not test_json.exists():
        errors.append(f"❌ Test JSON not found: {test_json}")
    
    # Check FMAE code
    fmae_script = Path("fmae/BP4D_finetune.py")
    if not fmae_script.exists():
        errors.append(f"❌ FMAE training script not found: {fmae_script}")
    
    return errors

def train_bp4d(model_path, fold, model_type=None, device='auto'):
    """Continue fine-tuning an existing BP4D model"""
    
    # Auto-detect model type from filename if not specified
    if model_type is None:
        model_name = Path(model_path).name.lower()
        if 'iat' in model_name:
            model_type = 'IAT'
        else:
            model_type = 'FMAE'
    
    print(f"\n{'='*76}")
    print(f"🚀 Continue fine-tuning {model_type} on BP4D - Fold {fold}")
    print(f"📁 Starting from: {model_path}")
    print(f"{'='*76}\n")
    
    # Check requirements
    print("🔍 Checking requirements...\n")
    errors = check_requirements(model_path, fold)
    if errors:
        print("❌ ERRORS:\n")
        for error in errors:
            print(error)
        print()
        return False
    
    print("✅ All requirements satisfied!\n")
    
    # Auto-detect device
    if device == 'auto':
        import torch
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    print(f"🖥️  Device: {device}\n")
    
    # Set parameters
    if model_type == 'FMAE':
        grad_reverse = 0  # No identity adversarial
        blr = 0.0002
        epochs = 20
        warmup = 2
    else:  # IAT
        grad_reverse = 2  # Identity adversarial lambda=2
        blr = 0.0005
        epochs = 30
        warmup = 3
    
    batch_size = 64
    seed = 0
    
    # Paths
    root_path = "BP4D/BP4D_cropped"
    train_path = f"BP4D/BP4D_train{fold}.json"
    test_path = f"BP4D/BP4D_test{fold}.json"
    output_dir = f"./training_output/{model_type}_fold{fold}_continued"
    
    # Build command - use the provided model as starting point
    cmd = f"""python fmae/BP4D_finetune.py \\
    --seed {seed} \\
    --blr {blr} \\
    --batch_size {batch_size} \\
    --epochs {epochs} \\
    --warmup_epochs {warmup} \\
    --nb_classes 12 \\
    --model vit_large_patch16 \\
    --finetune {model_path} \\
    --root_path {root_path} \\
    --train_path {train_path} \\
    --test_path {test_path} \\
    --output_dir {output_dir} \\
    --log_dir {output_dir} \\
    --device {device}
"""
    
    print("📋 Training configuration:")
    print(f"   • Model: {model_type}")
    print(f"   • Fold: {fold}")
    print(f"   • Epochs: {epochs}")
    print(f"   • Batch size: {batch_size}")
    print(f"   • Learning rate: {blr}")
    print(f"   • Identity adversarial: {'Yes (λ=2)' if grad_reverse > 0 else 'No'}")
    print()
    
    # Execute
    print("🏃 Starting training...\n")
    result = os.system(cmd)
    
    if result != 0:
        print(f"\n❌ Training failed!")
        return False
    
    # Copy checkpoint to models folder with "_v2" suffix
    print(f"\n✅ Training completed!\n")
    src = f"{output_dir}/checkpoint-best.pth"
    original_name = Path(model_path).stem  # e.g., "FMAE_BP4D_fold1"
    dst = f"models/{original_name}_v2.pth"
    
    if os.path.exists(src):
        import shutil
        os.makedirs("models", exist_ok=True)
        shutil.copy(src, dst)
        print(f"📦 Updated model saved to: {dst}")
        print(f"📦 Original model preserved at: {model_path}")
        print(f"{'='*76}\n")
        return True
    else:
        print(f"⚠️  Checkpoint not found at: {src}")
        return False

def main():
    """Run the fine-tuning with the configuration from the top of the file"""
    train_bp4d(
        model_path=MODEL_PATH,
        fold=FOLD,
        model_type=MODEL_TYPE,
        device=DEVICE
    )

if __name__ == "__main__":
    main()
