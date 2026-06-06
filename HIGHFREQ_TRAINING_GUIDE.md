# High-Frequency vs Low-Frequency Training Guide

## Overview

This update adds support for **frequency band selection** during training, allowing you to:
- Train on **low frequencies** (center of image, blurry) - original behavior
- Train on **high frequencies** (edges/details, outer region) - new behavior
- Train on **full image** (no masking)

This enables your supervisor's experiment: testing whether high frequencies alone can support AU detection and whether they encode subject identity information.

## Physics/Intuition

### Frequency Masking in Images

In the frequency domain (after FFT):
- **Center (low frequencies)**: Global structure, lighting, blur, slow variations
- **Outer (high frequencies)**: Edges, details, fine textures, rapid variations

### Radial Mask Types

**Low-Pass (keep center)**:
```
keep_ratio = 8-20%  →  Keep inner 8-20% of frequency spectrum  →  Blurry image
```

**High-Pass (keep outer)** - NEW:
```
keep_ratio = 15-100%  →  Remove inner 15%, keep outer 85-100%  →  Edge-emphasized image
```

## API Changes

### `train_lowfreq_fmae.py`

**New Argument:**
```bash
--mask_type {low,high,full}
  - low   : keep center frequencies (default, original behavior)
  - high  : keep outer frequencies (NEW)
  - full  : no masking
```

**Updated Function:**
```python
apply_random_frequency_mask(
    images,
    min_keep: float,
    max_keep: float,
    mask_type: str = "low"  # NEW parameter
) -> torch.Tensor
```

### `eval_lowfreq_masked_checkpoints.py`

Same `--mask_type` argument added for evaluation.

## Usage Examples

### 1. Train FMAE with High Frequencies

```bash
python train_lowfreq_fmae.py \
    --model_type FMAE \
    --model_path models/FMAE_BP4D_fold1.pth \
    --train_json BP4D/BP4D_train1.json \
    --test_json BP4D/BP4D_test1.json \
    --mask_type high \
    --min_keep 0.15 \
    --max_keep 1.0 \
    --epochs 30 \
    --batch_size 64 \
    --blr 5e-4 \
    --output_dir output/highfreq_fmae_fold1 \
    --fold 1
```

Parameters explained:
- `--mask_type high`: Remove center frequencies, keep outer
- `--min_keep 0.15`: Remove inner 15% of frequency spectrum
- `--max_keep 1.0`: Keep all the way to outer edge

### 2. Train IAT (Identity-Aware) with High Frequencies

```bash
python train_lowfreq_fmae.py \
    --model_type IAT \
    --model_path models/FMAE_IAT_BP4D_fold1.pth \
    --train_json BP4D/BP4D_train1.json \
    --test_json BP4D/BP4D_test1.json \
    --mask_type high \
    --min_keep 0.15 \
    --max_keep 1.0 \
    --epochs 30 \
    --batch_size 64 \
    --blr 5e-4 \
    --output_dir output/highfreq_iat_fold1 \
    --fold 1
```

### 3. Evaluate High-Frequency Checkpoints on AU Task

```bash
python eval_lowfreq_masked_checkpoints.py \
    --checkpoint_dir output/highfreq_fmae_fold1 \
    --checkpoint_glob 'fold1_epoch*.pth' \
    --model_type FMAE \
    --mask_type high \
    --min_keep 0.15 \
    --max_keep 1.0 \
    --test_json BP4D/BP4D_test1.json \
    --output_csv output/highfreq_au_results.csv
```

### 4. Test for ID Information in High-Frequency Features

```bash
python probe_lowfreq_checkpoints.py \
    --model_type FMAE \
    --checkpoint_paths \
        output/highfreq_fmae_fold1/fold1_epoch030.pth \
        models/FMAE_BP4D_fold1.pth \
    --checkpoint_labels epoch30_highfreq baseline_fullfreq \
    --train_json BP4D/BP4D_train1.json \
    --test_json BP4D/BP4D_test1.json \
    --output_csv output/highfreq_identity_probe_results.csv
```

This trains a linear probe for **subject identity** on frozen high-frequency FMAE backbone. If ID accuracy is high, high frequencies encode subject information.

## Supported Mask Types

| Type | Behavior | Use Case |
|------|----------|----------|
| `low` | Keep center, remove outer | Original low-frequency training |
| `high` | Remove center, keep outer | Your experiment: test high frequencies |
| `full` | No masking | Baseline: full image |

## Expected Behavior

### Low-Frequency Training (Original)
```
min_keep=0.08, max_keep=0.20, mask_type=low
→ Images appear blurry, global structure only
→ AU detection still works → AUs encoded in low frequencies
```

### High-Frequency Training (New)
```
min_keep=0.15, max_keep=1.0, mask_type=high
→ Images appear edge-emphasized, detail-only
→ Can AU detection still work? → Test your hypothesis
→ Does identity probe find ID? → If yes, ID in high frequencies
```

## Experiment Workflow

Run the complete experiment:
```bash
bash test_highfreq_training.sh
```

This will:
1. Train FMAE with high frequencies (epoch 30)
2. Train IAT with high frequencies (epoch 30)
3. Evaluate AU macro F1 on high-frequency test set
4. Run identity probe on high-frequency backbone
5. Generate results CSV files

## Output Files

After training:
- `output/highfreq_fmae_fold1/` - FMAE checkpoints
- `output/highfreq_fmae_fold1/fold1_fmae_metrics.csv` - Training metrics
- `output/highfreq_au_macro_f1_results.csv` - AU evaluation results
- `output/highfreq_identity_probe_results.csv` - ID probe results

## Comparison: What to Look For

Compare your results:

```
Low-Frequency AU Accuracy:  ✓ High (expected)
High-Frequency AU Accuracy: ? Unknown (hypothesis test)

Low-Frequency ID Accuracy:  ✓ High (expected)
High-Frequency ID Accuracy: ? Unknown (high-freq ID content?)
```

If high-frequency AU accuracy is low, it confirms **AU information is mostly in low frequencies**.

If high-frequency ID accuracy is high, it means **ID information is distributed across frequencies**, not just low frequencies.

## Backward Compatibility

- Existing code using `apply_random_lowpass_mask()` still works (wrapper function)
- Default `--mask_type low` preserves original behavior
- No breaking changes to existing scripts

## Notes

- Testing should use the **same masking type as training** for consistency
- You can mix: train on high-freq, test on low-freq (or vice versa) to study transfer
- For most AU detection tasks, stick with `min_keep=0.08, max_keep=0.20` or `min_keep=0.15, max_keep=1.0`
