# Frequency-Based AU Detection

This repository contains the code used for BP4D facial action unit (AU) experiments with frequency-domain masking. It combines a fine-tuned FMAE / IAT backbone, HFSS-style frequency shortcut search, low-frequency masking during training, and identity probing to study how much subject information remains in the model.

## What is in this repo

- `train_lowfreq_fmae.py` trains FMAE / IAT with random low-frequency masking on BP4D.
- `hfss_search_au.py` searches for frequency masks that change AU performance.
- `linear_probe_identity.py` measures how much subject identity is still recoverable from frozen features.
- `predict.py` runs a small single-image AU prediction demo with frequency filtering.
- `batch_analysis.py` applies frequency filters to many samples and saves aggregate metrics.
- `visualizations.py` creates plots and visual summaries for saved masks and AU drops.
- `stats.py` reports AU label distributions and mask diagnostics.
- `fmae/` contains the masked autoencoder backbone and finetuning code.
- `hfss/` contains the HFSS mask-search utilities and supporting code.

## Dataset layout

The code expects BP4D to be organized like this:

```text
BP4D/
  BP4D_train1.json
  BP4D_train2.json
  BP4D_train3.json
  BP4D_test1.json
  BP4D_test2.json
  BP4D_test3.json
  BP4D_cropped/
```

The default image root used by most scripts is `BP4D/BP4D_cropped/`.

## Installation

The repository is built around the FMAE codebase and older PyTorch/timm versions for compatibility.

```bash
pip install -r requirements.txt
```

If you already have a working environment, make sure the main dependencies are available: PyTorch, torchvision, timm, numpy, pandas, scikit-learn, matplotlib, seaborn, Pillow, and tqdm.

## Common workflows

### 1. Train a low-frequency masked model

```bash
python train_lowfreq_fmae.py \
  --model_path models/FMAE_ViT_base.pth \
  --train_json BP4D/BP4D_train1.json \
  --test_json BP4D/BP4D_test1.json \
  --output_dir output/lowfreq_fmae \
  --fold 1
```

Useful options:

- `--model_type FMAE|IAT`
- `--min_keep` and `--max_keep` control the low-frequency keep ratio.
- `--num_samples` lets you run on a smaller subset for debugging.

### 2. Search for frequency shortcuts with HFSS

```bash
python hfss_search_au.py \
  --model_path models/FMAE_BP4D_fold1.pth \
  --test_json BP4D/BP4D_test1.json \
  --stages stage1 stage2 stage3 \
  --search_mode random
```

Useful options:

- `--per_au_eval` to print per-AU drops.
- `--per_au_search` to search for a target AU.
- `--search_mode radial` for deterministic concentric masks.
- `--reuse_saved_stages` to continue from previously saved masks.

### 3. Run identity probing

```bash
python linear_probe_identity.py \
  --model_path models/FMAE_IAT_BP4D_fold1.pth \
  --train_json BP4D/BP4D_train1.json \
  --test_json BP4D/BP4D_test1.json \
  --dfm_dir hfss/DFM
```

This trains a frozen-backbone linear probe to predict subject IDs and evaluates performance under frequency masking.

### 4. Inspect one image manually

```bash
python predict.py
```

This script loads one sample, applies a low-pass or high-pass filter, and prints the predicted AUs.

### 5. Generate plots and diagnostics

```bash
python stats.py --fold_json BP4D/BP4D_test1.json
python visualizations.py --mask_pkl hfss/DFM/AU01/FMAE_stage3_AU01_DFMs.pkl
python batch_analysis.py
```

## Outputs

The repo writes results to a few main places:

- `models/` for trained checkpoints.
- `results/` for CSV outputs, metrics, and figures.
- `hfss/DFM/` for saved masks and identity probing results.
- `hfss/figures/` for HFSS visualizations.
- `hfss/logs/` for timestamped search logs.

## Notes

- The main AU labels used throughout the project are `AU01, AU02, AU04, AU06, AU07, AU10, AU12, AU14, AU15, AU17, AU23, AU24`.
- Most scripts default to GPU execution but fall back to CPU when needed.
- The code assumes BP4D JSON files store one sample per line in JSON format.

## Related subprojects

- `fmae/README.md` documents the underlying masked autoencoder implementation.
- `hfss/README.md` documents the frequency-shortcut search project this repo adapts.
