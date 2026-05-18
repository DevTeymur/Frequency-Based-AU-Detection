# Frequency-Based AU Detection

This repository contains the code used for BP4D facial action unit (AU) experiments with frequency-domain masking. It combines a fine-tuned FMAE / IAT backbone, HFSS-style frequency shortcut search, low-frequency masking during training, and identity probing to study how much subject information remains in the model.

## Repository Layout

The repository root is organized by function rather than by experiment stage:

- `batch_analysis.py`, `predict.py`, `stats.py`, and `visualizations.py` are utility and analysis scripts.
- `train_lowfreq_fmae.py`, `eval_lowfreq_masked_checkpoints.py`, `probe_lowfreq_checkpoints.py`, and `linear_probe_identity.py` cover training and evaluation workflows.
- `hfss_search_au.py` and `hfss_search_id.py` implement the frequency-shortcut search experiments.
- `scripts/` contains small helper entry points that support the main workflows.
- `results/`, `logs/`, `saved_masks/`, `models/`, and `All results/` are output or artifact directories.
- `notes/` stores working notes and result logs.
- `latex/` contains manuscript-related assets.
- `BP4D/` is the local dataset checkout and is ignored by Git.
- `hfss/` and `fmae/` are separate nested repositories and are intentionally left untouched in this cleanup branch.

## Dataset Layout

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

## Common Workflows

### Train a low-frequency masked model

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

### Search for frequency shortcuts with HFSS

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

### Run identity probing

```bash
python linear_probe_identity.py \
  --model_path models/FMAE_IAT_BP4D_fold1.pth \
  --train_json BP4D/BP4D_train1.json \
  --test_json BP4D/BP4D_test1.json \
  --dfm_dir hfss/DFM
```

This trains a frozen-backbone linear probe to predict subject IDs and evaluates performance under frequency masking.

### Inspect one image manually

```bash
python predict.py
```

This script loads one sample, applies a low-pass or high-pass filter, and prints the predicted AUs.

### Generate plots and diagnostics

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

## Related Subprojects

- `fmae/README.md` documents the underlying masked autoencoder implementation.
- `hfss/README.md` documents the frequency-shortcut search project this repo adapts.
