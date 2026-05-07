# Section 6.2: Preprocessing and Experimental Setup — Detailed Information

## Overview
This document contains all the preprocessing and experimental setup details extracted from your codebase for your thesis Section 6.2.

---

## 1. DATA LOADING AND PREPROCESSING PIPELINE

### 1.1 Image Format and Storage
- **Format:** JPEG images
- **Stored at:** `BP4D/BP4D_cropped/` (face-cropped version)
- **Loading method:** PIL Image.open() with RGB conversion
- **Total samples in dataset:** 439,962 frames from 41 subjects × 8 tasks

### 1.2 Image Resolution
- **Final input size to model:** 224 × 224 pixels (fixed)
- **Pre-stored image size:** Also 224 × 224 (already preprocessed)
- **Original video resolution before preprocessing:** Downsampled by 2× from raw BP4D videos
  - Extracted from original .avi files with specified AU-labeled frames
  - cv2.resize applied: `new_width = width // 2`, `new_height = height // 2`

### 1.3 Face Detection and Alignment
- **Detection method:** Not applied in your pipeline
- **Alignment:** Not applied in your pipeline
- **Cropping:** BP4D dataset provides pre-cropped face images; your preprocessing uses these directly
  - No additional face detection/alignment needed—images are already face-localized
  - Images contain full frontal face region

---

## 2. NORMALIZATION

### 2.1 Pixel Value Normalization
**Method:** Standard ImageNet normalization (applied via `transforms.Normalize`)

```
Mean (μ): (0.485, 0.456, 0.406)  # RGB channels
Std (σ):  (0.229, 0.224, 0.225)  # RGB channels
```

**Pipeline:**
1. Load image as uint8 RGB [0, 255]
2. ToTensor() → float32 [0.0, 1.0]
3. Normalize using above mean/std

### 2.2 Resizing Strategy
**Training transform:**
- Uses `create_transform()` from timm
- Applies RandomResizedCrop with aspect ratio preservation
- Interpolation: bicubic

**Evaluation transform:**
- Direct resize: `transforms.Resize([224, 224])`
- No aspect ratio preservation (direct scaling)
- Interpolation: bicubic (from create_transform default)

---

## 3. TRAIN/TEST SPLIT — CROSS-SUBJECT PROTOCOL

### 3.1 Split Strategy
- **Protocol:** Strict cross-subject (no subject overlap between train/test)
- **Number of folds:** 3-fold cross-validation
- **Total subjects:** 41 (23 female + 18 male)

### 3.2 Fold Details

#### **Fold 1 (train1 / test1)**
- **Train subjects:** 27 subjects
  - Female: F03, F04, F05, F06, F07, F11, F12, F13, F14, F15, F17, F19, F20, F21, F22 (15 total)
  - Male: M02, M03, M05, M06, M09, M10, M11, M13, M15, M16, M17, M18 (12 total)
  - Training samples: 96,049 frames

- **Test subjects:** 14 subjects
  - Female: F01, F02, F08, F09, F10, F16, F18, F23 (8 total)
  - Male: M01, M04, M07, M08, M12, M14 (6 total)
  - Test samples: 50,605 frames

#### **Fold 2 (train2 / test2)**
- **Train subjects:** Similar 27/14 split
- **Training samples:** 96,472 frames
- **Test samples:** 50,182 frames

#### **Fold 3 (train3 / test3)**
- **Train subjects:** Similar 27/14 split
- **Training samples:** 100,787 frames
- **Test samples:** 45,867 frames

### 3.3 Subject Distribution
- **Approximate ratio:** 2:1 (train:test)
- **Subjects per fold:**
  - Train: ~27 subjects (66%)
  - Test: ~14 subjects (34%)

---

## 4. ACTION UNIT (AU) SUBSET SELECTION

### 4.1 AU Set Used
Your experiments use a subset of **12 Action Units** from BP4D's 27 available AUs:

```
AU indices: [1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24]
```

### 4.2 AU Definitions (FACS)
| AU | Name |
|----|----|
| 1  | Inner Brow Raiser |
| 2  | Outer Brow Raiser |
| 4  | Brow Lowerer |
| 6  | Cheek Raiser |
| 7  | Lid Tightener |
| 10 | Upper Lip Raiser |
| 12 | Lip Corner Puller |
| 14 | Dimpler |
| 15 | Lip Corner Depressor |
| 17 | Chin Raiser |
| 23 | Lip Tightener |
| 24 | Lip Pressor |

### 4.3 Label Format
- **Storage:** JSON format (one sample per line)
- **Example entry:**
  ```json
  {"img_path": "2F15_T8/frame_1.jpg", "AUs": [999, 999, 4, 999, 999, 10, 999, 999, 999, 999, 999, 999]}
  ```
- **Label encoding:** 
  - 1 = AU present
  - 999 = AU absent (no AU)
  - Labels are in order of AU indices [1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24]

---

## 5. DATA AUGMENTATION

### 5.1 Training Augmentation
Applied via `timm.data.create_transform()`:

| Augmentation | Configuration |
|---|---|
| **AutoAugment Policy** | rand-m9-mstd0.5-inc1 |
| **Random Erase Probability** | 0.25 |
| **Random Erase Mode** | pixel |
| **Color Jitter** | Disabled (None) |
| **Mixup Alpha** | 0 (Disabled) |
| **CutMix Alpha** | 0 (Disabled) |

**What this means:**
- Automatic augmentation with moderate intensity
- 25% of training samples get random pixel erasure
- No color jittering, mixup, or cutmix in your setup

### 5.2 Evaluation (No Augmentation)
- Direct resize to 224×224
- No augmentation applied
- Single deterministic transform

---

## 6. BATCH SIZE AND TRAINING HYPERPARAMETERS

### 6.1 Batch Sizes
| Stage | Batch Size | Source |
|---|---|---|
| **AU Detection Training** | 32 | BP4D_finetune.py |
| **AU Detection Evaluation** | 32 | BP4D_finetune.py |
| **HFSS Search (AU)** | 24 | hfss_search_au.py |

### 6.2 Other Training Parameters
| Parameter | Value |
|---|---|
| Epochs | 20 (for AU fine-tuning) |
| Learning Rate (base) | 5e-4 |
| Learning Rate (actual) | Computed as: base_lr × batch_size / 256 |
| Warmup Epochs | 2 |
| Weight Decay | 0.05 |
| Layer Decay | 0.65 (AU), 0.75 (general) |
| Drop Path Rate | 0.1 |

---

## 7. DATA PIPELINE FOR YOUR EXPERIMENTS

### 7.1 FMAE (AU Detection) Pipeline
```
BP4D JSON (train/test split)
    ↓
BP4D_AU_dataset class loads:
  - Image path from JSON
  - AU binary labels (12 AUs)
  - Subject ID (extracted from path)
    ↓
Transform pipeline:
  - Training: RandomResizedCrop → AutoAugment → RandomErase → ToTensor → Normalize
  - Evaluation: Resize(224,224) → ToTensor → Normalize
    ↓
Model input: (batch_size, 3, 224, 224)
Model output: AU logits (batch_size, 12)
```

### 7.2 HFSS Search (AU/ID) Pipeline
Reuses the same BP4D_AU_dataset, applies frequency masks in FFT domain:
```
Image (224×224, RGB)
    ↓
FFT transformation
    ↓
Frequency mask sampling (various frequencies tested)
    ↓
Inverse FFT
    ↓
Model evaluation with masked input
```

### 7.3 Batch Analysis Pipeline
Configuration in batch_analysis.py:
- Processes 224×224 face images
- Applies optional frequency masks
- Tests on specified folds (default: test2)
- Generates predictions and performance metrics

---

## 8. SUBJECT IDENTITY (ID) LABELS

### 8.1 Identity Classes
- **Number of identities in BP4D:** 41 subjects
- **Female subjects:** 23 (F01–F23)
- **Male subjects:** 18 (M01–M18)
- **Identity extraction:** Automatically from image filename
  - Format: `2F15_T8/frame_1.jpg` → subject = `F15` (converted to `F15` from `2F15`)
  - Encoded as one-hot vector of length 41

---

## 9. IMPORTANT NOTES FOR YOUR PROPOSAL

### Key Points to Include:
1. **Cross-subject protocol ensures no subject leakage** between train and test
2. **BP4D dataset is pre-cropped** (no face detection step required)
3. **12 AU subset** is standard in the AU detection literature
4. **224×224 resolution** is industry standard (ViT patch size: 16×16 → 14×14 patches)
5. **Normalization uses ImageNet statistics** (standard for pre-trained models)
6. **Augmentation is minimal** (no mixup/cutmix) to preserve AU signal integrity

### Connection to HFSS Experiments:
- **hfss_search_au.py** uses identical data loading (`BP4D_AU_dataset`)
- **Same train/test splits** (fold 1, 2, 3)
- **Same batch size** in evaluation loop
- **Frequency masks tested in FFT domain**, not pixel space
- **Identity experiment (hfss_search_id.py)** uses IAT checkpoint (FMAE with ID head)

---

## 10. REPOSITORY LOCATIONS

| Component | Path |
|---|---|
| Data loader | `fmae/util/datasets.py` |
| AU detection training | `fmae/BP4D_finetune.py` |
| HFSS AU search | `hfss_search_au.py` |
| HFSS ID search | `hfss_search_id.py` |
| Batch analysis | `batch_analysis.py` |
| BP4D preprocessed data | `BP4D/BP4D_cropped/` |
| Train/test splits | `BP4D/BP4D_train{1,2,3}.json`, `BP4D/BP4D_test{1,2,3}.json` |

---

## Summary Table for Section 6.2

| Aspect | Details |
|---|---|
| **Dataset** | BP4D (face-cropped, 224×224) |
| **Total subjects** | 41 (23F, 18M) |
| **Total frames** | 439,962 |
| **Image resolution** | 224 × 224 pixels |
| **Image format** | JPEG, RGB |
| **AUs used** | 12 (IDs: 1,2,4,6,7,10,12,14,15,17,23,24) |
| **Protocol** | 3-fold cross-subject validation |
| **Train/test ratio** | ~66% train / 34% test (no subject overlap) |
| **Normalization** | ImageNet (μ=[0.485,0.456,0.406], σ=[0.229,0.224,0.225]) |
| **Augmentation** | AutoAugment (rand-m9-mstd0.5-inc1) + RandomErase (p=0.25) |
| **Batch size** | 32 (training), 24 (HFSS search) |
| **Processing** | Pre-cropped faces → Direct resize → Normalize → Model |

