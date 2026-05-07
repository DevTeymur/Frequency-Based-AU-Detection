"""
HFSS Search for AU Detection
Finds frequency shortcuts in AU detection models
"""

import sys
import pickle
import argparse
import time
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from sklearn.metrics import f1_score

import torch
from torch.utils.data import DataLoader, Subset
from types import SimpleNamespace
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

inference_decorator = torch.inference_mode if hasattr(torch, 'inference_mode') else torch.no_grad

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
DEFAULT_KEEP_RANGES = {
    'stage1': (0.60, 0.80),
    'stage2': (0.40, 0.60),
    'stage3': (0.20, 0.40),
    'stage4': (0.15, 0.30),
    'stage5': (0.08, 0.20),
    'stage6': (0.03, 0.10),
}

# Fixed runtime defaults (kept out of CLI to reduce argument noise).
FIXED_DATA_ROOT = 'BP4D/BP4D_cropped/'
FIXED_TOP_N = 10
FIXED_PROPORTION = 0.8
FIXED_STOP_DROP_PCT = 5.0
FIXED_F1_TOLERANCE_PCT = 1.0
FIXED_BATCH_SIZE = 24
FIXED_NUM_WORKERS = 4
FIXED_DEVICE = 'cuda'
FIXED_SEED = 42
FIXED_RADIAL_STEPS = 10
FIXED_RADIAL_START_KEEP_PCT =100.0
FIXED_RADIAL_DIRECTION = 'small_to_big'  # 'big_to_small' | 'small_to_big'


class TeeStream:
    """Write stream output to multiple targets (terminal + log file)."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def create_run_log_file(log_dir):
    """Create timestamped log file path for current run."""
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"hfss_run_{ts}.txt"


def parse_keep_ranges(config_str):
    """Parse keep ranges: 'stage1:0.6-0.8,stage2:0.4-0.6,...'."""
    ranges = dict(DEFAULT_KEEP_RANGES)
    if not config_str:
        return ranges
    items = [x.strip() for x in config_str.split(',') if x.strip()]
    for item in items:
        if ':' not in item:
            continue
        stage, val = item.split(':', 1)
        stage = stage.strip()
        if '-' not in val:
            continue
        lo, hi = val.split('-', 1)
        lo = float(lo)
        hi = float(hi)
        if lo > hi:
            lo, hi = hi, lo
        ranges[stage] = (max(0.0, lo), min(1.0, hi))
    return ranges


def load_model(model_path, model_type='FMAE', device='cuda'):
    """Load AU detection model
    
    Handles mixed checkpoint formats:
    - FMAE checkpoints work with FMAE model_type
    - IAT checkpoints work with IAT model_type
    - FMAE checkpoints can be loaded into IAT model (ID_head uninitialized)
    """
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
    
    try:
        # Try strict loading first (all keys must match exactly)
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        if 'ID_head' in str(e) and model_type == 'IAT':
            # Loading FMAE checkpoint into IAT model: missing ID_head is expected
            print(f"⚠ ID_head parameters not found in checkpoint (using FMAE weights for IAT model).")
            print(f"  Loading with strict=False. ID_head will use random initialization.")
            model.load_state_dict(state_dict, strict=False)
        else:
            # Other loading errors should be raised
            raise
    
    model = model.to(device).eval()
    
    print(f"✓ Loaded {model_type} model from {model_path}")
    return model


def set_global_seed(seed, device='cuda'):
    """Set RNG seeds for reproducible mask sampling and subset selection."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if str(device).startswith('cuda') and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Best-effort deterministic behavior for reproducible evaluation runs.
    if hasattr(torch, 'backends') and hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def generate_mask_candidates(num_candidates, proportion, stage, prev_masks=None, keep_ratio_range=None):
    """Generate random frequency mask candidates.

    Paper-aligned behavior:
    - Stage1: sample on full grid.
        - Stage>1: sample ONLY inside white/active regions of a selected parent
            mask from previous stage top-N, then multiply (black stays black).
        - Stage1 includes shifted-window variants (patch, patch+1, mixed),
            similar to original HFSS candidate generation.
    """
    # Each candidate becomes a 224x224 binary-like frequency mask (float array),
    # later wrapped by White_Mask for downstream use.
    candidates = []
    # stage1->4, stage2->8, stage3->16, ... controls grid resolution.
    # The generated integer mask lives on (patch x patch) first, then upsampled.
    patch = PATCHES[stage]
    parent_masks = prev_masks if prev_masks else None

    def _make_symmetric(mask_arr):
        max_n_h = IMG_SIZE // 2
        max_n_w = IMG_SIZE // 2
        for h_index in range(-max_n_h, 1):
            for w_index in range(-max_n_w, max_n_w):
                h_matrix_index = IMG_SIZE // 2 + h_index
                w_matrix_index = IMG_SIZE // 2 + w_index
                if h_index != 0:
                    mask_arr[IMG_SIZE - h_matrix_index - 1, IMG_SIZE - w_matrix_index - 1] = mask_arr[h_matrix_index, w_matrix_index]
        return mask_arr

    def _active_map_from_parent(parent_mask, grid_n):
        """Downsample parent full-res mask to grid cells (active/inactive)."""
        cell = IMG_SIZE // grid_n
        parent_bin = (parent_mask > 0.5).astype(np.float32)
        # max-pool over each grid cell: 1 means this cell is still active
        small = parent_bin.reshape(grid_n, cell, grid_n, cell).max(axis=(1, 3))
        return small

    def _sample_mask_int(grid_n, p, parent_mask=None, keep_range=None):
        freqs = gen_freqs_list(grid_n, grid_n)
        if parent_mask is None:
            # Stage1-style sampling on full grid.
            if keep_range is not None and len(freqs) > 0:
                lo, hi = keep_range
                target_ratio = np.random.uniform(lo, hi)
                k = int(round(target_ratio * len(freqs)))
                k = max(1, min(len(freqs), k))
                chosen_idx = np.random.choice(len(freqs), size=k, replace=False)
                chosen = [freqs[i] for i in chosen_idx]
            else:
                chosen = sample_frequency(p, freqs)
            return generate_mask(chosen, grid_n, grid_n), len(chosen), len(freqs)

        # Stage>1-style sampling: only cells still white in the parent are eligible.
        active_map = _active_map_from_parent(parent_mask, grid_n)
        active_freqs = [f for f in freqs if active_map[f[0], f[1]] > 0]

        if not active_freqs:
            return torch.zeros((grid_n, grid_n)), 0, 0

        if keep_range is not None:
            lo, hi = keep_range
            target_ratio = np.random.uniform(lo, hi)
            k = int(round(target_ratio * len(active_freqs)))
            k = max(1, min(len(active_freqs), k))
            chosen_idx = np.random.choice(len(active_freqs), size=k, replace=False)
            chosen = [active_freqs[i] for i in chosen_idx]
        else:
            chosen = sample_frequency(p, active_freqs)
        m = generate_mask(chosen, grid_n, grid_n)
        # ensure inactive parent cells can never turn on
        m = m * torch.tensor(active_map, dtype=m.dtype)
        return m, len(chosen), len(active_freqs)

    while len(candidates) < num_candidates:
        parent_mask = None
        if parent_masks:
            # Parent chosen uniformly from previous stage top-N masks.
            parent_mask = parent_masks[np.random.randint(len(parent_masks))]

        # Variant A: patch grid
        mask_int, chosen_count, eligible_count = _sample_mask_int(
            patch, proportion, parent_mask, keep_ratio_range
        )
        mask_a = np.kron(np.asarray(mask_int), np.ones((IMG_SIZE // patch, IMG_SIZE // patch)))

        if parent_mask is not None:
            # Permanent pruning rule: if parent is black at a frequency, child stays black.
            mask_a = parent_mask * mask_a
        mask_a = _make_symmetric(mask_a)
        t_a = White_Mask(mask_a)
        t_a.white_count = int(np.sum(mask_a > 0.5))
        t_a.keep_pct = float(chosen_count / eligible_count) if eligible_count > 0 else 0.0
        candidates.append(t_a)
        if len(candidates) >= num_candidates:
            break

        # For stage>1, only subdivide and sample inside parent white regions.
        # Do not run shifted/global variants to avoid resampling outside parent.
        if parent_mask is not None:
            continue

        # Variant B: shifted window (patch+1), if possible
        if patch < IMG_SIZE:
            patch_b = patch + 1
            mask_int_b, chosen_b, eligible_b = _sample_mask_int(
                patch_b, proportion, None, keep_ratio_range
            )
            step = patch_b - 1
            mask_b = np.kron(np.asarray(mask_int_b), np.ones((IMG_SIZE // step, IMG_SIZE // step)))
            crop = int(IMG_SIZE // step / 2)
            if crop > 0:
                mask_b = mask_b[crop:-crop, crop:-crop]

            if parent_mask is not None:
                mask_b = parent_mask * mask_b
            mask_b = _make_symmetric(mask_b)
            t_b = White_Mask(mask_b)
            t_b.white_count = int(np.sum(mask_b > 0.5))
            t_b.keep_pct = float(chosen_b / eligible_b) if eligible_b > 0 else 0.0
            candidates.append(t_b)
            if len(candidates) >= num_candidates:
                break

            # Variant C: mixed patch and shifted patch
            mask_c1, chosen_c1, eligible_c1 = _sample_mask_int(
                patch, proportion / 2.0, None, keep_ratio_range
            )
            mask_c1 = np.kron(np.asarray(mask_c1), np.ones((IMG_SIZE // patch, IMG_SIZE // patch)))

            mask_c2, chosen_c2, eligible_c2 = _sample_mask_int(
                patch_b, proportion / 2.0, None, keep_ratio_range
            )
            mask_c2 = np.kron(np.asarray(mask_c2), np.ones((IMG_SIZE // step, IMG_SIZE // step)))
            if crop > 0:
                mask_c2 = mask_c2[crop:-crop, crop:-crop]

            # Mixed variant combines two sampled masks then clips to {0,1}-like range.
            mask_c = np.clip(mask_c1 + mask_c2, 0, 1)
            if parent_mask is not None:
                mask_c = parent_mask * mask_c
            mask_c = _make_symmetric(mask_c)
            t_c = White_Mask(mask_c)
            t_c.white_count = int(np.sum(mask_c > 0.5))
            elig = eligible_c1 + eligible_c2
            t_c.keep_pct = float((chosen_c1 + chosen_c2) / elig) if elig > 0 else 0.0
            candidates.append(t_c)

    return candidates[:num_candidates]


def _sync_if_cuda(device, enabled):
    """Synchronize CUDA only when profiling is enabled."""
    if enabled and str(device).startswith('cuda') and torch.cuda.is_available():
        torch.cuda.synchronize()


def _timing_add(timing_state, key, value):
    """Accumulate numeric timing values."""
    if timing_state is None:
        return
    timing_state[key] = timing_state.get(key, 0.0) + float(value)


@inference_decorator()
def evaluate_mask(model, dataloader, mask_transform, device, timing_state=None):
    """Evaluate AU performance with frequency mask applied (mean F1 over AUs).

    This accumulates predictions across all batches then computes the macro
    average F1 across AU labels (equivalent to mean per-AU F1).
    """
    all_preds = []
    all_labels = []

    mask_t = None
    if hasattr(mask_transform, 'mask'):
        mask_np = mask_transform.mask
        if getattr(mask_transform, 'flip', False):
            mask_np = 1 - mask_np
        # mask_t shape: [1, 1, H, W] (here H=W=224).
        # It is broadcast over batch and channel dims when multiplying frequency tensor.
        mask_t = torch.as_tensor(mask_np, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    profile = timing_state is not None
    eval_t0 = time.perf_counter() if profile else None
    if profile:
        timing_state['calls'] = timing_state.get('calls', 0) + 1
        wait_t0 = time.perf_counter()

    for images, (au_labels, _) in dataloader:
        if profile:
            now = time.perf_counter()
            _timing_add(timing_state, 'data_wait_s', now - wait_t0)

            _sync_if_cuda(device, True)
            t_h2d = time.perf_counter()
        images = images.to(device, non_blocking=True)
        au_labels = au_labels.to(device, non_blocking=True)
        if profile:
            _sync_if_cuda(device, True)
            _timing_add(timing_state, 'h2d_s', time.perf_counter() - t_h2d)
            timing_state['batches'] = timing_state.get('batches', 0) + 1

        # Apply frequency mask in Fourier domain.
        # images shape is [B, C, H, W]; FFT is applied per sample and per channel on H,W.
        # FFT values are complex64/complex128 depending on input dtype.
        if mask_t is not None:
            if profile:
                _sync_if_cuda(device, True)
                t_fft = time.perf_counter()
            freq = torch.fft.fftshift(torch.fft.fft2(images, dim=(-2, -1)), dim=(-2, -1))
            if profile:
                _sync_if_cuda(device, True)
                _timing_add(timing_state, 'fft_s', time.perf_counter() - t_fft)

                _sync_if_cuda(device, True)
                t_mask = time.perf_counter()
            # Element-wise complex multiplication by a real mask.
            # Same spatial mask is applied to every channel.
            freq = freq * mask_t
            if profile:
                _sync_if_cuda(device, True)
                _timing_add(timing_state, 'mask_mul_s', time.perf_counter() - t_mask)

                _sync_if_cuda(device, True)
                t_ifft = time.perf_counter()
            # Inverse FFT back to spatial domain, then keep only real part.
            # Output returns to float tensor shape [B, C, H, W].
            images = torch.fft.ifft2(torch.fft.ifftshift(freq, dim=(-2, -1)), dim=(-2, -1)).real.float()
            if profile:
                _sync_if_cuda(device, True)
                _timing_add(timing_state, 'ifft_s', time.perf_counter() - t_ifft)

        # Forward pass
        if profile:
            _sync_if_cuda(device, True)
            t_infer = time.perf_counter()
        outputs = model(images)
        if profile:
            _sync_if_cuda(device, True)
            _timing_add(timing_state, 'infer_s', time.perf_counter() - t_infer)

            t_post = time.perf_counter()
        if isinstance(outputs, tuple):
            outputs = outputs[0]  # Take AU head only

        # Multi-label AU logits -> sigmoid probabilities -> threshold 0.5 per AU.
        preds = (torch.sigmoid(outputs) >= 0.5).float()

        all_preds.append(preds.cpu().numpy())
        all_labels.append(au_labels.cpu().numpy())

        if profile:
            _timing_add(timing_state, 'post_s', time.perf_counter() - t_post)
            wait_t0 = time.perf_counter()

    if not all_preds:
        return 0.0

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Macro F1 over 12 AU columns (each AU equally weighted).
    # all_labels/all_preds shape: [N, 12].
    f1 = f1_score(all_labels, all_preds, average='macro')
    if profile:
        _timing_add(timing_state, 'eval_total_s', time.perf_counter() - eval_t0)
    return float(f1)
    # return accuracy


# ============================================================
# OPTIONAL FUNCTION 1: Per-AU F1 evaluation
# Enable with --per_au_eval flag
# ============================================================
@inference_decorator()
def evaluate_mask_per_au(model, dataloader, mask_transform, baseline_per_au, device, timing_state=None):
    """Evaluate per-AU F1 and show drop vs baseline for each AU.
    Returns dict: {AU_label: {'f1': float, 'drop': float}}
    """
    all_preds = []
    all_labels = []

    mask_t = None
    if hasattr(mask_transform, 'mask'):
        mask_np = mask_transform.mask
        if getattr(mask_transform, 'flip', False):
            mask_np = 1 - mask_np
        # Same broadcast mask convention as evaluate_mask: [1,1,H,W].
        mask_t = torch.as_tensor(mask_np, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    profile = timing_state is not None
    eval_t0 = time.perf_counter() if profile else None
    if profile:
        timing_state['calls'] = timing_state.get('calls', 0) + 1
        wait_t0 = time.perf_counter()

    for images, (au_labels, _) in dataloader:
        if profile:
            now = time.perf_counter()
            _timing_add(timing_state, 'data_wait_s', now - wait_t0)

            _sync_if_cuda(device, True)
            t_h2d = time.perf_counter()
        images = images.to(device, non_blocking=True)
        au_labels = au_labels.to(device, non_blocking=True)
        if profile:
            _sync_if_cuda(device, True)
            _timing_add(timing_state, 'h2d_s', time.perf_counter() - t_h2d)
            timing_state['batches'] = timing_state.get('batches', 0) + 1

        if mask_t is not None:
            if profile:
                _sync_if_cuda(device, True)
                t_fft = time.perf_counter()
            freq = torch.fft.fftshift(torch.fft.fft2(images, dim=(-2, -1)), dim=(-2, -1))
            if profile:
                _sync_if_cuda(device, True)
                _timing_add(timing_state, 'fft_s', time.perf_counter() - t_fft)

                _sync_if_cuda(device, True)
                t_mask = time.perf_counter()
            # Per-channel complex frequency masking.
            freq = freq * mask_t
            if profile:
                _sync_if_cuda(device, True)
                _timing_add(timing_state, 'mask_mul_s', time.perf_counter() - t_mask)

                _sync_if_cuda(device, True)
                t_ifft = time.perf_counter()
            # Keep real part after inverse FFT reconstruction.
            images = torch.fft.ifft2(torch.fft.ifftshift(freq, dim=(-2, -1)), dim=(-2, -1)).real.float()
            if profile:
                _sync_if_cuda(device, True)
                _timing_add(timing_state, 'ifft_s', time.perf_counter() - t_ifft)

        if profile:
            _sync_if_cuda(device, True)
            t_infer = time.perf_counter()
        outputs = model(images)
        if profile:
            _sync_if_cuda(device, True)
            _timing_add(timing_state, 'infer_s', time.perf_counter() - t_infer)

            t_post = time.perf_counter()
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        all_preds.append(preds.cpu().numpy())
        all_labels.append(au_labels.cpu().numpy())

        if profile:
            _timing_add(timing_state, 'post_s', time.perf_counter() - t_post)
            wait_t0 = time.perf_counter()

    if not all_preds:
        return {}

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    per_au = {}
    # Per-AU binary F1 is computed independently for each AU column.
    for i, au in enumerate(AU_LABELS):
        au_f1 = f1_score(all_labels[:, i], all_preds[:, i], average='binary', zero_division=0)
        baseline_entry = baseline_per_au.get(au, 0.0) if baseline_per_au else 0.0
        baseline = baseline_entry.get('f1', 0.0) if isinstance(baseline_entry, dict) else baseline_entry
        drop = baseline - au_f1
        per_au[au] = {'f1': au_f1, 'drop': drop}
    if profile:
        _timing_add(timing_state, 'eval_total_s', time.perf_counter() - eval_t0)
    return per_au


def print_per_au_results(per_au, stage, mask_id):
    """Pretty-print per-AU F1 drops for a given mask."""
    print(f"   Per-AU breakdown — {stage} mask{mask_id}:")
    for au, vals in per_au.items():
        bar = '▓' * int(abs(vals['drop']) * 100 / 5)  # 1 block per 5% drop
        print(f"     AU{au:02d}: F1={vals['f1']*100:.1f}%  drop={vals['drop']*100:+.1f}%  {bar}")


@inference_decorator()
def evaluate_mask_identity(model, dataloader, mask_transform, device, model_type='FMAE', timing_state=None):
    """Evaluate subject ID classification accuracy under frequency mask.
    
    For IAT model: applies mask in Fourier domain, runs inference, extracts output[1] (ID head logits),
    computes classification accuracy against ground truth subject IDs.
    
    For FMAE model: returns empty dict (no ID head).
    
    Args:
        model: Vision Transformer (FMAE or IAT)
        dataloader: DataLoader yielding (image, (AU_labels, ID_labels))
        mask_transform: White_Mask object with .mask attribute
        device: CUDA or CPU
        model_type: 'FMAE' or 'IAT'
        timing_state: Optional dict to accumulate timing stats
    
    Returns:
        dict with 'accuracy' key, or empty dict for FMAE
    """
    if model_type != 'IAT':
        return {}
    
    all_preds = []
    all_labels = []

    # Extract mask and convert to device tensor (same pattern as evaluate_mask_per_au)
    mask_t = None
    if hasattr(mask_transform, 'mask'):
        mask_np = mask_transform.mask
        if getattr(mask_transform, 'flip', False):
            mask_np = 1 - mask_np
        # Broadcast mask: [1, 1, H, W] for per-channel multiplication
        mask_t = torch.as_tensor(mask_np, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    profile = timing_state is not None
    eval_t0 = time.perf_counter() if profile else None
    if profile:
        timing_state['calls'] = timing_state.get('calls', 0) + 1
        wait_t0 = time.perf_counter()

    for images, (au_labels, id_labels_onehot) in dataloader:
        if profile:
            now = time.perf_counter()
            _timing_add(timing_state, 'data_wait_s', now - wait_t0)
            _sync_if_cuda(device, True)
            t_h2d = time.perf_counter()
        
        images = images.to(device, non_blocking=True)
        id_labels_onehot = id_labels_onehot.to(device, non_blocking=True)
        id_labels = torch.argmax(id_labels_onehot, dim=1)  # Convert one-hot to class indices
        
        if profile:
            _sync_if_cuda(device, True)
            _timing_add(timing_state, 'h2d_s', time.perf_counter() - t_h2d)
            timing_state['batches'] = timing_state.get('batches', 0) + 1

        # Apply frequency mask in Fourier domain
        if mask_t is not None:
            if profile:
                _sync_if_cuda(device, True)
                t_fft = time.perf_counter()
            freq = torch.fft.fftshift(torch.fft.fft2(images, dim=(-2, -1)), dim=(-2, -1))
            if profile:
                _sync_if_cuda(device, True)
                _timing_add(timing_state, 'fft_s', time.perf_counter() - t_fft)

                _sync_if_cuda(device, True)
                t_mask = time.perf_counter()
            # Per-channel complex frequency masking
            freq = freq * mask_t
            if profile:
                _sync_if_cuda(device, True)
                _timing_add(timing_state, 'mask_mul_s', time.perf_counter() - t_mask)

                _sync_if_cuda(device, True)
                t_ifft = time.perf_counter()
            # Reconstruct image from masked frequency (keep real part)
            images = torch.fft.ifft2(torch.fft.ifftshift(freq, dim=(-2, -1)), dim=(-2, -1)).real.float()
            if profile:
                _sync_if_cuda(device, True)
                _timing_add(timing_state, 'ifft_s', time.perf_counter() - t_ifft)

        if profile:
            _sync_if_cuda(device, True)
            t_infer = time.perf_counter()
        
        # Run inference on masked images
        with torch.no_grad():
            output = model(images)
        
        if profile:
            _sync_if_cuda(device, True)
            _timing_add(timing_state, 'infer_s', time.perf_counter() - t_infer)
            t_post = time.perf_counter()

        # Extract ID head output (output[1] for IAT)
        id_logits = output[1]  # (batch_size, 41)
        id_preds = torch.argmax(id_logits, dim=1)
        
        all_preds.append(id_preds.cpu().numpy())
        all_labels.append(id_labels.cpu().numpy())
        
        if profile:
            _timing_add(timing_state, 'post_s', time.perf_counter() - t_post)
            wait_t0 = time.perf_counter()

    if not all_preds:
        return {}

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Compute classification accuracy
    accuracy = np.mean(all_preds == all_labels)
    
    if profile:
        _timing_add(timing_state, 'eval_total_s', time.perf_counter() - eval_t0)
    
    return {'accuracy': accuracy}


def save_per_au_grouped_bar(baseline_per_au, masked_per_au, stage, model_type, save_dir):
    """Save grouped bar chart: baseline vs masked per-AU F1 for one stage."""
    aus = AU_LABELS
    baseline_vals = []
    masked_vals = []

    for au in aus:
        b = baseline_per_au.get(au, 0.0) if baseline_per_au else 0.0
        b = b.get('f1', 0.0) if isinstance(b, dict) else b
        m = masked_per_au.get(au, {}) if masked_per_au else {}
        m = m.get('f1', 0.0) if isinstance(m, dict) else m
        baseline_vals.append(float(b))
        masked_vals.append(float(m))

    x = np.arange(len(aus))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, baseline_vals, width=width, label='Baseline F1', color='#4C78A8')
    ax.bar(x + width / 2, masked_vals, width=width, label='Masked F1', color='#F58518')

    ax.set_title(f"Per-AU F1 | {model_type} | {stage} (best mask)")
    ax.set_xlabel('Action Unit (AU)')
    ax.set_ylabel('F1 score')
    ax.set_xticks(x)
    ax.set_xticklabels([f"AU{au}" for au in aus], rotation=0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis='y', alpha=0.2)
    ax.legend(loc='best')

    plt.tight_layout()
    save_path = save_dir / f"{model_type}_{stage}_per_au_grouped_bar.png"
    plt.savefig(save_path, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"   📊 Saved per-AU grouped bar to {save_path}")


def validate_target_au(target_au):
    """Validate requested AU id."""
    if target_au is None:
        return None
    if target_au not in AU_LABELS:
        raise ValueError(f"Invalid --target_au={target_au}. Valid AUs: {AU_LABELS}")
    return target_au


def load_masks_from_pkl(pkl_path):
    """Load mask objects from saved PKL and return a normalized list."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in PKL, got {type(data)}")
    return data


def save_per_au_stage_heatmap(per_au_drop_by_stage, model_type, save_dir):
    """Save AU x stage heatmap where value is per-AU F1 drop (baseline - masked)."""
    if not per_au_drop_by_stage:
        return

    stages = list(per_au_drop_by_stage.keys())
    aus = AU_LABELS

    heat = np.array([
        [float(per_au_drop_by_stage.get(stage, {}).get(au, 0.0)) for stage in stages]
        for au in aus
    ])

    vmax = np.max(np.abs(heat)) if heat.size > 0 else 1.0
    vmax = max(vmax, 1e-6)

    fig, ax = plt.subplots(figsize=(1.8 * len(stages) + 4, 6))
    im = ax.imshow(heat, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')

    ax.set_title(f"Per-AU F1 Drop Heatmap | {model_type}")
    ax.set_xlabel('Stage')
    ax.set_ylabel('Action Unit (AU)')
    ax.set_xticks(np.arange(len(stages)))
    ax.set_xticklabels(stages)
    ax.set_yticks(np.arange(len(aus)))
    ax.set_yticklabels([f"AU{au}" for au in aus])

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('F1 drop (baseline - masked)')

    plt.tight_layout()
    save_path = save_dir / f"{model_type}_per_au_drop_heatmap.png"
    plt.savefig(save_path, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"   🌡️ Saved per-AU heatmap to {save_path}")


# ============================================================
# OPTIONAL FUNCTION 2: Advanced frequency mask visualizations
# Enable with --visualize_masks flag
# Shows: raw mask, ring overlay, frequency band, masked FFT
# ============================================================
def visualize_mask_advanced(mask_array, save_path, stage, mask_id, sample_image=None):
    """Visualize a frequency mask in 4 views:
    1. Raw mask (white=keep, black=remove)
    2. Ring/radial distance overlay
    3. Frequency band annotation (low/mid/high)
    4. Masked FFT magnitude (if sample_image provided)
    """
    size = mask_array.shape[0]
    center = size // 2

    # --- Build radial distance map ---
    ys, xs = np.ogrid[:size, :size]
    radial = np.sqrt((xs - center) ** 2 + (ys - center) ** 2)
    max_r = np.sqrt(2) * center

    # --- Frequency band labels (low < 20%, mid 20-50%, high > 50%) ---
    band = np.zeros((size, size), dtype=int)  # 0=low, 1=mid, 2=high
    band[radial > max_r * 0.5] = 2
    band[(radial >= max_r * 0.2) & (radial <= max_r * 0.5)] = 1

    fig, axes = plt.subplots(1, 4 if sample_image is not None else 3, figsize=(16, 4))
    fig.suptitle(f"{stage} | mask{mask_id}", fontsize=12)

    # Panel 1: Raw mask
    axes[0].imshow(mask_array, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Mask\n(white=keep)')
    axes[0].axis('off')

    # Panel 2: Ring overlay
    ring_vis = plt.cm.RdYlGn(mask_array)  # green=keep, red=remove
    axes[1].imshow(ring_vis)
    for r_frac in [0.2, 0.5]:
        circle = plt.Circle((center, center), max_r * r_frac,
                             color='white', fill=False, linewidth=1.5, linestyle='--')
        axes[1].add_patch(circle)
    axes[1].set_title('Ring overlay\n(dashed=band boundaries)')
    axes[1].axis('off')

    # Panel 3: Frequency band annotation
    masked_band = np.ma.masked_where(mask_array == 0, band.astype(float))
    axes[2].imshow(band, cmap='coolwarm', alpha=0.3, vmin=0, vmax=2)
    axes[2].imshow(masked_band, cmap='coolwarm', vmin=0, vmax=2)
    for label, y_frac in [('LOW', 0.85), ('MID', 0.55), ('HIGH', 0.1)]:
        axes[2].text(size * 0.02, size * y_frac, label, color='white',
                     fontsize=9, fontweight='bold')
    axes[2].set_title('Freq bands\n(low/mid/high)')
    axes[2].axis('off')

    # Panel 4: Masked FFT of sample image (optional)
    if sample_image is not None:
        img_t = torch.tensor(sample_image, dtype=torch.float32)
        if img_t.ndim == 3:
            img_t = img_t.mean(0)  # grayscale for FFT vis
        freq = torch.fft.fftshift(torch.fft.fft2(img_t))
        freq_masked = freq * torch.tensor(mask_array, dtype=torch.complex64)
        magnitude = torch.log(torch.abs(freq_masked) + 1).numpy()
        axes[3].imshow(magnitude, cmap='inferno')
        axes[3].set_title('Masked FFT\n(log magnitude)')
        axes[3].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()


def search_stage(
    model,
    dataloader,
    stage,
    num_candidates,
    proportion,
    prev_masks,
    device,
    top_n=10,
    keep_ratio_range=None,
    f1_tolerance_pct=1.0,
    baseline_f1=None,
    objective_mode='macro',
    target_au=None,
    baseline_per_au=None,
    profile_timing=False,
    precomputed_candidates=None,
):
    """Search for frequency shortcuts in one stage"""
    stage_t0 = time.perf_counter()
    timing_state = {} if profile_timing else None
    objective_label = 'mean F1 (macro)' if objective_mode == 'macro' else f'AU{target_au:02d} F1'
    print(
        f"\n🔍 Stage {stage}: Testing {num_candidates} candidates (P={proportion}) "
        f"| objective={objective_label}"
    )
    
    # Generate candidates (or reuse provided candidates)
    if precomputed_candidates is None:
        gen_t0 = time.perf_counter()
        candidates = generate_mask_candidates(
            num_candidates, proportion, stage, prev_masks, keep_ratio_range
        )
        gen_dt = time.perf_counter() - gen_t0
    else:
        candidates = precomputed_candidates
        gen_dt = 0.0
    
    # Test baseline objective (no mask), can be precomputed once
    if objective_mode == 'macro':
        if baseline_f1 is None:
            baseline_f1 = evaluate_mask(model, dataloader, lambda x: x, device, timing_state=timing_state)
        baseline_obj = baseline_f1
        print(f"   Baseline mean F1: {baseline_obj*100:.2f}% (mean per-AU)")
    else:
        if target_au is None:
            raise ValueError("target_au must be provided when objective_mode='per_au'")
        if baseline_per_au is None:
            baseline_per_au = evaluate_mask_per_au(model, dataloader, lambda x: x, {}, device, timing_state=timing_state)
        baseline_obj = float(baseline_per_au.get(target_au, {}).get('f1', 0.0))
        print(f"   Baseline AU{target_au:02d} F1: {baseline_obj*100:.2f}%")
    
    # Evaluate every candidate on the selected objective:
    # - macro mode: objective is mean F1 across all AUs
    # - per_au mode: objective is F1 of target AU only
    results = []
    for i, mask_transform in enumerate(tqdm(candidates, desc=f"   Testing masks")):
        if objective_mode == 'macro':
            f1 = evaluate_mask(model, dataloader, mask_transform, device, timing_state=timing_state)
        else:
            per_au = evaluate_mask_per_au(
                model, dataloader, mask_transform, baseline_per_au, device, timing_state=timing_state
            )
            f1 = float(per_au.get(target_au, {}).get('f1', 0.0))
        drop = baseline_obj - f1
        white_count = int(getattr(mask_transform, 'white_count', np.sum(mask_transform.mask > 0.5)))
        keep_pct = float(getattr(mask_transform, 'keep_pct', np.mean(mask_transform.mask > 0.5)))
        results.append((i, f1, drop, white_count, keep_pct, mask_transform))
    
    # "Best mask" is defined as the candidate with highest objective F1.
    # Sorting descending by objective F1 establishes ranking.
    results.sort(key=lambda x: x[1], reverse=True)
    top_k = results[:top_n]

    best_f1 = results[0][1] if results else 0.0
    best_drop_pct = (baseline_obj - best_f1) * 100.0

    # "Smallest mask within tolerance" means:
    # drop <= tolerance, then minimize white pixel count (most compact mask).
    viable = [r for r in results if (baseline_obj - r[1]) * 100.0 <= f1_tolerance_pct]
    smallest_viable = min(viable, key=lambda x: x[3]) if viable else None

    # Print ALL mask drops for this stage (one line each)
    print(f"\n   {stage} — all masks (sorted by masked F1, high→low):")
    for r in results:
        mask_id, f1_val, drop_val, white_count, keep_pct, _ = r
        marker = " ◀ best" if mask_id == results[0][0] else ""
        print(
            f"     mask{mask_id:>3}: F1={f1_val*100:.2f}%  |  "
            f"drop={drop_val*100:+.2f}%  |  white={white_count}  "
            f"| keep={keep_pct*100:.1f}%{marker}"
        )

    if smallest_viable is not None:
        sid, sf1, sdrop, swhite, skeep, _ = smallest_viable
        print(
            f"   Smallest mask within {f1_tolerance_pct:.1f}% of baseline: "
            f"mask{sid} | F1={sf1*100:.2f}% | drop={sdrop*100:+.2f}% "
            f"| white={swhite} | keep={skeep*100:.1f}%"
        )
    else:
        print(f"   No mask is within {f1_tolerance_pct:.1f}% of baseline.")

    # top_k masks are passed to next stage as parent candidates.
    summary = {
        'baseline_f1': baseline_obj,
        'best_f1': best_f1,
        'best_drop_pct': best_drop_pct,
        'smallest_viable': smallest_viable,
        'objective': objective_label,
    }

    if profile_timing and timing_state is not None:
        stage_total = time.perf_counter() - stage_t0
        eval_total = timing_state.get('eval_total_s', 0.0)
        calls = int(timing_state.get('calls', 0))
        batches = int(timing_state.get('batches', 0))
        print(f"\n   ⏱ Timing summary ({stage}, objective={objective_label}):")
        print(
            f"     stage_total={stage_total:.2f}s | mask_generation={gen_dt:.2f}s | "
            f"eval_total={eval_total:.2f}s | eval_calls={calls} | eval_batches={batches}"
        )
        for key, label in [
            ('data_wait_s', 'data_wait'),
            ('h2d_s', 'h2d_transfer'),
            ('fft_s', 'fft'),
            ('mask_mul_s', 'mask_apply'),
            ('ifft_s', 'ifft'),
            ('infer_s', 'model_infer'),
            ('post_s', 'postprocess'),
        ]:
            val = timing_state.get(key, 0.0)
            pct = (val / eval_total * 100.0) if eval_total > 0 else 0.0
            print(f"     - {label:<13}: {val:8.2f}s ({pct:5.1f}%)")

    return [r[5] for r in top_k], results, summary


def build_eval_dataloader(test_json, data_root, num_samples, batch_size, num_workers, device):
    """Shared data loading path used by random and radial search modes."""
    dataset_args = SimpleNamespace(
        root_path=data_root,
        input_size=IMG_SIZE,
        color_jitter=None,
        aa='rand-m9-mstd0.5-inc1',
        reprob=0.25,
        remode='pixel',
        recount=1,
    )
    dataset = BP4D_AU_dataset(test_json, is_train=False, args=dataset_args)
    if num_samples and num_samples < len(dataset):
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        dataset = Subset(dataset, indices)

    pin_mem = str(device).startswith('cuda')
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=(num_workers > 0),
    )
    return dataset, dataloader


def generate_radial_mask_candidates(
    num_steps,
    mode='low-pass',
    start_keep_pct=100.0,
    direction='big_to_small',
):
    """Create deterministic radial mask candidates wrapped as White_Mask objects."""
    if num_steps < 2:
        num_steps = 2

    if direction not in ('big_to_small', 'small_to_big'):
        raise ValueError(
            f"Invalid radial direction: {direction}. "
            "Use 'big_to_small' or 'small_to_big'."
        )

    h = IMG_SIZE
    w = IMG_SIZE
    cy, cx = h // 2, w // 2
    ys, xs = np.ogrid[:h, :w]
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)

    max_radius = float(dist.max())
    if mode == 'low-pass':
        start_keep_pct = float(np.clip(start_keep_pct, 0.1, 100.0))
        end_keep_pct = max(start_keep_pct / float(num_steps), 0.1)
        target_keep_pcts = np.linspace(start_keep_pct, end_keep_pct, num_steps)
        if direction == 'small_to_big':
            target_keep_pcts = target_keep_pcts[::-1]
        radii = [float(np.quantile(dist, kp / 100.0)) for kp in target_keep_pcts]
    else:
        max_remove_radius = min(h, w) / 2.0
        radii = np.linspace(max_remove_radius, 0.0, num_steps)

    masks = []
    for idx, r in enumerate(radii):
        if mode == 'low-pass':
            mask = (dist <= r).astype(np.float32)
        else:
            mask = (dist >= r).astype(np.float32)
        wm = White_Mask(mask)
        wm.white_count = int(np.sum(mask > 0.5))
        wm.keep_pct = float(np.mean(mask > 0.5))
        if mode == 'low-pass':
            wm.target_keep_pct = float(target_keep_pcts[idx])
        masks.append(wm)
    return masks


def run_random_hfss_search(
    args,
    model,
    dataloader,
    output_dir,
    figures_dir,
    baseline_f1_global,
    baseline_per_au_full,
    keep_ranges,
):
    """Current hierarchical random-mask HFSS search (preserved logic)."""
    all_results = {}
    if args.per_au_search:
        target_aus = [args.target_au] if args.target_au is not None else list(AU_LABELS)
        print(f"\n🎯 Per-AU search mode ON | targets: {target_aus}")
    else:
        target_aus = [None]

    # Shared candidate cache: in per-AU search, each stage's candidate list is
    # generated once (for the first AU encounter) and then reused for all AUs.
    stage_masks = {}

    for target_au in target_aus:
        prev_masks = None
        per_au_drop_by_stage = {}
        target_suffix = f"_AU{target_au:02d}" if target_au is not None else ""
        target_name = f"AU{target_au:02d}" if target_au is not None else "macro"
        target_output_dir = output_dir / target_name if target_au is not None else output_dir
        target_figures_dir = figures_dir / target_name if target_au is not None else figures_dir
        target_output_dir.mkdir(parents=True, exist_ok=True)
        target_figures_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*50}")
        print(f"Search target: {target_name}")
        print(f"{'='*50}")

        # Optional bootstrap: per-AU stage2+ can start from macro stage1 masks.
        if (
            args.per_au_search
            and args.bootstrap_from_macro_stage1
            and target_au is not None
            and 'stage1' not in args.stages
            and prev_masks is None
        ):
            macro_stage1_file = output_dir / f"{args.model_type}_stage1_DFMs.pkl"
            if macro_stage1_file.exists():
                try:
                    boot_masks = load_masks_from_pkl(macro_stage1_file)
                    prev_masks = [m.mask for m in boot_masks if hasattr(m, 'mask')]
                    print(
                        f"   ↪ Bootstrapped parents from macro stage1: {macro_stage1_file} "
                        f"(count={len(prev_masks)})"
                    )
                except Exception as e:
                    print(f"   ⚠ Failed to bootstrap from {macro_stage1_file}: {e}")
            else:
                print(f"   ⚠ Macro stage1 PKL not found for bootstrap: {macro_stage1_file}")

        all_results[target_name] = {}

        for stage in args.stages:
            stage_keep = keep_ranges.get(stage, DEFAULT_KEEP_RANGES.get(stage, (0.2, 0.4)))
            print(
                f"   Keep-ratio target for {stage}: "
                f"{stage_keep[0]*100:.1f}% - {stage_keep[1]*100:.1f}%"
            )

            output_file = target_output_dir / f"{args.model_type}_{stage}{target_suffix}_DFMs.pkl"
            reused_stage = False

            if args.reuse_saved_stages and output_file.exists():
                try:
                    top_masks = load_masks_from_pkl(output_file)
                    stage_results = []
                    stage_summary = None
                    reused_stage = True
                    print(f"   ♻ Reusing saved stage masks from {output_file} (count={len(top_masks)})")
                except Exception as e:
                    print(f"   ⚠ Failed to reuse {output_file}: {e}")
                    reused_stage = False

            if not reused_stage:
                shared_candidates = None
                if args.per_au_search:
                    shared_candidates = stage_masks.get(stage)
                    if shared_candidates is None:
                        shared_candidates = generate_mask_candidates(
                            args.num_candidates,
                            FIXED_PROPORTION,
                            stage,
                            prev_masks,
                            stage_keep,
                        )
                        stage_masks[stage] = shared_candidates

                top_masks, stage_results, stage_summary = search_stage(
                    model, dataloader, stage,
                    args.num_candidates, FIXED_PROPORTION,
                    prev_masks, FIXED_DEVICE, FIXED_TOP_N,
                    keep_ratio_range=stage_keep,
                    f1_tolerance_pct=FIXED_F1_TOLERANCE_PCT,
                    baseline_f1=baseline_f1_global,
                    objective_mode='per_au' if target_au is not None else 'macro',
                    target_au=target_au,
                    baseline_per_au=baseline_per_au_full,
                    profile_timing=args.profile_timing,
                    precomputed_candidates=shared_candidates,
                )

                with open(output_file, 'wb') as f:
                    pickle.dump(top_masks, f)
                print(f"   ✓ Saved DFMs to {output_file}")

            best_mask = top_masks[0]
            best_mask_array = best_mask.mask if hasattr(best_mask, 'mask') else None

            if args.per_au_eval and best_mask_array is not None:
                print(f"\n   🔬 Per-AU eval for best mask ({stage})...")
                best_per_au = evaluate_mask_per_au(
                    model, dataloader, best_mask, baseline_per_au_full, FIXED_DEVICE)
                print_per_au_results(best_per_au, stage, 'best')

                save_per_au_grouped_bar(
                    baseline_per_au=baseline_per_au_full,
                    masked_per_au=best_per_au,
                    stage=stage,
                    model_type=f"{args.model_type}{target_suffix}",
                    save_dir=target_figures_dir,
                )
                per_au_drop_by_stage[stage] = {
                    au: vals.get('drop', 0.0) for au, vals in best_per_au.items()
                }

            # Identity prediction evaluation (IAT only)
            if args.model_type == 'IAT' and best_mask_array is not None:
                print(f"   🔐 Identity evaluation for {stage}...")
                id_result = evaluate_mask_identity(
                    model, dataloader, best_mask, FIXED_DEVICE,
                    model_type=args.model_type)
                if id_result:
                    print(f"     Subject ID accuracy: {id_result['accuracy']*100:.1f}%")
                else:
                    print(f"     [ID evaluation skipped - model_type={args.model_type}]")

            if args.visualize_masks and best_mask_array is not None:
                vis_path = target_figures_dir / f"{args.model_type}_{stage}{target_suffix}_best_mask_advanced.png"
                visualize_mask_advanced(best_mask_array, vis_path, stage, 'best')
                print(f"   🖼  Advanced mask visualization saved to {vis_path}")

            prev_masks = [m.mask for m in top_masks if hasattr(m, 'mask')]
            all_results[target_name][stage] = stage_results

            if (stage_summary is not None) and (stage_summary['best_drop_pct'] > FIXED_STOP_DROP_PCT):
                print(
                    f"   ⛔ Early stop: best drop {stage_summary['best_drop_pct']:.2f}% "
                    f"> stop threshold {FIXED_STOP_DROP_PCT:.2f}%"
                )
                break

        if args.per_au_eval and per_au_drop_by_stage:
            save_per_au_stage_heatmap(
                per_au_drop_by_stage=per_au_drop_by_stage,
                model_type=f"{args.model_type}{target_suffix}",
                save_dir=target_figures_dir,
            )

    return all_results


def run_radial_hfss_search(
    args,
    model,
    dataloader,
    output_dir,
    figures_dir,
    baseline_f1_global,
    baseline_per_au_full,
):
    """Deterministic radial-mask HFSS search using the same AU evaluation stack.

    The stage list controls circle size progression:
    first stage = biggest circle, later stages = smaller circles.
    """
    print("\n🧭 Radial HFSS mode ON (deterministic masks)")
    radial_mode = 'low-pass'
    radial_direction = FIXED_RADIAL_DIRECTION
    stage_masks = generate_radial_mask_candidates(
        FIXED_RADIAL_STEPS,
        mode=radial_mode,
        start_keep_pct=FIXED_RADIAL_START_KEEP_PCT,
        direction=radial_direction,
    )
    print(
        f"   Direction: {radial_direction} "
        f"| steps={FIXED_RADIAL_STEPS} "
        f"| start_keep={FIXED_RADIAL_START_KEEP_PCT:.1f}%"
    )
    all_results = {}

    if args.per_au_search:
        target_aus = [args.target_au] if args.target_au is not None else list(AU_LABELS)
    else:
        target_aus = [None]

    for target_au in target_aus:
        target_suffix = f"_AU{target_au:02d}" if target_au is not None else ""
        target_name = f"AU{target_au:02d}" if target_au is not None else "macro"
        target_output_dir = output_dir / target_name if target_au is not None else output_dir
        target_figures_dir = figures_dir / target_name if target_au is not None else figures_dir
        target_output_dir.mkdir(parents=True, exist_ok=True)
        target_figures_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*50}")
        print(f"Search target: {target_name}")
        print(f"{'='*50}")

        all_results[target_name] = {}
        prev_masks = None
        radial_stage_names = [f"radial_step_{i + 1:02d}" for i in range(len(stage_masks))]

        for stage_index, stage_name in enumerate(radial_stage_names):
            stage_mask = stage_masks[min(stage_index, len(stage_masks) - 1)]
            print(
                f"   Radial stage {stage_index + 1}/{len(radial_stage_names)} ({stage_name}): "
                f"keep={stage_mask.keep_pct*100:.1f}% "
                f"| white={stage_mask.white_count}"
            )

            top_masks, stage_results, stage_summary = search_stage(
                model=model,
                dataloader=dataloader,
                stage=stage_name,
                num_candidates=1,
                proportion=FIXED_PROPORTION,
                prev_masks=prev_masks,
                device=FIXED_DEVICE,
                top_n=1,
                keep_ratio_range=None,
                f1_tolerance_pct=FIXED_F1_TOLERANCE_PCT,
                baseline_f1=baseline_f1_global,
                objective_mode='per_au' if target_au is not None else 'macro',
                target_au=target_au,
                baseline_per_au=baseline_per_au_full,
                profile_timing=args.profile_timing,
                precomputed_candidates=[stage_mask],
            )

            output_file = target_output_dir / f"{args.model_type}_{stage_name}{target_suffix}_DFMs.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(top_masks, f)
            print(f"   ✓ Saved radial DFMs to {output_file}")

            best_mask = top_masks[0] if top_masks else None
            if args.per_au_eval and best_mask is not None:
                best_per_au = evaluate_mask_per_au(
                    model, dataloader, best_mask, baseline_per_au_full, FIXED_DEVICE)
                print_per_au_results(best_per_au, stage_name, 'best')
                save_per_au_grouped_bar(
                    baseline_per_au=baseline_per_au_full,
                    masked_per_au=best_per_au,
                    stage=stage_name,
                    model_type=f"{args.model_type}{target_suffix}",
                    save_dir=target_figures_dir,
                )

            # Identity prediction evaluation (IAT only)
            if args.model_type == 'IAT' and best_mask is not None:
                print(f"   🔐 Identity evaluation for {stage_name}...")
                id_result = evaluate_mask_identity(
                    model, dataloader, best_mask, FIXED_DEVICE, 
                    model_type=args.model_type)
                if id_result:
                    print(f"     Subject ID accuracy: {id_result['accuracy']*100:.1f}%")
                else:
                    print(f"     [ID evaluation skipped - model_type={args.model_type}]")

            if args.visualize_masks and best_mask is not None and hasattr(best_mask, 'mask'):
                vis_path = target_figures_dir / f"{args.model_type}_{stage_name}{target_suffix}_best_mask_advanced.png"
                visualize_mask_advanced(best_mask.mask, vis_path, stage_name, 'best')
                print(f"   🖼  Advanced mask visualization saved to {vis_path}")

            prev_masks = [m.mask for m in top_masks if hasattr(m, 'mask')]
            all_results[target_name][stage_name] = {
                'rows': stage_results,
                'summary': stage_summary,
            }

            # Radial mode tracks the full F1-vs-keep-ratio curve intentionally;
            # early stopping would abort at step 1 (tiny circle → big F1 drop).

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='FMAE', choices=['FMAE', 'IAT'])
    parser.add_argument('--model_path', default='models/FMAE_BP4D_fold1.pth')
    parser.add_argument('--test_json', default='BP4D/BP4D_test1.json')
    parser.add_argument('--num_candidates', type=int, default=200)
    parser.add_argument('--num_samples', type=int, default=500, help='Number of training samples to use')
    parser.add_argument('--output_dir', default='hfss/DFM')
    parser.add_argument('--stages', nargs='+', default=['stage1', 'stage2', 'stage3'])
    parser.add_argument('--search_mode', default='random', choices=['random', 'radial'],
                        help='Mask search mode: random (current HFSS) or radial (deterministic)')
    # --- Optional feature flags (disable by omitting the flag) ---
    parser.add_argument('--per_au_eval', action='store_true',
                        help='After search, evaluate per-AU F1 drops for the best mask per stage')
    parser.add_argument('--visualize_masks', action='store_true',
                        help='Save advanced mask visualizations (ring, freq band, masked FFT) for top masks')
    parser.add_argument('--per_au_search', action='store_true',
                        help='Use per-AU objective for mask search (thesis mode)')
    parser.add_argument('--target_au', type=int, default=None,
                        help='Specific AU to optimize in per-AU search mode (default: all AUs)')
    parser.add_argument('--reuse_saved_stages', action='store_true',
                        help='Reuse existing stage PKLs and skip recomputing those stages')
    parser.add_argument('--bootstrap_from_macro_stage1', action='store_true',
                        help='For per-AU runs starting at stage2+, initialize parents from macro stage1 PKL')
    parser.add_argument('--profile_timing', action='store_true',
                        help='Print detailed timing breakdown for data/FFT/mask/inference during stage search')
    args = parser.parse_args()

    # Fixed runtime knobs (removed from CLI, values unchanged from previous defaults).
    args.data_root = FIXED_DATA_ROOT
    args.top_n = FIXED_TOP_N
    args.proportion = FIXED_PROPORTION
    args.stop_drop_pct = FIXED_STOP_DROP_PCT
    args.f1_tolerance_pct = FIXED_F1_TOLERANCE_PCT
    args.batch_size = FIXED_BATCH_SIZE
    args.num_workers = FIXED_NUM_WORKERS
    args.device = FIXED_DEVICE
    args.seed = FIXED_SEED

    keep_ranges = dict(DEFAULT_KEEP_RANGES)
    args.target_au = validate_target_au(args.target_au)

    # Reproducibility: controls numpy/random/torch sampling in this run.
    set_global_seed(args.seed, args.device)
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir.parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir.parent / "logs"

    log_file_path = create_run_log_file(logs_dir)
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    log_fp = open(log_file_path, 'w', encoding='utf-8')
    sys.stdout = TeeStream(orig_stdout, log_fp)
    sys.stderr = TeeStream(orig_stderr, log_fp)
    
    try:
        print("="*70)
        print(f"HFSS Search for AU Detection - {args.model_type}")
        print(f"Model: {args.model_path} | Device: {args.device}")
        print(f"Seed: {args.seed}")
        print(f"Log file: {log_file_path}")
        print("="*70)
        
        # Load model
        model = load_model(args.model_path, args.model_type, args.device)
        dataset, dataloader = build_eval_dataloader(
            test_json=args.test_json,
            data_root=args.data_root,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
        )
        print(f"✓ Loaded {len(dataset)} TEST samples (evaluating on unseen data)")

        # Baseline (no mask) computed once on the same evaluation dataset.
        # This is the reference for reporting drops.
        baseline_f1_global = evaluate_mask(model, dataloader, lambda x: x, args.device)
        baseline_per_au_full = None
        if args.per_au_eval or args.per_au_search:
            baseline_per_au_full = evaluate_mask_per_au(
                model, dataloader, lambda x: x, {}, args.device
            )

        if args.search_mode == 'random':
            all_results = run_random_hfss_search(
                args=args,
                model=model,
                dataloader=dataloader,
                output_dir=output_dir,
                figures_dir=figures_dir,
                baseline_f1_global=baseline_f1_global,
                baseline_per_au_full=baseline_per_au_full,
                keep_ranges=keep_ranges,
            )
        else:
            all_results = run_radial_hfss_search(
                args=args,
                model=model,
                dataloader=dataloader,
                output_dir=output_dir,
                figures_dir=figures_dir,
                baseline_f1_global=baseline_f1_global,
                baseline_per_au_full=baseline_per_au_full,
            )
        
        print("\n" + "="*70)
        print("✅ HFSS Search Complete!")
        print("="*70)
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        log_fp.close()


if __name__ == "__main__":
    main()


'''
Run from project root:
cd /Users/tima/Documents/DASC/Thesis/Frequency-Based-AU-Detection

--- RANDOM MODE ---

1) QUICK SMOKE (macro, subset)
python hfss_search_au.py --model_path models/FMAE_BP4D_fold1.pth \
    --test_json BP4D/BP4D_test1.json --num_samples 50 \
    --num_candidates 20 --stages stage1 stage2

2) FULL TEST SET, MACRO F1
python hfss_search_au.py --model_path models/FMAE_BP4D_fold1.pth \
    --test_json BP4D/BP4D_test1.json --num_samples 999999 \
    --num_candidates 200 --stages stage1 stage2 stage3

3) FULL TEST SET, ALL AUs (per-AU search + eval)
python hfss_search_au.py --model_path models/FMAE_BP4D_fold1.pth \
    --test_json BP4D/BP4D_test1.json --num_samples 999999 \
    --num_candidates 200 --stages stage1 stage2 stage3 \
    --per_au_search --per_au_eval

4) FULL TEST SET, SINGLE AU (e.g. AU12)
python hfss_search_au.py --model_path models/FMAE_BP4D_fold1.pth \
    --test_json BP4D/BP4D_test1.json --num_samples 999999 \
    --num_candidates 200 --stages stage1 stage2 stage3 \
    --per_au_search --target_au 12 --per_au_eval --visualize_masks

5) FAST REFINEMENT (reuse precomputed stage1 masks)
python hfss_search_au.py --model_path models/FMAE_BP4D_fold1.pth \
    --test_json BP4D/BP4D_test1.json --num_samples 200 \
    --stages stage2 stage3 --per_au_search --target_au 12 \
    --bootstrap_from_macro_stage1 --reuse_saved_stages

--- RADIAL MODE ---

6) RADIAL, FULL TEST SET, MACRO F1
python hfss_search_au.py --model_path models/FMAE_BP4D_fold1.pth \
    --test_json BP4D/BP4D_test1.json --num_samples 999999 \
    --search_mode radial

7) RADIAL, FULL TEST SET, ALL AUs
python hfss_search_au.py --model_path models/FMAE_BP4D_fold1.pth \
    --test_json BP4D/BP4D_test1.json --num_samples 999999 \
    --search_mode radial --per_au_search --per_au_eval

8) RADIAL, FULL TEST SET, SINGLE AU (e.g. AU12)
python hfss_search_au.py --model_path models/FMAE_BP4D_fold1.pth \
    --test_json BP4D/BP4D_test1.json --num_samples 999999 \
    --search_mode radial --per_au_search --target_au 12 \
    --per_au_eval --visualize_masks

Notes:
- CUDA unavailable? Append: --device cpu --num_workers 0  (device/workers are fixed constants; set in FIXED_* at top of file)
- Logs saved under: hfss/logs/
- Outputs under: hfss/DFM/  and  hfss/figures/
- Per-AU outputs separated by target: hfss/DFM/AUxx/  and  hfss/figures/AUxx/
- --stages and --num_candidates are ignored in radial mode (uses FIXED_RADIAL_STEPS=10)
'''

"""
The search target is only the AU being optimized. For example, with --target_au 04, the search score is based on AU04 only.

The reason you still see all AUs printed is that the script also runs a per-AU report for the same best mask. That extra printout is for debugging and analysis, not for the search objective. In hfss_search_au.py, this comes from the per-AU evaluation path that prints every AU in AU_LABELS.

If you want, I can also show you the exact line that controls this output and how to turn it off.

"""