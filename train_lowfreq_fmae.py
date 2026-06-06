"""Train FMAE for BP4D AU detection with GPU-side low-frequency masking.

Training applies a random radial low-pass filter per image after the standard
BP4D augmentations, while evaluation always uses clean images.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Optional

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from warnings import filterwarnings
filterwarnings("ignore")

import timm

if timm.__version__ != "0.3.2":
    print(f"Warning: expected timm==0.3.2, found {timm.__version__}")

PROJECT_ROOT = Path(__file__).resolve().parent


class TeeStream:
    """Write to both stdout and a log file."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        written = 0
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
                written = len(data)
            except Exception:
                # Ignore errors writing to any closed stream
                continue
        return written

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                # Stream may be closed already during interpreter shutdown
                continue
from fmae import models_vit  # noqa: E402
from fmae.util import lr_decay as lrd  # noqa: E402
from fmae.util import lr_sched, misc  # noqa: E402
from fmae.util.datasets import BP4D_AU_dataset  # noqa: E402
from fmae.util.misc import NativeScalerWithGradNormCount as NativeScaler  # noqa: E402
from fmae.util.pos_embed import interpolate_pos_embed  # noqa: E402


AU_LABELS = [1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24]
NUM_AUS = 12
IMG_SIZE = 224

DEFAULT_ROOT_PATH = str(PROJECT_ROOT / "BP4D" / "BP4D_cropped") + "/"
DEFAULT_TRANSFORM_ARGS = {
    "input_size": IMG_SIZE,
    "color_jitter": None,
    "aa": "rand-m9-mstd0.5-inc1",
    "reprob": 0.25,
    "remode": "pixel",
    "recount": 1,
}

_RADIAL_DISTANCE_CACHE: dict[tuple[str, int, int], torch.Tensor] = {}


def _radial_keep_mask(height: int, width: int, keep_pct: float, mask_type: str, device: torch.device) -> torch.Tensor:
    """Build a deterministic radial low/high-pass mask from an exact keep percentage."""
    if mask_type not in ("low", "high"):
        raise ValueError(f"mask_type must be 'low' or 'high' for exact radial masking, got {mask_type}")

    keep_pct = float(np.clip(keep_pct, 0.0, 100.0))
    dist = _distance_grid(height, width, device)

    if mask_type == "low":
        radius = float(torch.quantile(dist.flatten(), keep_pct / 100.0).item())
        mask = (dist <= radius)
    else:
        radius = float(torch.quantile(dist.flatten(), 1.0 - keep_pct / 100.0).item())
        mask = (dist >= radius)

    return mask.to(dtype=torch.float32)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True


def build_dataset_args(root_path: str) -> SimpleNamespace:
    return SimpleNamespace(root_path=root_path, **DEFAULT_TRANSFORM_ARGS)


def build_dataloaders(
    train_json: str,
    test_json: str,
    root_path: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    num_samples: Optional[int] = None,
) -> tuple[BP4D_AU_dataset, BP4D_AU_dataset, DataLoader, DataLoader]:
    dataset_args = build_dataset_args(root_path)
    train_dataset = BP4D_AU_dataset(train_json, is_train=True, args=dataset_args)
    test_dataset = BP4D_AU_dataset(test_json, is_train=False, args=dataset_args)

    if num_samples is not None and num_samples > 0:
        train_dataset.data = train_dataset.data[:num_samples]
        test_dataset.data = test_dataset.data[:num_samples]

    pin_memory = device.type == "cuda"
    
    # Efficiency: reduce worker overhead for small datasets
    is_small_dataset = len(train_dataset) < batch_size * 2
    workers = 0 if is_small_dataset else num_workers
    persistent = False if is_small_dataset else (num_workers > 0)
    # Efficiency: drop_last=False for small subsets to avoid empty dataloader
    train_drop_last = False if is_small_dataset else True
    
    # Build dataloaders with conditional prefetch_factor (only valid when num_workers > 0)
    train_loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": workers,
        "pin_memory": pin_memory,
        "drop_last": train_drop_last,
        "persistent_workers": persistent,
    }
    if workers > 0:
        train_loader_kwargs["prefetch_factor"] = 2 if is_small_dataset else 4
    
    test_loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": pin_memory,
        "drop_last": False,
        "persistent_workers": persistent,
    }
    if workers > 0:
        test_loader_kwargs["prefetch_factor"] = 2 if is_small_dataset else 4
    
    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    test_loader = DataLoader(test_dataset, **test_loader_kwargs)
    return train_dataset, test_dataset, train_loader, test_loader


def build_model(model_path: str, device: torch.device, model_type: str = "FMAE") -> nn.Module:
    num_subjects = 41 if model_type == "IAT" else 0
    grad_reverse = 1.0 if model_type == "IAT" else 0.0
    model = models_vit.vit_large_patch16(
        num_classes=NUM_AUS,
        num_subjects=num_subjects,
        drop_path_rate=0.1,
        global_pool=True,
        grad_reverse=grad_reverse,
    )

    checkpoint = torch.load(model_path, map_location="cpu")
    checkpoint_model = checkpoint.get("model", checkpoint)
    state_dict = model.state_dict()

    for key in ["head.weight", "head.bias"]:
        if key in checkpoint_model and checkpoint_model[key].shape != state_dict[key].shape:
            print(f"   Removing key {key} from checkpoint (shape mismatch)")
            del checkpoint_model[key]

    interpolate_pos_embed(model, checkpoint_model)
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(f"   Model type: {model_type} | Loaded from: {model_path}")

    # Defer model.to(device) to main() for better efficiency tracking
    return model


def _distance_grid(height: int, width: int, device: torch.device) -> torch.Tensor:
    cache_key = (str(device), height, width)
    cached = _RADIAL_DISTANCE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    cy, cx = height // 2, width // 2
    ys = torch.arange(height, device=device, dtype=torch.float32)
    xs = torch.arange(width, device=device, dtype=torch.float32)
    try:
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    except TypeError:
        yy, xx = torch.meshgrid(ys, xs)
    dist = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    _RADIAL_DISTANCE_CACHE[cache_key] = dist
    return dist


def apply_random_frequency_mask(
    images: torch.Tensor,
    min_keep: float,
    max_keep: float,
    mask_type: str = "low",
    exact_keep_pct: Optional[float] = None,
) -> torch.Tensor:
    """Apply frequency mask to images.

    Args:
        images: [batch, channels, height, width] tensor
        min_keep: minimum keep ratio (0.0 to 1.0)
        max_keep: maximum keep ratio (0.0 to 1.0)
        mask_type: 'low' (keep center low frequencies), 'high' (keep outer high frequencies), or 'full' (no masking)
        exact_keep_pct: optional exact keep percentage for deterministic masking
    """
    if mask_type == "full":
        return images

    if images.ndim != 4:
        raise ValueError(f"Expected 4D image batch, got shape {tuple(images.shape)}")
    if not (0.0 <= min_keep <= max_keep <= 1.0):
        raise ValueError("min_keep/max_keep must satisfy 0 <= min_keep <= max_keep <= 1")
    if mask_type not in ("low", "high"):
        raise ValueError(f"mask_type must be 'low', 'high', or 'full', got {mask_type}")

    batch_size, _, height, width = images.shape
    device = images.device

    if exact_keep_pct is not None:
        mask_2d = _radial_keep_mask(height, width, exact_keep_pct, mask_type, device)
        mask = mask_2d.unsqueeze(0).expand(batch_size, -1, -1)
    else:
        dist = _distance_grid(height, width, device)

        keep_ratios = torch.rand(batch_size, device=device, dtype=torch.float32)
        keep_ratios = keep_ratios * (max_keep - min_keep) + min_keep
        radius_scale = float(min(height, width)) / 2.0
        radii = keep_ratios * radius_scale

        if mask_type == "low":
            # Low-pass: keep center frequencies (dist <= radii)
            mask = (dist.unsqueeze(0) <= radii[:, None, None]).to(dtype=images.dtype)
        else:  # high
            # High-pass: keep outer frequencies (dist >= radii)
            mask = (dist.unsqueeze(0) >= radii[:, None, None]).to(dtype=images.dtype)

    if exact_keep_pct is not None:
        mask = mask.to(dtype=images.dtype).unsqueeze(1)
    else:
        mask = mask.unsqueeze(1)

    freq = torch.fft.fftshift(torch.fft.fft2(images, dim=(-2, -1)), dim=(-2, -1))
    freq = freq * mask
    filtered = torch.fft.ifft2(torch.fft.ifftshift(freq, dim=(-2, -1)), dim=(-2, -1)).real
    return filtered


def apply_random_lowpass_mask(
    images: torch.Tensor,
    min_keep: float,
    max_keep: float,
    exact_keep_pct: Optional[float] = None,
) -> torch.Tensor:
    """Backward-compatible wrapper for low-pass masking."""
    return apply_random_frequency_mask(images, min_keep, max_keep, mask_type="low", exact_keep_pct=exact_keep_pct)


def train_one_epoch_lowfreq(
    model: nn.Module,
    criterion: nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    args,
    min_keep: float,
    max_keep: float,
    loss_scaler: Optional[NativeScaler],
    mask_type: str = "low",
    exact_keep_pct: Optional[float] = None,
) -> dict[str, float]:
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    accum_iter = args.accum_iter
    use_amp = device.type == "cuda"

    optimizer.zero_grad()

    data_loader_len = len(data_loader)

    # Temporarily redirect stdout to stderr to keep progress bars out of log file
    sys.stdout = sys.__stdout__

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, 50, header)):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / max(data_loader_len, 1) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets[0] if isinstance(targets, (tuple, list)) else targets
        targets = targets.to(device, non_blocking=True)
        samples = apply_random_frequency_mask(
            samples,
            min_keep=min_keep,
            max_keep=max_keep,
            mask_type=mask_type,
            exact_keep_pct=exact_keep_pct,
        )

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(samples)
            logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            loss = criterion(logits, targets)

        loss_value = float(loss.item())
        if not math.isfinite(loss_value):
            raise FloatingPointError(f"Loss is not finite: {loss_value}")

        loss = loss / accum_iter
        if loss_scaler is not None and use_amp:
            loss_scaler(
                loss,
                optimizer,
                clip_grad=args.clip_grad,
                parameters=model.parameters(),
                create_graph=False,
                update_grad=(data_iter_step + 1) % accum_iter == 0,
            )
        else:
            loss.backward()
            if (data_iter_step + 1) % accum_iter == 0:
                if args.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        if use_amp:
            torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        current_lr = max(group["lr"] for group in optimizer.param_groups)
        metric_logger.update(lr=current_lr)

    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_au_macro_f1(
    model: nn.Module,
    data_loader: Iterable,
    device: torch.device,
    mask_type: str = "full",
    exact_keep_pct: Optional[float] = None,
) -> float:
    model.eval()
    all_probs = []
    all_targets = []

    for images, (au_labels, _) in data_loader:
        images = images.to(device, non_blocking=True)
        au_labels = au_labels.to(device, non_blocking=True)

        if exact_keep_pct is not None and mask_type in ("low", "high"):
            images = apply_random_frequency_mask(
                images,
                min_keep=0.0,
                max_keep=1.0,
                mask_type=mask_type,
                exact_keep_pct=exact_keep_pct,
            )

        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            outputs = model(images)
            logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).to(dtype=torch.int32)
        all_probs.append(preds.cpu().numpy())
        all_targets.append(au_labels.cpu().numpy())

    y_pred = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    per_class_f1 = [f1_score(y_true[:, idx], y_pred[:, idx], zero_division=0) for idx in range(y_true.shape[1])]
    return float(np.mean(per_class_f1))


def save_checkpoint(
    output_dir: Path,
    fold: int,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    args,
    loss_scaler: Optional[NativeScaler],
    is_best: bool = False,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = "best" if is_best else f"epoch{epoch:03d}"
    checkpoint_path = output_dir / f"fold{fold}_{tag}.pth"
    to_save = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "args": args,
    }
    if loss_scaler is not None:
        to_save["scaler"] = loss_scaler.state_dict()
    torch.save(to_save, checkpoint_path)
    return checkpoint_path


def write_csv_row(csv_path: Path, row: dict[str, object]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "test_au_macro_f1"])
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def print_summary_table(rows: list[dict[str, object]]) -> None:
    print("\n" + "=" * 62)
    print(f"{'Epoch':>6} | {'Train loss':>12} | {'Test AU macro F1':>17}")
    print("-" * 62)
    for row in rows:
        print(f"{int(row['epoch']):6d} | {float(row['train_loss']):12.6f} | {float(row['test_au_macro_f1']):17.6f}")
    print("=" * 62)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FMAE/IAT on BP4D with low-frequency masking")
    parser.add_argument("--model_type", type=str, default="FMAE", choices=["FMAE", "IAT"], help="Model type")
    parser.add_argument("--model_path", type=str, required=True, help="Path to pretrained checkpoint")
    parser.add_argument("--train_json", type=str, required=True, help="Path to BP4D train JSON")
    parser.add_argument("--test_json", type=str, required=True, help="Path to BP4D test JSON")
    parser.add_argument("--mask_type", type=str, default="low", choices=["low", "high", "full"], help="Frequency mask type: 'low' (keep center), 'high' (keep outer), 'full' (no mask)")
    parser.add_argument("--min_keep", type=float, default=0.08, help="Minimum keep ratio (default 8%)")
    parser.add_argument("--max_keep", type=float, default=0.20, help="Maximum keep ratio (default 20%)")
    parser.add_argument("--exact_keep_pct", type=float, default=None, help="Exact keep percentage for deterministic radial masking, e.g. 99 for high-pass keep-99%")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--blr", type=float, default=5e-4, help="Base learning rate")
    parser.add_argument("--output_dir", type=str, default="output/lowfreq_fmae", help="Output directory")
    parser.add_argument("--fold", type=int, default=1, help="Fold identifier")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--num_samples", type=int, default=None, help="Use subset of data (for testing)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--input_size", type=int, default=IMG_SIZE)
    parser.add_argument("--accum_iter", type=int, default=1)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--drop_path", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--layer_decay", type=float, default=0.75)
    parser.add_argument("--clip_grad", type=float, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f"fold{args.fold}_{args.model_type.lower()}_log.txt"
    metrics_csv = output_dir / f"fold{args.fold}_{args.model_type.lower()}_metrics.csv"

    orig_stdout = sys.stdout
    log_fp = open(log_file, "w", encoding="utf-8")
    sys.stdout = TeeStream(orig_stdout, log_fp)

    print(f"\n{'='*70}")
    print(f"Training {args.model_type} on BP4D fold {args.fold} with {args.mask_type}-frequency masking")
    print(f"{'='*70}")
    print(f"Device: {device} | Model: {args.model_type}")
    if args.exact_keep_pct is not None:
        print(f"Mask type: {args.mask_type} | Exact keep percentage: {args.exact_keep_pct:.1f}%")
    else:
        print(f"Mask type: {args.mask_type} | Keep ratio range: [{args.min_keep:.1%}, {args.max_keep:.1%}]")
    if args.num_samples:
        print(f"Using subset: {args.num_samples} samples (test mode)")
    print(f"Output: {output_dir}")

    train_dataset, test_dataset, train_loader, test_loader = build_dataloaders(
        args.train_json,
        args.test_json,
        DEFAULT_ROOT_PATH,
        args.batch_size,
        args.num_workers,
        device,
        args.num_samples,
    )
    train_batches = len(train_loader)
    test_batches = len(test_loader)
    print(f"Data: train={len(train_dataset)} ({train_batches} batches) | test={len(test_dataset)} ({test_batches} batches)")
    if train_batches == 0:
        raise ValueError(f"ERROR: Training dataloader is empty! Reduce batch_size from {args.batch_size} or increase num_samples.")

    model = build_model(args.model_path, device, args.model_type).to(device)

    eff_batch_size = args.batch_size * args.accum_iter
    if getattr(args, "lr", None) is None:
        args.lr = args.blr * eff_batch_size / 256
    print(f"Optimizer: AdamW | lr={args.lr:.2e} | weight_decay={args.weight_decay}")
    print(f"Training for {args.epochs} epochs | batch_size={args.batch_size}")
    print()

    param_groups = lrd.param_groups_lrd(
        model,
        args.weight_decay,
        no_weight_decay_list=model.no_weight_decay(),
        layer_decay=args.layer_decay,
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler() if device.type == "cuda" else None
    criterion = nn.BCEWithLogitsLoss()

    rows: list[dict[str, object]] = []
    best_f1 = 0.0
    start_time = time.time()
    
    # Efficiency note: Move model to GPU once before training loop
    model = model.to(device)

    for epoch in range(args.epochs):
        train_stats = train_one_epoch_lowfreq(
            model=model,
            criterion=criterion,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            args=args,
            min_keep=args.min_keep,
            max_keep=args.max_keep,
            loss_scaler=loss_scaler,
            mask_type=args.mask_type,
            exact_keep_pct=args.exact_keep_pct,
        )

        # Restore TeeStream for logging epoch summary
        sys.stdout = TeeStream(orig_stdout, log_fp)

        test_f1 = evaluate_au_macro_f1(
            model,
            test_loader,
            device,
            mask_type=args.mask_type,
            exact_keep_pct=args.exact_keep_pct,
        )

        row = {
            "epoch": epoch + 1,
            "train_loss": float(train_stats.get("loss", 0.0)),
            "test_au_macro_f1": float(test_f1),
        }
        rows.append(row)
        write_csv_row(metrics_csv, row)

        save_checkpoint(output_dir, args.fold, epoch + 1, model, optimizer, args, loss_scaler, is_best=False)
        if test_f1 > best_f1:
            best_f1 = test_f1
            save_checkpoint(output_dir, args.fold, epoch + 1, model, optimizer, args, loss_scaler, is_best=True)

        print(f"Epoch {epoch + 1:3d} | loss: {row['train_loss']:.4f} | F1: {test_f1:.4f} | best_F1: {best_f1:.4f}")

    total_time = dt.timedelta(seconds=int(time.time() - start_time))
    print(f"\n{'='*70}")
    print(f"Training completed in {total_time}")
    print(f"Best F1: {best_f1:.4f}")
    print(f"Logs saved to: {log_file}")
    print(f"Metrics saved to: {metrics_csv}")
    print(f"{'='*70}\n")
    print_summary_table(rows)

    # Restore original stdout before closing the logfile to avoid flushing closed streams
    try:
        sys.stdout = orig_stdout
    except Exception:
        pass
    log_fp.close()


if __name__ == "__main__":
    main()

'''
============================================================================
EXAMPLE COMMANDS
============================================================================

=== LOW-FREQUENCY TRAINING (Original - Default) ===

Quick test FMAE (50 samples, 2 epochs):
python train_lowfreq_fmae.py --model_type FMAE --model_path models/FMAE_BP4D_fold1.pth \
    --train_json BP4D/BP4D_train1.json --test_json BP4D/BP4D_test1.json \
    --mask_type low --min_keep 0.08 --max_keep 0.20 \
    --fold 1 --num_samples 50 --epochs 2

Full training FMAE fold 1 (low-frequency 8-20%):
python train_lowfreq_fmae.py --model_type FMAE --model_path models/FMAE_BP4D_fold1.pth \
    --train_json BP4D/BP4D_train1.json --test_json BP4D/BP4D_test1.json \
    --mask_type low --min_keep 0.08 --max_keep 0.20 \
    --fold 1 --epochs 30 --output_dir output/lowfreq_fmae_fold1

Quick test IAT (50 samples, 2 epochs):
python train_lowfreq_fmae.py --model_type IAT --model_path models/FMAE_IAT_BP4D_fold1.pth \
    --train_json BP4D/BP4D_train1.json --test_json BP4D/BP4D_test1.json \
    --mask_type low --min_keep 0.08 --max_keep 0.20 \
    --fold 1 --num_samples 50 --epochs 2

Full training IAT fold 1 (low-frequency 8-20%):
python train_lowfreq_fmae.py --model_type IAT --model_path models/FMAE_IAT_BP4D_fold1.pth \
    --train_json BP4D/BP4D_train1.json --test_json BP4D/BP4D_test1.json \
    --mask_type low --min_keep 0.08 --max_keep 0.20 \
    --fold 1 --epochs 30 --output_dir output/lowfreq_iat_fold1



=== HIGH-FREQUENCY TRAINING (New - For Supervisor's Experiment) ===

Full training FMAE fold 1 (high-frequency, remove center 15%):
python train_lowfreq_fmae.py --model_type FMAE --model_path models/FMAE_BP4D_fold1.pth --train_json BP4D/BP4D_train1.json --test_json BP4D/BP4D_test1.json --mask_type high --min_keep 0.15 --max_keep 1.0 --fold 1 --epochs 30 --output_dir output/highfreq_fmae_fold1

python train_lowfreq_fmae.py --model_type FMAE --model_path models/FMAE_BP4D_fold1.pth --train_json BP4D/BP4D_train1.json --test_json BP4D/BP4D_test1.json --mask_type high --exact_keep_pct 99 --fold 1 --epochs 30 --output_dir output/highfreq_fmae_fold1

Full training IAT fold 1 (high-frequency, remove center 15%):
python train_lowfreq_fmae.py --model_type IAT --model_path models/FMAE_IAT_BP4D_fold1.pth --train_json BP4D/BP4D_train1.json --test_json BP4D/BP4D_test1.json --mask_type high --min_keep 0.15 --max_keep 1.0 --fold 1 --epochs 30 --output_dir output/highfreq_iat_fold1

python train_lowfreq_fmae.py --model_type IAT --model_path models/FMAE_IAT_BP4D_fold1.pth --train_json BP4D/BP4D_train1.json --test_json BP4D/BP4D_test1.json --mask_type high --exact_keep_pct 99 --fold 1 --epochs 30 --output_dir output/highfreq_iat_fold1

=== NO MASKING (Baseline - Full Image) ===

Full training FMAE fold 1 (no masking, full image):
python train_lowfreq_fmae.py --model_type FMAE --model_path models/FMAE_BP4D_fold1.pth \
    --train_json BP4D/BP4D_train1.json --test_json BP4D/BP4D_test1.json \
    --mask_type full \
    --fold 1 --epochs 30 --output_dir output/fullfreq_fmae_fold1

Full training IAT fold 1 (no masking, full image):
python train_lowfreq_fmae.py --model_type IAT --model_path models/FMAE_IAT_BP4D_fold1.pth \
    --train_json BP4D/BP4D_train1.json --test_json BP4D/BP4D_test1.json \
    --mask_type full \
    --fold 1 --epochs 30 --output_dir output/fullfreq_iat_fold1

    
python train_lowfreq_fmae.py --model_type FMAE --model_path models/FMAE_BP4D_fold1.pth --train_json BP4D/BP4D_train1.json --test_json BP4D/BP4D_test1.json --mask_type low --min_keep 0.04 --max_keep 0.10 --fold 1 --epochs 30 --output_dir output/lowfreq_fmae_fold1_4_10


python train_lowfreq_fmae.py --model_type IAT --model_path models/FMAE_IAT_BP4D_fold1.pth --train_json BP4D/BP4D_train1.json --test_json BP4D/BP4D_test1.json --mask_type low --min_keep 0.04 --max_keep 0.10 --fold 1 --epochs 30 --output_dir output/lowfreq_iat_fold1_4_10
'''