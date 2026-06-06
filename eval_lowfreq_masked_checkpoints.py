"""Evaluate saved low-frequency checkpoints on randomly masked test images.

This script reuses the same radial low-pass masking used during low-frequency
training, then evaluates each saved epoch checkpoint on the BP4D test split.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from fmae import models_vit
from fmae.util.datasets import BP4D_AU_dataset
from fmae.util.pos_embed import interpolate_pos_embed


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_ROOT_PATH = str(PROJECT_ROOT / "BP4D" / "BP4D_cropped") + "/"
DEFAULT_TRANSFORM_ARGS = {
    "input_size": 224,
    "color_jitter": None,
    "aa": "rand-m9-mstd0.5-inc1",
    "reprob": 0.25,
    "remode": "pixel",
    "recount": 1,
}

_RADIAL_DISTANCE_CACHE: dict[tuple[str, int, int], torch.Tensor] = {}


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def build_dataset_args(root_path: str):
    return type("Args", (), {"root_path": root_path, **DEFAULT_TRANSFORM_ARGS})()


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
) -> torch.Tensor:
    """Apply frequency mask to images.
    
    Args:
        images: [batch, channels, height, width] tensor
        min_keep: minimum keep ratio (0.0 to 1.0)
        max_keep: maximum keep ratio (0.0 to 1.0)
        mask_type: 'low' (keep center low frequencies), 'high' (keep outer high frequencies), or 'full' (no masking)
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

    freq = torch.fft.fftshift(torch.fft.fft2(images, dim=(-2, -1)), dim=(-2, -1))
    freq = freq * mask.unsqueeze(1)
    filtered = torch.fft.ifft2(torch.fft.ifftshift(freq, dim=(-2, -1)), dim=(-2, -1)).real
    return filtered


def apply_random_lowpass_mask(
    images: torch.Tensor,
    min_keep: float,
    max_keep: float,
) -> torch.Tensor:
    """Backward-compatible wrapper for low-pass masking."""
    return apply_random_frequency_mask(images, min_keep, max_keep, mask_type="low")


def build_model(checkpoint_path: str | Path, device: torch.device, model_type: str = "FMAE") -> nn.Module:
    model = models_vit.vit_large_patch16(
        num_classes=12,
        num_subjects=41 if model_type == "IAT" else 0,
        drop_path_rate=0.1,
        global_pool=True,
        grad_reverse=1.0 if model_type == "IAT" else 0.0,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_model = checkpoint.get("model", checkpoint)

    if model_type == "IAT":
        checkpoint_model = {
            key: value
            for key, value in checkpoint_model.items()
            if not (
                key.startswith("ID_head.")
                or key.startswith("module.ID_head.")
                or key.endswith(".ID_head.weight")
                or key.endswith(".ID_head.bias")
            )
        }

    interpolate_pos_embed(model, checkpoint_model)
    model.load_state_dict(checkpoint_model, strict=False)
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def evaluate_au_macro_f1_masked(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    min_keep: float,
    max_keep: float,
    mask_type: str = "low",
) -> float:
    model.eval()
    all_preds = []
    all_targets = []

    for images, (au_labels, _) in data_loader:
        images = images.to(device, non_blocking=True)
        au_labels = au_labels.to(device, non_blocking=True)
        masked_images = apply_random_frequency_mask(images, min_keep=min_keep, max_keep=max_keep, mask_type=mask_type)

        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            outputs = model(masked_images)
            logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs

        preds = (torch.sigmoid(logits) >= 0.5).to(dtype=torch.int32)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(au_labels.cpu().numpy())

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    per_class_f1 = [f1_score(y_true[:, idx], y_pred[:, idx], zero_division=0) for idx in range(y_true.shape[1])]
    return float(np.mean(per_class_f1))


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"epoch(\d+)", path.name)
    return (int(match.group(1)) if match else 10**9, path.name)


def list_checkpoints(checkpoint_dir: Path, checkpoint_glob: str) -> list[Path]:
    checkpoints = sorted(checkpoint_dir.glob(checkpoint_glob), key=_checkpoint_sort_key)
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {checkpoint_dir} matching {checkpoint_glob}")
    return checkpoints


def write_csv(output_csv: Path, rows: list[dict[str, object]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "checkpoint_path", "masked_au_macro_f1"])
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, object]]) -> None:
    print("\n" + "=" * 86)
    print(f"{'label':>12} | {'masked AU macro F1':>18} | checkpoint")
    print("-" * 86)
    for row in rows:
        print(f"{row['label']:>12} | {row['masked_au_macro_f1']*100:>17.2f}% | {row['checkpoint_path']}")
    print("=" * 86)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved low-frequency checkpoints on masked test images")
    parser.add_argument("--checkpoint_dir", default=str(DEFAULT_OUTPUT_DIR / "lowfreq_fmae_fold1"))
    parser.add_argument("--checkpoint_glob", default="fold1_epoch*.pth")
    parser.add_argument("--labels", nargs="*", default=None, help="Optional labels matching the checkpoint order")
    parser.add_argument("--model_type", default="FMAE", choices=["FMAE", "IAT"])
    parser.add_argument("--mask_type", type=str, default="low", choices=["low", "high", "full"], help="Frequency mask type: 'low' (center), 'high' (outer), 'full' (no mask)")
    parser.add_argument("--test_json", default="BP4D/BP4D_test1.json")
    parser.add_argument("--output_csv", default=str(DEFAULT_OUTPUT_DIR / "lowfreq_masked_eval_results.csv"))
    parser.add_argument("--min_keep", type=float, default=0.08)
    parser.add_argument("--max_keep", type=float, default=0.20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = resolve_device(args.device)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoints = list_checkpoints(checkpoint_dir, args.checkpoint_glob)

    if args.labels is not None and len(args.labels) not in (0, len(checkpoints)):
        raise ValueError("--labels must be omitted or have the same length as the discovered checkpoints")

    test_dataset = BP4D_AU_dataset(args.test_json, is_train=False, args=build_dataset_args(DEFAULT_ROOT_PATH))
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=(args.num_workers > 0),
    )

    rows: list[dict[str, object]] = []
    for index, checkpoint_path in enumerate(checkpoints):
        label = args.labels[index] if args.labels else checkpoint_path.stem
        print(f"\nEvaluating {label}: {checkpoint_path}")
        model = build_model(checkpoint_path, device, model_type=args.model_type)
        masked_f1 = evaluate_au_macro_f1_masked(model, test_loader, device, args.min_keep, args.max_keep, mask_type=args.mask_type)

        rows.append(
            {
                "label": label,
                "checkpoint_path": str(checkpoint_path),
                "masked_au_macro_f1": round(masked_f1, 6),
            }
        )
        print(f"   Masked AU macro F1: {masked_f1*100:.2f}%")

    output_csv = Path(args.output_csv)
    write_csv(output_csv, rows)
    print(f"\nSaved results to {output_csv}")
    print_summary(rows)


if __name__ == "__main__":
    main()


"""
python eval_lowfreq_masked_checkpoints.py \
  --checkpoint_dir output/lowfreq_fmae_fold1 \
  --checkpoint_glob 'fold1_epoch*.pth' \
  --test_json BP4D/BP4D_test1.json \
  --output_csv output/lowfreq_masked_eval_results.csv
"""