"""
Compare multiple FMAE checkpoints with a fresh identity probe per checkpoint.

For each checkpoint:
- load the frozen FMAE backbone
- train a new BN + Linear probe for subject identity
- report test identity accuracy
- report AU macro F1 on the clean test split
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

from linear_probe_identity import (
    FIXED_BATCH_SIZE,
    FIXED_DATA_ROOT,
    FIXED_NUM_SUBJECTS,
    FIXED_PROBE_BLR,
    FIXED_PROBE_EPOCHS,
    FIXED_PROBE_WEIGHT_DECAY,
    LinearProbe,
    build_dataset_args,
    build_fixed_samples_per_subject_split,
    evaluate_au_macro_f1,
    evaluate_probe_on_loader,
    run_sanity_checks,
    set_global_seed,
    terminal_tqdm,
)
from util.lars import LARS  # pyright: ignore[reportMissingImports]
from util.datasets import BP4D_AU_dataset  # pyright: ignore[reportMissingImports]
import models_vit  # pyright: ignore[reportMissingImports]


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_OUTPUT_CSV = OUTPUT_DIR / "lowfreq_probe_results.csv"


def resolve_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_fmae_backbone(checkpoint_path: str | Path, device: torch.device, model_type: str = "FMAE"):
    num_subjects = FIXED_NUM_SUBJECTS if model_type == "IAT" else 0
    model = models_vit.vit_large_patch16(
        num_classes=12,
        num_subjects=num_subjects,
        drop_path_rate=0.0,
        global_pool=True,
        grad_reverse=0.0,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)

    # IAT checkpoints include an identity head that is not used by this probe script.
    if model_type == "IAT":
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if not (
                key.startswith("ID_head.")
                or key.startswith("module.ID_head.")
                or key.endswith(".ID_head.weight")
                or key.endswith(".ID_head.bias")
            )
        }

    model.load_state_dict(state_dict, strict=True)

    model = model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def train_linear_probe(backbone, probe, train_loader, epochs, lr, weight_decay, device):
    backbone.eval()
    probe = probe.to(device)

    optimizer = LARS(probe.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    warmup_epochs = 10

    def get_lr(epoch):
        if epoch < warmup_epochs:
            return lr * (epoch + 1) / warmup_epochs
        return lr

    for epoch in range(epochs):
        current_lr = get_lr(epoch)
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        probe.train()
        for images, (_, id_onehot) in terminal_tqdm(
            train_loader, desc=f"   Epoch {epoch + 1}/{epochs}", leave=False
        ):
            images = images.to(device, non_blocking=True)
            features = backbone.forward_features(images).detach()
            labels = torch.argmax(id_onehot.to(device, non_blocking=True), dim=1)

            optimizer.zero_grad(set_to_none=True)
            logits = probe(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

    return probe


@torch.no_grad()
def evaluate_probe_on_features(probe, features, labels, device):
    probe.eval()
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=FIXED_BATCH_SIZE, shuffle=False, num_workers=0)

    all_preds = []
    all_labels = []
    for batch_features, batch_labels in loader:
        batch_features = batch_features.to(device, non_blocking=True)
        logits = probe(batch_features)
        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(batch_labels.cpu())

    if not all_preds:
        return 0.0

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    return float((all_preds == all_labels).float().mean().item())


def write_results_csv(output_csv, rows):
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["label", "checkpoint_path", "test_id_accuracy", "au_macro_f1"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_summary_table(rows):
    print("\n" + "=" * 90)
    print(f"{'label':>12} | {'ID acc':>10} | {'AU macro F1':>12} | checkpoint")
    print("-" * 90)
    for row in rows:
        print(
            f"{row['label']:>12} | {row['test_id_accuracy']*100:>9.2f}% | "
            f"{row['au_macro_f1']*100:>11.2f}% | {row['checkpoint_path']}"
        )
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Compare multiple FMAE checkpoints with a fresh identity probe")
    parser.add_argument("--model_type", default="FMAE", choices=["FMAE", "IAT"])
    parser.add_argument("--checkpoint_paths", nargs="+", required=True, help="One or more .pth checkpoint paths")
    parser.add_argument("--checkpoint_labels", nargs="+", required=True, help="One label per checkpoint path")
    parser.add_argument("--train_json", default="BP4D/BP4D_train1.json")
    parser.add_argument("--test_json", default="BP4D/BP4D_test1.json")
    parser.add_argument("--output_csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if len(args.checkpoint_paths) != len(args.checkpoint_labels):
        raise ValueError("--checkpoint_paths and --checkpoint_labels must have the same length")

    device = resolve_device()
    set_global_seed(args.seed, str(device))

    train_dataset, test_dataset, sampling_audit = build_fixed_samples_per_subject_split(
        [args.train_json, args.test_json],
        FIXED_DATA_ROOT,
        train_samples=70,
        test_samples=30,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=FIXED_BATCH_SIZE,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=(args.num_workers > 0),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=FIXED_BATCH_SIZE,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=(args.num_workers > 0),
    )

    au_eval_dataset = BP4D_AU_dataset(args.test_json, is_train=False, args=build_dataset_args(FIXED_DATA_ROOT))
    au_eval_loader = DataLoader(
        au_eval_dataset,
        batch_size=FIXED_BATCH_SIZE,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=(args.num_workers > 0),
    )

    rows = []
    for checkpoint_path, label in zip(args.checkpoint_paths, args.checkpoint_labels):
        print("\n" + "=" * 90)
        print(f"Checkpoint: {label}")
        print(f"Path: {checkpoint_path}")
        print(f"Model type: {args.model_type}")
        print("=" * 90)

        backbone = load_fmae_backbone(checkpoint_path, device, model_type=args.model_type)
        feature_dim = backbone.head.in_features if hasattr(backbone, "head") else 1024
        probe = LinearProbe(feature_dim=feature_dim, num_subjects=FIXED_NUM_SUBJECTS)

        run_sanity_checks(backbone, probe, train_dataset, test_dataset, sampling_audit, 70, 30)

        probe_lr = FIXED_PROBE_BLR * FIXED_BATCH_SIZE / 256.0
        probe = train_linear_probe(
            backbone,
            probe,
            train_loader,
            epochs=FIXED_PROBE_EPOCHS,
            lr=probe_lr,
            weight_decay=FIXED_PROBE_WEIGHT_DECAY,
            device=device,
        )

        test_id_acc = evaluate_probe_on_loader(backbone, probe, test_loader, device)
        au_macro_f1 = evaluate_au_macro_f1(backbone, au_eval_loader, None, device)

        rows.append(
            {
                "label": label,
                "checkpoint_path": checkpoint_path,
                "test_id_accuracy": round(test_id_acc, 6),
                "au_macro_f1": round(au_macro_f1, 6),
            }
        )

        print(f"   ID accuracy: {test_id_acc*100:.2f}%")
        print(f"   AU macro F1: {au_macro_f1*100:.2f}%")

    write_results_csv(args.output_csv, rows)
    print(f"\nSaved CSV results to {args.output_csv}")
    print_summary_table(rows)


if __name__ == "__main__":
    main()


"""
python probe_lowfreq_checkpoints.py \
    --checkpoint_paths \
        output/lowfreq_fmae_fold1/fold1_epoch003.pth \
        output/lowfreq_fmae_fold1/fold1_epoch030.pth \
        models/FMAE_BP4D_fold1.pth \
    --checkpoint_labels epoch3 epoch30 baseline \
    --train_json BP4D/BP4D_train1.json \
    --test_json BP4D/BP4D_test1.json \
    --output_csv output/lowfreq_probe_results.csv

python probe_lowfreq_checkpoints.py \
    --model_type IAT \
    --checkpoint_paths \
        output/lowfreq_iat_fold1/fold1_epoch002.pth \
        output/lowfreq_iat_fold1/fold1_epoch030.pth \
        models/FMAE_IAT_BP4D_fold1.pth \
    --checkpoint_labels epoch2 epoch30 baseline_iat \
    --train_json BP4D/BP4D_train1.json \
    --test_json BP4D/BP4D_test1.json \
    --output_csv output/lowfreq_iat_probe_results.csv

"""