"""
Linear probe for BP4D subject identity analysis.

Trains a frozen FMAE backbone + linear classifier on subject IDs, then
evaluates the trained probe on radial frequency-masked test images while
also reporting AU macro F1 with the frozen AU head.
"""

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from warnings import filterwarnings
filterwarnings("ignore") 


PROJECT_ROOT = Path(__file__).resolve().parent
FMAE_PATH = PROJECT_ROOT / "fmae"
HFSS_PATH = PROJECT_ROOT / "hfss" / "hfss"
sys.path.insert(0, str(FMAE_PATH))
sys.path.insert(0, str(HFSS_PATH))

import models_vit  # pyright: ignore[reportMissingImports]
from util.datasets import BP4D_AU_dataset  # pyright: ignore[reportMissingImports]
from transforms_search_space import White_Mask  # pyright: ignore[reportMissingImports]

from hfss_search_au import (
    IMG_SIZE,
    TeeStream,
    load_masks_from_pkl,
    load_model,
    set_global_seed,
)


FIXED_DATA_ROOT = "BP4D/BP4D_cropped/"
FIXED_BATCH_SIZE = 64
FIXED_NUM_WORKERS = 4
FIXED_DEVICE = "cuda"
FIXED_SEED = 42
FIXED_NUM_SUBJECTS = 41
FIXED_PROBE_EPOCHS = 30
FIXED_PROBE_LR = 1e-3
FIXED_RADIAL_STEPS = 10
FIXED_RADIAL_START_KEEP_PCT = 100.0
FIXED_RADIAL_DIRECTION = "big_to_small"  # 'big_to_small' | 'small_to_big'


def create_linear_probe_log_file(log_dir):
    """Create timestamped log file path for linear probe runs."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"linear_probe_{ts}.txt"


@dataclass
class MaskRecord:
    step: int
    path: Path
    mask: White_Mask
    target_keep_pct: Optional[float] = None


class LinearProbe(nn.Module):
    def __init__(self, feature_dim, num_subjects):
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_subjects)

    def forward(self, features):
        return self.classifier(features)


def resolve_device():
    if FIXED_DEVICE.startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_dataset_args(data_root):
    return SimpleNamespace(
        root_path=data_root,
        input_size=IMG_SIZE,
        color_jitter=None,
        aa="rand-m9-mstd0.5-inc1",
        reprob=0.25,
        remode="pixel",
        recount=1,
    )


def build_dataloader(json_path, data_root, is_train, batch_size, num_workers, shuffle):
    dataset_args = build_dataset_args(data_root)
    dataset = BP4D_AU_dataset(json_path, is_train=is_train, args=dataset_args)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=resolve_device().type == "cuda",
        persistent_workers=(num_workers > 0),
    )
    return dataset, dataloader


def build_within_subject_split(all_json_paths, data_root, train_ratio=0.7, seed=42):
    """Build probe train/test splits by sampling frames within each subject.

    All JSON paths are combined first, then each subject's frames are split
    independently so both probe_train and probe_test contain all subjects.
    """
    if isinstance(all_json_paths, (str, Path)):
        all_json_paths = [all_json_paths]

    combined_data = []
    for json_path in all_json_paths:
        with open(json_path, "r", encoding="utf-8") as f:
            for line in f:
                combined_data.append(json.loads(line))

    if not combined_data:
        raise ValueError("No BP4D samples found in the provided JSON paths.")

    grouped_by_subject = {}
    for entry in combined_data:
        subject_id = entry["img_path"][1:4]
        grouped_by_subject.setdefault(subject_id, []).append(entry)

    rng = np.random.default_rng(seed)
    train_entries = []
    test_entries = []

    for subject_id in sorted(grouped_by_subject.keys()):
        subject_entries = grouped_by_subject[subject_id]
        if len(subject_entries) < 2:
            raise ValueError(f"Subject {subject_id} has fewer than 2 frames; cannot make a 70/30 split.")

        order = np.arange(len(subject_entries))
        rng.shuffle(order)

        train_count = int(round(len(subject_entries) * train_ratio))
        train_count = max(1, min(len(subject_entries) - 1, train_count))

        train_entries.extend(subject_entries[i] for i in order[:train_count])
        test_entries.extend(subject_entries[i] for i in order[train_count:])

    rng.shuffle(train_entries)
    rng.shuffle(test_entries)

    dataset_args = build_dataset_args(data_root)
    train_dataset = BP4D_AU_dataset(all_json_paths[0], is_train=True, args=dataset_args)
    test_dataset = BP4D_AU_dataset(all_json_paths[0], is_train=False, args=dataset_args)
    train_dataset.data = train_entries
    test_dataset.data = test_entries
    return train_dataset, test_dataset


def validate_subject_mapping(dataset, split_name):
    # BP4D labels are encoded as Fxx/Mxx in the dataset logic (img_path[1:4]).
    subject_ids = sorted({entry["img_path"][1:4] for entry in dataset.data})
    print(
        f"   {split_name} subject IDs: {len(subject_ids)} unique | "
        f"mapped classes: {len(dataset.IDs)}"
    )
    print(f"   Sample IDs: {', '.join(subject_ids[:5])}{' ...' if len(subject_ids) > 5 else ''}")
    if len(dataset.IDs) != FIXED_NUM_SUBJECTS:
        raise ValueError(f"Expected {FIXED_NUM_SUBJECTS} BP4D subject classes, got {len(dataset.IDs)}")
    return set(subject_ids)


def load_backbone(model_path, device):
    model = load_model(model_path, model_type="FMAE", device=str(device))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


@torch.no_grad()
def extract_features_and_labels(backbone, dataloader, device):
    features = []
    labels = []

    for images, (_, id_onehot) in tqdm(dataloader, desc="   Extracting features", leave=False):
        images = images.to(device, non_blocking=True)
        id_labels = torch.argmax(id_onehot.to(device, non_blocking=True), dim=1)
        pooled = backbone.forward_features(images)
        features.append(pooled.detach().cpu())
        labels.append(id_labels.detach().cpu())

    if not features:
        return torch.empty(0), torch.empty(0, dtype=torch.long)

    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def train_linear_probe(probe, train_features, train_labels, epochs, lr, device):
    probe = probe.to(device)
    train_dataset = TensorDataset(train_features, train_labels)
    train_loader = DataLoader(
        train_dataset,
        batch_size=FIXED_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        probe.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_features, batch_labels in tqdm(train_loader, desc=f"   Epoch {epoch + 1}/{epochs}", leave=False):
            batch_features = batch_features.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = probe(batch_features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * batch_labels.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_labels).sum().item()
            total += batch_labels.size(0)

        epoch_loss = running_loss / max(total, 1)
        epoch_acc = correct / max(total, 1)
        print(f"   Epoch {epoch + 1:02d}/{epochs}: loss={epoch_loss:.4f} | acc={epoch_acc*100:.2f}%")

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


def _mask_to_tensor(mask_record, device):
    mask_obj = mask_record.mask if isinstance(mask_record, MaskRecord) else mask_record
    if hasattr(mask_obj, "mask"):
        mask_np = mask_obj.mask
        if getattr(mask_obj, "flip", False):
            mask_np = 1 - mask_np
    else:
        mask_np = np.asarray(mask_obj)

    return torch.as_tensor(mask_np, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)


def apply_frequency_mask(images, mask_record, device):
    if mask_record is None:
        return images

    mask_t = _mask_to_tensor(mask_record, device)
    freq = torch.fft.fftshift(torch.fft.fft2(images, dim=(-2, -1)), dim=(-2, -1))
    freq = freq * mask_t
    return torch.fft.ifft2(torch.fft.ifftshift(freq, dim=(-2, -1)), dim=(-2, -1)).real.float()


def load_radial_masks(dfm_dir):
    dfm_dir = Path(dfm_dir)
    if not dfm_dir.exists():
        raise FileNotFoundError(f"DFM directory not found: {dfm_dir}")

    candidates = []
    for pkl_path in dfm_dir.rglob("*.pkl"):
        match = re.search(r"radial_step_(\d+)", pkl_path.name)
        if match is None:
            continue
        candidates.append((int(match.group(1)), pkl_path))

    candidates.sort(key=lambda item: item[0])
    if not candidates:
        raise FileNotFoundError(
            f"No radial_step_*.pkl files found under {dfm_dir}. "
            "Expected radial DFM masks saved by hfss_search_au.py."
        )

    records = []
    for step, pkl_path in candidates:
        loaded = load_masks_from_pkl(pkl_path)
        mask_obj = loaded[0] if isinstance(loaded, list) else loaded
        if not hasattr(mask_obj, "mask"):
            mask_obj = White_Mask(np.asarray(mask_obj))
        records.append(MaskRecord(step=step, path=pkl_path, mask=mask_obj))
    return records


def load_radial_masks_strict(dfm_dir):
    """Load only canonical macro radial masks: FMAE_radial_step_XX_DFMs.pkl."""
    dfm_dir = Path(dfm_dir)
    if not dfm_dir.exists():
        raise FileNotFoundError(f"DFM directory not found: {dfm_dir}")

    candidates = []
    pattern = re.compile(r"^FMAE_radial_step_(\d+)_DFMs\.pkl$")
    for pkl_path in dfm_dir.rglob("*.pkl"):
        match = pattern.match(pkl_path.name)
        if match is None:
            continue
        candidates.append((int(match.group(1)), pkl_path))

    candidates.sort(key=lambda item: item[0])
    if not candidates:
        raise FileNotFoundError(
            f"No canonical radial masks found under {dfm_dir}. "
            "Expected files named FMAE_radial_step_XX_DFMs.pkl"
        )

    records = []
    for step, pkl_path in candidates:
        loaded = load_masks_from_pkl(pkl_path)
        mask_obj = loaded[0] if isinstance(loaded, list) else loaded
        if not hasattr(mask_obj, "mask"):
            mask_obj = White_Mask(np.asarray(mask_obj))
        records.append(MaskRecord(step=step, path=pkl_path, mask=mask_obj))
    return records


def generate_radial_masks(num_steps, start_keep_pct=100.0, direction="small_to_big"):
    """Generate deterministic radial masks directly (preferred for reproducibility)."""
    if num_steps < 2:
        num_steps = 2

    if direction not in ("big_to_small", "small_to_big"):
        raise ValueError("direction must be 'big_to_small' or 'small_to_big'")

    h = IMG_SIZE
    w = IMG_SIZE
    cy, cx = h // 2, w // 2
    ys, xs = np.ogrid[:h, :w]
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)

    max_radius = float(dist.max())
    start_keep_pct = float(np.clip(start_keep_pct, 0.1, 100.0))
    end_keep_pct = max(start_keep_pct / float(num_steps), 0.1)
    target_keep_pcts = np.linspace(start_keep_pct, end_keep_pct, num_steps)
    if direction == "small_to_big":
        target_keep_pcts = target_keep_pcts[::-1]
    radii = np.sqrt(target_keep_pcts / 100.0) * max_radius

    records = []
    for idx, (r, target_keep) in enumerate(zip(radii, target_keep_pcts)):
        mask = (dist <= r).astype(np.float32)
        wm = White_Mask(mask)
        wm.white_count = int(mask.sum())
        wm.keep_pct = float(mask.mean())
        records.append(
            MaskRecord(
                step=idx + 1,
                path=Path(f"generated_radial_step_{idx + 1:02d}"),
                mask=wm,
                target_keep_pct=float(target_keep),
            )
        )
    return records


def _mask_keep_ratio_pct(mask_obj):
    mask_array = mask_obj.mask if hasattr(mask_obj, "mask") else np.asarray(mask_obj)
    return float(np.mean(mask_array > 0.5) * 100.0)


@torch.no_grad()
def evaluate_probe_and_au(backbone, probe, dataloader, mask_record, device):
    probe.eval()
    backbone.eval()

    id_preds_all = []
    id_labels_all = []
    au_preds_all = []
    au_labels_all = []

    for images, (au_labels, id_onehot) in dataloader:
        images = images.to(device, non_blocking=True)
        au_labels = au_labels.to(device, non_blocking=True)
        id_labels = torch.argmax(id_onehot.to(device, non_blocking=True), dim=1)
        images = apply_frequency_mask(images, mask_record, device)

        # Single backbone pass: reuse same frozen features for both ID probe and AU head.
        features = backbone.forward_features(images)
        id_logits = probe(features)
        id_preds = torch.argmax(id_logits, dim=1)
        au_logits = backbone.head(features)
        au_preds = (torch.sigmoid(au_logits) >= 0.5).float()

        id_preds_all.append(id_preds.cpu())
        id_labels_all.append(id_labels.cpu())
        au_preds_all.append(au_preds.cpu().numpy())
        au_labels_all.append(au_labels.cpu().numpy())

    if id_preds_all:
        id_preds_all = torch.cat(id_preds_all, dim=0)
        id_labels_all = torch.cat(id_labels_all, dim=0)
        id_accuracy = float((id_preds_all == id_labels_all).float().mean().item())
    else:
        id_accuracy = 0.0

    if au_preds_all:
        from sklearn.metrics import f1_score

        au_preds_all = np.concatenate(au_preds_all, axis=0)
        au_labels_all = np.concatenate(au_labels_all, axis=0)
        # Match HFSS evaluate_mask() exactly: sklearn multilabel macro F1.
        au_macro_f1 = float(f1_score(au_labels_all, au_preds_all, average="macro"))
    else:
        au_macro_f1 = 0.0

    return id_accuracy, au_macro_f1


def write_results_csv(output_csv, rows):
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["keep_ratio", "id_accuracy", "au_macro_f1", "au_f1_drop"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_summary_table(rows):
    print("\n" + "=" * 78)
    print(f"{'Keep':>8} | {'ID acc':>9} | {'AU macro F1':>11} | {'AU drop':>8}")
    print("-" * 78)
    for row in rows:
        print(
            f"{row['keep_ratio']:>7.1f}% | "
            f"{row['id_accuracy']*100:>8.2f}% | "
            f"{row['au_macro_f1']*100:>10.2f}% | "
            f"{row['au_f1_drop']*100:>7.2f}%"
        )
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(description="Linear probe identity analysis for BP4D/FMAE")
    parser.add_argument("--model_path", default="models/FMAE_BP4D_fold1.pth")
    parser.add_argument("--train_json", default="BP4D/BP4D_train1.json")
    parser.add_argument("--test_json", default="BP4D/BP4D_test1.json")
    parser.add_argument("--dfm_dir", default="hfss/DFM")
    parser.add_argument(
        "--mask_source",
        default="generated",
        choices=["generated", "dfm"],
        help="generated: create 10 deterministic radial masks in-script; dfm: load canonical FMAE radial pkl files",
    )
    parser.add_argument("--probe_save_path", default="hfss/DFM/linear_probe_identity.pth")
    parser.add_argument("--probe_epochs", type=int, default=FIXED_PROBE_EPOCHS)
    parser.add_argument("--probe_lr", type=float, default=FIXED_PROBE_LR)
    parser.add_argument("--output_csv", default="hfss/DFM/linear_probe_identity_results.csv")
    args = parser.parse_args()

    device = resolve_device()
    set_global_seed(FIXED_SEED, str(device))

    output_csv = Path(args.output_csv)
    probe_save_path = Path(args.probe_save_path)
    probe_save_path.parent.mkdir(parents=True, exist_ok=True)

    logs_dir = PROJECT_ROOT / "hfss" / "logs"
    log_file_path = create_linear_probe_log_file(logs_dir)
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    log_fp = open(log_file_path, "w", encoding="utf-8")
    sys.stdout = TeeStream(orig_stdout, log_fp)
    sys.stderr = TeeStream(orig_stderr, log_fp)

    try:
        print("=" * 70)
        print("Linear Probe Identity Analysis - BP4D")
        print(f"Model: {args.model_path} | Device: {device}")
        print(f"Seed: {FIXED_SEED}")
        print(f"Log file: {log_file_path}")
        print("=" * 70)

        backbone = load_backbone(args.model_path, device)
        feature_dim = backbone.head.in_features if hasattr(backbone, "head") else 1024
        print(f"   Backbone feature dim: {feature_dim}")

        print("\n[1] Building datasets")
        train_dataset, test_dataset = build_within_subject_split(
            [args.train_json, args.test_json],
            FIXED_DATA_ROOT,
            train_ratio=0.7,
            seed=FIXED_SEED,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=FIXED_BATCH_SIZE,
            shuffle=True,
            num_workers=FIXED_NUM_WORKERS,
            pin_memory=device.type == "cuda",
            persistent_workers=(FIXED_NUM_WORKERS > 0),
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=FIXED_BATCH_SIZE,
            shuffle=False,
            num_workers=FIXED_NUM_WORKERS,
            pin_memory=device.type == "cuda",
            persistent_workers=(FIXED_NUM_WORKERS > 0),
        )
        train_subjects = validate_subject_mapping(train_dataset, "Train")
        test_subjects = validate_subject_mapping(test_dataset, "Test")
        overlap = train_subjects.intersection(test_subjects)
        if len(overlap) != FIXED_NUM_SUBJECTS:
            raise ValueError(
                f"Within-subject split failed: expected all {FIXED_NUM_SUBJECTS} subjects in both splits, got overlap={len(overlap)}"
            )
        print(f"   Subject overlap train∩test: {len(overlap)} IDs")
        print(f"   Train samples: {len(train_dataset)}")
        print(f"   Test samples:  {len(test_dataset)}")

        print("\n[2] Precomputing frozen backbone features")
        train_features, train_labels = extract_features_and_labels(backbone, train_loader, device)
        test_features, test_labels = extract_features_and_labels(backbone, test_loader, device)
        print(f"   Train feature tensor: {tuple(train_features.shape)}")
        print(f"   Test feature tensor:  {tuple(test_features.shape)}")

        print("\n[3] Training linear probe")
        probe = LinearProbe(feature_dim=feature_dim, num_subjects=FIXED_NUM_SUBJECTS)
        probe = train_linear_probe(probe, train_features, train_labels, args.probe_epochs, args.probe_lr, device)

        torch.save(
            {
                "probe_state_dict": probe.state_dict(),
                "feature_dim": feature_dim,
                "num_subjects": FIXED_NUM_SUBJECTS,
                "model_path": args.model_path,
                "train_json": args.train_json,
                "test_json": args.test_json,
            },
            probe_save_path,
        )
        print(f"   ✓ Saved probe weights to {probe_save_path}")

        train_id_acc = evaluate_probe_on_features(probe, train_features, train_labels, device)
        print(f"\n   Train ID accuracy (sanity): {train_id_acc*100:.2f}%")
        test_id_acc = evaluate_probe_on_features(probe, test_features, test_labels, device)
        print(f"\n   Unmasked test ID accuracy: {test_id_acc*100:.2f}%")
        if len(overlap) == 0:
            print("   ⚠ This value is not comparable to closed-set identity numbers (e.g., ~83%).")

        print("\n[4] Computing AU baseline")
        _, baseline_macro_f1 = evaluate_probe_and_au(backbone, probe, test_loader, None, device)
        print(f"   Baseline AU macro F1: {baseline_macro_f1*100:.2f}%")

        print("\n[5] Preparing radial masks")
        if args.mask_source == "generated":
            mask_records = generate_radial_masks(
                num_steps=FIXED_RADIAL_STEPS,
                start_keep_pct=FIXED_RADIAL_START_KEEP_PCT,
                direction=FIXED_RADIAL_DIRECTION,
            )
            print(
                f"   Using generated masks: steps={FIXED_RADIAL_STEPS}, "
                f"start_keep={FIXED_RADIAL_START_KEEP_PCT:.1f}%, direction={FIXED_RADIAL_DIRECTION}"
            )
        else:
            mask_records = load_radial_masks_strict(args.dfm_dir)
            print("   Using DFM masks filtered to canonical macro files: FMAE_radial_step_XX_DFMs.pkl")
        print(f"   Loaded {len(mask_records)} radial mask files")

        rows = []
        for record in tqdm(mask_records, desc="   Radial steps", leave=False):
            keep_ratio = _mask_keep_ratio_pct(record.mask)
            target_keep_ratio = (
                float(record.target_keep_pct)
                if getattr(record, "target_keep_pct", None) is not None
                else keep_ratio
            )
            print(f"\n   Step {record.step:02d} | file={record.path.name}")
            id_accuracy, au_macro_f1 = evaluate_probe_and_au(
                backbone,
                probe,
                test_loader,
                record,
                device,
            )
            au_f1_drop = float(baseline_macro_f1 - au_macro_f1)
            print(
                f"   Keep(target/actual)={target_keep_ratio:.1f}%/{keep_ratio:.1f}% | "
                f"ID accuracy: {id_accuracy*100:.2f}% | "
                f"AU macro F1: {au_macro_f1*100:.2f}% | AU F1 drop: {au_f1_drop*100:+.2f}%"
            )
            rows.append(
                {
                    "keep_ratio": round(target_keep_ratio, 4),
                    "id_accuracy": round(id_accuracy, 6),
                    "au_macro_f1": round(au_macro_f1, 6),
                    "au_f1_drop": round(au_f1_drop, 6),
                }
            )

        write_results_csv(output_csv, rows)
        print(f"\n   ✓ Saved CSV results to {output_csv}")
        print_summary_table(rows)

    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        log_fp.close()


if __name__ == "__main__":
    main()


"""
python linear_probe_identity.py \
    --model_path models/FMAE_BP4D_fold1.pth \
    --train_json BP4D/BP4D_train1.json \
    --test_json BP4D/BP4D_test1.json \
    --dfm_dir hfss/DFM \
    --probe_save_path hfss/DFM/linear_probe_identity.pth \
    --probe_epochs 30 \
    --probe_lr 0.001 \
    --output_csv hfss/DFM/linear_probe_identity_results.csv

python linear_probe_identity.py \
  --model_path models/FMAE_BP4D_fold1.pth \
  --train_json BP4D/BP4D_train1.json \
  --test_json BP4D/BP4D_test1.json \
  --dfm_dir hfss/DFM \
  --mask_source dfm \
  --output_csv hfss/DFM/linear_probe_identity_results.csv
"""