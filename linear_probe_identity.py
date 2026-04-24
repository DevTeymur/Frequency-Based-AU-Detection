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
from warnings import filterwarnings, warn

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
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
FIXED_TRAIN_SAMPLES = 70
FIXED_TEST_SAMPLES = 30
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


@dataclass
class SamplingAudit:
    examples_by_subject: dict
    overlap_by_subject: dict


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


def _extract_frame_index(img_path):
    match = re.search(r"frame_(\d+)", img_path)
    return int(match.group(1)) if match else None


def build_fixed_samples_per_subject_split(all_json_paths, data_root, train_samples, test_samples, seed=42):
    """Sample exactly N train/test frames per subject from combined JSON files.

    This matches fixed-count per-subject sampling used by linear probe protocols.
    Subjects with insufficient frames are skipped with a warning.
    """
    if isinstance(all_json_paths, (str, Path)):
        all_json_paths = [all_json_paths]

    train_samples = int(train_samples)
    test_samples = int(test_samples)
    if train_samples <= 0 or test_samples <= 0:
        raise ValueError("train_samples and test_samples must be positive integers.")

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
    skipped_subjects = []
    examples_by_subject = {}
    overlap_by_subject = {}
    required = train_samples + test_samples

    for subject_id in sorted(grouped_by_subject.keys()):
        subject_entries = grouped_by_subject[subject_id]
        if len(subject_entries) < required:
            skipped_subjects.append((subject_id, len(subject_entries)))
            continue

        order = rng.permutation(len(subject_entries))
        selected = order[:required]
        train_idx = selected[:train_samples]
        test_idx = selected[train_samples:]

        train_subject_entries = [subject_entries[i] for i in train_idx]
        test_subject_entries = [subject_entries[i] for i in test_idx]
        train_entries.extend(train_subject_entries)
        test_entries.extend(test_subject_entries)

        train_paths = {e["img_path"] for e in train_subject_entries}
        test_paths = {e["img_path"] for e in test_subject_entries}
        overlap_by_subject[subject_id] = len(train_paths.intersection(test_paths))

        examples_by_subject[subject_id] = {
            "train_frame_indices": [_extract_frame_index(e["img_path"]) for e in train_subject_entries],
            "test_frame_indices": [_extract_frame_index(e["img_path"]) for e in test_subject_entries],
        }

    if skipped_subjects:
        preview = ", ".join([f"{sid}({count})" for sid, count in skipped_subjects[:8]])
        suffix = " ..." if len(skipped_subjects) > 8 else ""
        warn(
            f"Skipped {len(skipped_subjects)} subject(s) with < {required} frames: {preview}{suffix}",
            stacklevel=1,
        )

    if not train_entries or not test_entries:
        raise ValueError(
            "No valid subjects left after fixed-count sampling. "
            "Reduce --train_samples/--test_samples or verify JSON files."
        )

    rng.shuffle(train_entries)
    rng.shuffle(test_entries)

    dataset_args = build_dataset_args(data_root)
    train_dataset = BP4D_AU_dataset(all_json_paths[0], is_train=True, args=dataset_args)
    test_dataset = BP4D_AU_dataset(all_json_paths[0], is_train=False, args=dataset_args)
    train_dataset.data = train_entries
    test_dataset.data = test_entries
    audit = SamplingAudit(examples_by_subject=examples_by_subject, overlap_by_subject=overlap_by_subject)
    return train_dataset, test_dataset, audit


def _subject_set_from_json(json_path):
    subjects = set()
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            subjects.add(entry["img_path"][1:4])
    return subjects


def ensure_subject_exclusive_au_split(train_json, test_json):
    """Ensure AU evaluation uses a subject-exclusive test split."""
    train_subjects = _subject_set_from_json(train_json)
    test_subjects = _subject_set_from_json(test_json)
    overlap = train_subjects.intersection(test_subjects)

    print(f"   AU split subjects | train={len(train_subjects)} test={len(test_subjects)} overlap={len(overlap)}")
    if overlap:
        sample = ", ".join(sorted(list(overlap))[:8])
        raise ValueError(
            "AU evaluation split is not subject-exclusive. "
            f"Overlapping subjects ({len(overlap)}): {sample}"
        )


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
    if len(subject_ids) != FIXED_NUM_SUBJECTS:
        raise ValueError(
            f"{split_name} split does not contain all {FIXED_NUM_SUBJECTS} subjects. "
            f"Found {len(subject_ids)} unique subjects."
        )
    return set(subject_ids)


def print_subject_mapping(train_dataset, test_dataset):
    train_map = getattr(train_dataset, "ID_label2idx", {})
    test_map = getattr(test_dataset, "ID_label2idx", {})
    consistent = train_map == test_map

    print("\n   Full subject ID -> class index mapping:")
    for subject_id, class_idx in sorted(train_map.items(), key=lambda kv: kv[1]):
        print(f"     {subject_id} -> {class_idx}")

    print(f"   Subject mapping consistent (train vs test): {consistent}")
    if not consistent:
        raise ValueError("Train/test subject mapping mismatch detected.")

    return consistent


def load_backbone(model_path, model_type, device):
    model = models_vit.vit_large_patch16(
        num_classes=12,
        num_subjects=FIXED_NUM_SUBJECTS if model_type == "IAT" else 0,
        drop_path_rate=0.0,
        global_pool=True,
        grad_reverse=1.0 if model_type == "IAT" else 0.0,
    )

    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    loaded_keys = list(state_dict.keys())
    if not loaded_keys:
        raise ValueError("Checkpoint has no parameters in state_dict.")

    print("   Checkpoint keys (first 5):")
    for key in loaded_keys[:5]:
        print(f"     {key}")
    print("   Checkpoint keys (last 5):")
    for key in loaded_keys[-5:]:
        print(f"     {key}")

    model.load_state_dict(state_dict, strict=True)

    # For IAT linear-probe analysis we intentionally bypass ID path; keep GRL inactive.
    if model_type == "IAT":
        model.grad_reverse = 0.0
        print("   IAT grad_reverse set to 0.0 for inference-time probing (GRL inactive)")

    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    print(f"   ✓ Loaded {model_type} model with strict=True from {model_path}")
    print(f"   model.eval() set: {not model.training}")
    return model


@torch.no_grad()
def extract_features_and_labels(backbone, dataloader, device):
    if backbone.training:
        raise RuntimeError("Backbone is in training mode during feature extraction. Expected eval mode.")

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


def load_radial_masks_saved(dfm_dir, model_type):
    """Load saved canonical radial masks from PKL files."""
    dfm_dir = Path(dfm_dir)
    if not dfm_dir.exists():
        raise FileNotFoundError(f"DFM directory not found: {dfm_dir}")

    candidates = []
    pattern = re.compile(rf"^{re.escape(model_type)}_radial_step_(\d+)_DFMs\.pkl$")
    for pkl_path in dfm_dir.rglob("*.pkl"):
        match = pattern.match(pkl_path.name)
        if match is None:
            continue
        candidates.append((int(match.group(1)), pkl_path))

    if not candidates:
        fallback = re.compile(r"^radial_step_(\d+).*\.pkl$")
        for pkl_path in dfm_dir.rglob("*.pkl"):
            match = fallback.match(pkl_path.name)
            if match is None:
                continue
            candidates.append((int(match.group(1)), pkl_path))

    candidates.sort(key=lambda item: item[0])
    if not candidates:
        raise FileNotFoundError(
            f"No saved radial masks found under {dfm_dir}. "
            f"Expected files named {model_type}_radial_step_XX_DFMs.pkl"
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
def evaluate_probe_with_mask(backbone, probe, dataloader, mask_record, device):
    probe.eval()
    backbone.eval()

    id_preds_all = []
    id_labels_all = []

    for images, (_, id_onehot) in dataloader:
        images = images.to(device, non_blocking=True)
        id_labels = torch.argmax(id_onehot.to(device, non_blocking=True), dim=1)
        images = apply_frequency_mask(images, mask_record, device)

        # Single backbone pass: reuse same frozen features for both ID probe and AU head.
        features = backbone.forward_features(images)
        id_logits = probe(features)
        id_preds = torch.argmax(id_logits, dim=1)

        id_preds_all.append(id_preds.cpu())
        id_labels_all.append(id_labels.cpu())

    if id_preds_all:
        id_preds_all = torch.cat(id_preds_all, dim=0)
        id_labels_all = torch.cat(id_labels_all, dim=0)
        id_accuracy = float((id_preds_all == id_labels_all).float().mean().item())
    else:
        id_accuracy = 0.0

    return id_accuracy


@torch.no_grad()
def evaluate_au_macro_f1(backbone, dataloader, mask_record, device):
    """Match hfss_search_au.evaluate_mask(): sigmoid + threshold=0.5 + macro F1."""
    backbone.eval()
    au_preds_all = []
    au_labels_all = []

    for images, (au_labels, _) in dataloader:
        images = images.to(device, non_blocking=True)
        au_labels = au_labels.to(device, non_blocking=True)
        images = apply_frequency_mask(images, mask_record, device)

        features = backbone.forward_features(images)
        au_logits = backbone.head(features)
        au_preds = (torch.sigmoid(au_logits) >= 0.5).float()

        au_preds_all.append(au_preds.cpu().numpy())
        au_labels_all.append(au_labels.cpu().numpy())

    if not au_preds_all:
        return 0.0

    au_preds_all = np.concatenate(au_preds_all, axis=0)
    au_labels_all = np.concatenate(au_labels_all, axis=0)
    return float(f1_score(au_labels_all, au_preds_all, average="macro"))


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


def _subject_sample_stats(entries):
    counts = {}
    for entry in entries:
        subject_id = entry["img_path"][1:4]
        counts[subject_id] = counts.get(subject_id, 0) + 1
    values = list(counts.values())
    return counts, min(values), max(values), float(np.mean(values))


def run_sanity_checks(backbone, probe, train_dataset, test_dataset, sampling_audit, expected_train_samples, expected_test_samples):
    train_counts, train_min, train_max, train_mean = _subject_sample_stats(train_dataset.data)
    test_counts, test_min, test_max, test_mean = _subject_sample_stats(test_dataset.data)

    overlap_violations = {
        sid: cnt for sid, cnt in sampling_audit.overlap_by_subject.items() if cnt > 0
    }
    mapping_consistent = getattr(train_dataset, "ID_label2idx", {}) == getattr(test_dataset, "ID_label2idx", {})
    backbone_frozen = all(not p.requires_grad for p in backbone.parameters())
    in_features = probe.classifier.in_features
    out_features = probe.classifier.out_features

    print("\n=== SANITY CHECKS ===")
    print("Feature source: backbone.forward_features (global pooled transformer output, pre-head)")
    print(f"Feature dim: {in_features}")
    print(f"Backbone frozen: {backbone_frozen}")
    print(f"Probe architecture: Linear({in_features} -> {out_features})")
    print(
        f"Train samples per subject: min={train_min}, max={train_max}, mean={train_mean:.2f} "
        f"(should all be {expected_train_samples})"
    )
    print(
        f"Test samples per subject: min={test_min}, max={test_max}, mean={test_mean:.2f} "
        f"(should all be {expected_test_samples})"
    )
    if not overlap_violations:
        print("Train/test sample overlap per subject: 0 for all")
    else:
        print(f"Train/test sample overlap per subject: {overlap_violations}")
    print(f"Subject mapping consistent: {mapping_consistent}")
    print(f"model.training: {backbone.training} during feature extraction")
    print("=== END SANITY CHECKS ===")

    # Print three example sampled subjects for randomization audit.
    print("\n   Sampling examples (3 subjects):")
    shown = 0
    for subject_id in sorted(sampling_audit.examples_by_subject.keys()):
        if shown >= 3:
            break
        ex = sampling_audit.examples_by_subject[subject_id]
        print(f"     {subject_id} train frame indices (first 10): {ex['train_frame_indices'][:10]}")
        print(f"     {subject_id} test frame indices (first 10):  {ex['test_frame_indices'][:10]}")
        shown += 1

    expected_train = train_min == train_max == int(expected_train_samples)
    expected_test = test_min == test_max == int(expected_test_samples)
    if not (backbone_frozen and mapping_consistent and not overlap_violations and expected_train and expected_test):
        raise ValueError("Sanity checks failed. See printed diagnostics above.")


def main():
    parser = argparse.ArgumentParser(description="Linear probe identity analysis for BP4D (FMAE/IAT)")
    parser.add_argument("--model_type", default="FMAE", choices=["FMAE", "IAT"])
    parser.add_argument("--model_path", default="models/FMAE_BP4D_fold1.pth")
    parser.add_argument("--train_json", default="BP4D/BP4D_train1.json")
    parser.add_argument("--test_json", default="BP4D/BP4D_test1.json")
    parser.add_argument("--dfm_dir", default="hfss/DFM")
    parser.add_argument(
        "--mask_source",
        default="generated",
        choices=["generated", "saved"],
        help="generated: create deterministic radial masks in-script; saved: load radial masks from PKL",
    )
    parser.add_argument("--train_samples", type=int, default=FIXED_TRAIN_SAMPLES)
    parser.add_argument("--test_samples", type=int, default=FIXED_TEST_SAMPLES)
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
        print(f"Model type: {args.model_type} | Model: {args.model_path} | Device: {device}")
        print(f"Seed: {FIXED_SEED}")
        print(f"Log file: {log_file_path}")
        print("=" * 70)

        backbone = load_backbone(args.model_path, args.model_type, device)
        feature_dim = backbone.head.in_features if hasattr(backbone, "head") else 1024
        print(f"   Backbone feature dim: {feature_dim}")
        print("   Feature extraction uses: backbone.forward_features(images) (pre-AU/ID heads)")

        print("\n[1] Building probe datasets (fixed samples per subject)")
        train_dataset, test_dataset, sampling_audit = build_fixed_samples_per_subject_split(
            [args.train_json, args.test_json],
            FIXED_DATA_ROOT,
            train_samples=args.train_samples,
            test_samples=args.test_samples,
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
        print(f"   Subject overlap train∩test: {len(overlap)} IDs (expected for closed-set ID probe)")
        print(f"   Per-subject sampling: train={args.train_samples}, test={args.test_samples}")
        print(f"   Train samples: {len(train_dataset)}")
        print(f"   Test samples:  {len(test_dataset)}")
        mapping_consistent = print_subject_mapping(train_dataset, test_dataset)
        if not mapping_consistent:
            raise ValueError("Subject mapping mismatch detected.")

        print("\n[2] Building AU evaluation dataset (subject-exclusive check)")
        ensure_subject_exclusive_au_split(args.train_json, args.test_json)
        au_eval_dataset = BP4D_AU_dataset(args.test_json, is_train=False, args=build_dataset_args(FIXED_DATA_ROOT))
        au_eval_loader = DataLoader(
            au_eval_dataset,
            batch_size=FIXED_BATCH_SIZE,
            shuffle=False,
            num_workers=FIXED_NUM_WORKERS,
            pin_memory=device.type == "cuda",
            persistent_workers=(FIXED_NUM_WORKERS > 0),
        )
        print(f"   AU evaluation samples (official test split): {len(au_eval_dataset)}")

        print("\n[3] Precomputing frozen backbone features")
        train_features, train_labels = extract_features_and_labels(backbone, train_loader, device)
        test_features, test_labels = extract_features_and_labels(backbone, test_loader, device)
        print(f"   Train feature tensor: {tuple(train_features.shape)}")
        print(f"   Test feature tensor:  {tuple(test_features.shape)}")

        print("\n[4] Training linear probe")
        probe = LinearProbe(feature_dim=feature_dim, num_subjects=FIXED_NUM_SUBJECTS)
        print(f"   Probe check: {probe}")

        # Mandatory pre-experiment audit block.
        run_sanity_checks(
            backbone,
            probe,
            train_dataset,
            test_dataset,
            sampling_audit,
            expected_train_samples=args.train_samples,
            expected_test_samples=args.test_samples,
        )

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
        print("\n[5] Computing AU baseline (HFSS-aligned: sigmoid + thr=0.5 + macro F1)")
        baseline_macro_f1 = evaluate_au_macro_f1(backbone, au_eval_loader, None, device)
        print(f"   Baseline AU macro F1: {baseline_macro_f1*100:.2f}%")

        print("\n[6] Preparing radial masks")
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
            mask_records = load_radial_masks_saved(args.dfm_dir, args.model_type)
            print(f"   Using saved masks from PKL files (model_type={args.model_type})")
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
            id_accuracy = evaluate_probe_with_mask(
                backbone,
                probe,
                test_loader,
                record,
                device,
            )
            au_macro_f1 = evaluate_au_macro_f1(backbone, au_eval_loader, record, device)
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
        --model_type FMAE \
    --model_path models/FMAE_BP4D_fold1.pth \
    --train_json BP4D/BP4D_train1.json \
    --test_json BP4D/BP4D_test1.json \
    --dfm_dir hfss/DFM \
        --train_samples 70 \
        --test_samples 30 \
    --probe_save_path hfss/DFM/linear_probe_identity.pth \
    --probe_epochs 30 \
    --probe_lr 0.001 \
    --output_csv hfss/DFM/linear_probe_identity_results.csv

python linear_probe_identity.py \
    --model_type IAT \
    --model_path models/FMAE_IAT_BP4D_fold1.pth \
  --train_json BP4D/BP4D_train1.json \
  --test_json BP4D/BP4D_test1.json \
  --dfm_dir hfss/DFM \
    --mask_source saved \
    --train_samples 70 \
    --test_samples 30 \
  --output_csv hfss/DFM/linear_probe_identity_results.csv
"""