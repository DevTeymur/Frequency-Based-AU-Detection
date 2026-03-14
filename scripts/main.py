from __future__ import annotations

import csv
import inspect
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import f1_score, roc_auc_score
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader, Dataset, SequentialSampler, Subset
import torchvision.transforms as T

try:
	from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback if tqdm missing
	tqdm = None


def log(msg: str) -> None:
	print(f"[INFO] {msg}")


# --- user controls ---
# Choose which model to run: "FMAE" or "IAT"
MODEL_CHOICE = "FMAE"
# Choose which fold to evaluate: "test1", "test2", or "test3"
FOLD = "test1"
# Limit the number of samples for a quick run (set to None for full set)
SAMPLE_LIMIT: Optional[int] = None
# Data loader knobs
BATCH_SIZE = 64
NUM_WORKERS = 4


# Reference scores from supervisor (percentage values)
REFERENCE_RESULTS: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {
	"FMAE": {
		"test1": {"F1": 67.108, "AUC": 82.407},
		"test2": {"F1": 65.317, "AUC": 82.143},
		"test3": {"F1": 67.705, "AUC": 82.768},
	},
	"IAT": {
		"test1": {"F1": 66.209, "AUC": None},
		"test2": {"F1": 69.566, "AUC": None},
		"test3": {"F1": 65.385, "AUC": None},
	},
}


BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "BP4D" / "BP4D_cropped"
JSON_MAP = {
	"test1": BASE_DIR / "BP4D" / "BP4D_test1.json",
	"test2": BASE_DIR / "BP4D" / "BP4D_test2.json",
	"test3": BASE_DIR / "BP4D" / "BP4D_test3.json",
}
MODEL_PATHS = {
	"FMAE": {
		"test1": BASE_DIR / "models" / "FMAE_BP4D_fold1.pth",
		"test2": BASE_DIR / "models" / "FMAE_BP4D_fold2.pth",
		"test3": BASE_DIR / "models" / "FMAE_BP4D_fold3.pth",
	},
	"IAT": {
		"test1": BASE_DIR / "models" / "FMAE_IAT_BP4D_fold1.pth",
		"test2": BASE_DIR / "models" / "FMAE_IAT_BP4D_fold2.pth",
		"test3": BASE_DIR / "models" / "FMAE_IAT_BP4D_fold3.pth",
	},
}


# Make fmae/ modules importable (prepend to avoid clashing with other util packages)
sys.path.insert(0, str(BASE_DIR / "fmae"))
import models_vit  # type: ignore  # noqa: E402


def build_AU_transform(is_train: bool, args) -> T.Compose:
	if is_train:
		raise ValueError("build_AU_transform is only used for eval in this script")
	return T.Compose([
		T.Resize([args.input_size, args.input_size]),
		T.ToTensor(),
		T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
	])


class BP4D_dataset(Dataset):
	"""Minimal BP4D AU dataset reader for inference."""
	AUs = [1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24]
	IDs = [
		'F01', 'F02', 'F03', 'F04', 'F05', 'F06', 'F07', 'F08', 'F09', 'F10',
		'F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F17', 'F18', 'F19', 'F20',
		'F21', 'F22', 'F23',
		'M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10',
		'M11', 'M12', 'M13', 'M14', 'M15', 'M16', 'M17', 'M18'
	]

	def __init__(self, root_path: str, json_file: Path, transform=None):
		self.root_path = root_path
		self.transform = transform
		self.data = []
		with open(json_file, 'r') as f:
			for line in f:
				self.data.append(json.loads(line))
		self.AU_label2idx = {label: idx for idx, label in enumerate(self.AUs)}
		self.ID_label2idx = {label: idx for idx, label in enumerate(self.IDs)}

	def __len__(self) -> int:
		return len(self.data)

	def __getitem__(self, idx: int):
		record = self.data[idx]
		image_path = self.root_path + record['img_path']
		image = Image.open(image_path).convert('RGB')

		AUs = record['AUs']
		ID = record['img_path'][:3]

		AU_labels = torch.zeros(len(self.AUs))
		ID_labels = torch.zeros(len(self.IDs))
		for au in AUs:
			if au == 999:
				continue
			AU_labels[self.AU_label2idx[au]] = 1

		# Some JSON entries (e.g., F00) are absent from the ID list; skip unknown IDs instead of crashing.
		if ID in self.ID_label2idx:
			ID_labels[self.ID_label2idx[ID]] = 1

		if self.transform:
			image = self.transform(image)

		return image, (AU_labels, ID_labels)


@dataclass
class InferenceArgs:
	root_path: str
	input_size: int = 224


def load_model(model_path: Path, device: torch.device) -> Tuple[torch.nn.Module, object]:
	checkpoint = torch.load(model_path, map_location=device)
	ckpt_args = checkpoint.get("args")
	model_fn = models_vit.__dict__[ckpt_args.model]

	# Build kwargs compatible with the model_fn signature (some environments lack num_subjects/grad_reverse)
	requested_kwargs = {
		"num_classes": getattr(ckpt_args, "nb_classes", 0),
		"num_subjects": getattr(ckpt_args, "nb_subjects", 0),
		"drop_path_rate": getattr(ckpt_args, "drop_path", 0.0),
		"global_pool": getattr(ckpt_args, "global_pool", True),
		"grad_reverse": getattr(ckpt_args, "grad_reverse", 0.0),
	}
	# Align with the VisionTransformer constructor actually available in this environment
	vt_sig = inspect.signature(models_vit.VisionTransformer.__init__)
	allowed_params = {k for k in vt_sig.parameters.keys() if k != "self"}
	filtered_kwargs = {k: v for k, v in requested_kwargs.items() if k in allowed_params}
	model = model_fn(**filtered_kwargs)

	# Allow head shape mismatch by stripping incompatible keys before loading
	state_dict = checkpoint["model"]
	model_state = model.state_dict()
	pruned_state = {}
	for k, v in state_dict.items():
		if k in model_state and model_state[k].shape == v.shape:
			pruned_state[k] = v
		else:
			continue
	msg = model.load_state_dict(pruned_state, strict=False)
	log(f"Loaded weights from {model_path.name}")
	if msg.missing_keys:
		log(f"Missing keys while loading: {msg.missing_keys}")
	if msg.unexpected_keys:
		log(f"Unexpected keys while loading: {msg.unexpected_keys}")
	model.to(device)
	model.eval()
	return model, ckpt_args


def prepare_dataset(fold: str, sample_limit: Optional[int]) -> Tuple[BP4D_dataset, List[str]]:
	json_path = JSON_MAP[fold]
	args = InferenceArgs(root_path=str(DATA_ROOT) + "/")
	transform = build_AU_transform(is_train=False, args=args)
	dataset = BP4D_dataset(args.root_path, json_path, transform=transform)
	indices = list(range(len(dataset))) if sample_limit is None else list(range(min(sample_limit, len(dataset))))
	dataset = dataset if sample_limit is None else Subset(dataset, indices)
	if isinstance(dataset, Subset):
		paths = [dataset.dataset.data[i]["img_path"] for i in indices]
		aus = dataset.dataset.AUs
	else:
		paths = [record["img_path"] for record in dataset.data]
		aus = dataset.AUs
	return dataset, paths  # aus kept on dataset if needed later


def run_inference(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, float]:
	sigmoid = torch.nn.Sigmoid()
	criterion = torch.nn.BCEWithLogitsLoss()
	probs_list: List[np.ndarray] = []
	targets_list: List[np.ndarray] = []
	running_loss = 0.0
	total = 0

	pbar = None
	if tqdm is not None:
		pbar = tqdm(total=len(dataloader.dataset), desc="Infer", ncols=100, unit="img")

	with torch.no_grad():
		for batch in dataloader:
			images, labels = batch
			au_labels = labels[0]
			images = images.to(device, non_blocking=True)
			au_labels = au_labels.to(device, non_blocking=True)

			# Some checkpoints/models return a tuple, others return a bare tensor.
			output = model(images)
			logits = output[0] if isinstance(output, (tuple, list)) else output
			loss = criterion(logits, au_labels)
			probs = sigmoid(logits)

			batch_size = images.size(0)
			running_loss += loss.item() * batch_size
			total += batch_size

			probs_list.append(probs.cpu().numpy())
			targets_list.append(au_labels.cpu().numpy())

			if pbar is not None:
				pbar.update(batch_size)

	if pbar is not None:
		pbar.close()

	mean_loss = running_loss / max(total, 1)
	y_probs = np.concatenate(probs_list, axis=0)
	y_true = np.concatenate(targets_list, axis=0)
	return y_true, y_probs, mean_loss


def compute_metrics(y_true: np.ndarray, y_probs: np.ndarray) -> Dict[str, object]:
	thresholds = np.linspace(0.01, 0.99, 99)
	f1_grid = []
	for thr in thresholds:
		preds = (y_probs >= thr).astype(int)
		f1_scores = [f1_score(y_true[:, i], preds[:, i], zero_division=0) for i in range(y_true.shape[1])]
		f1_grid.append(f1_scores)
	f1_grid = np.array(f1_grid)
	mean_f1_per_thr = f1_grid.mean(axis=1)
	best_idx = int(mean_f1_per_thr.argmax())
	best_thr = float(thresholds[best_idx])
	best_f1_mean = float(mean_f1_per_thr[best_idx])
	best_f1_per_class = f1_grid[best_idx].tolist()

	preds_half = (y_probs >= 0.5).astype(int)
	f1_half = [f1_score(y_true[:, i], preds_half[:, i], zero_division=0) for i in range(y_true.shape[1])]
	f1_half_mean = float(np.mean(f1_half))

	auc_scores = []
	for i in range(y_true.shape[1]):
		try:
			auc_scores.append(roc_auc_score(y_true[:, i], y_probs[:, i]))
		except ValueError:
			auc_scores.append(float("nan"))
	auc_mean = float(np.nanmean(auc_scores))

	return {
		"best_threshold": best_thr,
		"f1_mean_best": best_f1_mean,
		"f1_per_class_best": best_f1_per_class,
		"f1_mean_at_05": f1_half_mean,
		"f1_per_class_05": f1_half,
		"auc_mean": auc_mean,
		"auc_per_class": auc_scores,
	}


def save_metrics(summary_path: Path, per_class_path: Path, fold: str, model_choice: str, sample_count: int, metrics: Dict[str, object]) -> None:
	summary_path.parent.mkdir(parents=True, exist_ok=True)

	with summary_path.open("w", newline="") as f:
		writer = csv.writer(f)
		writer.writerow([
			"test_set",
			"model_type",
			"samples",
			"best_threshold",
			"f1_mean_best",
			"f1_mean_at_0.5",
			"auc_mean",
		])
		writer.writerow([
			fold,
			model_choice,
			sample_count,
			metrics["best_threshold"],
			metrics["f1_mean_best"],
			metrics["f1_mean_at_05"],
			metrics["auc_mean"],
		])

	aus = [1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24]
	with per_class_path.open("w", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(["AU", "f1_best", "f1_at_0.5", "auc"])
		for idx, au in enumerate(aus):
			writer.writerow([
				au,
				metrics["f1_per_class_best"][idx],
				metrics["f1_per_class_05"][idx],
				metrics["auc_per_class"][idx],
			])


def save_predictions(pred_path: Path, paths: List[str], y_true: np.ndarray, y_probs: np.ndarray) -> None:
	pred_path.parent.mkdir(parents=True, exist_ok=True)
	aus = [1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24]
	headers = ["img_path"] + [f"AU{au}_prob" for au in aus] + [f"AU{au}_label" for au in aus]

	with pred_path.open("w", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(headers)
		for idx, img_path in enumerate(paths):
			row = [img_path]
			row += [float(v) for v in y_probs[idx]]
			row += [int(v) for v in y_true[idx]]
			writer.writerow(row)


def log_reference(model_choice: str, fold: str) -> None:
	ref = REFERENCE_RESULTS.get(model_choice, {}).get(fold, {})
	if not ref:
		log("No reference results recorded.")
		return
	log(f"Reference {model_choice} {fold}: F1={ref.get('F1')} AUC={ref.get('AUC')}")


def main() -> None:
	assert MODEL_CHOICE in MODEL_PATHS, f"MODEL_CHOICE must be one of {list(MODEL_PATHS.keys())}"
	assert FOLD in JSON_MAP, f"FOLD must be one of {list(JSON_MAP.keys())}"

	model_path = MODEL_PATHS[MODEL_CHOICE][FOLD]
	json_path = JSON_MAP[FOLD]
	assert model_path.exists(), f"Missing model file: {model_path}"
	assert json_path.exists(), f"Missing json file: {json_path}"
	assert DATA_ROOT.exists(), f"Missing data root: {DATA_ROOT}"

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	log(f"Running on device: {device}")
	log(f"Model: {MODEL_CHOICE} | Fold: {FOLD} | Sample limit: {SAMPLE_LIMIT}")
	log_reference(MODEL_CHOICE, FOLD)

	model, ckpt_args = load_model(model_path, device)
	dataset, paths = prepare_dataset(FOLD, SAMPLE_LIMIT)
	log(f"Dataset size: {len(paths)} samples")

	sampler = SequentialSampler(dataset)
	dataloader = DataLoader(
		dataset,
		sampler=sampler,
		batch_size=BATCH_SIZE,
		num_workers=NUM_WORKERS,
		pin_memory=True,
		drop_last=False,
	)

	start = time.time()
	y_true, y_probs, mean_loss = run_inference(model, dataloader, device)
	metrics = compute_metrics(y_true, y_probs)
	elapsed = time.time() - start

	log(f"Eval loss: {mean_loss:.4f}")
	log(f"Best F1 mean: {metrics['f1_mean_best']*100:.3f}% at threshold {metrics['best_threshold']:.2f}")
	log(f"F1 mean @0.5: {metrics['f1_mean_at_05']*100:.3f}%")
	log(f"AUC mean: {metrics['auc_mean']*100:.3f}%")
	log(f"Elapsed: {elapsed/60:.2f} mins")

	results_dir = BASE_DIR / "results"
	summary_path = results_dir / f"{FOLD}_{MODEL_CHOICE}_metrics.csv"
	per_class_path = results_dir / f"{FOLD}_{MODEL_CHOICE}_per_class.csv"
	preds_path = results_dir / f"{FOLD}_{MODEL_CHOICE}_preds.csv"

	save_metrics(summary_path, per_class_path, FOLD, MODEL_CHOICE, len(paths), metrics)
	save_predictions(preds_path, paths, y_true, y_probs)
	log(f"Saved metrics to {summary_path}")
	log(f"Saved per-AU metrics to {per_class_path}")
	log(f"Saved predictions to {preds_path}")


if __name__ == "__main__":
	main()
