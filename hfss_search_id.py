"""
HFSS Search for Identity Prediction

Separate identity-focused experiment path that reuses the same
FFT -> mask -> iFFT pipeline as AU HFSS search.
"""

import argparse
import pickle
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
FMAE_PATH = PROJECT_ROOT / "fmae"
HFSS_PATH = PROJECT_ROOT / "hfss" / "hfss"
sys.path.insert(0, str(FMAE_PATH))
sys.path.insert(0, str(HFSS_PATH))

import models_vit
from util.datasets import BP4D_AU_dataset

from hfss_search_au import (
	AU_LABELS,
	IMG_SIZE,
	TeeStream,
	create_run_log_file,
	generate_mask_candidates,
	parse_keep_ranges,
)


inference_decorator = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad


def set_global_seed(seed, device="cuda"):
	"""Set RNG seeds for reproducible mask sampling and subset selection."""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if str(device).startswith("cuda") and torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
	if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False


def load_iat_model(model_path, num_subjects=41, device="cuda"):
	"""Load IAT model with identity head enabled."""
	model = models_vit.vit_large_patch16(
		num_classes=12,
		num_subjects=num_subjects,
		drop_path_rate=0.0,
		global_pool=True,
		grad_reverse=1.0,
	)
	checkpoint = torch.load(model_path, map_location="cpu")
	state_dict = checkpoint.get("model", checkpoint)
	model.load_state_dict(state_dict, strict=True)
	model = model.to(device).eval()
	print(f"✓ Loaded IAT model for identity ({num_subjects} subjects)")
	return model


def _extract_identity_logits(outputs):
	"""Extract identity logits from model outputs."""
	if isinstance(outputs, tuple):
		if len(outputs) >= 2:
			return outputs[1]
	raise RuntimeError(
		"Identity logits were not found in model output. "
		"Use an IAT checkpoint with ID head enabled."
	)


def _extract_au_logits(outputs):
	"""Extract AU logits from model outputs."""
	if isinstance(outputs, tuple):
		return outputs[0]
	return outputs


@inference_decorator()
def evaluate_mask_identity(mask_transform, model, dataloader, device, compute_top5=False):
	"""Evaluate identity prediction under a frequency mask.

	Uses pipeline: FFT -> mask -> iFFT -> model input.
	Returns accuracy and macro-F1 in [0,1], plus optional top-5 accuracy.
	"""
	all_pred = []
	all_true = []
	top5_correct = 0
	total = 0

	mask_t = None
	if hasattr(mask_transform, "mask"):
		mask_np = mask_transform.mask
		if getattr(mask_transform, "flip", False):
			mask_np = 1 - mask_np
		mask_t = torch.as_tensor(mask_np, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

	for images, (_, subject_labels) in dataloader:
		images = images.to(device, non_blocking=True)
		subject_labels = subject_labels.to(device, non_blocking=True)

		if mask_t is not None:
			freq = torch.fft.fftshift(torch.fft.fft2(images, dim=(-2, -1)), dim=(-2, -1))
			freq = freq * mask_t
			images = torch.fft.ifft2(torch.fft.ifftshift(freq, dim=(-2, -1)), dim=(-2, -1)).real.float()

		outputs = model(images)
		id_logits = _extract_identity_logits(outputs)
		pred_idx = torch.argmax(id_logits, dim=1)
		true_idx = torch.argmax(subject_labels, dim=1)

		all_pred.append(pred_idx.detach().cpu().numpy())
		all_true.append(true_idx.detach().cpu().numpy())

		if compute_top5:
			k = min(5, id_logits.shape[1])
			topk = torch.topk(id_logits, k=k, dim=1).indices
			top5_correct += (topk == true_idx.unsqueeze(1)).any(dim=1).sum().item()
			total += true_idx.numel()

	if not all_pred:
		result = {"acc": 0.0, "f1": 0.0}
		if compute_top5:
			result["top5_acc"] = 0.0
		return result

	y_pred = np.concatenate(all_pred, axis=0)
	y_true = np.concatenate(all_true, axis=0)

	result = {
		"acc": float(accuracy_score(y_true, y_pred)),
		"f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
	}
	if compute_top5:
		result["top5_acc"] = float(top5_correct / max(total, 1))
	return result


@inference_decorator()
def evaluate_mask_au_batch_style(mask_transform, model, dataloader, device, mode="best_threshold", fixed_threshold=0.5):
	"""Evaluate AU macro-F1 with batch-analysis style thresholding.

	mode:
	- best_threshold: sweep thresholds 0.01..0.99 and take best mean per-AU F1.
	- fixed: use fixed_threshold directly.
	"""
	all_probs = []
	all_labels = []

	mask_t = None
	if hasattr(mask_transform, "mask"):
		mask_np = mask_transform.mask
		if getattr(mask_transform, "flip", False):
			mask_np = 1 - mask_np
		mask_t = torch.as_tensor(mask_np, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

	for images, (au_labels, _) in dataloader:
		images = images.to(device, non_blocking=True)
		au_labels = au_labels.to(device, non_blocking=True)

		if mask_t is not None:
			freq = torch.fft.fftshift(torch.fft.fft2(images, dim=(-2, -1)), dim=(-2, -1))
			freq = freq * mask_t
			images = torch.fft.ifft2(torch.fft.ifftshift(freq, dim=(-2, -1)), dim=(-2, -1)).real.float()

		outputs = model(images)
		au_logits = _extract_au_logits(outputs)
		probs = torch.sigmoid(au_logits)

		all_probs.append(probs.detach().cpu().numpy())
		all_labels.append(au_labels.detach().cpu().numpy())

	if not all_probs:
		return {"f1": 0.0, "threshold": fixed_threshold}

	y_prob = np.concatenate(all_probs, axis=0)
	y_true = np.concatenate(all_labels, axis=0)

	if mode == "fixed":
		thresholds = [float(fixed_threshold)]
	else:
		thresholds = [i / 100.0 for i in range(1, 100)]

	best = {"f1": -1.0, "threshold": float(fixed_threshold), "per_au_f1": None}
	for threshold in thresholds:
		y_pred = (y_prob >= threshold).astype(np.float32)
		per_au_f1 = [
			f1_score(y_true[:, i], y_pred[:, i], average="binary", zero_division=0)
			for i in range(y_true.shape[1])
		]
		macro_f1 = float(np.mean(per_au_f1))
		if macro_f1 > best["f1"]:
			best = {
				"f1": macro_f1,
				"threshold": float(threshold),
				"per_au_f1": per_au_f1,
			}

	return best


def search_stage_identity(
	model,
	dataloader,
	stage,
	num_candidates,
	proportion,
	prev_masks,
	device,
	keep_ratio_range,
	top_n,
	baseline_identity,
	objective_metric="f1",
	include_top5=False,
	task="id",
	au_eval_mode="best_threshold",
	au_fixed_threshold=0.5,
	au_drop_threshold_pct=1.0,
):
	"""Search one stage using identity objective while optionally reporting AU too.
	
	Mask selection logic (when task='both' or 'au'):
	- Filter masks where AU_drop > au_drop_threshold_pct
	- From those, select mask with LOWEST ID_f1 (worst identity)
	- Break ties by preferring higher AU_f1 (better AU preservation)
		"""
	print(f"\n🔍 Stage {stage}: Testing {num_candidates} candidates | objective=ID-{objective_metric}")
	candidates = generate_mask_candidates(
		num_candidates=num_candidates,
		proportion=proportion,
		stage=stage,
		prev_masks=prev_masks,
		keep_ratio_range=keep_ratio_range,
	)

	baseline_obj = baseline_identity[objective_metric]
	print(
		f"   Baseline ID: acc={baseline_identity['acc']*100:.2f}% | "
		f"f1={baseline_identity['f1']*100:.2f}%"
		+ (
			f" | top5={baseline_identity['top5_acc']*100:.2f}%"
			if include_top5 and "top5_acc" in baseline_identity
			else ""
		)
	)

	baseline_au = None
	if task in {"au", "both"}:
		baseline_au = evaluate_mask_au_batch_style(
			mask_transform=lambda x: x,
			model=model,
			dataloader=dataloader,
			device=device,
			mode=au_eval_mode,
			fixed_threshold=au_fixed_threshold,
		)
		print(
			f"   Baseline AU macro-F1: {baseline_au['f1']*100:.2f}% "
			f"(threshold={baseline_au['threshold']:.2f}, mode={au_eval_mode})"
		)

	rows = []
	for i, mask_transform in enumerate(tqdm(candidates, desc="   Testing masks")):
		id_metrics = evaluate_mask_identity(
			mask_transform=mask_transform,
			model=model,
			dataloader=dataloader,
			device=device,
			compute_top5=include_top5,
		)
		obj = id_metrics[objective_metric]
		obj_drop = baseline_obj - obj
		white_count = int(getattr(mask_transform, "white_count", np.sum(mask_transform.mask > 0.5)))
		keep_pct = float(getattr(mask_transform, "keep_pct", np.mean(mask_transform.mask > 0.5)))

		au_val = None
		au_drop = None
		au_threshold = None
		if task in {"au", "both"}:
			au_eval = evaluate_mask_au_batch_style(
				mask_transform=mask_transform,
				model=model,
				dataloader=dataloader,
				device=device,
				mode=au_eval_mode,
				fixed_threshold=au_fixed_threshold,
			)
			au_val = float(au_eval["f1"])
			au_threshold = float(au_eval["threshold"])
			au_drop = float(baseline_au["f1"] - au_val)

		rows.append(
			{
				"mask_id": i,
				"mask_transform": mask_transform,
				"id_acc": id_metrics["acc"],
				"id_f1": id_metrics["f1"],
				"id_top5": id_metrics.get("top5_acc", None),
				"id_obj": obj,
				"id_obj_drop": obj_drop,
				"au_f1": au_val,
				"au_drop": au_drop,
				"au_threshold": au_threshold,
				"white": white_count,
				"keep_pct": keep_pct,
			}
		)

	rows.sort(key=lambda r: r["id_obj"], reverse=True)

	print(f"\n   {stage} — all masks (sorted by ID-{objective_metric}, high→low):")
	for r in rows:
		line = (
			f"     mask{r['mask_id']:>3}: ID_acc={r['id_acc']*100:.2f}% | "
			f"ID_f1={r['id_f1']*100:.2f}% | "
			f"ID_drop={r['id_obj_drop']*100:+.2f}% | "
			f"white={r['white']} | keep={r['keep_pct']*100:.1f}%"
		)
		if include_top5 and r["id_top5"] is not None:
			line += f" | ID_top5={r['id_top5']*100:.2f}%"
		if r["au_f1"] is not None:
			line += (
				f" | AU_f1={r['au_f1']*100:.2f}%"
				f" | AU_drop={r['au_drop']*100:+.2f}%"
				f" | AU_th={r['au_threshold']:.2f}"
			)
		print(line)

	# New mask selection logic: prioritize AU drop + minimize identity
	best_mask_id = rows[0]["mask_id"] if rows else -1
	selected_row = rows[0] if rows else None

	if task in {"au", "both"} and baseline_au is not None:
		# Filter masks with sufficient AU drop
		candidates = [
			r for r in rows
			if r["au_f1"] is not None and r["au_drop"] is not None
			and r["au_drop"] * 100.0 > au_drop_threshold_pct
		]

		if candidates:
			# From candidates, select lowest ID_f1 (worst identity)
			# Break ties with highest AU_f1
			selected_row = min(candidates, key=lambda r: (r["id_f1"], -r["au_f1"]))
			best_mask_id = selected_row["mask_id"]
			print(
				f"\n   ✓ Selected mask{best_mask_id:>3} (AU-based selection):"
				f" AU_f1={selected_row['au_f1']*100:.2f}% | "
				f" AU_drop={selected_row['au_drop']*100:.2f}% | "
				f" ID_f1={selected_row['id_f1']*100:.2f}% (lowest)"
			)
			print(f"   (Filtered {len(candidates)} masks with AU_drop > {au_drop_threshold_pct:.1f}%)")
		else:
			print(
				f"\n   ⚠️  No masks with AU_drop > {au_drop_threshold_pct:.1f}%. "
				f"Using highest ID_f1 instead: mask{best_mask_id}"
			)

	# Select top_n masks: start with selected_row, then add others
	if selected_row is not None:
		# Put selected at front, then others
		other_rows = [r for r in rows if r["mask_id"] != selected_row["mask_id"]]
		top_rows = [selected_row] + other_rows[:top_n-1]
	else:
		top_rows = rows[:top_n]

	top_masks = [r["mask_transform"] for r in top_rows]
	summary = {
		"baseline_id": baseline_identity,
		"selected_mask_id": best_mask_id,
		"selected_au_f1": float(selected_row["au_f1"]) if selected_row and selected_row["au_f1"] is not None else None,
		"selected_au_drop_pct": float(selected_row["au_drop"] * 100.0) if selected_row and selected_row["au_drop"] is not None else None,
		"selected_id_f1": float(selected_row["id_f1"]) if selected_row else 0.0,
		"objective": f"id_{objective_metric}",
	}
	return top_masks, rows, summary


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_path", required=True)
	parser.add_argument("--test_json", default="BP4D/BP4D_test1.json")
	parser.add_argument("--data_root", default="BP4D/BP4D_cropped/")
	parser.add_argument("--num_subjects", type=int, default=41)
	parser.add_argument("--num_candidates", type=int, default=30)
	parser.add_argument("--top_n", type=int, default=10)
	parser.add_argument("--proportion", type=float, default=0.8)
	parser.add_argument(
		"--keep_ranges",
		default="stage1:0.6-0.8,stage2:0.4-0.6,stage3:0.2-0.4,stage4:0.15-0.3,stage5:0.08-0.2,stage6:0.03-0.1",
	)
	parser.add_argument("--num_samples", type=int, default=500)
	parser.add_argument("--batch_size", type=int, default=24)
	parser.add_argument("--num_workers", type=int, default=4)
	parser.add_argument("--device", default="cuda")
	parser.add_argument("--stages", nargs="+", default=["stage1", "stage2", "stage3"])
	parser.add_argument("--task", choices=["id", "au", "both"], default="id")
	parser.add_argument("--objective_metric", choices=["acc", "f1"], default="f1")
	parser.add_argument("--compute_top5", action="store_true")
	parser.add_argument(
		"--au_eval_mode",
		choices=["best_threshold", "fixed"],
		default="best_threshold",
		help="AU reporting mode when task includes AU: best_threshold (batch-style) or fixed",
	)
	parser.add_argument(
		"--au_fixed_threshold",
		type=float,
		default=0.5,
		help="Used only when --au_eval_mode fixed",
	)
	parser.add_argument(
		"--au_drop_threshold_pct",
		type=float,
		default=1.0,
		help="Min AU_drop %% to accept mask (when task includes AU); select mask with lowest ID_f1 from valid set",
	)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--output_dir", default="hfss/DFM_ID")
	args = parser.parse_args()

	if args.task == "au":
		print("⚠️ --task au selected in identity script. Objective remains ID; AU metrics are reported only.")

	set_global_seed(args.seed, args.device)
	keep_ranges = parse_keep_ranges(args.keep_ranges)

	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	logs_dir = output_dir.parent / "logs"
	logs_dir.mkdir(parents=True, exist_ok=True)

	log_file_path = create_run_log_file(logs_dir)
	orig_stdout = sys.stdout
	orig_stderr = sys.stderr
	log_fp = open(log_file_path, "w", encoding="utf-8")
	sys.stdout = TeeStream(orig_stdout, log_fp)
	sys.stderr = TeeStream(orig_stderr, log_fp)

	try:
		print("=" * 70)
		print("HFSS Search for Identity Prediction (separate path)")
		print(f"Model: {args.model_path} | Device: {args.device} | Seed: {args.seed}")
		print(f"Task: {args.task} | Objective: ID-{args.objective_metric}")
		print(f"Log file: {log_file_path}")
		print("=" * 70)

		model = load_iat_model(args.model_path, num_subjects=args.num_subjects, device=args.device)

		dataset_args = SimpleNamespace(
			root_path=args.data_root,
			input_size=IMG_SIZE,
			color_jitter=None,
			aa="rand-m9-mstd0.5-inc1",
			reprob=0.25,
			remode="pixel",
			recount=1,
		)
		dataset = BP4D_AU_dataset(args.test_json, is_train=False, args=dataset_args)
		if args.num_samples and args.num_samples < len(dataset):
			indices = np.random.choice(len(dataset), args.num_samples, replace=False)
			dataset = Subset(dataset, indices)

		pin_mem = str(args.device).startswith("cuda")
		dataloader = DataLoader(
			dataset,
			batch_size=args.batch_size,
			shuffle=False,
			num_workers=args.num_workers,
			pin_memory=pin_mem,
			persistent_workers=(args.num_workers > 0),
		)
		print(f"✓ Loaded {len(dataset)} evaluation samples")

		baseline_identity = evaluate_mask_identity(
			mask_transform=lambda x: x,
			model=model,
			dataloader=dataloader,
			device=args.device,
			compute_top5=args.compute_top5,
		)
		print(
			f"\nBaseline identity: acc={baseline_identity['acc']*100:.2f}% | "
			f"f1={baseline_identity['f1']*100:.2f}%"
			+ (
				f" | top5={baseline_identity['top5_acc']*100:.2f}%"
				if args.compute_top5 and "top5_acc" in baseline_identity
				else ""
			)
		)

		prev_masks = None
		stage_summaries = {}
		for stage in args.stages:
			keep_range = keep_ranges.get(stage)
			top_masks, stage_rows, summary = search_stage_identity(
				model=model,
				dataloader=dataloader,
				stage=stage,
				num_candidates=args.num_candidates,
				proportion=args.proportion,
				prev_masks=prev_masks,
				device=args.device,
				keep_ratio_range=keep_range,
				top_n=args.top_n,
				baseline_identity=baseline_identity,
				objective_metric=args.objective_metric,
				include_top5=args.compute_top5,
				task=args.task,
				au_eval_mode=args.au_eval_mode,
				au_fixed_threshold=args.au_fixed_threshold,
				au_drop_threshold_pct=args.au_drop_threshold_pct,
			)
			prev_masks = [m.mask for m in top_masks]
			stage_summaries[stage] = {
				"summary": summary,
				"rows": stage_rows,
			}

			out_pkl = output_dir / f"IAT_{stage}_ID_DFMs.pkl"
			with open(out_pkl, "wb") as f:
				pickle.dump(top_masks, f)
			print(f"💾 Saved top-{len(top_masks)} masks to: {out_pkl}")

		summary_file = output_dir / "identity_search_summary.pkl"
		with open(summary_file, "wb") as f:
			pickle.dump(stage_summaries, f)
		print(f"\n📦 Saved stage summary: {summary_file}")
		print("✅ Identity HFSS search complete")

	finally:
		sys.stdout = orig_stdout
		sys.stderr = orig_stderr
		log_fp.close()


if __name__ == "__main__":
	main()


"""
python3 hfss_search_id.py \
  --model_path models/FMAE_IAT_BP4D_fold1.pth \
  --test_json BP4D/BP4D_test1.json \
  --data_root BP4D/BP4D_cropped/ \
  --num_samples 200 \
  --num_candidates 30 \
  --top_n 10 \
  --stages stage1 stage2 stage3 \
  --task both \
  --objective_metric f1 \
  --au_eval_mode best_threshold \
  --au_drop_threshold_pct 1.0 \
  --device cuda \
  --num_workers 4
"""