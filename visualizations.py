"""
Visualization helpers for HFSS frequency masking analysis.

This script does not change HFSS search logic. It only reads saved masks,
evaluates baseline vs masked AU performance, and produces presentation-ready plots.
"""

import argparse
import json
import pickle
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = Path(__file__).resolve().parent
FMAE_PATH = PROJECT_ROOT / "fmae"
HFSS_PATH = PROJECT_ROOT / "hfss" / "hfss"
sys.path.insert(0, str(FMAE_PATH))
sys.path.insert(0, str(HFSS_PATH))

from util.datasets import BP4D_AU_dataset
from hfss_search_au import AU_LABELS, IMG_SIZE, evaluate_mask_per_au, load_model


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def set_global_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)


def load_mask_from_pkl(pkl_path, mask_index=0):
	"""Load one mask object from a saved DFM pickle."""
	with open(pkl_path, "rb") as f:
		data = pickle.load(f)
	if not isinstance(data, list) or not data:
		raise ValueError(f"Expected non-empty list in {pkl_path}")
	if mask_index < 0 or mask_index >= len(data):
		raise IndexError(f"mask_index={mask_index} out of range for {pkl_path} (len={len(data)})")
	mask_obj = data[mask_index]
	if not hasattr(mask_obj, "mask"):
		raise ValueError(f"Selected item has no .mask attribute in {pkl_path}")
	return mask_obj


def visualize_mask(mask, save_path, title="Frequency Mask"):
	"""Save a standalone grayscale visualization of the frequency mask."""
	plt.figure(figsize=(6, 6))
	plt.imshow(mask, cmap="gray", vmin=0, vmax=1)
	plt.title(f"{title} (white=keep, black=remove)")
	plt.colorbar()
	plt.tight_layout()
	plt.savefig(save_path, dpi=140, bbox_inches="tight")
	plt.close()


def apply_fft_and_reconstruct(image, mask, device):
	"""Apply FFT masking and inverse reconstruction for one image tensor.

	Args:
		image: torch.Tensor [3, H, W], normalized
		mask: np.ndarray [H, W], where 1=keep and 0=remove
	Returns:
		torch.Tensor [3, H, W], reconstructed normalized tensor
	"""
	image_b = image.unsqueeze(0).to(device)
	mask_t = torch.as_tensor(mask, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

	freq = torch.fft.fftshift(torch.fft.fft2(image_b, dim=(-2, -1)), dim=(-2, -1))
	freq = freq * mask_t
	recon = torch.fft.ifft2(torch.fft.ifftshift(freq, dim=(-2, -1)), dim=(-2, -1)).real.float()
	return recon.squeeze(0).cpu()


def denorm_to_numpy(image_t):
	"""Convert normalized CHW tensor to displayable HWC uint8-like float image."""
	x = image_t.detach().cpu() * IMAGENET_STD + IMAGENET_MEAN
	x = torch.clamp(x, 0.0, 1.0)
	return x.permute(1, 2, 0).numpy()


def plot_au_performance(baseline_f1, masked_f1, save_path, title_suffix=""):
	"""Plot baseline and masked per-AU F1 on the same chart."""
	xs = np.arange(len(AU_LABELS))
	labels = [f"AU{a}" for a in AU_LABELS]

	plt.figure(figsize=(10, 4.8))
	plt.plot(xs, baseline_f1, marker="o", linewidth=2.0, label="Baseline F1")
	plt.plot(xs, masked_f1, marker="o", linewidth=2.0, label="Masked F1")
	plt.xticks(xs, labels)
	plt.ylim(0.0, 1.0)
	plt.xlabel("Action Unit")
	plt.ylabel("F1 score")
	plt.title(f"Per-AU Performance {title_suffix}".strip())
	plt.grid(alpha=0.25)
	plt.legend(loc="best")
	plt.tight_layout()
	plt.savefig(save_path, dpi=140, bbox_inches="tight")
	plt.close()


def plot_au_drop(baseline_f1, masked_f1, save_path, title_suffix=""):
	"""Plot per-AU F1 drop: baseline - masked."""
	xs = np.arange(len(AU_LABELS))
	labels = [f"AU{a}" for a in AU_LABELS]
	drops = np.asarray(baseline_f1) - np.asarray(masked_f1)

	colors = ["#d95f02" if d > 0 else "#1b9e77" for d in drops]
	plt.figure(figsize=(10, 4.8))
	plt.bar(xs, drops, color=colors)
	plt.axhline(0.0, color="black", linewidth=1.0)
	plt.xticks(xs, labels)
	plt.xlabel("Action Unit")
	plt.ylabel("F1 drop (baseline - masked)")
	plt.title(f"Per-AU Sensitivity {title_suffix}".strip())
	plt.grid(axis="y", alpha=0.25)
	plt.tight_layout()
	plt.savefig(save_path, dpi=140, bbox_inches="tight")
	plt.close()


def load_json_entries(json_path):
	entries = []
	with open(json_path, "r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			entries.append(json.loads(line))
	return entries


def sample_entries_for_au(entries, au, n, seed):
	candidates = [e for e in entries if au in e.get("AUs", [])]
	if not candidates:
		return []
	rng = random.Random(seed + int(au))
	if len(candidates) <= n:
		return candidates
	return rng.sample(candidates, n)


def build_eval_dataloader(test_json, data_root, num_samples, batch_size, num_workers, device):
	dataset_args = SimpleNamespace(
		root_path=data_root,
		input_size=IMG_SIZE,
		color_jitter=None,
		aa="rand-m9-mstd0.5-inc1",
		reprob=0.25,
		remode="pixel",
		recount=1,
	)
	dataset = BP4D_AU_dataset(test_json, is_train=False, args=dataset_args)
	if num_samples and num_samples < len(dataset):
		idx = np.random.choice(len(dataset), num_samples, replace=False)
		dataset = Subset(dataset, idx)
	pin_mem = str(device).startswith("cuda")
	dataloader = DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=pin_mem,
		persistent_workers=(num_workers > 0),
	)
	return dataset, dataloader


def render_image_grid_for_au(entries, au, data_root, mask, samples_per_au, save_path, device):
	"""Create rows=samples, cols=[original | mask | reconstructed]."""
	rows = sample_entries_for_au(entries, au, samples_per_au, seed=42)
	if not rows:
		print(f"⚠ No samples found for AU{au:02d} in provided JSON.")
		return

	tfm = transforms.Compose(
		[
			transforms.Resize((IMG_SIZE, IMG_SIZE)),
			transforms.ToTensor(),
			transforms.Normalize(mean=IMAGENET_MEAN.flatten().tolist(), std=IMAGENET_STD.flatten().tolist()),
		]
	)

	fig, axes = plt.subplots(len(rows), 3, figsize=(9, 3 * len(rows)))
	if len(rows) == 1:
		axes = np.expand_dims(axes, axis=0)

	for r, entry in enumerate(rows):
		img_path = Path(data_root) / entry["img_path"]
		if not img_path.exists():
			raise FileNotFoundError(f"Missing image: {img_path}")

		pil = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
		original_np = np.asarray(pil).astype(np.float32) / 255.0

		img_t = tfm(pil)
		recon_t = apply_fft_and_reconstruct(img_t, mask, device)
		recon_np = denorm_to_numpy(recon_t)

		axes[r, 0].imshow(original_np)
		axes[r, 0].set_title("Original" if r == 0 else "")
		axes[r, 0].axis("off")

		axes[r, 1].imshow(mask, cmap="gray", vmin=0, vmax=1)
		axes[r, 1].set_title("Mask" if r == 0 else "")
		axes[r, 1].axis("off")

		axes[r, 2].imshow(recon_np)
		axes[r, 2].set_title("Reconstructed" if r == 0 else "")
		axes[r, 2].axis("off")

	fig.suptitle(f"AU{au:02d} samples | original vs mask vs reconstructed", fontsize=12)
	plt.tight_layout()
	plt.savefig(save_path, dpi=140, bbox_inches="tight")
	plt.close()


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_type", default="FMAE", choices=["FMAE", "IAT"])
	parser.add_argument("--model_path", default="models/FMAE_BP4D_fold1.pth")
	parser.add_argument("--test_json", default="BP4D/BP4D_test1.json")
	parser.add_argument("--data_root", default="BP4D/BP4D_cropped/")
	parser.add_argument("--mask_pkl", required=True, help="Path to saved mask PKL (e.g., stage3 DFM file)")
	parser.add_argument("--mask_index", type=int, default=0, help="Index in PKL list; 0 is usually best")
	parser.add_argument("--num_samples_eval", type=int, default=500)
	parser.add_argument("--batch_size", type=int, default=24)
	parser.add_argument("--num_workers", type=int, default=4)
	parser.add_argument("--device", default="cuda")
	parser.add_argument("--au_targets", nargs="+", type=int, default=[4, 12])
	parser.add_argument("--samples_per_au", type=int, default=3)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--plot_drop", action="store_true")
	parser.add_argument("--output_dir", default="hfss/DFM/visualizations")
	args = parser.parse_args()

	set_global_seed(args.seed)

	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	mask_obj = load_mask_from_pkl(args.mask_pkl, args.mask_index)
	mask = mask_obj.mask
	if getattr(mask_obj, "flip", False):
		mask = 1 - mask

	mask_name = Path(args.mask_pkl).stem + f"_idx{args.mask_index}"
	visualize_mask(mask, output_dir / f"{mask_name}_mask.png", title=mask_name)
	print(f"✓ Saved mask plot: {output_dir / f'{mask_name}_mask.png'}")

	model = load_model(args.model_path, args.model_type, args.device)
	dataset, dataloader = build_eval_dataloader(
		test_json=args.test_json,
		data_root=args.data_root,
		num_samples=args.num_samples_eval,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		device=args.device,
	)
	print(f"✓ Loaded {len(dataset)} eval samples for per-AU metric plots")

	baseline = evaluate_mask_per_au(model, dataloader, lambda x: x, {}, args.device)
	masked = evaluate_mask_per_au(model, dataloader, mask_obj, baseline, args.device)

	baseline_f1 = [float(baseline.get(au, {}).get("f1", 0.0)) for au in AU_LABELS]
	masked_f1 = [float(masked.get(au, {}).get("f1", 0.0)) for au in AU_LABELS]

	perf_path = output_dir / f"{mask_name}_per_au_f1.png"
	plot_au_performance(
		baseline_f1=baseline_f1,
		masked_f1=masked_f1,
		save_path=perf_path,
		title_suffix=f"| mask={mask_name}",
	)
	print(f"✓ Saved per-AU performance plot: {perf_path}")

	if args.plot_drop:
		drop_path = output_dir / f"{mask_name}_per_au_drop.png"
		plot_au_drop(
			baseline_f1=baseline_f1,
			masked_f1=masked_f1,
			save_path=drop_path,
			title_suffix=f"| mask={mask_name}",
		)
		print(f"✓ Saved per-AU drop plot: {drop_path}")

	entries = load_json_entries(args.test_json)
	for au in args.au_targets:
		if au not in AU_LABELS:
			print(f"⚠ Skipping AU{au:02d}: not in configured AU labels {AU_LABELS}")
			continue
		grid_path = output_dir / f"{mask_name}_AU{au:02d}_grid.png"
		render_image_grid_for_au(
			entries=entries,
			au=au,
			data_root=args.data_root,
			mask=mask,
			samples_per_au=args.samples_per_au,
			save_path=grid_path,
			device=args.device,
		)
		print(f"✓ Saved image grid for AU{au:02d}: {grid_path}")

	print("✅ Visualization generation complete")


if __name__ == "__main__":
	main()

"""
1. Does not rerun HFSS.
2. Reads saved masks from PKL files.
3. Evaluates baseline vs masked performance per AU.

Outputs:
1. Mask image
2. Per-AU performance plot (baseline vs masked)
3. Per-AU drop plot (optional)
4. Image grids for selected AUs showing original vs masked vs reconstructed.

cd /Users/tima/Documents/DASC/Thesis/Frequency-Based-AU-Detection

python3 visualizations.py \
  --model_type IAT \
  --model_path models/FMAE_IAT_BP4D_fold1.pth \
  --test_json BP4D/BP4D_test1.json \
  --data_root BP4D/BP4D_cropped/ \
  --mask_pkl hfss/DFM/DFM_ID/IAT_stage3_ID_DFMs.pkl \
  --mask_index 0 \
  --au_targets 4 12 \
  --samples_per_au 3 \
  --num_samples_eval 500 \
  --batch_size 24 \
  --num_workers 4 \
  --device cuda \
  --plot_drop \
  --output_dir hfss/DFM/visualizations
""" 
