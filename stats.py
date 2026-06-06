"""
Report BP4D AU distributions and inspect saved HFSS mask artifacts.
Also renders summary plots for fold JSON files and PKL masks.
"""

import argparse
import json
import math
import pickle
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Defaults for AU distribution mode.
FOLD_JSON_PATH = Path("BP4D/BP4D_test1.json")
AU_LABELS = [1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24]
PLOT_PATH = Path("results") / "au_distribution.png"


class _WhiteMaskStub:
	"""Fallback class used only for unpickling saved White_Mask objects."""
	pass


class _MaskUnpickler(pickle.Unpickler):
	"""Allow loading saved mask PKLs without importing hfss modules."""

	def find_class(self, module, name):
		if name == "White_Mask" and (
			module == "transforms_search_space"
			or module.endswith(".transforms_search_space")
		):
			return _WhiteMaskStub
		return super().find_class(module, name)


def load_pickle_with_fallback(pkl_path):
	"""Load pickle with fallback for White_Mask class path differences."""
	with pkl_path.open("rb") as f:
		try:
			return pickle.load(f)
		except ModuleNotFoundError as e:
			# Common on servers where HFSS modules are not on PYTHONPATH.
			if "transforms_search_space" not in str(e):
				raise

	with pkl_path.open("rb") as f:
		return _MaskUnpickler(f).load()


def save_distribution_plot(pos_counts, total, save_path, fold_name):
	labels = [f"AU{au:02d}" for au in AU_LABELS]
	percents = [((100.0 * pos_counts[au] / total) if total else 0.0) for au in AU_LABELS]

	save_path.parent.mkdir(parents=True, exist_ok=True)
	plt.figure(figsize=(9, 4.8))
	plt.bar(labels, percents, color="#4C78A8")
	plt.ylabel("Positive rate (%)")
	plt.xlabel("Action Unit")
	plt.title(f"AU distribution | {fold_name}")
	plt.grid(axis="y", alpha=0.25)
	plt.tight_layout()
	plt.savefig(save_path, dpi=180, bbox_inches="tight")
	plt.close()


def run_au_distribution(fold_json_path, plot_path):
	if not fold_json_path.exists():
		raise FileNotFoundError(f"File not found: {fold_json_path}")

	total = 0
	pos_counts = {au: 0 for au in AU_LABELS}

	with fold_json_path.open("r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			item = json.loads(line)
			aus = item.get("AUs", [])
			total += 1
			for au in AU_LABELS:
				if au in aus:
					pos_counts[au] += 1

	print(f"Fold: {fold_json_path}")
	print(f"Total samples: {total}")
	print("\nAU distribution (positive count and percentage):")
	for au in AU_LABELS:
		c = pos_counts[au]
		pct = (100.0 * c / total) if total else 0.0
		print(f"AU{au:02d}: {c:6d} ({pct:6.2f}%)")

	save_distribution_plot(pos_counts, total, plot_path, fold_json_path.name)
	print(f"\nSaved plot: {plot_path}")


def load_mask_arrays(mask_pkl_path):
	"""Load mask arrays from saved DFM PKL entries (White_Mask or ndarray-like)."""
	if not mask_pkl_path.exists():
		raise FileNotFoundError(f"Mask PKL not found: {mask_pkl_path}")

	data = load_pickle_with_fallback(mask_pkl_path)

	if not isinstance(data, list):
		raise ValueError(f"Expected list in PKL, got: {type(data)}")

	mask_arrays = []
	for item in data:
		if hasattr(item, "mask"):
			arr = np.asarray(item.mask, dtype=np.float32)
		else:
			arr = np.asarray(item, dtype=np.float32)
		if arr.ndim != 2:
			continue
		mask_arrays.append(arr)

	if not mask_arrays:
		raise ValueError("No 2D masks found in PKL")

	return mask_arrays


def save_mask_gallery(mask_arrays, save_path, max_masks=30, cols=6):
	"""Save one image containing many masks for quick visual inspection."""
	count = min(len(mask_arrays), max_masks)
	cols = max(1, cols)
	rows = int(math.ceil(count / float(cols)))

	fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.2 * rows))
	if rows == 1 and cols == 1:
		axes = np.array([[axes]])
	elif rows == 1:
		axes = np.array([axes])
	elif cols == 1:
		axes = np.array([[ax] for ax in axes])

	for i in range(rows * cols):
		r = i // cols
		c = i % cols
		ax = axes[r][c]
		ax.axis("off")
		if i >= count:
			continue

		arr = mask_arrays[i]
		center = arr.shape[0] // 2
		center_included = bool(arr[center, center] > 0.5)
		keep_pct = float((arr > 0.5).mean() * 100.0)

		ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
		ax.plot(center, center, "ro", markersize=2)
		ax.set_title(
			f"mask{i} | keep={keep_pct:.1f}% | center={'IN' if center_included else 'OUT'}",
			fontsize=8,
		)

	plt.tight_layout()
	save_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(save_path, dpi=180, bbox_inches="tight")
	plt.close()


def parse_stage_f1_lines(log_path):
	"""Parse lines like: mask 12: F1=0.00% from a stage log snippet file."""
	if not log_path.exists():
		raise FileNotFoundError(f"F1 log not found: {log_path}")

	pattern = re.compile(r"mask\s+(\d+):\s+F1=([0-9]*\.?[0-9]+)%")
	rows = []
	with log_path.open("r", encoding="utf-8") as f:
		for line in f:
			m = pattern.search(line)
			if m:
				rows.append((int(m.group(1)), float(m.group(2))))
	return rows


def save_selected_mask_gallery(selected, save_path, cols=5):
	"""Save gallery for selected masks.

	selected: list of tuples (title, mask_array)
	"""
	count = len(selected)
	if count == 0:
		raise ValueError("No masks selected for plotting")

	cols = max(1, cols)
	rows = int(math.ceil(count / float(cols)))

	fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.6 * rows))
	if rows == 1 and cols == 1:
		axes = np.array([[axes]])
	elif rows == 1:
		axes = np.array([axes])
	elif cols == 1:
		axes = np.array([[ax] for ax in axes])

	for i in range(rows * cols):
		r = i // cols
		c = i % cols
		ax = axes[r][c]
		ax.axis("off")
		if i >= count:
			continue

		title, arr = selected[i]
		center = arr.shape[0] // 2
		ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
		ax.plot(center, center, "ro", markersize=2)
		ax.set_title(title, fontsize=8)

	plt.tight_layout()
	save_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(save_path, dpi=200, bbox_inches="tight")
	plt.close()


def summarize_mask_centers(mask_arrays):
	"""Print center-in/out summary for quick sanity checks."""
	print("\nCenter check for loaded masks:")
	for i, arr in enumerate(mask_arrays):
		c = arr.shape[0] // 2
		center = float(arr[c, c])
		keep_pct = float((arr > 0.5).mean() * 100.0)
		state = "IN" if center > 0.5 else "OUT"
		print(f"mask{i:02d}: center={center:.3f} ({state}) | keep={keep_pct:.2f}%")


def run_mask_visualization(
	mask_pkl_path,
	mask_out_dir,
	max_masks=30,
	cols=6,
	lowest_n=0,
	f1_log_path=None,
):
	mask_arrays = load_mask_arrays(mask_pkl_path)
	print(f"Loaded {len(mask_arrays)} masks from {mask_pkl_path}")
	summarize_mask_centers(mask_arrays)

	gallery_path = mask_out_dir / "mask_gallery.png"
	save_mask_gallery(mask_arrays, gallery_path, max_masks=max_masks, cols=cols)
	print(f"Saved mask gallery: {gallery_path}")

	if lowest_n and lowest_n > 0:
		lowest_n = min(lowest_n, len(mask_arrays))

		# Saved PKLs are usually already in descending F1 order (best->worst).
		# For that common case, the last N entries are the lowest-F1 masks.
		selected_indices = list(range(len(mask_arrays) - lowest_n, len(mask_arrays)))
		selected = []

		rows = None
		if f1_log_path:
			rows = parse_stage_f1_lines(f1_log_path)
			if len(rows) < len(mask_arrays):
				print(
					f"Warning: parsed only {len(rows)} F1 rows for {len(mask_arrays)} masks; "
					"continuing with index-only labels"
				)

		for idx in selected_indices:
			arr = mask_arrays[idx]
			keep_pct = float((arr > 0.5).mean() * 100.0)
			if rows is not None and idx < len(rows):
				mask_id, f1_pct = rows[idx]
				title = f"idx{idx} | mask{mask_id} | F1={f1_pct:.2f}% | keep={keep_pct:.1f}%"
			else:
				title = f"idx{idx} | keep={keep_pct:.1f}%"
			selected.append((title, arr))

		lowest_path = mask_out_dir / f"lowest_{lowest_n}_mask_gallery.png"
		save_selected_mask_gallery(selected, lowest_path, cols=min(cols, lowest_n))
		print(f"Saved lowest-{lowest_n} gallery: {lowest_path}")


def parse_args():
	parser = argparse.ArgumentParser(description="AU distribution and saved-mask debug utilities")
	parser.add_argument("--fold_json", default=str(FOLD_JSON_PATH), help="Path to BP4D fold json")
	parser.add_argument("--plot_path", default=str(PLOT_PATH), help="Output AU distribution plot path")

	# Optional mask-debug mode.
	parser.add_argument("--mask_pkl", default=None, help="Path to saved DFM PKL to inspect")
	parser.add_argument("--mask_out_dir", default="results/mask_debug", help="Where to save mask visuals")
	parser.add_argument("--max_masks", type=int, default=30, help="Max masks to show in gallery")
	parser.add_argument("--mask_cols", type=int, default=6, help="Columns in mask gallery")
	parser.add_argument("--lowest_n", type=int, default=0, help="Also save gallery with N lowest-F1 masks")
	parser.add_argument(
		"--f1_log_path",
		default=None,
		help="Optional text file containing stage lines like 'mask X: F1=Y%' for labeling",
	)
	return parser.parse_args()


"""
Example runs:
python stats.py --fold_json BP4D/BP4D_test1.json
python stats.py --mask_pkl hfss/DFM/AU01/FMAE_stage3_AU01_DFMs.pkl --lowest_n 10
"""
def main():
	args = parse_args()

	run_au_distribution(Path(args.fold_json), Path(args.plot_path))

	if args.mask_pkl:
		run_mask_visualization(
			mask_pkl_path=Path(args.mask_pkl),
			mask_out_dir=Path(args.mask_out_dir),
			max_masks=args.max_masks,
			cols=args.mask_cols,
			lowest_n=args.lowest_n,
			f1_log_path=Path(args.f1_log_path) if args.f1_log_path else None,
		)


if __name__ == "__main__":
	main()

