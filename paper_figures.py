"""Minimal figure helpers for HFSS random and radial masking."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_PATH = PROJECT_ROOT / "path_to_your_image.png"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "paper_figures_output.png"

sys.path.insert(0, str(PROJECT_ROOT / "hfss" / "hfss"))
from transforms_search_space import gen_freqs_list, generate_mask, sample_frequency  # noqa: E402


IMG_SIZE = 224
PATCHES = {"stage1": 4, "stage2": 8, "stage3": 16, "stage4": 28, "stage5": 56, "stage6": 112}
DEFAULT_KEEP_RANGES = {
    1: (0.60, 0.80),
    2: (0.40, 0.60),
    3: (0.20, 0.40),
    4: (0.15, 0.30),
    5: (0.08, 0.20),
    6: (0.03, 0.10),
}


def _to_image_tensor(image: str | Path | np.ndarray | torch.Tensor | Image.Image) -> torch.Tensor:
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")
    if isinstance(image, Image.Image):
        image = image.resize((IMG_SIZE, IMG_SIZE))
        image = np.asarray(image, dtype=np.float32) / 255.0
    if isinstance(image, np.ndarray):
        array = image.astype(np.float32)
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=2)
        if array.shape[-1] == 3:
            tensor = torch.from_numpy(array).permute(2, 0, 1)
        elif array.shape[0] == 3:
            tensor = torch.from_numpy(array)
        else:
            raise ValueError(f"Unsupported image array shape: {array.shape}")
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        return tensor.contiguous().float()
    if torch.is_tensor(image):
        tensor = image.detach().clone().float()
        if tensor.ndim == 4:
            tensor = tensor[0]
        if tensor.ndim != 3:
            raise ValueError(f"Expected a 3D tensor image, got shape {tuple(tensor.shape)}")
        if tensor.shape[0] != 3 and tensor.shape[-1] == 3:
            tensor = tensor.permute(2, 0, 1)
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        return tensor.contiguous().float()
    raise TypeError(f"Unsupported image type: {type(image)!r}")


def load_image(image_path: str | Path = IMAGE_PATH) -> torch.Tensor:
    return _to_image_tensor(image_path)


def _make_symmetric(mask_arr: np.ndarray) -> np.ndarray:
    max_n_h = IMG_SIZE // 2
    max_n_w = IMG_SIZE // 2
    for h_index in range(-max_n_h, 1):
        for w_index in range(-max_n_w, max_n_w):
            h_matrix_index = IMG_SIZE // 2 + h_index
            w_matrix_index = IMG_SIZE // 2 + w_index
            if h_index != 0:
                mask_arr[IMG_SIZE - h_matrix_index - 1, IMG_SIZE - w_matrix_index - 1] = mask_arr[h_matrix_index, w_matrix_index]
    return mask_arr


def _active_map_from_parent(parent_mask: np.ndarray, grid_n: int) -> np.ndarray:
    cell = IMG_SIZE // grid_n
    parent_bin = (parent_mask > 0.5).astype(np.float32)
    return parent_bin.reshape(grid_n, cell, grid_n, cell).max(axis=(1, 3))


def _sample_stage_mask(
    grid_n: int,
    proportion: float,
    parent_mask: np.ndarray | None,
    keep_ratio_range: tuple[float, float] | None,
) -> np.ndarray:
    freqs = gen_freqs_list(grid_n, grid_n)

    if parent_mask is None:
        eligible_freqs = freqs
        active_map = None
    else:
        active_map = _active_map_from_parent(parent_mask, grid_n)
        eligible_freqs = [freq for freq in freqs if active_map[freq[0], freq[1]] > 0]

    if not eligible_freqs:
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

    if keep_ratio_range is not None:
        lo, hi = keep_ratio_range
        target_ratio = np.random.uniform(lo, hi)
        keep_count = int(round(target_ratio * len(eligible_freqs)))
        keep_count = max(1, min(len(eligible_freqs), keep_count))
        chosen_idx = np.random.choice(len(eligible_freqs), size=keep_count, replace=False)
        chosen = [eligible_freqs[idx] for idx in chosen_idx]
    else:
        chosen = sample_frequency(proportion, eligible_freqs)

    mask_int = generate_mask(chosen, grid_n, grid_n)
    if active_map is not None:
        mask_int = mask_int * torch.tensor(active_map, dtype=mask_int.dtype)

    mask = np.kron(np.asarray(mask_int), np.ones((IMG_SIZE // grid_n, IMG_SIZE // grid_n), dtype=np.float32))
    if parent_mask is not None:
        mask = parent_mask * mask
    return _make_symmetric(mask).astype(np.float32)


def _stage_name(stage: int | str) -> str:
    if isinstance(stage, str):
        return stage if stage.startswith("stage") else f"stage{stage}"
    return f"stage{int(stage)}"


def mask_random(
    image_path: str | Path = IMAGE_PATH,
    stage: int | str = 3,
    proportion: float = 0.8,
    keep_ratio_range: tuple[float, float] | None = None,
    seed: int | None = 42,
):
    image = load_image(image_path)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    stage_label = _stage_name(stage)
    if stage_label not in PATCHES:
        raise ValueError(f"Unknown stage: {stage}")

    stage_number = int(stage_label.replace("stage", ""))
    parent_mask = None
    for current_stage in range(1, stage_number + 1):
        current_label = f"stage{current_stage}"
        grid_n = PATCHES[current_label]
        stage_mask = _sample_stage_mask(grid_n, proportion, parent_mask, keep_ratio_range or DEFAULT_KEEP_RANGES[current_stage])
        parent_mask = _make_symmetric(stage_mask)

    masked_image = _apply_frequency_mask(image, parent_mask)
    return masked_image, parent_mask


def mask_radial(
    image_path: str | Path = IMAGE_PATH,
    mode: str = "low-pass",
    keep_pct: float = 10.0,
):
    image = load_image(image_path)
    if mode not in ("low-pass", "high-pass"):
        raise ValueError("mode must be 'low-pass' or 'high-pass'")

    size = IMG_SIZE
    center = size // 2
    ys, xs = np.ogrid[:size, :size]
    dist = np.sqrt((xs - center) ** 2 + (ys - center) ** 2)

    keep_pct = float(np.clip(keep_pct, 0.1, 100.0))
    if mode == "low-pass":
        radius = float(np.quantile(dist, keep_pct / 100.0))
        mask = (dist <= radius).astype(np.float32)
    else:
        radius = float(np.quantile(dist, 1.0 - keep_pct / 100.0))
        mask = (dist >= radius).astype(np.float32)

    masked_image = _apply_frequency_mask(image, mask)
    return masked_image, mask, radius


def _apply_frequency_mask(image: torch.Tensor, mask: np.ndarray) -> torch.Tensor:
    mask_t = torch.as_tensor(mask, dtype=torch.float32, device=image.device).unsqueeze(0).unsqueeze(0)
    freq = torch.fft.fftshift(torch.fft.fft2(image.unsqueeze(0), dim=(-2, -1)), dim=(-2, -1))
    freq = freq * mask_t
    return torch.fft.ifft2(torch.fft.ifftshift(freq, dim=(-2, -1)), dim=(-2, -1)).real.squeeze(0).float()


def _to_image_array(image: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        array = image.detach().cpu().float().permute(1, 2, 0).numpy()
    else:
        array = np.asarray(image, dtype=np.float32)
    return np.clip(array, 0.0, 1.0)


def _fft_spectrum(image: torch.Tensor | np.ndarray) -> np.ndarray:
    tensor = image if isinstance(image, torch.Tensor) else torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1)
    freq = torch.fft.fftshift(torch.fft.fft2(tensor.unsqueeze(0), dim=(-2, -1)), dim=(-2, -1))
    return torch.log1p(torch.abs(freq).mean(dim=1).squeeze(0)).detach().cpu().numpy()


def plot_mask_figure(
    original_image: torch.Tensor | np.ndarray,
    masked_image: torch.Tensor | np.ndarray,
    mode: str = "pair",
    save_path: str | Path | None = None,
    title: str | None = None,
):
    original_img = _to_image_array(original_image)
    masked_img = _to_image_array(masked_image)

    if mode == "pair":
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(original_img)
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        axes[1].imshow(masked_img)
        axes[1].set_title("Masked Image")
        axes[1].axis("off")
    elif mode == "spectrum":
        original_spec = _fft_spectrum(original_img)
        masked_spec = _fft_spectrum(masked_img)
        fig, axes = plt.subplots(2, 2, figsize=(9, 9))
        axes[0, 0].imshow(original_img)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis("off")
        axes[0, 1].imshow(original_spec, cmap="magma")
        axes[0, 1].set_title("Original Frequency Spectrum")
        axes[0, 1].axis("off")
        axes[1, 0].imshow(masked_img)
        axes[1, 0].set_title("Masked Image")
        axes[1, 0].axis("off")
        axes[1, 1].imshow(masked_spec, cmap="magma")
        axes[1, 1].set_title("Masked Frequency Spectrum")
        axes[1, 1].axis("off")
    else:
        raise ValueError("mode must be 'pair' or 'spectrum'")

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    output_path = Path(save_path) if save_path is not None else DEFAULT_OUTPUT_PATH
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal HFSS paper-figure helper")
    parser.add_argument("--image_path", default=str(IMAGE_PATH))
    parser.add_argument("--mask", choices=["random", "radial"], default="random")
    parser.add_argument("--stage", type=int, default=3)
    parser.add_argument("--proportion", type=float, default=0.8)
    parser.add_argument("--keep_pct", type=float, default=10.0)
    parser.add_argument("--radial_mode", choices=["low-pass", "high-pass"], default="low-pass")
    parser.add_argument("--figure_mode", choices=["pair", "spectrum"], default="spectrum")
    parser.add_argument("--output_path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    image = load_image(args.image_path)
    if args.mask == "random":
        masked, _ = mask_random(args.image_path, stage=args.stage, proportion=args.proportion, seed=args.seed)
    else:
        masked, _, _ = mask_radial(args.image_path, mode=args.radial_mode, keep_pct=args.keep_pct)

    output_path = plot_mask_figure(image, masked, mode=args.figure_mode, save_path=args.output_path)
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()

"""
python3 paper_figures.py --mask radial --image_path BP4D/BP4D_cropped_v1/M007/T3/14.jpg --radial_mode low-pass --keep_pct 3 --figure_mode spectrum --output_path results/radial_mask.png

python3 paper_figures.py --mask random --image_path BP4D/BP4D_cropped_v1/M007/T3/14.jpg --stage 3 --figure_mode spectrum --output_path results/random_mask.png
"""