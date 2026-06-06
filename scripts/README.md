# scripts/

This directory contains small helper entry points that support the main frequency-based AU workflows.

## Files

- `main.py` provides a lightweight script entry point for shared utilities.
- `finetune.py` wraps finetuning-related helpers.
- `hfss.py` contains script-level helpers for HFSS experiments.
- `hfss_search_accuracy.py` checks search accuracy and related evaluation paths.

## Usage

Run these helpers from the repository root so the shared paths for `BP4D/`, `models/`, and `results/` resolve consistently.