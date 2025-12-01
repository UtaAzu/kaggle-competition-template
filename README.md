# Kaggle Competition Template

This repository provides a starter template for organizing Kaggle-style competitions, notebooks, scripts, and experiment tracking. It is intentionally minimal and configurable -- use the `examples/archive/` folder for competition-specific examples.

What's included:
- `train.py` and `predict.py` examples for training/predicting models (config-driven)
- `experiments/` skeleton for storing experiment artifacts, OOF/metrics, run.json
- `docs/` templates and guides. Place competition-specific content under `examples/archive/<COMPETITION>/`
- `scripts/` helpers for common workflows: recording artifacts, CV/LB analysis, etc.
- `src/` structured code and `src/metrics/` for metric helpers

Quick start:
1. Set your dataset path in `--data-dir` or environment variables (see `train.py` / `validate.py` options)
2. Use `train.py --config configs/<your-config>` to run training
3. Save your artifacts into `experiments/<EXP_ID>/<exp_id>-artifacts` and run `validate.py --artifacts-dir <path>` to summarize

Customization:
- Replace evaluation code in `src/metrics/` with your competition's metric
- Add dataset-specific pre-processing under `src/` as needed

For competition-specific docs and archived examples, see the `examples/archive/` directory.