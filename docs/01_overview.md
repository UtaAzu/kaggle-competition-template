Overview

This repository is a generic Kaggle competition template. It provides an opinionated project structure and scripts for training, inference, validation, EDA, and experiment management. The examples and historical competition artifacts were archived into `examples/archive/`.

Quick Start

- Place your dataset into a directory and either set common environment variables (e.g. `TEST_DIR`, `TRAIN_CSV`) or provide CLI options to the scripts `train.py`, `validate.py`, and `analysis.py`.
- Put experiment artifacts into `experiments/<EXP_ID>/<exp_id>-artifacts` (or provide `--artifacts-dir` to `validate.py`).
- See `docs/TEMPLATE_QUICKSTART.md` for a generic quickstart workflow.

Where to find example competitions:

- `examples/archive/` — archived competitions (for reference only)
- `examples/` — example configurations and scripts that show how to adapt the template to your competition

Evaluation & Submissions

This repository does not enforce a particular evaluation metric. Example metrics and notebooks are provided under `examples/archive/` for historical competitions. Use `src/metrics/` to add any evaluation code you need.

If you were using this repo directly in a competitive environment, update the scripts' CLI args or environment variables to match your dataset and evaluation pipeline.