# Product-Manifold Projection

This project hosts the token-level product-manifold model that used to live alongside the Expert MoE codebase. It keeps the same Hydra/Dora workflow but is now isolated so it can evolve independently.

## Setup

1. Create/activate a Python 3.11 environment and install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Materialise the dataset caches once (they are shared with the Expert MoE repo):
   ```bash
   python tools/build_dataset.py --dataset wikiann --splits train validation --subset 1.0
   ```

## Training

Launch training locally with Dora:
```bash
dora --package src --main_module train run
```
Override any Hydra config key via the command line, e.g. `train.epochs=30 product_model.num_subspaces=6`.

To submit a SLURM job, adapt and run `tools/slurm/training/train_product.sbatch` (the script derives paths relative to this directory and writes logs under `logs/training/`).

## Project Structure

- `src/` – data pipeline, utilities, model, trainer, and Hydra configs.
- `tools/` – dataset helper and SLURM scripts.
- `requirements.txt` – runtime dependencies shared with the Expert MoE project.

The code mirrors the original implementation so that checkpoints/configs remain compatible with past experiments.
