# SLURM Utilities

Batch helpers for the product-manifold project live here.

- `training/train_product.sbatch`: launch `dora --package src --main_module train` on the cluster.
- `common.sh`: shared environment bootstrap (module loads, virtualenv activation, cache env vars).

All paths are derived relative to this directory, so you can clone the project anywhere. Logs land in `logs/training/` under the project root.
