#!/usr/bin/env bash
set -euo pipefail

python scripts/00_inspect_dataset.py --config configs/default.yaml
python scripts/01_extract_features.py --config configs/default.yaml
python scripts/02_select_coreset.py --config configs/default.yaml --method all
python scripts/03_train_mlp.py --config configs/default.yaml --method random
python scripts/03_train_mlp.py --config configs/default.yaml --method action_change
python scripts/03_train_mlp.py --config configs/default.yaml --method visual_change
python scripts/03_train_mlp.py --config configs/default.yaml --method visual_action
python scripts/03_train_mlp.py --config configs/default.yaml --method brain_coreset
python scripts/03_train_mlp.py --config configs/default.yaml --method full
python scripts/04_evaluate.py --config configs/default.yaml --method all
python scripts/05_make_plots.py --config configs/default.yaml

