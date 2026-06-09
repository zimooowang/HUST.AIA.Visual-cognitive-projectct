#!/usr/bin/env bash
set -euo pipefail

configs=(
  "configs/ablations/14d_norm_only.yaml"
  "configs/ablations/14d_joint_diversity_only.yaml"
  "configs/ablations/14d_low_coordination.yaml"
  "configs/ablations/14d_low_action_diversity.yaml"
)

methods=(
  "random"
  "action_change"
  "visual_change"
  "visual_action"
  "brain_coreset"
  "full"
)

for config in "${configs[@]}"; do
  echo "==> Running ablation config: ${config}"
  python scripts/02_select_coreset.py --config "${config}" --method all
  for method in "${methods[@]}"; do
    python scripts/03_train_mlp.py --config "${config}" --method "${method}"
  done
  python scripts/04_evaluate.py --config "${config}" --method all
  python scripts/05_make_plots.py --config "${config}"
done
