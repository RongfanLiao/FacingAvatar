#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/miniconda3/envs/avatar/bin/python}"

cd "$ROOT_DIR"

exec "$PYTHON_BIN" train_motion_diffusion.py \
  --predefined_splits_dir data/LookingFace/dataset_splits \
  --checkpoint_dir checkpoints/motion_diffusion_lookingface_predefined \
  "$@"
