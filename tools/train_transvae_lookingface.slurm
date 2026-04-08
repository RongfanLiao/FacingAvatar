#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=42
#SBATCH --mem-per-cpu=3850
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --account=su004
#SBATCH --job-name=transvae_lf
#SBATCH --output=logs/transvae_lf_%j.out

# Train TransVAE on full LookingFace dataset with predefined splits.
#
# Usage:
#   cd /gpfs/home/r/rongfan/head_avatar
#   mkdir -p logs
#   sbatch scripts/train_transvae_lookingface.slurm
#

cd /gpfs/home/r/rongfan/head_avatar
export PYTHONPATH=$PWD:$PYTHONPATH
PY="/gpfs/home/r/rongfan/micromamba/envs/qwen_vl/bin/python"

echo "JOBID=${SLURM_JOB_ID}"
echo "=== [$(date)] Starting TransVAE training ==="

${PY} train_motion_transvae.py \
    --predefined_splits_dir data/LookingFace/dataset_splits \
    --epochs 5 \
    --batch_size 2 \
    --lr 1e-5 \
    --val_interval 1 \
    --checkpoint_dir checkpoints/motion_transvae_lookingface_5ep \
    --target_variant content

echo "=== [$(date)] Done ==="
