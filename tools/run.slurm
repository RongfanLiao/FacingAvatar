#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=42
#SBATCH --mem-per-cpu=3850
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --partition=gpulowpri
#SBATCH --time=24:00:00
#SBATCH --account=su004-gpulowpri

# Full pipeline: preprocess LookingFace dataset + train.
#
# Usage:
#   cd /gpfs/home/r/rongfan/head_avatar
#   sbatch scripts/run.slurm
#

cd /gpfs/home/r/rongfan/head_avatar
export PYTHONPATH=$PWD:$PYTHONPATH
PY="/gpfs/home/r/rongfan/micromamba/envs/qwen_vl/bin/python"

echo "JOBID=${SLURM_JOB_ID}"
echo "=== [$(date)] Starting pipeline ==="

${PY} run.py

echo "=== [$(date)] Done ==="
