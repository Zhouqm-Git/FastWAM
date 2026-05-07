#!/usr/bin/env bash
#SBATCH -J test
#SBATCH --account=proj_paper_zhouqinming_20260303
#SBATCH --qos=important
#SBATCH -p a100_global
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH --begin=now
#SBATCH -o /mnt/users/zhouqm-20251002/logs/%x-%j.out
#SBATCH -e /mnt/users/zhouqm-20251002/logs/%x-%j.err

set -euo pipefail
mkdir -p /mnt/users/zhouqm-20251002/logs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_HOME=/mnt/users/zhouqm-20251002/.cache_job/torch
export HF_HOME=/mnt/users/zhouqm-20251002/.cache_job/huggingface
export DIFFSYNTH_MODEL_BASE_PATH=/mnt/users/zhouqm-20251002/FastWAM/checkpoints

CONDA_BASE=/mnt/public/apps/miniconda3
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate fastwam

cd /mnt/users/zhouqm-20251002/FastWAM

python scripts/train.py \
    task=libero_uncond_2cam224_1e-4 \
    max_steps=4000 \
    save_every=1000 \
    wandb.enabled=false \
    2>&1 | tee logs/train_4k.log
