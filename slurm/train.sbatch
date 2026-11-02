#!/bin/bash
#SBATCH --account=torch_pr_421_courant
#SBATCH --partition=h200_courant
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --job-name=celeba_train
#SBATCH --output=slurm/%x_%j.out
# Generic launcher: sbatch [--partition=...] slurm/train.sbatch <python module> [args...]
#   e.g. sbatch slurm/train.sbatch celebaflow.train --out runs/celeba/uncond --compile
# Data lives on the GPU and training resumes from <out>/ckpt.pt, so a requeue just continues.
cd /scratch/eb4727/L2MD_Project
source .venv/bin/activate
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader -l 300 &
python -u -m "$@"
