#!/bin/bash
#SBATCH --account=torch_pr_421_courant
#SBATCH --partition=h200_courant,b200_courant,h200_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --job-name=celeba_stage2
#SBATCH --output=slurm/%x_%j.out
# Fine-tune one stage-2 configuration, then run the composition experiments on it.
#   sbatch slurm/stage2.sbatch <run name> <pair> [extra celebaflow.train args...]
#   e.g. sbatch slurm/stage2.sbatch blond_male_pj0.1 blond_male --p_joint 0.1
set -e
NAME=$1; PAIR=$2; shift 2
cd /scratch/eb4727/L2MD_Project
source .venv/bin/activate
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader -l 300 &
python -u -m celebaflow.train --pair "$PAIR" --clean --init runs/celeba/uncond/ckpt.pt --out "runs/celeba/$NAME" \
    --steps 20000 --lr 1e-4 --ema 0.999 --sample_every 5000 --compile "$@"
python -u -m celebaflow.experiments --ckpt "runs/celeba/$NAME/ckpt.pt" --out "figures/celeba/$NAME" --n 4000
