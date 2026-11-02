#!/bin/bash
#SBATCH --account=torch_pr_421_courant
#SBATCH --partition=h200_courant
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --job-name=hold_gpu
#SBATCH --output=slurm/%x_%j.out
# Placeholder allocation for interactive testing: run commands with
#   srun --jobid=<id> --overlap bash -c '...'
sleep infinity
