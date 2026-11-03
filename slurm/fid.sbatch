#!/bin/bash
#SBATCH --account=torch_pr_421_courant
#SBATCH --partition=h200_courant,b200_courant,h200_public,l40s_courant
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --job-name=celeba_fid
#SBATCH --output=slurm/%x_%j.out
# FID (10k samples) for every stage-1 EMA snapshot, then the stage-1 report figure.
cd /scratch/eb4727/L2MD_Project
source .venv/bin/activate
rm -f runs/celeba/uncond/fid.jsonl
for ck in runs/celeba/uncond/ema_step*.pt; do python -u -m celebaflow.fid --ckpt "$ck" --n 10000 --n_steps 100; done
python -u -m celebaflow.report_stage1 --run runs/celeba/uncond --out figures/celeba/stage1
