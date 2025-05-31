#!/bin/sh
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -p opengpu.p
#SBATCH -w korn
#SBATCH --gres=gpu:1
#SBATCH -o slurm_logs/log_tinyimagenet_vit.out
#SBATCH -e slurm_logs/err_tinyimagenet_vit.out

for alpha in 5e-8 1e-7 5e-7 1e-6 5e-6 1e-5; do
    python train.py --model tinyimagenet_vit --dataset tinyimagenet --pretrained --fine-tune --alpha "$alpha"
done
