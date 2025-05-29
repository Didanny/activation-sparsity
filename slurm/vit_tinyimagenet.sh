#!/bin/sh 
#SBATCH -p opengpu.p
#SBATCH --gres=gpu:1 
#SBATCH -o slurm_logs/log_tinyimagenet_vit.out 
#SBATCH -e slurm_logs/err_tinyimagenet_vit.out

python train.py --model tinyimagenet_vit --dataset tinyimagenet --label-smoothing 0.1
