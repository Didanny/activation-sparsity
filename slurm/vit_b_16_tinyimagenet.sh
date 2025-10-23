#!/bin/sh 
#SBATCH -p opengpu.p
#SBATCH -w korn
#SBATCH --gres=gpu:1 
#SBATCH -o slurm_logs/log_tinyimagenet_vit.out 
#SBATCH -e slurm_logs/err_tinyimagenet_vit.out

python train.py --model tinyimagenet_vit_b_16 --dataset tinyimagenet --initial-lr 1e-2 --label-smoothing 0.1
