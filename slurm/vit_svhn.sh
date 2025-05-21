#!/bin/sh 
#SBATCH -p opengpu.p
#SBATCH -w poison
#SBATCH --gres=gpu:1 
#SBATCH -o slurm_logs/log_svhn_vit.out 
#SBATCH -e slurm_logs/err_svhn_vit.out

python train.py --model svhn_vit --dataset svhn --label-smoothing 0.1
