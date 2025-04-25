#!/bin/sh 
#SBATCH -n 1 
#SBATCH -N 1 
#SBATCH -p opengpu.p 
#SBATCH -w korn
#SBATCH --gres=gpu:1 
#SBATCH -o slurm_logs/log_cifar100_vit.out 
#SBATCH -e slurm_logs/err_cifar100_vit.out

python train.py --model cifar100_vit --dataset cifar100 --label-smoothing 0.25
