#!/bin/sh
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -p opengpu.p
#SBATCH -w korn
#SBATCH --gres=gpu:1
#SBATCH -o slurm_logs/log_cifar10_vit.out
#SBATCH -e slurm_logs/err_cifar10_vit.out

for alpha in 1e-6 5e-6 1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2 5e-2 1e-1 5e-1 1 5 1e1 5e1; do
    python train.py --model cifar10_vit --dataset cifar10 --pretrained --fine-tune --sparsity-type semi-structured --alpha "$alpha"
done
