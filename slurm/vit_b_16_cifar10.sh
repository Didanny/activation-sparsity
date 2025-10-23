#!/bin/sh 
#SBATCH -p opengpu.p
#SBATCH --gres=gpu:1 
#SBATCH -o slurm_logs/log_cifar10_vit.out 
#SBATCH -e slurm_logs/err_cifar10_vit.out

python train.py --model cifar10_vit_b_16 --dataset cifar10 --optimizer adam --initial-lr 1e-3 --label-smoothing 0.1
# python train.py --model cifar100_vit --dataset cifar100 --label-smoothing 0.1
