#!/bin/sh 
#SBATCH -p opengpu.p
#SBATCH --gres=gpu:1 
#SBATCH -o slurm_logs/log_cifar10_vit.out 
#SBATCH -e slurm_logs/err_cifar10_vit.out

python train.py --model cifar10_vit --dataset cifar10 --optimizer adam --lr-scheduler cosine-warmup --initial-lr 1e-3 --final-lr 1e-5 --warmup-epochs 5 --label-smoothing 0.1
