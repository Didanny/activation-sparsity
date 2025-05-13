#!/bin/sh 
#SBATCH -p opengpu.p
#SBATCH --gres=gpu:1 
#SBATCH -o slurm_logs/log_cifar100_vit.out 
#SBATCH -e slurm_logs/err_cifar100_vit.out

python train.py --model cifar100_vit --dataset cifar100 --optimizer adam --lr-scheduler cosine-warmup --initial-lr 1e-3 --final-lr 1e-5 --warmup-epochs 5 --label-smoothing 0.1
# python train.py --model cifar100_vit --dataset cifar100 --label-smoothing 0.1
