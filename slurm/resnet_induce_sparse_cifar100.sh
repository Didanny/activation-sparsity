#!/bin/sh
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -p opengpu.p
#SBATCH -w korn
#SBATCH --gres=gpu:1
#SBATCH -o slurm_logs/log_cifar100_resnet.out
#SBATCH -e slurm_logs/err_cifar100_resnet.out

for alpha in 1e-6 5e-6 1e-5 5e-5 1e-4; do
    python train.py --model cifar100_resnet20 --dataset cifar100 --pretrained --fine-tune --alpha "$alpha"
done
# for alpha in 1e-6 5e-6 1e-5 5e-5 1e-4; do
#     python train.py --model cifar100_resnet32 --dataset cifar100 --pretrained --fine-tune --alpha "$alpha"
# done
# for alpha in 1e-6 5e-6 1e-5 5e-5 1e-4; do
#     python train.py --model cifar100_resnet44 --dataset cifar100 --pretrained --fine-tune --alpha "$alpha"
# done
# for alpha in 1e-6 5e-6 1e-5 5e-5 1e-4; do
#     python train.py --model cifar100_resnet56 --dataset cifar100 --pretrained --fine-tune --alpha "$alpha"
# done
