#!/bin/sh 
#SBATCH -n 1 
#SBATCH -N 1 
#SBATCH -p opengpu.p 
#SBATCH --gres=gpu:1 
#SBATCH -o slurm_logs/log_cifar100_resnet.out 
#SBATCH -e slurm_logs/err_cifar100_resnet.out

python train.py --model resnet18 --dataset cifar100
python train.py --model resnet34 --dataset cifar100
python train.py --model resnet50 --dataset cifar100
python train.py --model resnet101 --dataset cifar100
python train.py --model resnet152 --dataset cifar100
