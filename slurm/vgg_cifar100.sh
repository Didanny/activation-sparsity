#!/bin/sh 
#SBATCH -n 1 
#SBATCH -N 1 
#SBATCH -p opengpu.p 
#SBATCH --gres=gpu:1 
#SBATCH -o slurm_logs/log_cifar100_resnet.out 
#SBATCH -e slurm_logs/err_cifar100_resnet.out

python train.py --model vgg11_bn --dataset cifar100
python train.py --model vgg13_bn --dataset cifar100
python train.py --model vgg16_bn --dataset cifar100
python train.py --model vgg19_bn --dataset cifar100
