#!/bin/sh 
#SBATCH -p opengpu.p
#SBATCH --gres=gpu:1 
#SBATCH -o slurm_logs/log_tinyimagenet_vit_s_16.out 
#SBATCH -e slurm_logs/err_tinyimagenet_vit_s_16.out

# python train.py --model tinyimagenet_vit_s_16 --dataset tinyimagenet --optimizer adam --initial-lr 1e-3 --label-smoothing 0.1 --epochs 100
python train.py --model tinyimagenet_vit_s_16 --dataset tinyimagenet --optimizer sgd --initial-lr 1e-3 --label-smoothing 0.1 --epochs 100