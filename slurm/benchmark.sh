#!/bin/sh 
#SBATCH -p opengpu.p
#SBATCH -w korn
#SBATCH --gres=gpu:1

python benchmark.py > results/rtx400_benchmark.out