#!/bin/bash -l
#SBATCH --time=12:00:00
#SBATCH --mem=12GB
#SBATCH --partition=gpu-v100-16g
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --output=.out/train.out
#SBATCH --error=.out/error.out

module load scicomp-python-env

python test-CIFAR-10-C.py