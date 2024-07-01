#!/bin/bash -l
#SBATCH --time=20:00:00
#SBATCH --mem=16GB
#SBATCH --partition=gpu-v100-16g
#SBATCH --cpus-per-task=8
#SBATCH --output=.out/train.out
#SBATCH --error=.out/error.out

module load scicomp-python-env

python run-script.py