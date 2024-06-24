#!/bin/bash -l
#SBATCH --time=00:30:00
#SBATCH --mem=10GB
#SBATCH --partition=gpu-debug
#SBATCH --cpus-per-task=4
#SBATCH --output=train.out
#SBATCH --error=error.out

module load scicomp-python-env

python run-script.py