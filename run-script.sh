#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH --mem=16GB
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --output=.out/train.out
#SBATCH --error=.out/error.out

module load scicomp-python-env

python run-script.py