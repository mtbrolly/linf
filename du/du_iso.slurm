#!/bin/bash
# Slurm options
#SBATCH --partition=gpu-cascade
#SBATCH --qos=gpu
#SBATCH --gres=gpu:4
#SBATCH --time=23:59:00
#SBATCH --account=sc073

# Job commands

source /work/sc073/sc073/mbrolly/.bashrc
conda activate tf

python du/modelling.py