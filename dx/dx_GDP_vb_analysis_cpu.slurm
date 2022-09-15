#!/bin/bash
# Slurm options
#SBATCH --partition=standard
#SBATCH --qos=highpriority
#SBATCH --exclusive
#SBATCH --nodes=16
#SBATCH --time=00:59:00
#SBATCH --account=sc073

# Job commands

source /work/sc073/sc073/mbrolly/.bashrc
conda activate tf

python dx/analysis_vb.py