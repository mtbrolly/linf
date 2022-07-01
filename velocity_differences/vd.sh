#!/bin/sh
# Grid Engine options (lines prefixed with #$)

# Name of job
#$ -N vd0107_lr5em6
# Directory in which to run code (-cwd or -wd <path-to-wd>)
#$ -wd /home/s1511699/git/linf
# Requested runtime allowance
#$ -l h_rt=47:59:00
# Requested number of cores in gpu parallel environment
#$ -pe gpu-titanx 2
# Requested memory per core
#$ -l h_vmem=32G
# Email address for notifications
#$ -notify
#$ -M m.brolly@ed.ac.uk
# Option to request resource reservation
#$ -R y

# Initialise the environment modules
. /etc/profile.d/modules.sh

# Load conda
module load anaconda/5.3.1
# Activate conda environment
source activate gpu

# Run the program
python velocity_differences/do_everything.py --model_name vd0107_lr5em6
