#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=0:10:0
#SBATCH --job-name="trivial"

cd $SLURM_SUBMIT_DIR

./trivial
