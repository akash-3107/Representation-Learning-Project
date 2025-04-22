#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:v100:1 -p v100
#SBATCH --job-name=generateCaptionv100
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

module load python
conda activate control

python envCheck.py

python generate_image_captions.py