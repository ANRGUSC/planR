#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=8
#SBATCH --job-name=LizardA.job
#SBATCH --output=.out/LizardA.out
#SBATCH --error=.out/LizardB.err
#SBATCH --time=2-00:00
#SBATCH --mem=12000
#SBATCH --qos=normal
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leezmiriam@gmail.com

python3 main.py