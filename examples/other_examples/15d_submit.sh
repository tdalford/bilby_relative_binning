#!/usr/bin/env bash
#
#SBATCH --job-name=15d
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=1G
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=END --mail-user=email@moritz-huebner.de

srun python 15d_gaussian.py $1