#!/bin/bash
#SBATCH --job-name=BTP
#SBATCH -N 1
#SBATCH --mail-user=aditya.jain.19031@iitgoa.ac.in
#SBATCH --ntasks-per-node=2
#SBATCH --time=0-01:00:00
#SBATCH --mail-type=END,FAIL

module purge
module load anaconda3/2021.11
conda activate pytorch

python test_sparse_bnn_moon_classification.py > output_cpu.txt

exit 0