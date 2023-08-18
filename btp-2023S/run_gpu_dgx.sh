#!/bin/bash
#SBATCH --job-name=BTP
#SBATCH --partition=DGX
#SBATCH --mail-user=aditya.jain.19031@iitgoa.ac.in
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --time=0-01:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL

module purge
module load anaconda3/2021.11
conda activate pytorch

python test_sparse_bnn_moon_classification.py > output_gpu_dgx.txt

exit 0