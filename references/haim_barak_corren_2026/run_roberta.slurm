#!/bin/bash
#SBATCH --job-name=classify_house
#SBATCH --partition=gpu-tad
#SBATCH --account=gpu-tad-users
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=classify_house_%j.out
#SBATCH --error=classify_house_%j.err

# --- load environment ---
module load cuda
module load anaconda

# activate your conda env (CHANGE NAME IF NEEDED)
source activate your_env_name

# go to project directory
cd /path/to/empirical_evidence_corren

# run script
python classify_house_members.py
