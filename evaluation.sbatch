#!/bin/bash

#SBATCH --job-name=vlm_robustness
#SBATCH --gres=gpu:4               # Request 4 GPUs
#SBATCH --nodes=1
#SBATCH --ntasks=1                 # Number of tasks (1 task that uses multiple GPUs)
#SBATCH --cpus-per-task=64         # Adjust CPUs per task based on the total available cores
#SBATCH --time=24:00:00            # Time limit (24 hours)
#SBATCH --mem=128G                  # Total memory for the job
#SBATCH --qos=normal
#SBATCH --output=vlm_robustness_output_%j.log
#SBATCH --error=vlm_robustness_error_%j.log


# Load necessary modules and activate conda environment
source ~/.bashrc
conda activate vlm_env

# Print the Python executable path for debugging
which python

# Run the evaluation script
python evaluate_vlms_datasets.py
