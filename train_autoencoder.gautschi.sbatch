#!/bin/bash -l
# FILENAME: train_autoencoder.sbatch

#SBATCH --account=davisjam
#SBATCH --nodes=1
#SBATCH --ntasks=14
#SBATCH --gpus-per-node=1
#SBATCH --partition=ai
#SBATCH --mem=80G
#SBATCH --qos=normal
#SBATCH --time=10:00:00
#SBATCH --job-name=train_similarity_transformer
#SBATCH --output=/home/patil185/joboutput/train_autoencoder.%j.out
#SBATCH --error=/home/patil185/joboutput/train_autoencoder.%j.err

set -euo pipefail

# -----------------------------
# Setup
# -----------------------------
module load cuda

# Activate your Python environment (adjust for your setup).
# Example using Conda:
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate ptm-recommendation

# Ensure we run from the directory where sbatch was submitted.
cd "${SLURM_SUBMIT_DIR}"

export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"
export HF_HOME="/scratch/gautschi/patil185/"
source .env

JOB_OUTPUT_DIR=/home/patil185/joboutput
mkdir -p "${JOB_OUTPUT_DIR}"

# Install dependencies for this job.
python -m pip install --user -r requirements.txt

echo "Running on node: $(hostname)"
echo "Job start: $(date)"
echo ""

# -----------------------------
# Train model
# -----------------------------
# python -u train.py ModelAutoEncoderTrainer
python -u train.py SimilarityTransformerTrainer

# -----------------------------
# Evaluate model
# -----------------------------
# python -u evaluate_model_autoencoder.py

echo ""
echo "Job end: $(date)"
