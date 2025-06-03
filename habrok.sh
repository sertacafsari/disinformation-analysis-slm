#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=mistral_initial_run
#SBATCH --time=04:00:00
#SBATCH --output=mistral_initial_run_%j.out
#SBATCH --error=mistral_initial_run_%j.err
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=20G

cd "$SLURM_SUBMIT_DIR"

module load Python/3.11.5-GCCcore-13.2.0

source .venv/bin/activate

export OMP_NUM_THREADS=1

export MASTER_ADDR=$(hostname)
export MASTER_PORT=$((29500 + SLURM_PROCID))

torchrun \
  --nproc_per_node=$SLURM_NTASKS_PER_NODE \
  --master_port=$MASTER_PORT \
  src/roberta/roberta_finetune.py \
    --run_name roberta-runs-sweeping \
    --batch_size 64

deactivate

