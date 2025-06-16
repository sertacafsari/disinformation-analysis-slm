#!/bin/bash
#SBATCH --partition=digitallab
#SBATCH --job-name=roberta-run
#SBATCH --time=04:00:00
#SBATCH --output=roberta_argilla%j.out
#SBATCH --error=roberta_argilla%j.err
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G

cd "$SLURM_SUBMIT_DIR"

module load Python/3.11.5-GCCcore-13.2.0

source .venv/bin/activate

export OMP_NUM_THREADS=1

export MASTER_ADDR=$(hostname)
export MASTER_PORT=$((29500 + SLURM_PROCID))

torchrun \
  --nproc_per_node=$SLURM_NTASKS_PER_NODE \
  --master_port=$MASTER_PORT \
  src/main.py \
  --seed 184 \
  --epochs 5 \
  --model_name qwen \
  --model_type vision \
  --dataset_name faux \
  --batch_size 32 \
  --lr 2e-5 \


deactivate