#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=smol-run
#SBATCH --time=36:00:00
#SBATCH --output=smol_argilla%j.out
#SBATCH --error=smol_argilla%j.err
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=20G

cd "$SLURM_SUBMIT_DIR"

module load Python/3.11.5-GCCcore-13.2.0

source .venv/bin/activate

export OMP_NUM_THREADS=1

export MASTER_ADDR=$(hostname)
export MASTER_PORT=$((29500 + SLURM_PROCID))

python \
  --nproc_per_node=$SLURM_NTASKS_PER_NODE \
  --master_port=$MASTER_PORT \
  src/main.py \
  --seed 184 \
  --epochs 5 \
  --model_name smol \
  --model_type text \
  --dataset_name argilla \
  --batch_size 4 \
  --lr 5e-5 \


deactivate