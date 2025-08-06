#!/bin/bash
#SBATCH --partition=digitallab
#SBATCH --job-name=qwen-test-run
#SBATCH --time=10:00:00
#SBATCH --output=test-qwen-test-argilla%j.out
#SBATCH --error=test-qwen-test-argilla%j.err
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=20G

cd "$SLURM_SUBMIT_DIR"

module load Python/3.11.5-GCCcore-13.2.0

source .venv/bin/activate

python \
  src/testing/cross.py \
  --model_name qwen \
  --model_type text \
  --dataset_name argilla \
  --batch_size 8 \
  --lr 5e-6 \


deactivate