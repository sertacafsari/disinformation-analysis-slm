#!/bin/bash
#SBATCH --partition=digitallab
#SBATCH --job-name=qwen-run
#SBATCH --time=20:00:00
#SBATCH --output=qwen_liar2_test_1%j.out
#SBATCH --error=qwen_liar2_test_1%j.err
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=20G

cd "$SLURM_SUBMIT_DIR"

module load Python/3.11.5-GCCcore-13.2.0

source .venv/bin/activate

python \
  src/main.py \
  --seed 326 \
  --epochs 5 \
  --model_name qwen \
  --model_type text \
  --dataset_name liar2 \
  --batch_size 32 \
  --lr 5e-6 \


deactivate