#!/bin/bash
#SBATCH --partition=digitallab
#SBATCH --job-name=smol-vision-run
#SBATCH --time=10:00:00
#SBATCH --output=smol-vision-faux%j.out
#SBATCH --error=smol-vision-faux%j.err
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G

cd "$SLURM_SUBMIT_DIR"

module load Python/3.11.5-GCCcore-13.2.0

source .venv/bin/activate

python \
  src/main.py \
  --seed 184 \
  --epochs 5 \
  --model_name smol \
  --model_type vision \
  --dataset_name faux \
  --batch_size 8 \
  --lr 5e-6 \


deactivate