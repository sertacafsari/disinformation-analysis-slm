#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=mistral_initial_run
#SBATCH --time=04:00:00
#SBATCH --output=mistral_initial_run_%j.out
#SBATCH --error=mistral_initial_run_%j.err
#SBATCH --gres=gpu:v100:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G

cd "$SLURM_SUBMIT_DIR"

module purge
module load Python/3.11.5-GCCcore-13.2.0

source .venv/bin/activate

srun python -m torch.distributed.run \
      --nproc_per_node=$SLURM_NTASKS_PER_NODE \
      src/mistral/mistral_finetune.py \
        --run_name initial-run1 \

deactivate

