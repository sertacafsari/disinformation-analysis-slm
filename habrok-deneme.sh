#!/bin/bash
#SBATCH --job-name=gpt2_wide   
#SBATCH --output=gpt2_wide_%j.out
#SBATCH --error=gpt2_wide_%j.err
#SBATCH --time=71:30:00
#SBATCH --mem=8G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu

# Ensure the job starts in the submission directory

# <main_dir>/job_scripts/xlstm/moses/job_script.sh
cd "$SLURM_SUBMIT_DIR"

# Load the modules
module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# Activate the virtual environment
source .venv/bin/activate

export PYTHONPATH=$(python -c 'import site; print(site.getsitepackages()[0])'):$PYTHONPATH


python -m src.main \
    --run_name gpt2Grokked \
    --seed 42 \
    --model_arch gpt2 \
    --layers 4 \
    --n_dim 512 \
    --n_heads 8 \
    --benchmark moses \
    --device cuda \
    --epochs 1000 \
    --learning_rate 1e-3 \
    --batch_size 1024 \
    --early_stop 0 \
    --warmup_steps 0 \



# Deactivate virtual environment
deactivate