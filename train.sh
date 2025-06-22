
module load Python/3.11.5-GCCcore-13.2.0

source .venv/bin/activate

# export OMP_NUM_THREADS=1

# export MASTER_ADDR=$(hostname)
# export MASTER_PORT=$((29500 + SLURM_PROCID))

torchrun \
  src/main.py \
  --seed 184 \
  --epochs 5 \
  --model_name smol \
  --model_type vision \
  --dataset_name faux \
  --batch_size 32 \
  --lr 2e-5 \


deactivate