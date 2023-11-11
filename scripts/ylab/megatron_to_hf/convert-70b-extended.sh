#!/bin/bash
#YBATCH -r dgx-a100_8
#SBATCH --job-name=convert
#SBATCH --time=12:00:00
#SBATCH --output outputs/checkpoint-convert/%j.out
#SBATCH --error errors/checkpoint-convertk/%j.err
. /etc/profile.d/modules.sh
module load cuda/11.8
module load cudnn/cuda-11.x/8.9.0
module load nccl/cuda-11.7/2.14.3
module load openmpi/4.0.5

set -e

# python virtualenv
cd /home/kazuki/llama/Megatron-LM
source .env/bin/activate

# TP > 1, PP > 1の場合は、TP=1, PP=1になるように scripts/abci/change_tp_pp.sh を実行してからconvertしてください
BASE_TENSOR_PARALLEL_SIZE=1   # fixed
BASE_PIPELINE_PARALLEL_SIZE=1 # fixed

ITERATION=2000
FORMATTED_ITERATION=$(printf "%07d" $ITERATION)

SAVE_DIR=/mnt/nfs/Users/kazuki/checkpoints/llama/hf_checkpoints/Llama-2-70b-extended/iter_${FORMATTED_ITERATION}
mkdir -p ${SAVE_DIR}

# change latest_checkpointed_iteration.txt
echo $ITERATION >/mnt/nfs/Users/kazuki/checkpoints/llama/megatron_checkpoints/Llama-2-70b-extended/tp${BASE_TENSOR_PARALLEL_SIZE}-pp${BASE_PIPELINE_PARALLEL_SIZE}/latest_checkpointed_iteration.txt

python scripts/abci/megatron_to_hf/llama_checkpoint_conversion.py \
  --convert_checkpoint_from_megatron_to_transformers \
  --load_path /mnt/nfs/Users/kazuki/checkpoints/llama/megatron_checkpoints/Llama-2-70b-extended/tp${BASE_TENSOR_PARALLEL_SIZE}-pp${BASE_PIPELINE_PARALLEL_SIZE} \
  --save_path $SAVE_DIR \
  --target_params_dtype "bf16" \
  --print-checkpoint-structure \
  --megatron-path /home/kazuki/llama/Megatron-LM
