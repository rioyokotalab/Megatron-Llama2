#!/bin/bash
#YBATCH -r dgx-a100_8
#SBATCH --job-name=megatron-hf-convert
#SBATCH --time=6:00:00
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

# distributed settings
TARGET_TENSOR_PARALLEL_SIZE=1   # fixed
TARGET_PIPELINE_PARALLEL_SIZE=1 # fixed

BASE_TENSOR_PARALLEL_SIZE=8
BASE_PIPELINE_PARALLEL_SIZE=8

# model config
BASE_CHECKPOINT_DIR=/home/kazuki/checkpoints/llama-2-70b-base/tp${BASE_TENSOR_PARALLEL_SIZE}-pp${BASE_PIPELINE_PARALLEL_SIZE}
TARGET_CHECKPOINT_DIR=/home/kazuki/checkpoints/llama-2-70b-base/tp${TARGET_TENSOR_PARALLEL_SIZE}-pp${TARGET_PIPELINE_PARALLEL_SIZE}

mkdir -p ${TARGET_CHECKPOINT_DIR}

# tokenizer config
TOKENIZER_MODEL=/home/kazuki/hf_models/Llama-2-7b-hf/tokenizer.model

# change latest_checkpointed_iteration.txt
ITERATION=5000
echo $ITERATION >${BASE_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt

# convert
python tools/checkpoint/util.py \
  --model-type GPT \
  --loader megatron \
  --saver megatron \
  --megatron-path /home/kazuki/llama/Megatron-LM \
  --target-tensor-parallel-size ${TARGET_TENSOR_PARALLEL_SIZE} \
  --target-pipeline-parallel-size ${TARGET_PIPELINE_PARALLEL_SIZE} \
  --load-dir ${BASE_CHECKPOINT_DIR} \
  --save-dir ${TARGET_CHECKPOINT_DIR}
