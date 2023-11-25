#!/bin/bash
#$ -l rt_AF=1
#$ -l h_rt=5:00:00
#$ -j y
#$ -o outputs/convert/
#$ -cwd

set -e

# module load
source /etc/profile.d/modules.sh
module load cuda/11.8/11.8.0
module load cudnn/8.9/8.9.2
module load nccl/2.16/2.16.2-1
module load hpcx/2.12

# python virtualenv
cd /bb/llm/gaf51275/llama/Megatron-LM
source .env/bin/activate

# TP > 1, PP > 1の場合は、TP=1, PP=1になるように scripts/abci/change_tp_pp.sh を実行してからconvertしてください
BASE_TENSOR_PARALLEL_SIZE=1   # fixed
BASE_PIPELINE_PARALLEL_SIZE=1 # fixed

ITERATION=5000
FORMATTED_ITERATION=$(printf "%07d" $ITERATION)

SAVE_DIR=/bb/llm/gaf51275/llama/from_megatron_hf_checkpoints/hf_checkpoints/Llama2-7b-base-extended-en-updated/okazaki_lab_cc/iter_${FORMATTED_ITERATION}
mkdir -p ${SAVE_DIR}

# change latest_checkpointed_iteration.txt
echo $ITERATION >/bb/llm/gaf51275/llama/from_megatron_hf_checkpoints/megatron_checkpoints/Llama2-7b-base-extended-en-updated/okazaki_lab_cc/tp${BASE_TENSOR_PARALLEL_SIZE}-pp${BASE_PIPELINE_PARALLEL_SIZE}/latest_checkpointed_iteration.txt

python scripts/abci/megatron_to_hf/llama_checkpoint_conversion.py \
  --convert_checkpoint_from_megatron_to_transformers \
  --load_path /bb/llm/gaf51275/llama/from_megatron_hf_checkpoints/megatron_checkpoints/Llama2-7b-base-extended-en-updated/okazaki_lab_cc/tp${BASE_TENSOR_PARALLEL_SIZE}-pp${BASE_PIPELINE_PARALLEL_SIZE} \
  --save_path $SAVE_DIR \
  --target_params_dtype "bf16" \
  --print-checkpoint-structure \
  --megatron-path /bb/llm/gaf51275/llama/Megatron-LM
