#!/bin/bash
#PJM -L rscgrp=tutorial2-a
#PJM -L node=1
#PJM -N convert
#PJM -L elapse=06:00:00
#PJM -g gt01

# module load
module load cuda/11.8
module load cudnn/8.8.0
module load nccl/2.16.5
module load gcc/8.3.1
module load aquarius
module load hpcx/2.10

# python virtualenv
cd /work/gt01/GROUP_215-01/work/Megatron-LM
source .env/bin/activate

# distributed settings
TENSOR_PARALLEL_SIZE=2  # fixed
PIPELINE_PARALLEL_SIZE=4 # num layers 32: Llama-2 7B

# model config
HF_CHECKPOINT_DIR=/work/gt01/GROUP_215-01/work/hf-checkpoints/Llama-2-7b-chat-hf
MEGATRON_CHECKPOINT_DIR=/work/gt01/GROUP_215-01/work/megatron-checkpoints/Llama-2-7b-chat/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}

mkdir -p ${MEGATRON_CHECKPOINT_DIR}

# tokenizer config
TOKENIZER_MODEL=/work/gt01/GROUP_215-01/work/hf-checkpoints/Llama-2-7b-chat-hf/tokenizer.model

export TRANSFORMERS_CACHE=/work/gt01/GROUP_215-01/.cache/huggingface/transformers

# convert
python tools/checkpoint/util.py \
  --model-type GPT \
  --loader llama2_hf \
  --saver megatron \
  --target-tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
  --target-pipeline-parallel-size ${PIPELINE_PARALLEL_SIZE} \
  --load-dir ${HF_CHECKPOINT_DIR} \
  --save-dir ${MEGATRON_CHECKPOINT_DIR} \
  --tokenizer-model ${TOKENIZER_MODEL}
