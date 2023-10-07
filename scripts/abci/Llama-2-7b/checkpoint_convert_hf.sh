#!/bin/bash
#$ -l rt_AF=1
#$ -l h_rt=5:00:00
#$ -j y
#$ -o outputs/checkpoint_convert/
#$ -cwd

# module load
source /etc/profile.d/modules.sh
module load cuda/11.8/11.8.0
module load cudnn/8.9/8.9.2
module load nccl/2.16/2.16.2-1
module load hpcx/2.12

# python virtualenv
cd /bb/llm/gaf51275/llama/Megatron-LM
source .env/bin/activate

# model config
HF_CHECKPOINT_DIR=/bb/llm/gaf51275/llama/huggingface-checkpoint/Llama-2-7b-hf
MEGATRON_CHECKPOINT_DIR=/bb/llm/gaf51275/llama/llama-megatron-convert-checkpoint-hf/Llama-2-7b

mkdir -p ${MEGATRON_CHECKPOINT_DIR}

# tokenizer config
TOKENIZER_MODEL=/bb/llm/gaf51275/jalm/jalm-tokenizer-private/tokenizer/jalm_llama_clueweb/merged_tokenizer_hf

# convert
python tools/checkpoint/util.py \
  --model-type GPT \
  --loader llama2_hf \
  --save megatron \
  --target-tensor-parallel-size 1 \
  --load-dir ${HF_CHECKPOINT_DIR} \
  --save-dir ${MEGATRON_CHECKPOINT_DIR} \
  --tokenizer-model ${TOKENIZER_MODEL}
