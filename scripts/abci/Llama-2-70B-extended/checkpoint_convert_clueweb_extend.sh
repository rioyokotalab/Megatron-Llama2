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

# distributed settings
TENSOR_PARALLEL_SIZE=8   # fixed
PIPELINE_PARALLEL_SIZE=8 # num layers 80: Llama-2 70B

# model config
HF_CHECKPOINT_DIR=/bb/llm/gaf51275/jalm/modified-llama2-base-tokenizer-nfkc-aligned-8/llama2-70b-base-merged-tokenizer-nfkc-16k-hf
MEGATRON_CHECKPOINT_DIR=/bb/llm/gaf51275/llama/llama-megatron-convert-checkpoint-hf/Llama-2-70b-extended/clueweb/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}

mkdir -p ${MEGATRON_CHECKPOINT_DIR}

# tokenizer config
TOKENIZER_MODEL=/bb/llm/gaf51275/jalm/jalm-tokenizer-private/tokenizer/jalm_llama_clueweb_nfkc_16k_aligned_8/merged_tokenizer_sp/jalm_llama.model

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
