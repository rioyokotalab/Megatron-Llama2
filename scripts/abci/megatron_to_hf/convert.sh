#!/bin/bash
#$ -l rt_AF=1
#$ -l h_rt=5:00:00
#$ -j y
#$ -o outputs/convert/
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

# convert to hf format
python scripts/abci/megatron_to_hf/convert_megatron_to_hf.py \
  --convert_checkpoint_from_megatron_to_transformers \
  --megatron-path /bb/llm/gaf51275/llama/Megatron-LM \
  --load_path /bb/llm/gaf51275/llama/checkpoints/llama-2-7b-chat-megatron/tp2-pp2-lr-low/iter_0000600 \
  --save_path /bb/llm/gaf51275/llama/huggingface-checkpoint/Llama-2-7b-chat-megatron \
  --tokenizer_name meta-llama/Llama-2-7b-chat-hf
