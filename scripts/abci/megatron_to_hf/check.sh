#!/bin/bash
#$ -l rt_AF=1
#$ -l h_rt=5:00:00
#$ -j y
#$ -o outputs/check/
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

python scripts/abci/megatron_to_hf/check.py \
  --base-hf-model-path /bb/llm/gaf51275/llama/from_megatron_hf_checkpoints/hf_checkpoints/Llama2-13b-base-extended-llm-jp/iter_0010000 \
  --converted-hf-model-path /bb/llm/gaf51275/llama/from_megatron_hf_checkpoints/hf_checkpoints/Llama2-13b-base-extended-llm-jp/iter_0015000
