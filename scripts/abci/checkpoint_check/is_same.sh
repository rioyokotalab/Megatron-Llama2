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

python scripts/abci/checkpoint_check/is_same.py \
  --model-path-1 /bb/llm/gaf51275/llama/checkpoints/llama-2-13b-base-extended-megatron/llm-jp/tp2-pp4/iter_0005000/mp_rank_00_000/model_optim_rng.pt \
  --model-path-2 /bb/llm/gaf51275/llama/checkpoints/llama-2-13b-base-extended-megatron/llm-jp/tp2-pp4/iter_0015000/mp_rank_00_000/model_optim_rng.pt \
  --megatron-path /bb/llm/gaf51275/llama/Megatron-LM
