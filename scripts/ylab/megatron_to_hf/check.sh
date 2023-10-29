#!/bin/bash
#YBATCH -r epyc-7502_8
#SBATCH --job-name=check
#SBATCH --time=6:00:00
#SBATCH --output outputs/checkpoint-check/%j.out
#SBATCH --error errors/checkpoint-check/%j.err

# python virtualenv
cd /home/kazuki/llama/Megatron-LM
source .env/bin/activate

python scripts/ylab/megatron_to_hf/check.py \
  --base-hf-model-path /mnt/nfs/Users/kazuki/hf_checkpoints/Llama-2-70b-hf \
  --converted_hf_model_path /mnt/nfs/Users/kazuki/checkpoints/llama/huggingface-checkpoint/Llama-2-70b-megatron/tp8-pp8/
