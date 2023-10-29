#!/bin/bash
#!/bin/bash
#$ -l rt_AG.small=1
#$ -l h_rt=03:30:00
#$ -j y
#$ -o outputs/inference/
#$ -cwd

# module load
source /etc/profile.d/modules.sh
module load cuda/11.8/11.8.0
module load cudnn/8.9/8.9.2
module load nccl/2.16/2.16.2-1
module load hpcx/2.12

# swich virtual env
cd /bb/llm/gaf51275/llama/Megatron-LM
source .env/bin/activate

### YOUR HUGGINGFACE TOKEN ###
export HF_TOKEN=""
export HF_HOME=/bb/llm/gaf51275/.cache/huggingface

# inference

HF_MODEL=/bb/llm/gaf51275/llama/huggingface-checkpoint/Llama-2-7b-chat-megatron
python scripts/abci/inference/inference.py \
  --hf-model-path $HF_MODEL \
  --hf-tokenizer-path /bb/llm/gaf51275/llama/huggingface-checkpoint/Llama-2-7b-chat-hf/tokenizer.model \
  --hf-cache-dir $HF_HOME
