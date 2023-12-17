#!/bin/bash
#YBATCH -r dgx-a100_8
#SBATCH --job-name=check
#SBATCH --time=6:00:00
#SBATCH --output outputs/check/%j.out
#SBATCH --error errors/check/%j.err
. /etc/profile.d/modules.sh
module load cuda/11.8
module load cudnn/cuda-11.x/8.9.0
module load nccl/cuda-11.7/2.14.3
module load openmpi/4.0.5

# python virtualenv
cd /home/kazuki/llama/Megatron-LM
source .env/bin/activate

python scripts/ylab/megatron_to_hf/check.py \
  --base-hf-model-path /mnt/nfs/Users/kazuki/hf_checkpoints/Llama-2-70b-hf \
  --converted-hf-model-path /mnt/nfs/Users/kazuki/checkpoints/llama/huggingface-checkpoint/Llama-2-70b-megatron/tp8-pp8/
