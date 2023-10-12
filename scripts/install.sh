#!/bin/bash
#$ -l rt_AG.small=1
#$ -l h_rt=1:00:00
#$ -j y
#$ -o outputs/install/
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

# pip install
pip install --upgrade pip
pip install -r requirements.txt

pip install ninja wheel packaging

# apex install
git clone git@github.com:NVIDIA/apex.git
cd apex
git checkout 50ac8425403b98147cbb66aea9a2a27dd3fe7673

pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

# flash-attention install
pip install flash-attn==2.3.0 --no-build-isolation

# huggingface install
pip install transformers==4.32.0

pip install accelerate

# distributed checkpointing
pip install zarr

pip install tensorstore
