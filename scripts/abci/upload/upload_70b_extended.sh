#!/bin/bash

set -e

start=2000
end=8000
increment=2000

tokenizer_path=/bb/llm/gaf51275/llama/huggingface-checkpoint/Llama-2-7b-hf/tokenizer.model

upload_base_dir=/mnt/nfs/Users/kazuki/checkpoints/llama/hf_checkpoints/Llama-2-70b-extended

for ((i = start; i <= end; i += increment)); do
  upload_dir=$upload_base_dir/iter_$(printf "%07d" $i)

  python scripts/abci/upload/upload.py \
    --ckpt-path $upload_dir \
    --repo-name tokyotech-llm/Llama2-70b-base-extended-cc-megatron-iter$(printf "%07d" $i)
done
