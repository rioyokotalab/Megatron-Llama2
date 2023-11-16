#!/bin/bash

set -e

start=5000
end=10000
increment=5000

tokenizer_path=/bb/llm/gaf51275/jalm/jalm-tokenizer-private/tokenizer/jalm_llama_clueweb_nfkc_16k_aligned_8/merged_tokenizer_sp/jalm_llama.model

upload_base_dir=/bb/llm/gaf51275/llama/from_megatron_hf_checkpoints/hf_checkpoints/Llama2-13b-base-extended-clueweb

for ((i = start; i <= end; i += increment)); do
  upload_dir=$upload_base_dir/iter_$(printf "%07d" $i)

  python scripts/abci/upload/upload.py \
    --ckpt-path $upload_dir \
    --repo-name tokyotech-llm/Llama2-13b-base-extended-clueweb-megatron-iter$(printf "%07d" $i)
done
