#!/bin/bash

set -e

python scripts/abci/upload/upload.py \
  --ckpt-path /bb/llm/gaf51275/llama/from_megatron_hf_checkpoints/Llama-2-7b-chat/tp2-pp2-exp \
  --repo-name tokyotech-llm/llama2-7b-chat-4node-gbs1024-megatron-lm-taishi-data-en50-ja50-maxlr1e-4_min3.3e-6-iter_0005600
