#!/bin/bash
#YBATCH -r epyc-7502_1
#SBATCH --job-name=tokenize
#SBATCH --time=12:00:00
#SBATCH --output outputs/tokenize/%j.out
#SBATCH --error errors/tokenize/%j.err

set -e

# python virtualenv
cd /home/kazuki/llama/Megatron-LM
source .env/bin/activate

MERGED_DIR=/mnt/nfs/Users/kazuki/dataset/merged/okazaki_lab_cc_600
OUTPUT_DIR=/mnt/nfs/Users/kazuki/dataset/binarized/okazaki_lab_cc_600_default_Llama_tokenizer
MODEL_PATH=/mnt/nfs/Users/kazuki/hf_checkpoints/Llama-2-70b-hf/tokenizer.model

# tokenize ja cc
python tools/preprocess_data.py \
  --input $MERGED_DIR/ja-cc.jsonl \
  --output-prefix $OUTPUT_DIR/ja_cc \
  --tokenizer-model $MODEL_PATH \
  --tokenizer-type Llama2Tokenizer \
  --workers 64 \
  --append-eod
