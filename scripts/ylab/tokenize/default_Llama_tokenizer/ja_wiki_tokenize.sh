#!/bin/bash
#YBATCH -r epyc-7502_1
#SBATCH --job-name=tokenize
#SBATCH --time=6:00:00
#SBATCH --output outputs/tokenize/%j.out
#SBATCH --error errors/tokenize/%j.err

set -e

# python virtualenv
cd /home/kazuki/llama/Megatron-LM
source .env/bin/activate

DATASET_DIR=/mnt/nfs/Users/tn/datasets/llm-jp-corpus/v1.0.2-merge/ja_wiki
OUTPUT_DIR=/mnt/nfs/Users/kazuki/dataset/binarized/okazaki_lab_cc_600_default_Llama_tokenizer
MODEL_PATH=/mnt/nfs/Users/kazuki/hf_checkpoints/Llama-2-70b-hf/tokenizer.model

mkdir -p ${OUTPUT_DIR}

# tokenize ja wiki
python tools/preprocess_data.py \
  --input $DATASET_DIR/ja_wiki_merged_train.jsonl \
  --output-prefix $OUTPUT_DIR/ja_wiki \
  --tokenizer-model $MODEL_PATH \
  --tokenizer-type Llama2Tokenizer \
  --workers 64 \
  --append-eod
