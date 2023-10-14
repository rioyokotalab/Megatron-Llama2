#!/bin/bash
#PJM -L rscgrp=tutorial-share
#PJM -L gpu=1
#PJM -N tokenize
#PJM -L elapse=06:00:00
#PJM -g gt01

# module load
module load cuda/11.8
module load cudnn/8.8.0
module load nccl/2.16.5
module load gcc/8.3.1
module load aquarius
module load hpcx/2.10

set -e

# python virtualenv
cd /work/gt01/GROUP_215-01/work/Megatron-LM
source .env/bin/activate

DATASET_DIR=/work/gt01/GROUP_215-01/work/datasets
MERGED_DIR=/work/gt01/GROUP_215-01/work/datasets/merged
OUTPUT_DIR=/work/gt01/GROUP_215-01/work/datasets/binarized
MODEL_PATH=/work/gt01/GROUP_215-01/work/hf-checkpoints/Llama-2-7b-chat-hf/tokenizer.model

# merge ja wiki
MERGED_FILE_PATH="$MERGED_DIR/ja/ja_wiki.jsonl"

if [ -f "$MERGED_FILE_PATH" ]; then
  # ファイルの内容を初期化
  >"$MERGED_FILE_PATH"
fi

INPUT_DIR="$DATASET_DIR/ja"

for file in $INPUT_DIR/*wiki_ja*; do
  if [ -f "$file" ]; then
    cat "$file" >>$MERGED_FILE_PATH
  fi
done

echo "merged ja wiki: $MERGED_FILE_PATH"

mkdir -p ${OUTPUT_DIR}

# tokenize japanese wiki
python tools/preprocess_data.py \
  --input $MERGED_FILE_PATH \
  --output-prefix $OUTPUT_DIR/ja_wiki \
  --tokenizer-model $MODEL_PATH \
  --tokenizer-type Llama2Tokenizer \
  --workers 64 \
  --append-eod

# merge ja oscar
MERGED_FILE_PATH="$MERGED_DIR/ja/ja_oscar.jsonl"

if [ -f "$MERGED_FILE_PATH" ]; then
  # ファイルの内容を初期化
  >"$MERGED_FILE_PATH"
fi

INPUT_DIR="$DATASET_DIR/ja"

for file in $INPUT_DIR/*oscar*; do
  if [ -f "$file" ]; then
    cat "$file" >>$MERGED_FILE_PATH
  fi
done

echo "merged ja oscar: $MERGED_FILE_PATH"

mkdir -p ${OUTPUT_DIR}

# tokenize japanese oscar
python tools/preprocess_data.py \
  --input $MERGED_FILE_PATH \
  --output-prefix $OUTPUT_DIR/ja_oscar \
  --tokenizer-model $MODEL_PATH \
  --tokenizer-type Llama2Tokenizer \
  --workers 64 \
  --append-eod
