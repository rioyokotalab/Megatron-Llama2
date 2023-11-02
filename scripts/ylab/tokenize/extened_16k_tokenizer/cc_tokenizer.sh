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

DATASET_DIR=/mnt/nfs/Users/tn/datasets/okazaki_lab_cc_600
MERGED_DIR=/mnt/nfs/Users/kazuki/dataset/merged/okazaki_lab_cc_600
OUTPUT_DIR=/mnt/nfs/Users/kazuki/dataset/binarized/okazaki_lab_cc_600
MODEL_PATH=/mnt/nfs/Users/kazuki/tokenizer/jalm_llama_clueweb_nfkc_16k_aligned_8/merged_tokenizer_sp/jalm_llama.model

# merge ja cc
MERGED_FILE_PATH="$MERGED_DIR/ja-cc.jsonl"

if [ -f "$MERGED_FILE_PATH" ]; then
  # ファイルの内容を初期化
  >"$MERGED_FILE_PATH"
fi

INPUT_DIR="$DATASET_DIR"

for file in $INPUT_DIR/*CC-MAIN-*; do
  if [ -f "$file" ]; then
    cat "$file" >>$MERGED_FILE_PATH
  fi
done

echo "merged cc ja: $MERGED_FILE_PATH"

mkdir -p ${OUTPUT_DIR}

# tokenize ja cc
python tools/preprocess_data.py \
  --input $MERGED_FILE_PATH \
  --output-prefix $OUTPUT_DIR/ja_cc \
  --tokenizer-model $MODEL_PATH \
  --tokenizer-type Llama2Tokenizer \
  --workers 64 \
  --append-eod
