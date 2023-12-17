#!/bin/bash
#$ -l rt_AF=1
#$ -l h_rt=5:00:00
#$ -j y
#$ -o outputs/tokenize/
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

DATASET_DIR=/bb/llm/gaf51275/llama/datasets/llama2-llm-jp-corpus/v1.0.2/sample
OUTPUT_DIR=/bb/llm/gaf51275/llama/datasets/llama2-llm-jp-corpus/v1.0.2/tokenized/sentencepiece

mkdir -p ${OUTPUT_DIR}

# tokenize japanese cc
for ((i = 0; i <= 37; i++)); do
  INPUT_FILE=${DATASET_DIR}/ja_cc/merged_train_${i}.jsonl

  python tools/preprocess_data.py \
    --input ${INPUT_FILE} \
    --output-prefix ${OUTPUT_DIR}/ja_cc_${i} \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model /bb/llm/gaf51275/jalm/jalm-tokenizer-private/tokenizer/jalm_llama_clueweb/merged_tokenizer_sp/jalm_llama.model \
    --vocab-file /bb/llm/gaf51275/jalm/jalm-tokenizer-private/tokenizer/jalm_llama_clueweb/merged_tokenizer_sp/jalm_llama.vocab \
    --append-eod \
    --workers 64
done

# tokenize japanese wikipedia
python tools/preprocess_data.py \
  --input ${DATASET_DIR}/ja_wiki/merged_train_0.jsonl \
  --output-prefix ${OUTPUT_DIR}/ja_wiki \
  --tokenizer-type Llama2Tokenizer \
  --tokenizer-model /bb/llm/gaf51275/jalm/jalm-tokenizer-private/tokenizer/jalm_llama_clueweb/merged_tokenizer_sp/jalm_llama.model \
  --vocab-file /bb/llm/gaf51275/jalm/jalm-tokenizer-private/tokenizer/jalm_llama_clueweb/merged_tokenizer_sp/jalm_llama.vocab \
  --append-eod \
  --workers 64
