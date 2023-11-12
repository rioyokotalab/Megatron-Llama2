#!/bin/bash

# python virtualenv
cd /home/taishi/Megatron-LM
source .env/bin/activate

# distributed settings
export MASTER_ADDR=10.130.184.10
export MASTER_PORT=12802

echo "MASTER_ADDR=${MASTER_ADDR}"

NODE_TYPE="a100"
export NUM_GPU_PER_NODE=8

HOSTFILE_NAME=/home/taishi/Megatron-LM/hostfile/16node


NUM_NODES=16
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))
export LOGLEVEL=INFO
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=WARN
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0

# model config
# llama-2-7b: https://huggingface.co/meta-llama/Llama-2-7b-hf/blob/main/config.json
HIDDEN_SIZE=4096
FFN_HIDDEN_SIZE=11008 # intermediate size (HuggingFace)
NUM_LAYERS=32
NUM_HEADS=32
SEQ_LENGTH=4096

# distributed settings
TENSOR_PARALLEL_SIZE=2   # fixed
PIPELINE_PARALLEL_SIZE=2 # num layers 32: Llama-2 7B
DATA_PARALLEL_SIZE=$((${NUM_GPUS} / (${TENSOR_PARALLEL_SIZE} * ${PIPELINE_PARALLEL_SIZE})))

# training config
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=1024
TRAIN_STEPS=61000 # e.g. llama: 1T tokens / 4M tokens_per_batch = 250000 steps
# 今回は約270B Tokens #実際の数値は250B Tokens

LR=3e-4
MIN_LR=3e-5
LR_WARMUP_STEPS=2000
WEIGHT_DECAY=0.1
GRAD_CLIP=1

# model config
TOKENIZER_MODEL=/model/taishi/llm-jp-tokenizer/models/ver2.2/code20K_en40K_ja60K.ver2.2.model
CHECKPOINT_SAVE_DIR=/data/checkpoints_7b/v1.0.1_code20K_en40K_ja60K_ver2.2

mkdir -p ${CHECKPOINT_SAVE_DIR}

# data config
DATASET_DIR=/data/llm-jp-corpus/v1.0.1/merge/binarized/ver2.2/code20K_en40K_ja60K

TRAIN_DATA_PATH=""
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1442858730 ${DATASET_DIR}/ja_wiki/ja_wiki_merge_1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4697570389 ${DATASET_DIR}/en_wiki/en_wiki_merge_1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 8670888578 ${DATASET_DIR}/code_stack/code_stack_merge_1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 16802680411 ${DATASET_DIR}/en_pile/en_pile_merge_1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 16814033199 ${DATASET_DIR}/en_pile/en_pile_merge_2_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 16941076697 ${DATASET_DIR}/en_pile/en_pile_merge_3_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 16933151184 ${DATASET_DIR}/en_pile/en_pile_merge_4_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 16863221834 ${DATASET_DIR}/en_pile/en_pile_merge_5_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 16928046076 ${DATASET_DIR}/en_pile/en_pile_merge_6_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 8477979936 ${DATASET_DIR}/en_pile/en_pile_merge_7_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 18910928683 ${DATASET_DIR}/ja_cc/ja_cc_merge_1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 18929141567 ${DATASET_DIR}/ja_cc/ja_cc_merge_2_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 18917563475 ${DATASET_DIR}/ja_cc/ja_cc_merge_3_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 18919892586 ${DATASET_DIR}/ja_cc/ja_cc_merge_4_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 18923027590 ${DATASET_DIR}/ja_cc/ja_cc_merge_5_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 18936260793 ${DATASET_DIR}/ja_cc/ja_cc_merge_6_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 16510926846 ${DATASET_DIR}/ja_cc/ja_cc_merge_7_text_document"

VALID_DATA_PATH=""

VALID_DATA_PATH="${VALID_DATA_PATH} 74718249 /data/llm-jp-v2_2-code20k_en40k_ja60k-binarized/validation/code_stack_validation_0_text_document"
VALID_DATA_PATH="${VALID_DATA_PATH} 58182538 /data/llm-jp-v2_2-code20k_en40k_ja60k-binarized/validation/en_pile_validation_0_text_document"
VALID_DATA_PATH="${VALID_DATA_PATH} 953560 /data/llm-jp-v2_2-code20k_en40k_ja60k-binarized/validation/en_wiki_validation_0_text_document"
VALID_DATA_PATH="${VALID_DATA_PATH} 142519640 /data/llm-jp-v2_2-code20k_en40k_ja60k-binarized/validation/ja_cc_validation_0_text_document"
VALID_DATA_PATH="${VALID_DATA_PATH} 1060860 /data/llm-jp-v2_2-code20k_en40k_ja60k-binarized/validation/ja_wiki_validation_0_text_document"


# job name
JOB_NAME="7b-${NODE_TYPE}-${NUM_NODES}node-${NUM_GPUS}gpu-${SEQ_LENGTH}s-DP=${DATA_PARALLEL_SIZE}-TP=${TENSOR_PARALLEL_SIZE}-PP=${PIPELINE_PARALLEL_SIZE}-BS=${GLOBAL_BATCH_SIZE}-LR=${LR}-MINLR=${MIN_LR}-WARMUP=${LR_WARMUP_STEPS}-WD=${WEIGHT_DECAY}-GC=${GRAD_CLIP}"

# --norm-epsilon 1e-5 : conifg.json (RMS norm)

# checkpoint load
if [[ -f "${CHECKPOINT_SAVE_DIR}/latest_checkpointed_iteration.txt" ]]; then
  # resume training
  CHECKPOINT_ARGS="--load ${CHECKPOINT_SAVE_DIR}"
else
  # first training
  CHECKPOINT_ARGS="--load ${CHECKPOINT_DIR} --no-load-rng --no-load-optim"
fi

# run
mpirun -np $NUM_GPUS \
  --npernode $NUM_GPU_PER_NODE \
  -hostfile $HOSTFILE_NAME \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
  -x NCCL_IB_GID_INDEX=3 -x NCCL_IB_TC=106 \
  -bind-to none -map-by slot \
  -x PATH \
  python pretrain_gpt.py \
  --tensor-model-parallel-size ${TENSOR_PARALLEL_SIZE} \
  --pipeline-model-parallel-size ${PIPELINE_PARALLEL_SIZE} \
  --sequence-parallel \
  --use-distributed-optimizer \
  --num-layers ${NUM_LAYERS} \
  --hidden-size ${HIDDEN_SIZE} \
  --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
  --num-attention-heads ${NUM_HEADS} \
  --seq-length ${SEQ_LENGTH} \
  --max-position-embeddings ${SEQ_LENGTH} \
  --micro-batch-size ${MICRO_BATCH_SIZE} \
  --global-batch-size ${GLOBAL_BATCH_SIZE} \
  --train-iters ${TRAIN_STEPS} \
  --tokenizer-type SentencePieceTokenizer \
  --tokenizer-model ${TOKENIZER_MODEL} \
  --save ${CHECKPOINT_SAVE_DIR} \
  --train-data-path ${TRAIN_DATA_PATH} \
  --valid-data-path ${VALID_DATA_PATH} \
  --test-data-path ${VALID_DATA_PATH} \
  --distributed-backend nccl \
  --init-method-std 0.02 \
  --lr ${LR} \
  --min-lr ${MIN_LR} \
  --lr-decay-style cosine \
  --weight-decay ${WEIGHT_DECAY} \
  --clip-grad ${GRAD_CLIP} \
  --lr-warmup-iters ${LR_WARMUP_STEPS} \
  --optimizer adam \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --log-interval 1 \
  --save-interval 500 \
  --eval-interval 100 \
  --eval-iters 10 \
  --bf16 \
  --untie-embeddings-and-output-weights \
  --use-rotary-position-embeddings \
  --normalization RMSNorm \
  --norm-epsilon 1e-5 \
  --no-position-embedding \
  --no-masked-softmax-fusion \
  --no-query-key-layer-scaling \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --disable-bias-linear \
  --no-bias-gelu-fusion \
  --group-query-attention \
  --num-query-groups 32 \
  --swiglu \
  --use-flash-attn \
  --recompute-activations \
  --recompute-granularity "selective" \
  --use-mpi \
  --wandb-name ${JOB_NAME} \
  --wandb-project "megatron-lm-7B-2023-1112" \
  --wandb-entity "llm-jp"