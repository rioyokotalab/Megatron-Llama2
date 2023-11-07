#!/bin/bash

# python virtualenv
cd /home/taishi/Megatron-LM
source .env/bin/activate

# distributed settings
export MASTER_ADDR=10.130.184.12
export MASTER_PORT=12813

echo "MASTER_ADDR=${MASTER_ADDR}"

NODE_TYPE="a100"
export NUM_GPU_PER_NODE=8

HOSTFILE_NAME=/home/taishi/ABCI-llama-recipes/hostfile/8node


NUM_NODES=8
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))
export LOGLEVEL=INFO
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=WARN
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0

# model config
# llama-2-13b: https://huggingface.co/meta-llama/Llama-2-13b-hf/blob/main/config.json
HIDDEN_SIZE=5120
FFN_HIDDEN_SIZE=13824 # intermediate size (HuggingFace)
NUM_LAYERS=40
NUM_HEADS=40
SEQ_LENGTH=4096

# distributed settings
TENSOR_PARALLEL_SIZE=2   # fixed
PIPELINE_PARALLEL_SIZE=4 # num layers 40: Llama-2 13B
DATA_PARALLEL_SIZE=$((${NUM_GPUS} / (${TENSOR_PARALLEL_SIZE} * ${PIPELINE_PARALLEL_SIZE})))

# training config
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=1024
TRAIN_STEPS=25000 # e.g. llama: 1T tokens / 4M tokens_per_batch = 250000 steps
# 今回は約100B Tokensなので 1/10

LR=1e-4
MIN_LR=3.3e-6
LR_WARMUP_STEPS=1000
WEIGHT_DECAY=0.1
GRAD_CLIP=18
# model config
TOKENIZER_MODEL=/home/taishi/jalm-tokenizer-private/tokenizer/jalm_llama_clueweb_nfkc_16k_aligned_8/merged_tokenizer_sp/jalm_llama.model
CHECKPOINT_DIR=/home/taishi/llama-megatron-convert-checkpoint-hf/Llama-2-13b-extended/clueweb/tp2-pp4/
CHECKPOINT_SAVE_DIR=/data/taishi/llama-2-13b-extended-clueweb-16k-hf-megatron-tp2-pp4/

mkdir -p ${CHECKPOINT_SAVE_DIR}

# data config
DATASET_DIR=/data/taishi/performance/ClueWeb22Ja_sample1200_1107_clueweb_nfkc_16k_aligned_8

DATA_PATH=""

DATA_PATH="${DATA_PATH} 9266136291 ${DATASET_DIR}/ClueWeb22Ja_sample1200_1107_split_0_text_document"
DATA_PATH="${DATA_PATH} 12318833677 ${DATASET_DIR}/ClueWeb22Ja_sample1200_1107_split_1_text_document"
DATA_PATH="${DATA_PATH} 33799731557 ${DATASET_DIR}/ClueWeb22Ja_sample1200_1107_split_2_text_document"
DATA_PATH="${DATA_PATH} 32912297127 ${DATASET_DIR}/ClueWeb22Ja_sample1200_1107_split_3_text_document"
DATA_PATH="${DATA_PATH} 1703001349 ${DATASET_DIR}/ja_wiki_merged_text_document"
DATA_PATH="${DATA_PATH} 5000000000 ${DATASET_DIR}/lumi_en_arxiv_merge_text_document"
DATA_PATH="${DATA_PATH} 5000000000 ${DATASET_DIR}/lumi_en_falcon_merge_text_document"


# job name
JOB_NAME="llama-2-13b-base-clueweb_16k-${NODE_TYPE}-${NUM_NODES}node-${NUM_GPUS}gpu-${SEQ_LENGTH}s-DP=${DATA_PARALLEL_SIZE}-TP=${TENSOR_PARALLEL_SIZE}-PP=${PIPELINE_PARALLEL_SIZE}-BS=${GLOBAL_BATCH_SIZE}-LR=${LR}-MINLR=${MIN_LR}-WARMUP=${LR_WARMUP_STEPS}-WD=${WEIGHT_DECAY}-GC=${GRAD_CLIP}"





mkdir -p ${CHECKPOINT_SAVE_DIR}

# --norm-epsilon 1e-5 : conifg.json (RMS norm)

# checkpoint load
if [[ -f "${CHECKPOINT_SAVE_DIR}/latest_checkpointed_iteration.txt" ]]; then
  # resume training
  CHECKPOINT_ARGS="--load ${CHECKPOINT_SAVE_DIR}"
else
  # first training
  CHECKPOINT_ARGS="--load ${CHECKPOINT_DIR} --no-load-rng --no-load-optim"
fi

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
  --tokenizer-type Llama2Tokenizer \
  --tokenizer-model ${TOKENIZER_MODEL} \
  --use-checkpoint-args \
  ${CHECKPOINT_ARGS} \
  --save ${CHECKPOINT_SAVE_DIR} \
  --data-path ${DATA_PATH} \
  --split 949,50,1 \
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
  --swiglu \
  --use-flash-attn \
  --recompute-activations \
  --recompute-granularity "selective" \
  --use-mpi \
  --wandb-name ${JOB_NAME} \
  --wandb-project "Llama-2-13B" \
  --wandb-entity "prj-jalm"

