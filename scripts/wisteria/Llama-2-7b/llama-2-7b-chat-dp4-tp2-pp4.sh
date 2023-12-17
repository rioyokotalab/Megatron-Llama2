#!/bin/bash
#PJM -L rscgrp=tutorial2-a
#PJM -L node=4
#PJM -N gpt
#PJM --mpi proc=32
#PJM -L elapse=06:00:00
#PJM -g gt01

set -e

# module load
module load cuda/11.8
module load nvidia/22.11
module load cudnn/8.8.0
module load nccl/2.16.5
module load nvmpi/22.11

# bashrc
source /work/gt01/GROUP_215-01/.bashrc

# python virtualenv
cd /work/gt01/GROUP_215-01/work/Megatron-LM
source .env/bin/activate

export TRANSFORMERS_CACHE=/work/gt01/GROUP_215-01/.cache/huggingface/transformers

# distributed settings
export MASTER_ADDR=$(head -n 1 $PJM_O_NODEINF)
export MASTER_PORT=16500

echo "MASTER_ADDR=${MASTER_ADDR}"

NUM_GPU_PER_NODE=`nvidia-smi -L | wc -l`

NUM_NODES=4
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

# model config
# llama-2-7b: https://huggingface.co/meta-llama/Llama-2-7b-hf/blob/main/config.json
HIDDEN_SIZE=4096
FFN_HIDDEN_SIZE=11008 # intermediate size (HuggingFace)
NUM_LAYERS=32
NUM_HEADS=32
SEQ_LENGTH=4096

# distributed settings
TENSOR_PARALLEL_SIZE=2   # fixed
PIPELINE_PARALLEL_SIZE=4 # num layers 32: Llama-2 7B
DATA_PARALLEL_SIZE=$((${NUM_GPUS} / (${TENSOR_PARALLEL_SIZE} * ${PIPELINE_PARALLEL_SIZE})))

echo -e "\nDP=${DATA_PARALLEL_SIZE}, TP=${TENSOR_PARALLEL_SIZE}, PP=${PIPELINE_PARALLEL_SIZE}\n"

# training config
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=1024
TRAIN_STEPS=25000 # e.g. llama: 1T tokens / 4M tokens_per_batch = 250000 steps

LR=3e-5
MIN_LR=1e-5
LR_WARMUP_STEPS=2000
WEIGHT_DECAY=0.1
GRAD_CLIP=1

# model config
TOKENIZER_MODEL=/work/gt01/GROUP_215-01/work/hf-checkpoints/Llama-2-7b-chat-hf/tokenizer.model
CHECKPOINT_DIR=/work/gt01/GROUP_215-01/work/megatron-checkpoints/Llama-2-7b-chat/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}
CHECKPOINT_SAVE_DIR=/work/gt01/GROUP_215-01/work/checkpoints/llama-2-7b-chat-hf/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}

mkdir -p ${CHECKPOINT_SAVE_DIR}

# data config

# total token: 98082857551 token (= 98.1B token) llama-2 tokenizer
DATASET_DIR=/work/gt01/GROUP_215-01/work/datasets/binarized

DATA_PATH=""

# ja mc4
DATA_PATH="${DATA_PATH} 45914576536 ${DATASET_DIR}/mc4_text_document"
# ja aozora
DATA_PATH="${DATA_PATH} 160213881 ${DATASET_DIR}/ja_aozora_text_document"
# ja cc100
DATA_PATH="${DATA_PATH} 6308767651 ${DATASET_DIR}/ja_cc100_text_document"
# ja wiki
DATA_PATH="${DATA_PATH} 2923100972 ${DATASET_DIR}/ja_wiki_text_document"
# ja oscar
DATA_PATH="${DATA_PATH} 6093447282 ${DATASET_DIR}/ja_oscar_text_document"

# en arxiv
DATA_PATH="${DATA_PATH} 14378861379 ${DATASET_DIR}/en_arxiv_text_document"
# en bookcorpus
DATA_PATH="${DATA_PATH} 22303889850 ${DATASET_DIR}/en_books_text_document"


# job name
JOB_NAME="llama-2-7b-chat-wosteroa-a100-${NUM_NODES}node-${NUM_GPUS}gpu-${SEQ_LENGTH}s-DP=${DATA_PARALLEL_SIZE}-TP=${TENSOR_PARALLEL_SIZE}-PP=${PIPELINE_PARALLEL_SIZE}-BS=${GLOBAL_BATCH_SIZE}-LR=${LR}-MINLR=${MIN_LR}-WARMUP=${LR_WARMUP_STEPS}-WD=${WEIGHT_DECAY}-GC=${GRAD_CLIP}"

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
  -machinefile ${PJM_O_NODEINF} \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
  -bind-to none \
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
  --save-interval 100 \
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
  --wandb-project "megatron-lm-llama" \
  --wandb-entity "prj-jalm"
