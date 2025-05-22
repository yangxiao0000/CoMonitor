#!/bin/bash
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
# export MODELSCOPE_CACHE="/root/.cache/000030/Reference/ColossalAI-main/applications/Colossal-LLaMA_msft_multi/Modelscope_cache"
# export MODELSCOPE_CACHE="Reference/ColossalAI-main/applications/Colossal-LLaMA_msft_multi/Modelscope_cache"

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29688
export HCCL_WHITELIST_DISABLE=1

NPUS=($(seq 0 7))
export RANK_SIZE=${#NPUS[@]}
rank=0
i=0
export DEVICE_ID=${i}
export RANK_ID=${rank}
echo run process ${rank}

# PROJECT_NAME="llama2_sft"
# PARENT_SAVE_DIR="/root/.cache/000030/Reference/ColossalAI-main/applications/Colossal-LLaMA_msft_multi/save"
# PARENT_TENSORBOARD_DIR="/root/.cache/000030/Reference/ColossalAI-main/applications/Colossal-LLaMA_msft_multi/tensorboard"
# PARENT_CONFIG_FILE="/root/.cache/000030/Reference/ColossalAI-main/applications/Colossal-LLaMA_msft_multi/config"
# PRETRAINED_MODEL_PATH="/root/.cache/modelscope/hub/colossalai/Colossal-LLaMA-2-7b-base"

PROJECT_NAME="llama2_sft"
PARENT_SAVE_DIR="save/"
PARENT_TENSORBOARD_DIR="tensorboard/"
PARENT_CONFIG_FILE="config"
PRETRAINED_MODEL_PATH="/root/.cache/modelscope/hub/colossalai/Colossal-LLaMA-2-7b-base"

# declare -a dataset=(
#     "/root/.cache/000030/Reference/ColossalAI-main/applications/Colossal-LLaMA_msft_multi/dataset"
# )


declare -a dataset=(
    "/root/.cache/000030/Reference/ColossalAI-main/applications/Colossal-LLaMA_msft_multi/dataset/arrow/part-00000"
    "/root/.cache/000030/Reference/ColossalAI-main/applications/Colossal-LLaMA_msft_multi/dataset/arrow/part-00001"
)


TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
FULL_PROJECT_NAME="${PROJECT_NAME}-${TIMESTAMP}"
SAVE_DIR="${PARENT_SAVE_DIR}${FULL_PROJECT_NAME}"
TENSORBOARD_DIR="${PARENT_TENSORBOARD_DIR}${FULL_PROJECT_NAME}"
CONFIG_FILE="${PARENT_CONFIG_FILE}${FULL_PROJECT_NAME}.json"

colossalai run --nproc_per_node 2 --master_port 29688  train_npu.py \
    --pretrained $PRETRAINED_MODEL_PATH \
    --dataset ${dataset[@]} \
    --plugin "ddp" \
    --save_interval 400 \
    --save_dir $SAVE_DIR \
    --tensorboard_dir $TENSORBOARD_DIR \
    --config_file $CONFIG_FILE \
    --num_epochs 1 \
    --accumulation_steps 8 \
    --lr 5e-5 \
    --mixed_precision "fp16" \
    --grad_clip 1.0 \
    --weight_decay 0.01 \
    --warmup_steps 100 \
    --use_flash_attn \
    --pad_token "eos"\
    --lora_rank 1\
    --batch_size 1
