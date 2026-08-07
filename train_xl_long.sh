#!/usr/bin/env bash
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate idm

set -a
if [[ -f .env ]]; then
  source .env
fi
set +a

export LD_LIBRARY_PATH="${CONDA_PREFIX:-$HOME/miniconda3}/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:64}"

mkdir -p "${OUTPUT_DIR:-/home/uceda/Documents/IDM-VTON/result_train_long}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" accelerate launch train_xl.py \
  --pretrained_model_name_or_path="${PRETRAINED_MODEL_NAME_OR_PATH:-diffusers/stable-diffusion-xl-1.0-inpainting-0.1}" \
  --pretrained_garmentnet_path="${PRETRAINED_GARMENTNET_PATH:-stabilityai/stable-diffusion-xl-base-1.0}" \
  --pretrained_ip_adapter_path="${PRETRAINED_IP_ADAPTER_PATH:-ckpt/ip_adapter/ip-adapter-plus_sdxl_vit-h.bin}" \
  --image_encoder_path="${IMAGE_ENCODER_PATH:-ckpt/image_encoder}" \
  --low_vram_training \
  --train_ip_adapter_only \
  --gradient_checkpointing \
  --enable_xformers_memory_efficient_attention \
  --use_8bit_adam \
  --mixed_precision="${MIXED_PRECISION:-fp16}" \
  --train_batch_size="${TRAIN_BATCH_SIZE:-1}" \
  --test_batch_size="${TEST_BATCH_SIZE:-1}" \
  --train_num_workers="${TRAIN_NUM_WORKERS:-2}" \
  --test_num_workers="${TEST_NUM_WORKERS:-1}" \
  --num_train_epochs="${NUM_TRAIN_EPOCHS:-3}" \
  --max_train_steps="${MAX_TRAIN_STEPS:-100}" \
  --logging_steps="${LOGGING_STEPS:-20}" \
  --checkpointing_epoch="${CHECKPOINT_EPOCH:-10}" \
  --width="${TRAIN_WIDTH:-448}" \
  --height="${TRAIN_HEIGHT:-576}" \
  --output_dir="${OUTPUT_DIR:-/home/uceda/Documents/IDM-VTON/result_train_long}" \
  --data_dir="${DATA_DIR:-/home/uceda/Documents/IDM-VTON/dataset/DATA_DIR_PREP}"
