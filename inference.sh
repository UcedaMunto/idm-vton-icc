#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$ROOT_DIR/.env"

if [[ -f "$ENV_FILE" ]]; then
    set -a
    source "$ENV_FILE"
    set +a
fi

MODEL_PATH="${IDMVTON_MODEL_PATH:-yisol/IDM-VTON}"
echo "[inference] using model: $MODEL_PATH"

# VITON-HD
## paired setting
accelerate launch inference.py --pretrained_model_name_or_path "$MODEL_PATH" \
    --width 768 --height 1024 --num_inference_steps 30 \
    --output_dir "result" --data_dir "/home/omnious/workspace/yisol/Dataset/zalando" \
    --seed 42 --test_batch_size 2 --guidance_scale 2.0

## unpaired setting
accelerate launch inference.py --pretrained_model_name_or_path "$MODEL_PATH" \
    --width 768 --height 1024 --num_inference_steps 30 \
    --output_dir "result" --unpaired --data_dir "/home/omnious/workspace/yisol/Dataset/zalando" \
    --seed 42 --test_batch_size 2 --guidance_scale 2.0

# DressCode
## upper_body
accelerate launch inference_dc.py --pretrained_model_name_or_path "$MODEL_PATH" \
    --width 768 --height 1024 --num_inference_steps 30 \
    --output_dir "result" --unpaired --data_dir "/home/omnious/workspace/yisol/DressCode" \
    --seed 42 --test_batch_size 2 --guidance_scale 2.0 --category "upper_body"

## lower_body
accelerate launch inference_dc.py --pretrained_model_name_or_path "$MODEL_PATH" \
    --width 768 --height 1024 --num_inference_steps 30 \
    --output_dir "result" --unpaired --data_dir "/home/omnious/workspace/yisol/DressCode" \
    --seed 42 --test_batch_size 2 --guidance_scale 2.0 --category "lower_body"

## dresses
accelerate launch inference_dc.py --pretrained_model_name_or_path "$MODEL_PATH" \
    --width 768 --height 1024 --num_inference_steps 30 \
    --output_dir "result" --unpaired --data_dir "/home/omnious/workspace/yisol/DressCode" \
    --seed 42 --test_batch_size 2 --guidance_scale 2.0 --category "dresses"