#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/uceda/Documents/IDM-VTON"
CONFIG_FILE="${ROOT_DIR}/configuracion-v3-entrenamiento/config_v3.env"

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "ERROR: falta ${CONFIG_FILE}"
  exit 1
fi

set -a
source "${CONFIG_FILE}"
set +a

if pgrep -fa 'train_xl.py|gradio_demo/app.py' >/dev/null; then
  echo "ERROR: hay entrenamiento o Gradio activo; libera la GPU antes de iniciar."
  exit 1
fi

if [[ ! -f "${CONTINUATION_CHECKPOINT}/unet/config.json" ]]; then
  echo "ERROR: checkpoint base no valido: ${CONTINUATION_CHECKPOINT}"
  exit 1
fi

free_gib="$(df -BG --output=avail "${ROOT_DIR}" | tail -n 1 | tr -dc '0-9')"
if (( free_gib < MIN_FREE_GIB )); then
  echo "ERROR: espacio libre insuficiente: ${free_gib} GiB; se requieren ${MIN_FREE_GIB} GiB."
  exit 1
fi

source /home/uceda/miniconda3/etc/profile.d/conda.sh
conda activate idm

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:64}"

run_id="smoke10_$(date '+%Y%m%d_%H%M%S')"
output_dir="${TRAINING_ROOT}/${run_id}"
log_file="${LOG_ROOT}/${run_id}.log"

mkdir -p "${output_dir}" "${LOG_ROOT}"

CUDA_VISIBLE_DEVICES=0 accelerate launch "${ROOT_DIR}/train_xl.py" \
  --pretrained_model_name_or_path="${CONTINUATION_CHECKPOINT}" \
  --pretrained_garmentnet_path="${PRETRAINED_GARMENTNET_PATH}" \
  --pretrained_ip_adapter_path="${PRETRAINED_IP_ADAPTER_PATH}" \
  --image_encoder_path="${IMAGE_ENCODER_PATH}" \
  --data_dir="${DATA_DIR}" \
  --output_dir="${output_dir}" \
  --mixed_precision="${MIXED_PRECISION}" \
  --gradient_checkpointing \
  --enable_xformers_memory_efficient_attention \
  --low_vram_training \
  --train_ip_adapter_only \
  --use_8bit_adam \
  --width="${TRAIN_WIDTH}" \
  --height="${TRAIN_HEIGHT}" \
  --train_batch_size="${TRAIN_BATCH_SIZE}" \
  --test_batch_size="${TEST_BATCH_SIZE}" \
  --train_num_workers="${TRAIN_NUM_WORKERS}" \
  --test_num_workers="${TEST_NUM_WORKERS}" \
  --learning_rate="${LEARNING_RATE}" \
  --seed="${SEED}" \
  --max_train_steps="${MAX_TRAIN_STEPS}" \
  --checkpointing_steps="${CHECKPOINTING_STEPS}" \
  --logging_steps=1 \
  > "${log_file}" 2>&1

echo "Resultado: ${output_dir}"
echo "Log: ${log_file}"
