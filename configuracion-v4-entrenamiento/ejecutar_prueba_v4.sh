#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/uceda/Documents/IDM-VTON"
CONFIG_FILE="${ROOT_DIR}/configuracion-v4-entrenamiento/config_v4.env"
set -a
source "${CONFIG_FILE}"
set +a

if pgrep -fa 'train_xl.py|gradio_demo/app.py' >/dev/null; then
  echo "ERROR: existe otro proceso de entrenamiento o Gradio"
  exit 1
fi
if [[ ! -f "${CONTINUATION_CHECKPOINT}/unet/config.json" ]]; then
  echo "ERROR: checkpoint base invalido"
  exit 1
fi
free_gib="$(df -BG --output=avail "${ROOT_DIR}" | tail -n 1 | tr -dc '0-9')"
if (( free_gib < MIN_FREE_GIB )); then
  echo "ERROR: espacio insuficiente: ${free_gib} GiB"
  exit 1
fi

source /home/uceda/miniconda3/etc/profile.d/conda.sh
conda activate idm
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64"

run_id="v4_test_$(date '+%Y%m%d_%H%M%S')"
output_dir="${TRAINING_ROOT}/${run_id}"
log_file="${LOG_ROOT}/${run_id}.log"
resume_root="/home/uceda/Documents/IDM-VTON/result_train_v3"
resume_checkpoint="$(find "${resume_root}" -type f -name 'trainable_state.pt' -printf '%T@ %h\n' | sort -nr | awk 'NR==1 {print $2}')"
mkdir -p "${output_dir}" "${LOG_ROOT}"

args=(
  --pretrained_model_name_or_path="${CONTINUATION_CHECKPOINT}"
  --pretrained_garmentnet_path="${PRETRAINED_GARMENTNET_PATH}"
  --pretrained_ip_adapter_path="${PRETRAINED_IP_ADAPTER_PATH}"
  --image_encoder_path="${IMAGE_ENCODER_PATH}"
  --data_dir="${DATA_DIR}"
  --output_dir="${output_dir}"
  --mixed_precision="${MIXED_PRECISION}"
  --gradient_checkpointing
  --low_vram_training
  --train_ip_adapter_only
  --width="${TRAIN_WIDTH}"
  --height="${TRAIN_HEIGHT}"
  --train_batch_size="${TRAIN_BATCH_SIZE}"
  --test_batch_size="${TEST_BATCH_SIZE}"
  --train_num_workers="${TRAIN_NUM_WORKERS}"
  --test_num_workers="${TEST_NUM_WORKERS}"
  --learning_rate="${LEARNING_RATE}"
  --seed="${SEED}"
  --max_train_steps="${MAX_TRAIN_STEPS}"
  --checkpointing_steps="${CHECKPOINTING_STEPS}"
  --logging_steps=1
  --garmentnet_dtype="${GARMENTNET_DTYPE}"
)
if [[ -n "${resume_checkpoint}" ]]; then
  args+=(--resume_from_checkpoint="${resume_checkpoint}")
fi
if [[ "${RESUME_OPTIMIZER_STATE}" == "1" ]]; then
  args+=(--resume_optimizer_state)
fi

CUDA_VISIBLE_DEVICES=0 accelerate launch "${ROOT_DIR}/train_xl.py" "${args[@]}" > "${log_file}" 2>&1
status=$?
echo "exit_code=${status}"
echo "log=${log_file}"
echo "output=${output_dir}"
tail -n 40 "${log_file}"
exit "${status}"
