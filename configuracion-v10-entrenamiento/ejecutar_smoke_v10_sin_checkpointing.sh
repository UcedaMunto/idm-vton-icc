#!/usr/bin/env bash
# Ejecucion supervisada, manual, del smoke test V10 (perfilado Tier 1, 1 update).
# NO se instala en cron. Incluye monitoreo de RAM/swap en paralelo tras el
# hallazgo F-21 (thrashing observado en V6). Usa --profile_step_timing: solo
# imprime tiempos por etapa, no cambia ningun computo ni resultado.
set -euo pipefail

ROOT_DIR="/home/uceda/Documents/IDM-VTON"
CONFIG_FILE="${ROOT_DIR}/configuracion-v10-entrenamiento/config_v10.env"
set -a
# shellcheck disable=SC1090
source "${CONFIG_FILE}"
set +a

if pgrep -fa 'train_xl.py|accelerate launch|gradio_demo/app.py' >/dev/null; then
  echo "ERROR: existe otro proceso de entrenamiento o Gradio"
  exit 1
fi
if [[ ! -f "${CONTINUATION_CHECKPOINT}/unet/config.json" ]]; then
  echo "ERROR: checkpoint base invalido: ${CONTINUATION_CHECKPOINT}"
  exit 1
fi
if [[ ! -f "${RESUME_CHECKPOINT}/manifest.json" || ! -f "${RESUME_CHECKPOINT}/trainable_state.pt" ]]; then
  echo "ERROR: checkpoint de reanudacion invalido: ${RESUME_CHECKPOINT}"
  exit 1
fi
free_gib="$(df -BG --output=avail "${ROOT_DIR}" | tail -n 1 | tr -dc '0-9')"
if (( free_gib < MIN_FREE_GIB )); then
  echo "ERROR: espacio insuficiente: ${free_gib} GiB (minimo ${MIN_FREE_GIB})"
  exit 1
fi

source /home/uceda/miniconda3/etc/profile.d/conda.sh
conda activate idm
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64"

run_id="smoke_v10nogc_$(date '+%Y%m%d_%H%M%S')"
output_dir="${TRAINING_ROOT}/${run_id}"
log_file="${LOG_ROOT}/${run_id}.log"
monitor_file="${LOG_ROOT}/${run_id}.monitor.log"
mkdir -p "${output_dir}" "${LOG_ROOT}"

args=(
  --pretrained_model_name_or_path="${CONTINUATION_CHECKPOINT}"
  --pretrained_garmentnet_path="${PRETRAINED_GARMENTNET_PATH}"
  --pretrained_ip_adapter_path="${PRETRAINED_IP_ADAPTER_PATH}"
  --image_encoder_path="${IMAGE_ENCODER_PATH}"
  --data_dir="${DATA_DIR}"
  --output_dir="${output_dir}"
  --mixed_precision="${MIXED_PRECISION}"
  --low_vram_training
  --train_ip_adapter_only
  --hybrid_small_models_gpu
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
  --resume_from_checkpoint="${RESUME_CHECKPOINT}"
)
if [[ "${PROFILE_STEP_TIMING:-0}" == "1" ]]; then
  args+=(--profile_step_timing)
fi

echo "run_id=${run_id}"
echo "output_dir=${output_dir}"
echo "log_file=${log_file}"
echo "monitor_file=${monitor_file}"

# Monitor de RAM/swap en segundo plano, cada 5s, para detectar thrashing temprano.
(
  while true; do
    printf '[%s] ' "$(date '+%H:%M:%S')" >> "${monitor_file}"
    free -m | awk '/^Mem:/ {printf "mem_used=%sMiB mem_free=%sMiB "; } /^Swap:/ {printf "swap_used=%sMiB\n", $3}' >> "${monitor_file}" 2>/dev/null
    sleep 5
  done
) &
MONITOR_PID=$!
trap 'kill "${MONITOR_PID}" 2>/dev/null || true' EXIT

set +e
CUDA_VISIBLE_DEVICES=0 accelerate launch "${ROOT_DIR}/train_xl.py" "${args[@]}" > "${log_file}" 2>&1
status=$?
set -e

kill "${MONITOR_PID}" 2>/dev/null || true

expected_checkpoint="${output_dir}/checkpoint-${MAX_TRAIN_STEPS}"
echo "exit_code=${status}"
echo "expected_checkpoint=${expected_checkpoint}"
tail -n 40 "${log_file}"
echo "--- ultimas lecturas de memoria ---"
tail -n 10 "${monitor_file}" 2>/dev/null || true

if [[ ${status} -eq 0 && -f "${expected_checkpoint}/manifest.json" && -f "${expected_checkpoint}/trainable_state.pt" ]]; then
  echo "SMOKE_V7_OK"
  exit 0
else
  echo "SMOKE_V7_FALLIDO"
  exit 1
fi
