#!/usr/bin/env bash
set -euo pipefail

# Watchdog V10 de entrenamiento continuo para IDM-VTON.
# Se ejecuta cada minuto y, si no hay entrenamiento activo, lanza un bloque de
# MAX_TRAIN_STEPS pasos desde el ultimo checkpoint compacto, usando la mejor
# configuracion validada en configuracion-v10-entrenamiento (Tier1 GPU hibrido +
# --garmentnet_dtype=float32 + --cpu_threads=12; ~55-58s/paso en regimen
# estacionario, ver configuracion-v10-entrenamiento/02_PRUEBA_ESTABILIDAD_20_PASOS.md).

PROJECT_ROOT="${PROJECT_ROOT:-/home/uceda/Documents/IDM-VTON}"
CONDA_SH="${CONDA_SH:-/home/uceda/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-idm}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-0}"

DATA_DIR="${DATA_DIR:-/home/uceda/Documents/IDM-VTON/dataset/DATA_DIR_PREP}"
GARMENTNET_PATH="${GARMENTNET_PATH:-stabilityai/stable-diffusion-xl-base-1.0}"
IP_ADAPTER_PATH="${IP_ADAPTER_PATH:-ckpt/ip_adapter/ip-adapter-plus_sdxl_vit-h.bin}"
IMAGE_ENCODER_PATH="${IMAGE_ENCODER_PATH:-ckpt/image_encoder}"

BASE_CHECKPOINT="${BASE_CHECKPOINT:-/home/uceda/Documents/IDM-VTON/result_train_night/checkpoint-250}"
COMPACT_CHECKPOINT_ROOT="${COMPACT_CHECKPOINT_ROOT:-/home/uceda/Documents/IDM-VTON/result_train_v10/produccion_continua}"

AUTO_OUTPUT_ROOT="${AUTO_OUTPUT_ROOT:-/home/uceda/Documents/IDM-VTON/result_train_v10/produccion_continua}"
LOG_ROOT="${LOG_ROOT:-/home/uceda/Documents/IDM-VTON/logs/produccion_continua}"
WATCHDOG_LOG="${WATCHDOG_LOG:-${LOG_ROOT}/watchdog.log}"
LOCK_FILE="${LOCK_FILE:-${LOG_ROOT}/watchdog.lock}"
PAUSE_FILE="${PAUSE_FILE:-/home/uceda/Documents/IDM-VTON/entrenamiento continuo/PAUSAR_WATCHDOG}"
FAILURE_PAUSE_FILE="${FAILURE_PAUSE_FILE:-/home/uceda/Documents/IDM-VTON/entrenamiento continuo/PAUSAR_POR_ERROR}"

MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-500}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-100}"
LOGGING_STEPS="${LOGGING_STEPS:-100}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-1}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-1}"
TEST_NUM_WORKERS="${TEST_NUM_WORKERS:-1}"
WIDTH="${WIDTH:-448}"
HEIGHT="${HEIGHT:-576}"
MIN_FREE_GIB="${MIN_FREE_GIB:-90}"
MIN_AVAILABLE_RAM_GIB="${MIN_AVAILABLE_RAM_GIB:-3}"
CPU_THREADS="${CPU_THREADS:-12}"
# V12: continuidad del optimizador entre bloques (usa el optimizer_state.pt que
# ya se persiste en cada checkpoint). 1=activado, vacio/0=desactivado.
RESUME_OPTIMIZER_STATE="${RESUME_OPTIMIZER_STATE:-1}"
# V12: learning rate del fine-tune. Default 5e-5 (RECETA V12 PRODUCCION).
# 1e-5 (default original) mueve los pesos ~0.7% por bloque de 500 pasos y el
# entrenamiento queda practicamente sin efecto visible (ver configuracion-v12-
# entrenamiento/01_DIAGNOSTICO_ENTRENAMIENTO_DEBIL.md). 5e-5 es el valor
# recomendado para fine-tune de solo IP-Adapter con batch 1 en este hardware.
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
REVIEW_ROOT="${REVIEW_ROOT:-/home/uceda/Documents/IDM-VTON/entrenamiento continuo/imagenes_revision}"
REVIEW_SAMPLE_LIMIT="${REVIEW_SAMPLE_LIMIT:-2}"
REVIEW_INFERENCE_STEPS="${REVIEW_INFERENCE_STEPS:-20}"

mkdir -p "${AUTO_OUTPUT_ROOT}" "${LOG_ROOT}" "${REVIEW_ROOT}"

# Evita ejecuciones concurrentes del watchdog (cron cada minuto).
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  exit 0
fi

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${WATCHDOG_LOG}" >/dev/null
}

if [[ -f "${PAUSE_FILE}" ]]; then
  log "Watchdog pausado por ${PAUSE_FILE}."
  exit 0
fi

if [[ -f "${FAILURE_PAUSE_FILE}" ]]; then
  log "Watchdog pausado por fallo previo. Revisar ${FAILURE_PAUSE_FILE}."
  exit 0
fi

# Si cualquier proceso train_xl.py o Gradio esta activo, no lanzamos otro.
if pgrep -fa 'train_xl.py|gradio_demo/app.py' >/dev/null; then
  log "Entrenamiento o Gradio activo detectado. No se lanza nueva corrida."
  exit 0
fi

if [[ ! -f "${BASE_CHECKPOINT}/unet/config.json" ]]; then
  log "ERROR: checkpoint base invalido: ${BASE_CHECKPOINT}"
  exit 1
fi

FREE_GIB="$(df -BG --output=avail "${PROJECT_ROOT}" | tail -n 1 | tr -dc '0-9')"
if (( FREE_GIB < MIN_FREE_GIB )); then
  log "ERROR: espacio libre insuficiente: ${FREE_GIB} GiB; minimo=${MIN_FREE_GIB} GiB"
  exit 1
fi

AVAILABLE_RAM_GIB="$(free -g | awk '/^Mem:/ {print $7}')"
if (( AVAILABLE_RAM_GIB < MIN_AVAILABLE_RAM_GIB )); then
  log "ERROR: RAM disponible insuficiente: ${AVAILABLE_RAM_GIB} GiB; minimo=${MIN_AVAILABLE_RAM_GIB} GiB"
  exit 1
fi

# Busca el checkpoint compacto mas reciente para encadenar el entrenamiento.
RESUME_CHECKPOINT="$(find "${COMPACT_CHECKPOINT_ROOT}" -type f -name 'trainable_state.pt' -printf '%T@ %h\n' 2>/dev/null | sort -nr | awk 'NR==1 {print $2}')"
if [[ -n "${RESUME_CHECKPOINT}" && ! -f "${RESUME_CHECKPOINT}/manifest.json" ]]; then
  RESUME_CHECKPOINT=""
fi

RUN_ID="run_$(date '+%Y%m%d_%H%M%S')"
RUN_OUTPUT_DIR="${AUTO_OUTPUT_ROOT}/${RUN_ID}"
RUN_LOG="${LOG_ROOT}/train_${RUN_ID}.log"
RUN_META="${LOG_ROOT}/train_${RUN_ID}.meta"
EXPECTED_CHECKPOINT="${RUN_OUTPUT_DIR}/checkpoint-${MAX_TRAIN_STEPS}"

mkdir -p "${RUN_OUTPUT_DIR}"

log "No hay entrenamiento activo. Lanzando ${MAX_TRAIN_STEPS} steps; base=${BASE_CHECKPOINT}; resume=${RESUME_CHECKPOINT:-ninguno}; espacio=${FREE_GIB}GiB; ram_disponible=${AVAILABLE_RAM_GIB}GiB"

{
  echo "run_id=${RUN_ID}"
  echo "started_at=$(date '+%Y-%m-%d %H:%M:%S')"
  echo "base_checkpoint=${BASE_CHECKPOINT}"
  echo "resume_from=${RESUME_CHECKPOINT:-none}"
  echo "output_dir=${RUN_OUTPUT_DIR}"
  echo "log_file=${RUN_LOG}"
  echo "max_train_steps=${MAX_TRAIN_STEPS}"
  echo "checkpointing_steps=${CHECKPOINTING_STEPS}"
  echo "free_gib_at_start=${FREE_GIB}"
  echo "available_ram_gib_at_start=${AVAILABLE_RAM_GIB}"
} > "${RUN_META}"

# Corre en un subshell desacoplado (nohup + disown) que SI espera a que termine
# accelerate launch, para poder registrar el codigo de salida real y pausar el
# watchdog automaticamente (PAUSAR_POR_ERROR) si la corrida falla o no produce
# el checkpoint esperado. El propio cron/watchdog no espera: retorna de inmediato.
nohup bash -c '
  set +e
  source "'"${CONDA_SH}"'"
  conda activate "'"${CONDA_ENV}"'"
  cd "'"${PROJECT_ROOT}"'"

  CUDA_VISIBLE_DEVICES="'"${CUDA_VISIBLE_DEVICES_VALUE}"'" accelerate launch train_xl.py \
    --pretrained_model_name_or_path="'"${BASE_CHECKPOINT}"'" \
    --pretrained_garmentnet_path="'"${GARMENTNET_PATH}"'" \
    --pretrained_ip_adapter_path="'"${IP_ADAPTER_PATH}"'" \
    --image_encoder_path="'"${IMAGE_ENCODER_PATH}"'" \
    --data_dir="'"${DATA_DIR}"'" \
    --output_dir="'"${RUN_OUTPUT_DIR}"'" \
    --mixed_precision=fp16 \
    --gradient_checkpointing \
    --low_vram_training \
    --train_ip_adapter_only \
    --hybrid_small_models_gpu \
    --cpu_threads="'"${CPU_THREADS}"'" \
    --width="'"${WIDTH}"'" \
    --height="'"${HEIGHT}"'" \
    --train_batch_size="'"${TRAIN_BATCH_SIZE}"'" \
    --test_batch_size="'"${TEST_BATCH_SIZE}"'" \
    --train_num_workers="'"${TRAIN_NUM_WORKERS}"'" \
    --test_num_workers="'"${TEST_NUM_WORKERS}"'" \
    --max_train_steps="'"${MAX_TRAIN_STEPS}"'" \
    --checkpointing_steps="'"${CHECKPOINTING_STEPS}"'" \
    --logging_steps="'"${LOGGING_STEPS}"'" \
    --learning_rate="'"${LEARNING_RATE}"'" \
    --garmentnet_dtype=float32 \
    '"${RESUME_CHECKPOINT:+--resume_from_checkpoint=\"${RESUME_CHECKPOINT}\"}"' \
    '"${RESUME_OPTIMIZER_STATE:+--resume_optimizer_state}"' \
    > "'"${RUN_LOG}"'" 2>&1
  run_status=$?

  {
    echo "run_id='"${RUN_ID}"'"
    echo "finished_at=$(date "+%Y-%m-%d %H:%M:%S")"
    echo "exit_code=${run_status}"
    echo "expected_checkpoint='"${EXPECTED_CHECKPOINT}"'"
    echo "log_file='"${RUN_LOG}"'"
  } >> "'"${RUN_META}"'"

  if [[ ${run_status} -ne 0 || ! -f "'"${EXPECTED_CHECKPOINT}"'/manifest.json" ]]; then
    {
      echo "run_id='"${RUN_ID}"'"
      echo "exit_code=${run_status}"
      echo "expected_checkpoint='"${EXPECTED_CHECKPOINT}"'"
      echo "log_file='"${RUN_LOG}"'"
      echo "failed_at=$(date "+%Y-%m-%d %H:%M:%S")"
    } > "'"${FAILURE_PAUSE_FILE}"'"
    printf "[%s] Corrida '"${RUN_ID}"' fallo o no genero checkpoint valido (exit_code=%s). Watchdog pausado via '"${FAILURE_PAUSE_FILE}"'.\n" "$(date "+%Y-%m-%d %H:%M:%S")" "${run_status}" >> "'"${WATCHDOG_LOG}"'"
  else
    printf "[%s] Corrida '"${RUN_ID}"' finalizada OK (exit_code=0, checkpoint valido en '"${EXPECTED_CHECKPOINT}"').\n" "$(date "+%Y-%m-%d %H:%M:%S")" >> "'"${WATCHDOG_LOG}"'"

    # Genera imagenes de muestra (checkpoint base vs. este checkpoint) para revision visual.
    # No bloquea el encadenamiento: si falla, solo se pierde la muestra, no el checkpoint.
    review_dir="'"${REVIEW_ROOT}"'/'"${RUN_ID}"'"
    mkdir -p "${review_dir}"
    PYTHONPATH="'"${PROJECT_ROOT}"'" python3 "'"${PROJECT_ROOT}"'/configuracion-v9-entrenamiento/comparar_calidad_v9.py" \
      --pretrained_model_name_or_path="'"${BASE_CHECKPOINT}"'" \
      --compact_checkpoint="'"${EXPECTED_CHECKPOINT}"'" \
      --data_dir="'"${DATA_DIR}"'" \
      --output_dir="${review_dir}" \
      --width="'"${WIDTH}"'" --height="'"${HEIGHT}"'" \
      --num_inference_steps="'"${REVIEW_INFERENCE_STEPS}"'" --seed=42 --limit="'"${REVIEW_SAMPLE_LIMIT}"'" \
      > "${review_dir}/generar_muestra.log" 2>&1
    if [[ $? -eq 0 ]]; then
      printf "[%s] Muestra visual de '"${RUN_ID}"' guardada en %s\n" "$(date "+%Y-%m-%d %H:%M:%S")" "${review_dir}" >> "'"${WATCHDOG_LOG}"'"
    else
      printf "[%s] WARNING: no se pudo generar la muestra visual de '"${RUN_ID}"' (ver %s/generar_muestra.log)\n" "$(date "+%Y-%m-%d %H:%M:%S")" "${review_dir}" >> "'"${WATCHDOG_LOG}"'"
    fi

    # Libera espacio: solo el ultimo checkpoint de esta corrida hace falta para encadenar la siguiente.
    for old_ckpt in "'"${RUN_OUTPUT_DIR}"'"/checkpoint-*; do
      if [[ "${old_ckpt}" != "'"${EXPECTED_CHECKPOINT}"'" && -d "${old_ckpt}" ]]; then
        rm -rf "${old_ckpt}"
      fi
    done
    printf "[%s] Checkpoints intermedios de '"${RUN_ID}"' eliminados; se conserva solo '"${EXPECTED_CHECKPOINT}"'.\n" "$(date "+%Y-%m-%d %H:%M:%S")" >> "'"${WATCHDOG_LOG}"'"
  fi
' > /dev/null 2>&1 < /dev/null &
disown

log "Corrida lanzada en segundo plano. run_id=${RUN_ID} (el watchdog no espera; el resultado se registra en ${RUN_META} y en ${WATCHDOG_LOG} al terminar). Al finalizar OK, se guardara una imagen de muestra en ${REVIEW_ROOT}/${RUN_ID}."
