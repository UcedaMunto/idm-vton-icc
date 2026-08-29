#!/usr/bin/env bash
# Validacion estatica V7 (Fase 7 paso 1, analoga a V6). NO lanza accelerate ni
# train_xl.py. Verifica precondiciones y construye el comando efectivo para
# inspeccion, incluyendo la nueva bandera --hybrid_small_models_gpu (Tier 1).
set -uo pipefail

ROOT_DIR="/home/uceda/Documents/IDM-VTON"
CONFIG_FILE="${ROOT_DIR}/configuracion-v7-entrenamiento/config_v7.env"
PAUSE_FILE="${ROOT_DIR}/entrenamiento continuo/PAUSAR_WATCHDOG"
FAILURE_PAUSE_FILE="${ROOT_DIR}/entrenamiento continuo/PAUSAR_POR_ERROR"

FAILED=0
pass() { printf '  [OK] %s\n' "$1"; }
fail() { printf '  [FALLO] %s\n' "$1"; FAILED=1; }
warn() { printf '  [ADVERTENCIA] %s\n' "$1"; }

printf '%s\n' '=== 1. Configuracion ==='
if [[ ! -f "${CONFIG_FILE}" ]]; then
  fail "No existe ${CONFIG_FILE}"
  exit 1
fi
set -a
# shellcheck disable=SC1090
source "${CONFIG_FILE}"
set +a
pass "Config cargada: ${CONFIG_FILE}"
printf '      hash=%s\n' "$(sha256sum "${CONFIG_FILE}" | cut -d' ' -f1)"
printf '      line_id=%s parent_checkpoint_id=%s mode=%s hybrid_tier=%s\n' "${LINE_ID}" "${PARENT_CHECKPOINT_ID}" "${MODE}" "${HYBRID_TIER}"

printf '%s\n' '=== 2. Procesos y pausas (solo lectura) ==='
if pgrep -fa 'train_xl.py|accelerate launch|gradio_demo/app.py' >/dev/null; then
  fail "Hay un proceso de entrenamiento o Gradio activo; no continuar."
else
  pass "Sin procesos de entrenamiento ni Gradio activos."
fi
[[ -f "${PAUSE_FILE}" ]] && warn "PAUSAR_WATCHDOG presente (pausa manual activa)." || warn "PAUSAR_WATCHDOG ausente."
[[ -f "${FAILURE_PAUSE_FILE}" ]] && warn "PAUSAR_POR_ERROR presente (disyuntor activo)." || warn "PAUSAR_POR_ERROR ausente."

printf '%s\n' '=== 3. Checkpoints y linaje ==='
if [[ -f "${CONTINUATION_CHECKPOINT}/unet/config.json" ]]; then
  pass "Base valida: ${CONTINUATION_CHECKPOINT}"
else
  fail "Base invalida o incompleta: ${CONTINUATION_CHECKPOINT}"
fi
if [[ -f "${RESUME_CHECKPOINT}/manifest.json" && -f "${RESUME_CHECKPOINT}/trainable_state.pt" ]]; then
  pass "Checkpoint de reanudacion valido: ${RESUME_CHECKPOINT}"
else
  fail "Checkpoint de reanudacion incompleto: ${RESUME_CHECKPOINT}"
fi
warn "MODE=${MODE}: esta corrida NO se presenta como continuacion exacta (sin RNG/optimizer garantizado)."

printf '%s\n' '=== 4. Recursos (RAM y disco; V7 anade guardia de RAM tras el hallazgo F-21) ==='
FREE_GIB="$(df -BG --output=avail "${ROOT_DIR}" | tail -n 1 | tr -dc '0-9')"
if (( FREE_GIB >= MIN_FREE_GIB )); then
  pass "Disco libre: ${FREE_GIB} GiB (minimo ${MIN_FREE_GIB} GiB)"
else
  fail "Disco libre insuficiente: ${FREE_GIB} GiB (minimo ${MIN_FREE_GIB} GiB)"
fi
FREE_RAM_GIB="$(free -g | awk '/^Mem:/ {print $7}')"
if (( FREE_RAM_GIB >= MIN_FREE_RAM_GIB )); then
  pass "RAM disponible: ${FREE_RAM_GIB} GiB (minimo ${MIN_FREE_RAM_GIB} GiB)"
else
  fail "RAM disponible insuficiente: ${FREE_RAM_GIB} GiB (minimo ${MIN_FREE_RAM_GIB} GiB)"
fi
SWAP_USED_GIB="$(free -g | awk '/^Swap:/ {print $3}')"
if (( SWAP_USED_GIB > 2 )); then
  warn "Ya hay ${SWAP_USED_GIB} GiB de swap en uso antes de empezar; alto riesgo de repetir el thrashing de F-21."
else
  pass "Swap en uso antes de empezar: ${SWAP_USED_GIB} GiB"
fi

printf '%s\n' '=== 5. Coherencia de parametros ==='
if (( TRAIN_WIDTH % 8 == 0 && TRAIN_HEIGHT % 8 == 0 )); then
  pass "Resolucion ${TRAIN_WIDTH}x${TRAIN_HEIGHT} es multiplo de 8"
else
  fail "Resolucion ${TRAIN_WIDTH}x${TRAIN_HEIGHT} no es multiplo de 8"
fi
if (( CHECKPOINTING_STEPS <= MAX_TRAIN_STEPS )); then
  pass "checkpointing_steps (${CHECKPOINTING_STEPS}) <= max_train_steps (${MAX_TRAIN_STEPS})"
else
  fail "checkpointing_steps (${CHECKPOINTING_STEPS}) > max_train_steps (${MAX_TRAIN_STEPS})"
fi

printf '%s\n' '=== 6. Directorio de salida ==='
RUN_ID="smoke_v7_$(date '+%Y%m%d_%H%M%S')"
RUN_OUTPUT_DIR="${TRAINING_ROOT}/${RUN_ID}"
if [[ -e "${RUN_OUTPUT_DIR}" ]]; then
  fail "El directorio de salida ya existe: ${RUN_OUTPUT_DIR}"
else
  pass "Directorio de salida nuevo propuesto: ${RUN_OUTPUT_DIR}"
fi

printf '%s\n' '=== 7. Comando efectivo (NO se ejecuta) ==='
CMD=(accelerate launch train_xl.py
  --pretrained_model_name_or_path="${CONTINUATION_CHECKPOINT}"
  --pretrained_garmentnet_path="${PRETRAINED_GARMENTNET_PATH}"
  --pretrained_ip_adapter_path="${PRETRAINED_IP_ADAPTER_PATH}"
  --image_encoder_path="${IMAGE_ENCODER_PATH}"
  --data_dir="${DATA_DIR}"
  --output_dir="${RUN_OUTPUT_DIR}"
  --mixed_precision="${MIXED_PRECISION}"
  --gradient_checkpointing
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
  --garmentnet_dtype="${GARMENTNET_DTYPE}"
  --resume_from_checkpoint="${RESUME_CHECKPOINT}")
printf '      '; printf '%q ' "${CMD[@]}"; printf '\n'

printf '\n%s\n' '=== Resultado ==='
if (( FAILED == 0 )); then
  echo "VALIDACION_ESTATICA_OK (no se lanzo ningun proceso de entrenamiento)"
  exit 0
else
  echo "VALIDACION_ESTATICA_FALLIDA"
  exit 1
fi
