#!/usr/bin/env bash
# Validacion estatica V6 (Fase 7, paso 1). NO lanza accelerate ni train_xl.py.
# Solo verifica precondiciones y construye el comando efectivo para inspeccion.
set -uo pipefail

ROOT_DIR="/home/uceda/Documents/IDM-VTON"
CONFIG_FILE="${ROOT_DIR}/configuracion-v6-entrenamiento/config_v6.env"
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
printf '      line_id=%s parent_checkpoint_id=%s mode=%s\n' "${LINE_ID}" "${PARENT_CHECKPOINT_ID}" "${MODE}"

printf '%s\n' '=== 2. Procesos y pausas (solo lectura, no se modifican) ==='
if pgrep -fa 'train_xl.py|accelerate launch|gradio_demo/app.py' >/dev/null; then
  fail "Hay un proceso de entrenamiento o Gradio activo; no continuar."
else
  pass "Sin procesos de entrenamiento ni Gradio activos."
fi
[[ -f "${PAUSE_FILE}" ]] && warn "PAUSAR_WATCHDOG presente (pausa manual activa)." || warn "PAUSAR_WATCHDOG ausente."
[[ -f "${FAILURE_PAUSE_FILE}" ]] && warn "PAUSAR_POR_ERROR presente (disyuntor activo, revisar antes de reanudar cron)." || warn "PAUSAR_POR_ERROR ausente."

printf '%s\n' '=== 3. Checkpoints y linaje ==='
if [[ -f "${CONTINUATION_CHECKPOINT}/unet/config.json" ]]; then
  pass "Base valida: ${CONTINUATION_CHECKPOINT}"
else
  fail "Base invalida o incompleta: ${CONTINUATION_CHECKPOINT}"
fi

if [[ -f "${RESUME_CHECKPOINT}/manifest.json" && -f "${RESUME_CHECKPOINT}/trainable_state.pt" ]]; then
  pass "Checkpoint de reanudacion valido: ${RESUME_CHECKPOINT}"
  printf '      hash_trainable_state=%s\n' "$(sha256sum "${RESUME_CHECKPOINT}/trainable_state.pt" | cut -d' ' -f1)"
  if command -v python3 >/dev/null; then
    python3 - "${RESUME_CHECKPOINT}/manifest.json" <<'PYEOF'
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    data = json.load(f)
print(f"      manifest.base_checkpoint={data.get('base_checkpoint')}")
print(f"      manifest.completed_optimizer_updates={data.get('completed_optimizer_updates')}")
print(f"      manifest.checkpoint_type={data.get('checkpoint_type')}")
args = data.get("arguments", {})
print(f"      manifest.width={args.get('width')} height={args.get('height')}")
print(f"      manifest.train_ip_adapter_only={args.get('train_ip_adapter_only')}")
print(f"      manifest.garmentnet_dtype={args.get('garmentnet_dtype')}")
PYEOF
  fi
else
  fail "Checkpoint de reanudacion incompleto: ${RESUME_CHECKPOINT}"
fi

if [[ "${MODE}" == "continue_exact" ]]; then
  if [[ -f "${RESUME_CHECKPOINT}/optimizer_state.pt" ]]; then
    pass "optimizer_state.pt presente (requerido por continue_exact)."
  else
    fail "MODE=continue_exact pero falta optimizer_state.pt; use warm_start_weights o aporte el estado."
  fi
else
  warn "MODE=${MODE}: esta corrida NO se presenta como continuacion exacta (sin RNG/optimizer garantizado)."
fi

printf '%s\n' '=== 4. Recursos ==='
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
RUN_ID="smoke_v6_$(date '+%Y%m%d_%H%M%S')"
RUN_OUTPUT_DIR="${TRAINING_ROOT}/${RUN_ID}"
if [[ -e "${RUN_OUTPUT_DIR}" ]]; then
  fail "El directorio de salida ya existe: ${RUN_OUTPUT_DIR}"
else
  pass "Directorio de salida nuevo propuesto: ${RUN_OUTPUT_DIR}"
fi

printf '%s\n' '=== 7. Parametros solicitados vs efectivos (low_vram_training) ==='
printf '      %-28s %-12s %-12s\n' "parametro" "solicitado" "efectivo"
printf '      %-28s %-12s %-12s\n' "mixed_precision" "${MIXED_PRECISION}" "no (forzado por low_vram_training)"
printf '      %-28s %-12s %-12s\n' "optimizer" "adamw_8bit_flag_no_incluido" "torch.optim.AdamW"
printf '      %-28s %-12s %-12s\n' "xformers" "no_solicitado_en_v6" "omitido en low_vram_training"
printf '      %-28s %-12s %-12s\n' "trainable_scope" "ip_adapter_only" "attn_processors+encoder_hid_proj+conv_in"
warn "Estos valores efectivos difieren del original (ver 02_MATRIZ_PARIDAD_Y_PARAMETROS.md); no se presentan como equivalentes."

printf '%s\n' '=== 8. Comando efectivo (NO se ejecuta) ==='
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
if [[ "${RESUME_OPTIMIZER_STATE}" == "1" ]]; then
  CMD+=(--resume_optimizer_state)
fi
printf '      '; printf '%q ' "${CMD[@]}"; printf '\n'

printf '\n%s\n' '=== Resultado ==='
if (( FAILED == 0 )); then
  echo "VALIDACION_ESTATICA_OK (no se lanzo ningun proceso de entrenamiento)"
  exit 0
else
  echo "VALIDACION_ESTATICA_FALLIDA"
  exit 1
fi
