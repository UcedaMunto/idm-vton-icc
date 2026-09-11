#!/usr/bin/env bash
# V14 (2026-09-10) - Lanzador de experimentos A/B de la Pista A (doc 04).
#
# Para que sirve: cambiar UNA variable de la receta (que se entrena, augmentacion,
# resolucion) y medirla contra el modelo oficial y contra el checkpoint anterior, SIN
# tocar la cadena de produccion: escribe en result_train_v14/experimentos/<tag>/.
#
# Regla de oro (doc 04, 4.1): una variable por experimento y gate A/B antes de acumular
# pasos. El objetivo es DE-RIESGAR la receta, no producir volumen.
#
# Uso:
#   bash experimento_v14.sh --tag smoke --steps 1 --dry-run    # ver el comando exacto
#   bash experimento_v14.sh --tag smoke --steps 1              # smoke: 1 paso
#   bash experimento_v14.sh --tag c1_base --steps 100          # gate 1 de la receta actual
#   bash experimento_v14.sh --tag aug_hue0 --steps 100         # hue 0.0 (config por defecto)
#   TRAIN_GARMENTNET=1 bash experimento_v14.sh --tag c2 --steps 100   # receta C2
#
# Requisitos (se comprueban antes de lanzar):
#   1. PAUSAR_WATCHDOG presente: si no, el cron lanza produccion en paralelo.
#   2. Sin app web: 12 GB de VRAM no dan para entrenar y servir a la vez.
#   3. Disco libre >= MIN_FREE_GIB (cada checkpoint son ~4,9 GiB).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${SCRIPT_DIR}/config_v14_experimento.env}"
[[ -f "${CONFIG_FILE}" ]] || { echo "ERROR: falta ${CONFIG_FILE}" >&2; exit 1; }
# El entorno tiene PRIORIDAD sobre el fichero de config: asi se puede lanzar
#   TRAIN_GARMENTNET=1 WIDTH=576 HEIGHT=768 bash experimento_v14.sh --tag c2 ...
# sin editar el fichero. Los valores del entorno se reaplican despues del source.
declare -A _ENV_OVERRIDE=()
while IFS= read -r _name; do
  if [[ -n "${!_name+set}" ]]; then _ENV_OVERRIDE["${_name}"]="${!_name}"; fi
done < <(grep -oE '^[A-Za-z_][A-Za-z0-9_]*=' "${CONFIG_FILE}" | tr -d '=' | sort -u)
set -a
# shellcheck disable=SC1090
source "${CONFIG_FILE}"
set +a
for _name in "${!_ENV_OVERRIDE[@]}"; do
  export "${_name}=${_ENV_OVERRIDE[${_name}]}"
done
unset _ENV_OVERRIDE _name

CONDA_SH="${CONDA_SH:-/home/uceda/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-idm}"
PAUSE_FILE="${PAUSE_FILE:-/home/uceda/Documents/IDM-VTON/entrenamiento continuo/PAUSAR_WATCHDOG}"
TAG=""; STEPS=""; DRY_RUN=0; RESUME_FROM=""
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --tag) TAG="${2:-}"; shift 2 ;;
    --tag=*) TAG="${1#*=}"; shift ;;
    --steps) STEPS="${2:-}"; shift 2 ;;
    --steps=*) STEPS="${1#*=}"; shift ;;
    --resume-from) RESUME_FROM="${2:-}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) sed -n '2,27p' "$0"; exit 0 ;;
    *) echo "ERROR: argumento no reconocido: $1" >&2; exit 2 ;;
  esac
done
[[ -n "${TAG}" ]] || { echo "ERROR: falta --tag <nombre> (ej: --tag aug_hue0)" >&2; exit 2; }
[[ "${TAG}" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "ERROR: tag invalido: ${TAG}" >&2; exit 2; }
STEPS="${STEPS:-${MAX_TRAIN_STEPS}}"

log() { echo "[experimento] $*"; }

# --- Precondiciones ----------------------------------------------------------
if [[ ! -f "${PAUSE_FILE}" ]]; then
  echo "ERROR: falta ${PAUSE_FILE}" >&2
  echo "  Sin esa pausa, el cron del watchdog lanzaria el entrenamiento de produccion" >&2
  echo "  en menos de 1 minuto y las dos corridas competirian por GPU y RAM." >&2
  echo "  Crearla primero:  touch \"${PAUSE_FILE}\"" >&2
  exit 1
fi
if pgrep -f 'gradio_demo/app.py' >/dev/null 2>&1; then
  echo "ERROR: la app web esta levantada. Pararla antes (12 GB de VRAM no dan para las dos)." >&2
  exit 1
fi
if pgrep -f 'train_xl.py' >/dev/null 2>&1; then
  echo "ERROR: ya hay un entrenamiento en marcha (pgrep -f train_xl.py)." >&2
  exit 1
fi
FREE_GIB="$(df -BG --output=avail "${PROJECT_ROOT}" | tail -1 | tr -dc '0-9')"
if [[ -z "${FREE_GIB}" || "${FREE_GIB}" -lt "${MIN_FREE_GIB}" ]]; then
  echo "ERROR: solo ${FREE_GIB:-?} GiB libres (MIN_FREE_GIB=${MIN_FREE_GIB}). Libera espacio con:" >&2
  echo "  bash \"${SCRIPT_DIR}/podar_disco_v14.sh\"" >&2
  exit 1
fi

# --- Comando del experimento -------------------------------------------------
OUT_DIR="${EXPERIMENT_ROOT}/${TAG}"
LOG_DIR="${EXPERIMENT_LOG_ROOT}"
LOG_FILE="${LOG_DIR}/${TAG}.log"
META_FILE="${LOG_DIR}/${TAG}.meta"
[[ ! -e "${OUT_DIR}" ]] || { echo "ERROR: ${OUT_DIR} ya existe: elige otro --tag" >&2; exit 1; }

ARGS=(
  train_xl.py
  --pretrained_model_name_or_path="${BASE_CHECKPOINT}"
  --pretrained_garmentnet_path="${GARMENTNET_PATH}"
  --pretrained_ip_adapter_path="${IP_ADAPTER_PATH}"
  --image_encoder_path="${IMAGE_ENCODER_PATH}"
  --data_dir="${DATA_DIR}"
  --output_dir="${OUT_DIR}"
  --mixed_precision=fp16
  --gradient_checkpointing
  --low_vram_training
  --hybrid_small_models_gpu
  --cpu_threads="${CPU_THREADS}"
  --width="${WIDTH}"
  --height="${HEIGHT}"
  --train_batch_size="${TRAIN_BATCH_SIZE}"
  --test_batch_size="${TEST_BATCH_SIZE}"
  --train_num_workers="${TRAIN_NUM_WORKERS}"
  --test_num_workers="${TEST_NUM_WORKERS}"
  --gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}"
  --max_train_steps="${STEPS}"
  --checkpointing_steps="${CHECKPOINTING_STEPS}"
  --logging_steps="${LOGGING_STEPS}"
  --learning_rate="${LEARNING_RATE}"
  --max_grad_norm="${MAX_GRAD_NORM}"
  --num_tokens="${NUM_TOKENS}"
  --seed="${SEED}"
  --garmentnet_dtype="${GARMENTNET_DTYPE}"
)
if [[ "${TRAIN_IP_ADAPTER_ONLY}" == "1" ]]; then ARGS+=(--train_ip_adapter_only); fi
if [[ "${TRAIN_GARMENTNET}" == "1" ]]; then ARGS+=(--train_garmentnet); fi
if [[ "${COLOR_JITTER_ENABLED}" == "1" ]]; then
  ARGS+=(
    --color_jitter_prob="${COLOR_JITTER_PROB}"
    --color_jitter_brightness="${COLOR_JITTER_BRIGHTNESS}"
    --color_jitter_contrast="${COLOR_JITTER_CONTRAST}"
    --color_jitter_saturation="${COLOR_JITTER_SATURATION}"
    --color_jitter_hue="${COLOR_JITTER_HUE}"
  )
fi
if [[ "${RESUME_OPTIMIZER_STATE}" == "1" ]]; then ARGS+=(--resume_optimizer_state); fi
if [[ -n "${RESUME_FROM}" ]]; then ARGS+=(--resume_from_checkpoint="${RESUME_FROM}"); fi
if [[ -n "${EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  ARGS+=(${EXTRA_ARGS})
fi

if [[ "${TRAIN_GARMENTNET}" == "1" ]]; then
  log "perfil: IP-Adapter + GarmentNet (receta C2) -- exige VRAM para entrenar el GarmentNet"
else
  log "perfil: solo IP-Adapter (receta C1, la que corre en produccion)"
fi
if [[ "${COLOR_JITTER_ENABLED}" == "1" ]]; then
  log "augmentacion: prob=${COLOR_JITTER_PROB} brightness=${COLOR_JITTER_BRIGHTNESS} contrast=${COLOR_JITTER_CONTRAST} saturation=${COLOR_JITTER_SATURATION} hue=${COLOR_JITTER_HUE}"
else
  log "augmentacion: defaults de train_xl.py (prob 0.5, hue 0.1)"
fi
log "resolucion ${WIDTH}x${HEIGHT}, ${STEPS} pasos, lr=${LEARNING_RATE}, salida ${OUT_DIR}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} accelerate launch \\"
  printf '  %q \\\n' "${ARGS[@]}"
  echo
  log "DRY-RUN: no se ha lanzado nada ni se ha creado el log"
  exit 0
fi

mkdir -p "${LOG_DIR}" "${EXPERIMENT_ROOT}"
{
  echo "tag=${TAG}"
  echo "started_at=$(date '+%Y-%m-%d %H:%M:%S')"
  echo "steps=${STEPS}"
  echo "train_ip_adapter_only=${TRAIN_IP_ADAPTER_ONLY}"
  echo "train_garmentnet=${TRAIN_GARMENTNET}"
  echo "width=${WIDTH}"
  echo "height=${HEIGHT}"
  echo "learning_rate=${LEARNING_RATE}"
  echo "color_jitter_enabled=${COLOR_JITTER_ENABLED}"
  echo "color_jitter_hue=${COLOR_JITTER_HUE}"
  echo "resume_from=${RESUME_FROM:-none}"
  echo "log=${LOG_FILE}"
  printf 'command=accelerate launch'
  printf ' %q' "${ARGS[@]}"
  echo
} > "${META_FILE}"

log "lanzando (log: ${LOG_FILE})"
log "para detenerlo: pkill -f '[t]rain_xl.py'"
# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
cd "${PROJECT_ROOT}"
set +e
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" accelerate launch "${ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"
STATUS="${PIPESTATUS[0]}"
set -e

{
  echo "finished_at=$(date '+%Y-%m-%d %H:%M:%S')"
  echo "exit_code=${STATUS}"
} >> "${META_FILE}"

echo
log "exit_code=${STATUS}"
log "ritmo de los ultimos pasos (s/it):"
grep -o '[0-9]*\.[0-9]*s/it' "${LOG_FILE}" | tail -10 || true
log "loss final:"
grep -o 'loss[^,]*' "${LOG_FILE}" | tail -3 || true
log "checkpoints: $(ls -d "${OUT_DIR}"/checkpoint-* 2>/dev/null | sort -V | tr '\n' ' ' || true)"
log "siguiente paso: gate A/B del doc 04 -> exportar con exportar_checkpoint_para_demo.py y comparar con comparar_calidad_v9.py"
exit "${STATUS}"
