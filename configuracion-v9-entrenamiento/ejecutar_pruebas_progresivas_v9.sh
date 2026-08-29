#!/usr/bin/env bash
# V9: ejecuta pruebas progresivas de num_inference_steps (5, 10, 15), cada una
# en su propio log con progreso por paso (no solo al final). Se ejecutan de
# forma secuencial (nunca en paralelo) para no competir por CPU/RAM/GPU.
set -euo pipefail

ROOT_DIR="/home/uceda/Documents/IDM-VTON"
V9_DIR="${ROOT_DIR}/configuracion-v9-entrenamiento"
LOG_ROOT="${V9_DIR}/logs"
OUT_ROOT="${V9_DIR}/comparacion_progresiva"
mkdir -p "${LOG_ROOT}" "${OUT_ROOT}"

source /home/uceda/miniconda3/etc/profile.d/conda.sh
conda activate idm
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

for STEPS in 5 10 15; do
  if pgrep -fa 'train_xl.py|comparar_calidad_v9.py|gradio_demo/app.py' >/dev/null; then
    echo "ERROR: hay otro proceso de entrenamiento/inferencia activo; abortando antes de steps=${STEPS}"
    exit 1
  fi
  run_id="steps${STEPS}_$(date '+%Y%m%d_%H%M%S')"
  out_dir="${OUT_ROOT}/${run_id}"
  log_file="${LOG_ROOT}/${run_id}.log"
  mkdir -p "${out_dir}"
  echo "=== lanzando steps=${STEPS} -> log=${log_file} ==="
  python3 "${V9_DIR}/comparar_calidad_v9.py" \
    --pretrained_model_name_or_path="${ROOT_DIR}/result_train_night/checkpoint-250" \
    --data_dir="${ROOT_DIR}/dataset/DATA_DIR_PREP" \
    --output_dir="${out_dir}" \
    --width=448 --height=576 --num_inference_steps="${STEPS}" --seed=42 --limit=1 \
    > "${log_file}" 2>&1
  status=$?
  echo "steps=${STEPS} exit_code=${status} log=${log_file} output=${out_dir}"
  if [[ ${status} -ne 0 ]]; then
    echo "ERROR: fallo steps=${STEPS}; deteniendo la secuencia progresiva"
    tail -n 30 "${log_file}"
    exit "${status}"
  fi
done

echo "PROGRESION_V9_COMPLETA"
