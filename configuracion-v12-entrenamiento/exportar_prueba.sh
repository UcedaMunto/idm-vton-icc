#!/usr/bin/env bash
# Exporta un checkpoint (preservado o de la cadena) a un pipeline completo
# listo para usar con la app (IDMVTON_MODEL_PATH). Util para probar resultados
# iniciales del entrenamiento sin esperar al modelo final.
#
# Uso:
#   bash exportar_prueba.sh <dir_checkpoint> [nombre_salida]
#   bash exportar_prueba.sh /home/uceda/Documents/IDM-VTON/result_train_v10/pruebas_checkpoints/run_20260829_094044/checkpoint-100
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/uceda/Documents/IDM-VTON}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-${PROJECT_ROOT}/result_train_night/checkpoint-250}"
CKPT="${1:?Uso: exportar_prueba.sh <dir_checkpoint> [nombre_salida]}"
NAME="${2:-prueba_$(date '+%Y%m%d_%H%M%S')}"
OUT="${PROJECT_ROOT}/result_train_v10/demos/${NAME}"

if [[ ! -f "${CKPT}/manifest.json" || ! -f "${CKPT}/trainable_state.pt" ]]; then
  echo "ERROR: ${CKPT} no parece un checkpoint compacto (faltan manifest.json/trainable_state.pt)" >&2
  exit 1
fi

echo "Cumulative steps del checkpoint:"
python3 -c "import json,sys; d=json.load(open('${CKPT}/manifest.json')); print(' ', d.get('cumulative_steps'), 'pasos acumulados')"

python3 "${PROJECT_ROOT}/exportar_checkpoint_para_demo.py" \
  --base_checkpoint "${BASE_CHECKPOINT}" \
  --compact_checkpoint "${CKPT}" \
  --output_dir "${OUT}"

echo
echo "Modelo listo: ${OUT}"
echo "Para usarlo en la app: editar IDMVTON_MODEL_PATH en ${PROJECT_ROOT}/.env apuntando a esa ruta y reiniciar la app."
