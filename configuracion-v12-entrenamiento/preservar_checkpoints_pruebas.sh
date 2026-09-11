#!/usr/bin/env bash
# Preserva trainable_state.pt + manifest.json de los checkpoints de la corrida
# activa para poder probar resultados iniciales sin esperar al bloque completo.
#
# Por que hace falta: el watchdog borra los checkpoints intermedios (100/200/300/400)
# cuando un bloque de 500 pasos termina, dejando solo el checkpoint-500. Si se
# quiere probar el cambio con un checkpoint intermedio, hay que copiarlo antes.
#
# Se instala como cron cada 5 minutos (ver instalar_preservacion_pruebas.sh).
# Solo copia trainable_state.pt + manifest.json (~1.6 GiB por checkpoint);
# el optimizer_state.pt (3.2 GiB) NO hace falta para pruebas ni exportacion.
set -euo pipefail

# V14 (2026-09-08): seguimiento de la cadena nueva desde el modelo oficial.
PROJECT_ROOT="${PROJECT_ROOT:-/home/uceda/Documents/IDM-VTON}"
COMPACT_ROOT="${COMPACT_ROOT:-${PROJECT_ROOT}/result_train_v14/produccion_continua}"
PRESERVED_ROOT="${PRESERVED_ROOT:-${PROJECT_ROOT}/result_train_v14/pruebas_checkpoints}"

mkdir -p "${PRESERVED_ROOT}"

# La corrida activa = la mas reciente por mtime.
RUN_DIR="$(ls -td "${COMPACT_ROOT}"/run_* 2>/dev/null | head -1 || true)"
if [[ -z "${RUN_DIR}" || ! -d "${RUN_DIR}" ]]; then
  exit 0
fi

copied=0
for ckpt in "${RUN_DIR}"/checkpoint-*; do
  [[ -d "${ckpt}" ]] || continue
  name="$(basename "${ckpt}")"
  dest="${PRESERVED_ROOT}/$(basename "${RUN_DIR}")/${name}"
  if [[ ! -f "${dest}/manifest.json" ]]; then
    mkdir -p "${dest}"
    cp -p "${ckpt}/trainable_state.pt" "${ckpt}/manifest.json" "${dest}/"
    echo "[preservar] $(date '+%Y-%m-%d %H:%M:%S') copiado ${name} -> ${dest}"
    copied=$((copied + 1))
  fi
done

if [[ "${copied}" -gt 0 ]]; then
  echo "[preservar] ${copied} checkpoint(s) copiados a ${PRESERVED_ROOT}"
fi
