#!/usr/bin/env bash
# Instala la preservacion de checkpoints intermedios como cron (cada 5 minutos).
# Desinstalar: borrar la linea con 'preservar_checkpoints_pruebas.sh' de `crontab -l`.
set -euo pipefail

SCRIPT="/home/uceda/Documents/IDM-VTON/configuracion-v12-entrenamiento/preservar_checkpoints_pruebas.sh"
LOG="/home/uceda/Documents/IDM-VTON/logs/produccion_continua/preservar_checkpoints.log"

chmod +x "${SCRIPT}"

CRON_LINE="*/5 * * * * /bin/bash \"${SCRIPT}\" >> \"${LOG}\" 2>&1"

CURRENT_CRON="$(crontab -l 2>/dev/null || true)"

if printf '%s\n' "${CURRENT_CRON}" | grep -Fq "preservar_checkpoints_pruebas.sh"; then
  echo "Preservacion ya instalada."
  exit 0
fi

{
  printf '%s\n' "${CURRENT_CRON}"
  printf '%s\n' "${CRON_LINE}"
} | sed '/^$/N;/^\n$/D' | crontab -

echo "Preservacion instalada (cada 5 min)."
echo "Linea: ${CRON_LINE}"
