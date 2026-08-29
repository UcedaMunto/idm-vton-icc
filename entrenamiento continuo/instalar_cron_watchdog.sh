#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WATCHDOG_SCRIPT="${SCRIPT_DIR}/watchdog_entrenamiento.sh"
LOG_DIR="/home/uceda/Documents/IDM-VTON/logs/produccion_continua"
CRON_LOG="${LOG_DIR}/cron_watchdog.log"

mkdir -p "${LOG_DIR}"

if [[ ! -f "${WATCHDOG_SCRIPT}" ]]; then
  echo "ERROR: no existe ${WATCHDOG_SCRIPT}"
  exit 1
fi

chmod +x "${WATCHDOG_SCRIPT}"

CRON_LINE="* * * * * /bin/bash \"${WATCHDOG_SCRIPT}\" >> \"${CRON_LOG}\" 2>&1"

CURRENT_CRON="$(crontab -l 2>/dev/null || true)"

# Evita duplicados exactos.
if printf '%s\n' "${CURRENT_CRON}" | grep -Fq "${WATCHDOG_SCRIPT}"; then
  echo "Cron ya instalado para ${WATCHDOG_SCRIPT}"
  exit 0
fi

{
  printf '%s\n' "${CURRENT_CRON}"
  printf '%s\n' "${CRON_LINE}"
} | sed '/^$/N;/^\n$/D' | crontab -

echo "Cron instalado correctamente."
echo "Linea instalada: ${CRON_LINE}"
echo "Para validar: crontab -l"
