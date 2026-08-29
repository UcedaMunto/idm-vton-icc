#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WATCHDOG_SCRIPT="${SCRIPT_DIR}/watchdog_entrenamiento.sh"

CURRENT_CRON="$(crontab -l 2>/dev/null || true)"
NEW_CRON="$(printf '%s\n' "${CURRENT_CRON}" | grep -Fv "${WATCHDOG_SCRIPT}" || true)"

printf '%s\n' "${NEW_CRON}" | crontab -

echo "Cron eliminado para ${WATCHDOG_SCRIPT}"
