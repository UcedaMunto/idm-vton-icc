#!/usr/bin/env bash
set -u

ROOT_DIR="/home/uceda/Documents/IDM-VTON"
ACTIVE_CONFIG="${ROOT_DIR}/configuracion-v4-entrenamiento/config_v4.env"
FAILURE_PAUSE="${ROOT_DIR}/entrenamiento continuo/PAUSAR_POR_ERROR"
MANUAL_PAUSE="${ROOT_DIR}/entrenamiento continuo/PAUSAR_WATCHDOG"

printf '%s\n' '=== CONFIGURACION ACTIVA ==='
printf '%s\n' "${ACTIVE_CONFIG}"

printf '%s\n' '=== CRON ==='
crontab -l 2>/dev/null | grep -F 'watchdog_entrenamiento.sh' || echo 'watchdog no instalado'

printf '%s\n' '=== PROCESOS ==='
training_pids="$(pgrep -d, -f 'train_xl.py|accelerate launch' || true)"
if [[ -n "${training_pids}" ]]; then
  ps -o pid,ppid,state,%cpu,%mem,rss,etime,comm -p "${training_pids}"
else
  echo 'sin entrenamiento activo'
fi

printf '%s\n' '=== PAUSAS ==='
[[ -f "${MANUAL_PAUSE}" ]] && echo 'pausa manual activa' || echo 'pausa manual inactiva'
if [[ -f "${FAILURE_PAUSE}" ]]; then
  echo 'disyuntor por error activo:'
  cat "${FAILURE_PAUSE}"
else
  echo 'disyuntor por error inactivo'
fi

printf '%s\n' '=== RECURSOS ==='
free -h
df -h "${ROOT_DIR}"
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader 2>/dev/null || true
swap_used_kib="$(free -k | awk '/^Swap:/ {print $3}')"
if (( swap_used_kib > 8 * 1024 * 1024 )); then
  echo 'ADVERTENCIA: mas de 8 GiB de swap en uso; el update puede avanzar muy lentamente.'
fi

printf '%s\n' '=== ULTIMO CHECKPOINT V4 ==='
find "${ROOT_DIR}/result_train_v4" -type f -name manifest.json -printf '%T@ %h\n' 2>/dev/null \
  | sort -nr | awk 'NR==1 {$1=""; sub(/^ /, ""); print; found=1} END {if (!found) print "ninguno"}'

printf '%s\n' '=== ULTIMO LOG V4 ==='
latest_log="$(find "${ROOT_DIR}/logs/training_v4" -maxdepth 1 -type f -name 'v4_run_*.log' -printf '%T@ %p\n' 2>/dev/null \
  | sort -nr | awk 'NR==1 {$1=""; sub(/^ /, ""); print}')"
if [[ -n "${latest_log}" ]]; then
  echo "${latest_log}"
  log_mtime="$(stat -c %Y "${latest_log}")"
  log_age_seconds=$(( $(date +%s) - log_mtime ))
  echo "ultima escritura hace ${log_age_seconds}s"
  tail -n 25 "${latest_log}"
else
  echo 'ninguno'
fi