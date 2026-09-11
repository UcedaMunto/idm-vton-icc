#!/usr/bin/env bash
# V14 (2026-09-10) - Retencion de disco.
#
# Contexto (doc 09): el disco NO limita la velocidad, limita la CAPACIDAD. Cada bloque
# de 500 pasos deja ~4,9 GiB en produccion_continua y el watchdog exige MIN_FREE_GIB=40
# para arrancar: con el disco casi lleno la cadena se para por espacio, no por tiempo.
#
# Politica de retencion (doc 06.6), configurable por entorno:
#   KEEP_RUNS=2        run_* mas recientes de produccion_continua (el activo + el que
#                      sirve de reanudacion limpia)
#   KEEP_PRESERVED=2   run_* mas recientes de pruebas_checkpoints
# Nunca toca: base_oficial, demos/, experimentos/, backups/ ni el run que es origen del
# resume_from del ultimo bloque.
#
# Uso:
#   bash podar_disco_v14.sh                    # DRY-RUN: solo lista (no borra nada)
#   bash podar_disco_v14.sh --apply            # borra (pide confirmacion)
#   FORCE=1 bash podar_disco_v14.sh --apply    # borra sin preguntar
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/uceda/Documents/IDM-VTON}"
V14_ROOT="${V14_ROOT:-${PROJECT_ROOT}/result_train_v14}"
COMPACT_ROOT="${COMPACT_ROOT:-${V14_ROOT}/produccion_continua}"
PRESERVED_ROOT="${PRESERVED_ROOT:-${V14_ROOT}/pruebas_checkpoints}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/produccion_continua_v14}"
KEEP_RUNS="${KEEP_RUNS:-2}"
KEEP_PRESERVED="${KEEP_PRESERVED:-2}"

APPLY=0
[[ "${1:-}" == "--apply" ]] && APPLY=1
log() { echo "[disco] $*"; }

if pgrep -f 'train_xl.py' >/dev/null 2>&1 && [[ "${FORCE:-0}" != "1" ]]; then
  echo "ERROR: hay entrenamiento en marcha (pgrep -f train_xl.py): pausalo antes de podar." >&2
  exit 1
fi

# El run del que reanuda el ultimo bloque es la red de seguridad de la cadena: no se borra.
PROTECTED=""
if [[ -d "${LOG_ROOT}" ]]; then
  PROTECTED="$(grep -h '^resume_from=' "${LOG_ROOT}"/train_run_*.meta 2>/dev/null \
    | sed 's|^resume_from=||' | grep -v '^none$' | sort -u | tail -1 || true)"
  PROTECTED="${PROTECTED%%/checkpoint-*}"
fi

df -h "${PROJECT_ROOT}" | tail -1
[[ -n "${PROTECTED}" ]] && log "protegido por resume_from: ${PROTECTED}"
log "retencion: ${KEEP_RUNS} run(s) en produccion_continua, ${KEEP_PRESERVED} en pruebas_checkpoints"

candidates=()
collect() { # $1=raiz  $2=cuantos conservar
  local root="$1" keep="$2" i=0 d
  [[ -d "${root}" ]] || return 0
  while IFS= read -r d; do
    [[ -n "${d}" ]] || continue
    i=$((i + 1))
    if [[ -n "${PROTECTED}" && "${d}" == "${PROTECTED}" ]]; then
      log "  conservar  $(basename "${d}")  (resume_from del ultimo bloque)"
    elif [[ "${i}" -le "${keep}" ]]; then
      log "  conservar  $(basename "${d}")"
    else
      candidates+=("${d}")
    fi
  done < <(ls -td "${root}"/run_* 2>/dev/null || true)
}

collect "${COMPACT_ROOT}" "${KEEP_RUNS}"
collect "${PRESERVED_ROOT}" "${KEEP_PRESERVED}"

if [[ "${#candidates[@]}" -eq 0 ]]; then
  log "no hay candidatos: nada que podar"
  exit 0
fi

total_mb=0
for d in "${candidates[@]}"; do
  size_mb="$(du -sm "${d}" 2>/dev/null | cut -f1 || echo 0)"
  total_mb=$((total_mb + size_mb))
  echo "  BORRAR     $(basename "$(dirname "${d}")")/$(basename "${d}")  ($(du -sh "${d}" 2>/dev/null | cut -f1))"
done
log "liberables: ~$((total_mb / 1024)) GiB en ${#candidates[@]} carpetas"

if [[ "${APPLY}" -eq 0 ]]; then
  echo
  log "DRY-RUN: no se ha borrado nada. Para aplicarlo: bash $0 --apply"
  exit 0
fi

if [[ "${FORCE:-0}" != "1" ]]; then
  read -r -p "Borrar ${#candidates[@]} carpetas (~$((total_mb / 1024)) GiB)? [s/N] " answer
  [[ "${answer}" =~ ^[sS]$ ]] || { log "cancelado"; exit 0; }
fi

for d in "${candidates[@]}"; do
  rm -rf -- "${d}"
  log "borrado ${d}"
done
df -h "${PROJECT_ROOT}" | tail -1
log "hecho. Recuerda: nunca borrar base_oficial, demos/ (las usa la app) ni backups/."
