#!/usr/bin/env bash
# V14 (2026-09-10) - Respaldo verificable del entrenamiento.
#
# Por que existe: los pesos entrenables (trainable_state.pt, ~1,7 GiB) son el UNICO
# artefacto que NO se puede regenerar. El watchdog poda checkpoints intermedios y la
# politica de disco (doc 06.6) borra run_* antiguos: cualquiera de los dos puede
# llevarse por delante la punta de la cadena. Este script la respalda con HARDLINKS
# (mismo inodo -> 0 bytes extra de disco, pero sobrevive a un rm -rf del origen) y
# deja un MANIFEST + sha256 para poder demostrar despues que sigue intacta.
#
# Por que hardlinks y no copias: el disco es el limitante del entrenamiento continuo
# (doc 09). Copiar realmente 2 checkpoints son ~9,5 GiB; el hardlink cuesta 0.
# Limitacion: solo vale dentro del mismo sistema de ficheros (los hardlinks no cruzan FS).
#
# Uso:
#   bash respaldo_entrenamiento_v14.sh              # crea respaldo + manifest
#   bash respaldo_entrenamiento_v14.sh --verify     # verifica el ultimo respaldo (sha256)
#   bash respaldo_entrenamiento_v14.sh --list       # lista los respaldos existentes
#
# Variables: PROJECT_ROOT, COMPACT_ROOT, BACKUP_ROOT, PYTHON_BIN, TAG
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/uceda/Documents/IDM-VTON}"
COMPACT_ROOT="${COMPACT_ROOT:-${PROJECT_ROOT}/result_train_v14/produccion_continua}"
BACKUP_ROOT="${BACKUP_ROOT:-${PROJECT_ROOT}/backups}"
PYTHON_BIN="${PYTHON_BIN:-/home/uceda/miniconda3/envs/idm/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi
MODE="${1:-crear}"
FILES=(trainable_state.pt optimizer_state.pt manifest.json)

die() { echo "ERROR: $*" >&2; exit 1; }
log() { echo "[respaldo] $*"; }

cumulative_of() {
  "${PYTHON_BIN}" - "$1/manifest.json" <<'PY' 2>/dev/null || echo "?"
import json, sys
try:
    print(json.load(open(sys.argv[1], encoding="utf-8")).get("cumulative_steps", "?"))
except Exception:
    print("?")
PY
}

list_backups() {
  local dir
  for dir in "${BACKUP_ROOT}"/v14_estado_*; do
    [[ -d "${dir}" ]] || continue
    echo "== $(basename "${dir}") ($(du -sh "${dir}" 2>/dev/null | cut -f1) aparentes)"
    find "${dir}" -maxdepth 1 -mindepth 1 -type d -printf '   %f\n'
  done
}

verify_backup() {
  local dir="${1:-$(ls -td "${BACKUP_ROOT}"/v14_estado_* 2>/dev/null | head -1 || true)}"
  [[ -n "${dir}" && -d "${dir}" ]] || die "no hay ningun respaldo v14_estado_* en ${BACKUP_ROOT}"
  local tsv="${dir}/verificacion.tsv"
  [[ -f "${tsv}" ]] || die "falta ${tsv} (respaldo creado por una version anterior del script)"
  log "verificando $(basename "${dir}") (relee todos los ficheros: tarda ~1 min)"
  local fails=0 lines=0 rel src expected backup got links
  while IFS=$'\t' read -r rel src expected; do
    [[ -n "${rel}" ]] || continue
    lines=$((lines + 1))
    backup="${dir}/${rel}"
    if [[ ! -f "${backup}" ]]; then
      echo "  FALLO  ${rel}: no existe en el respaldo"
      fails=$((fails + 1)); continue
    fi
    got="$(sha256sum "${backup}" | cut -d' ' -f1)"
    if [[ ! -f "${src}" ]]; then
      echo "  AVISO  ${rel}: el origen ya no existe (${src}); solo se comprueba el respaldo"
      if [[ "${got}" == "${expected}" ]]; then
        echo "  OK     ${rel} (respaldo intacto)"
      else
        echo "  FALLO  ${rel}: sha256 del respaldo no coincide"; fails=$((fails + 1))
      fi
      continue
    fi
    links="$(stat -c %h "${backup}")"
    if [[ "${got}" == "${expected}" ]]; then
      echo "  OK     ${rel} (links=${links}, sha256=${got:0:16}...)"
    else
      echo "  FALLO  ${rel}: esperado ${expected:0:16}... obtenido ${got:0:16}..."
      fails=$((fails + 1))
    fi
  done < "${tsv}"
  log "${lines} entradas verificadas, ${fails} fallos"
  [[ "${fails}" -eq 0 ]] || exit 1
}  # fin de verify_backup

create_backup() {
  local tag="${TAG:-$(date '+%Y%m%d_%H%M%S')}"
  local dest="${BACKUP_ROOT}/v14_estado_${tag}"
  [[ ! -e "${dest}" ]] || die "${dest} ya existe (usa TAG=otro para forzar otro nombre)"

  local runs=() ckpts=() prev_ckpts=() prepared=()
  mapfile -t runs < <(ls -td "${COMPACT_ROOT}"/run_* 2>/dev/null || true)
  [[ "${#runs[@]}" -gt 0 ]] || die "no hay runs en ${COMPACT_ROOT}"

  # Punta de la cadena = ultimo checkpoint del run mas reciente (bloque en curso).
  mapfile -t ckpts < <(ls -d "${runs[0]}"/checkpoint-* 2>/dev/null | sort -V || true)
  [[ "${#ckpts[@]}" -gt 0 ]] || die "no hay checkpoints en ${runs[0]}"
  prepared+=("${ckpts[-1]}")

  # Ultimo bloque completo = ultimo checkpoint del run anterior: es el punto de
  # reanudacion "limpio" de la cadena (500/500 pasos hechos y verificados).
  if [[ "${#runs[@]}" -gt 1 ]]; then
    mapfile -t prev_ckpts < <(ls -d "${runs[1]}"/checkpoint-* 2>/dev/null | sort -V || true)
    [[ "${#prev_ckpts[@]}" -gt 0 ]] && prepared+=("${prev_ckpts[-1]}")
  fi

  mkdir -p "${dest}"
  : > "${dest}/verificacion.tsv"
  log "respaldando ${#prepared[@]} checkpoint(s) en ${dest} (hardlinks, coste 0 bytes)"
  local ckpt file rel cum dest_dir
  for ckpt in "${prepared[@]}"; do
    cum="$(cumulative_of "${ckpt}")"
    dest_dir="${dest}/ckpt_${cum}_$(basename "$(dirname "${ckpt}")")"
    mkdir -p "${dest_dir}"
    for file in "${FILES[@]}"; do
      if [[ ! -f "${ckpt}/${file}" ]]; then
        log "  omitido (no existe): ${ckpt}/${file}"
        continue
      fi
      ln "${ckpt}/${file}" "${dest_dir}/${file}"
      rel="$(basename "${dest_dir}")/${file}"
      printf '%s\t%s\t%s\n' "${rel}" "${ckpt}/${file}" "$(sha256sum "${ckpt}/${file}" | cut -d' ' -f1)" >> "${dest}/verificacion.tsv"
      log "  ${rel} <- ${ckpt}/${file}"
    done
  done

  {
    echo "Respaldo del estado de entrenamiento V14"
    echo "generado=$(date '+%Y-%m-%d %H:%M:%S')"
    echo "tipo=hardlinks (mismo inodo; coste 0 bytes; sobrevive a rm -rf del origen)"
    echo "verificacion=verificacion.tsv (rel<TAB>origen<TAB>sha256) y 'bash $0 --verify'"
    echo
    cat "${dest}/verificacion.tsv"
    echo
    echo "No versionado en git: result_*/ esta en .gitignore (los pesos no entran al repo)."
  } > "${dest}/MANIFEST.txt"

  log "disco tras el respaldo (no debe cambiar): $(df -h "${BACKUP_ROOT}" | tail -1)"
  verify_backup "${dest}"
}

mkdir -p "${BACKUP_ROOT}"
case "${MODE}" in
  crear|--crear) create_backup ;;
  --verify|verificar) verify_backup "${2:-}" ;;
  --list|listar) list_backups ;;
  *) die "modo no reconocido: ${MODE} (usa: crear | --verify | --list)" ;;
esac
