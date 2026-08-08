#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$ROOT_DIR/.env"

DEFAULT_MODEL="yisol/IDM-VTON"
TRAINED_MODEL="$ROOT_DIR/result_train_night/checkpoint-250"

usage() {
  cat <<'EOF'
Usage:
  ./switch_model_version.sh status
  ./switch_model_version.sh current
  ./switch_model_version.sh trained
  ./switch_model_version.sh custom <absolute_or_relative_path>

Actions:
  status   Show active model path from .env
  current  Switch to baseline model (yisol/IDM-VTON)
  trained  Switch to trained checkpoint (result_train_night/checkpoint-250)
  custom   Switch to a custom model/checkpoint path
EOF
}

ensure_env_file() {
  if [[ ! -f "$ENV_FILE" ]]; then
    touch "$ENV_FILE"
  fi
}

get_current_model() {
  if [[ -f "$ENV_FILE" ]] && grep -q '^IDMVTON_MODEL_PATH=' "$ENV_FILE"; then
    grep '^IDMVTON_MODEL_PATH=' "$ENV_FILE" | tail -n 1 | cut -d'=' -f2-
  else
    echo "$DEFAULT_MODEL"
  fi
}

set_env_var() {
  local key="$1"
  local value="$2"

  if grep -q "^${key}=" "$ENV_FILE"; then
    sed -i "s|^${key}=.*|${key}=${value}|" "$ENV_FILE"
  else
    printf '\n%s=%s\n' "$key" "$value" >> "$ENV_FILE"
  fi
}

resolve_model_path() {
  local input_path="$1"
  if [[ "$input_path" == /* ]] || [[ "$input_path" == *":"* ]]; then
    echo "$input_path"
  else
    echo "$ROOT_DIR/$input_path"
  fi
}

switch_model() {
  local model_path="$1"
  ensure_env_file
  set_env_var "IDMVTON_MODEL_PATH" "$model_path"

  echo "[switch] Active model set to: $model_path"
  echo "[switch] Updated file: $ENV_FILE"
  echo "[switch] Restart app or rerun inference scripts to apply the change."
}

main() {
  local action="${1:-status}"

  case "$action" in
    status)
      echo "[switch] Active model: $(get_current_model)"
      ;;
    current)
      switch_model "$DEFAULT_MODEL"
      ;;
    trained)
      if [[ ! -d "$TRAINED_MODEL" ]]; then
        echo "[switch] ERROR: trained model not found at $TRAINED_MODEL"
        echo "[switch] Use: ./switch_model_version.sh custom <path>"
        exit 1
      fi
      switch_model "$TRAINED_MODEL"
      ;;
    custom)
      if [[ $# -lt 2 ]]; then
        echo "[switch] ERROR: missing path for custom mode"
        usage
        exit 1
      fi
      local custom_path
      custom_path="$(resolve_model_path "$2")"
      if [[ ! -d "$custom_path" ]]; then
        echo "[switch] ERROR: directory not found: $custom_path"
        exit 1
      fi
      switch_model "$custom_path"
      ;;
    -h|--help|help)
      usage
      ;;
    *)
      echo "[switch] ERROR: unknown action '$action'"
      usage
      exit 1
      ;;
  esac
}

main "$@"
