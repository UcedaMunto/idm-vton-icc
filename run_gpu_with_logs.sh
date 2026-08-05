#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$ROOT_DIR/.env"
LOG_DIR="$ROOT_DIR/logs"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/idmvton_gpu_${TS}.log"
LATEST_LINK="$LOG_DIR/latest.log"
APP_FILE="$ROOT_DIR/gradio_demo/app.py"

mkdir -p "$LOG_DIR"

if [[ -f "$ENV_FILE" ]]; then
	set -a
	source "$ENV_FILE"
	set +a
fi

if [[ ! -f "$APP_FILE" ]]; then
	echo "ERROR: app file not found at $APP_FILE"
	exit 1
fi

if ss -ltn 2>/dev/null | grep -q ':7860 '; then
	echo "ERROR: port 7860 is already in use."
	echo "Stop previous app first, for example:"
	echo "  pkill -f 'python .*gradio_demo/app.py'"
	exit 1
fi

if command -v conda >/dev/null 2>&1; then
	CONDA_CMD="conda"
elif [[ -x "/home/uceda/miniconda3/bin/conda" ]]; then
	CONDA_CMD="/home/uceda/miniconda3/bin/conda"
else
	echo "ERROR: conda command not found."
	exit 1
fi

{
	echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting IDM-VTON (GPU mode)"
	echo "Root: $ROOT_DIR"
	echo "Log: $LOG_FILE"
		echo "IDMVTON_LOW_RAM=${IDMVTON_LOW_RAM:-1} IDMVTON_CPU_OFFLOAD=${IDMVTON_CPU_OFFLOAD:-1} IDMVTON_SHARE=${IDMVTON_SHARE:-0} IDMVTON_FORCE_CPU=${IDMVTON_FORCE_CPU:-0}"
	echo "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:64}"
} | tee "$LOG_FILE"

ln -sfn "$(basename "$LOG_FILE")" "$LATEST_LINK"

if command -v nvidia-smi >/dev/null 2>&1; then
	nvidia-smi | tee -a "$LOG_FILE"
else
	echo "WARN: nvidia-smi not found. GPU status check skipped." | tee -a "$LOG_FILE"
fi

export IDMVTON_LOW_RAM="${IDMVTON_LOW_RAM:-1}"
export IDMVTON_CPU_OFFLOAD="${IDMVTON_CPU_OFFLOAD:-1}"
export IDMVTON_SHARE="${IDMVTON_SHARE:-0}"
export IDMVTON_FORCE_CPU="${IDMVTON_FORCE_CPU:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:64}"
export PYTHONUNBUFFERED=1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching gradio_demo/app.py" | tee -a "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Open: http://127.0.0.1:7860" | tee -a "$LOG_FILE"

cd "$ROOT_DIR"
"$CONDA_CMD" run -n idm python "$APP_FILE" 2>&1 | tee -a "$LOG_FILE"
