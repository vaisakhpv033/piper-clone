#!/usr/bin/env bash
# ==============================================================================
# RunPod Startup Script for High-Concurrency Piper TTS Server (20+ Concurrency)
# ==============================================================================
set -e

echo "=== [1/5] Setting up Environment ==="
# Work in current repo or /workspace
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}}"
export WORKERS="${WORKERS:-4}"
export PORT="${PORT:-5000}"
export HOST="${HOST:-0.0.0.0}"

# Default models requested: Hindi, English, Malayalam, Telugu
export VOICE_MODELS="${VOICE_MODELS:-hi_IN-rohan-medium,en_US-lessac-medium,ml_IN-meera-medium,te_IN-padmavathi-medium}"

echo "Working directory: ${SCRIPT_DIR}"
echo "Data/Models directory: ${DATA_DIR}"
echo "Workers count: ${WORKERS}"
echo "Port: ${PORT}"
echo "Voice models to load: ${VOICE_MODELS}"

echo "=== [2/5] Checking GPU & CUDA Availability ==="
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    export USE_CUDA="true"
else
    echo "No GPU detected or CUDA driver not accessible. Running on CPU mode."
    export USE_CUDA="false"
fi

echo "=== [3/5] Installing Production Dependencies ==="
pip install -r requirements-prod.txt --no-cache-dir

if [ "${USE_CUDA}" = "true" ]; then
    # Ensure onnxruntime-gpu is present
    pip install onnxruntime-gpu --no-cache-dir
fi

echo "=== [4/5] Downloading Voice Models (if not already present) ==="
IFS=',' read -ra MODELS_ARRAY <<< "${VOICE_MODELS}"
for model in "${MODELS_ARRAY[@]}"; do
    model="$(echo "${model}" | xargs)" # trim whitespace
    onnx_file="${DATA_DIR}/${model}.onnx"
    if [ -f "${onnx_file}" ]; then
        echo "✓ Model '${model}' already exists at ${onnx_file}"
    else
        echo "↓ Downloading model '${model}' to ${DATA_DIR}..."
        python3 -m piper.download_voices --data-dir "${DATA_DIR}" "${model}"
    fi
done

echo "=== [5/5] Launching Gunicorn (${WORKERS} Workers) ==="
echo "Server will listen on http://${HOST}:${PORT}"
echo "Health check endpoint: http://${HOST}:${PORT}/health"
echo "Synthesize endpoint: http://${HOST}:${PORT}/synthesize"

exec gunicorn production_server:app \
    --workers "${WORKERS}" \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind "${HOST}:${PORT}" \
    --timeout 60 \
    --keep-alive 10 \
    --access-logfile - \
    --error-logfile -
