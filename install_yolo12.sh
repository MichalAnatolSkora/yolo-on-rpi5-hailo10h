#!/bin/bash
set -euo pipefail

# Downloads a pre-compiled YOLOv12n HEF model for Hailo-10H and installs
# Python dependencies for inference. Idempotent — safe to run multiple times.

MODEL_DIR="${HOME}/hailo_models"
MODEL_NAME="yolov11n"
HEF_FILE="${MODEL_DIR}/${MODEL_NAME}.hef"
HEF_URL="https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/${MODEL_NAME}.hef"
VENV_DIR="${HOME}/hailo_venv"

log() {
    echo "[$(date '+%H:%M:%S')] $1"
}

error_exit() {
    echo "ERROR: $1" >&2
    exit 1
}

# --- Pre-flight checks ---

if [[ "$(uname -m)" != "aarch64" ]]; then
    error_exit "This script must be run on a 64-bit Raspberry Pi OS (aarch64)."
fi

if [[ $EUID -eq 0 ]]; then
    error_exit "Do not run this script as root. It will use sudo when needed."
fi

# Detect Hailo-10H (45c4) vs Hailo-8 (2864) and check correct package
HAILO_DEV_ID=$(lspci -nn 2>/dev/null | grep -i hailo | grep -o '\[[0-9a-fA-F]\{4\}:[0-9a-fA-F]\{4\}\]' | head -1 | tr -d '[]' | cut -d: -f2)
if [[ "$HAILO_DEV_ID" == "45c4" ]]; then
    HAILO_PKG="hailo-h10-all"
else
    HAILO_PKG="hailo-all"
fi

if ! dpkg -s "$HAILO_PKG" &>/dev/null; then
    error_exit "$HAILO_PKG is not installed. Run ./install_hailo.sh first."
fi

echo "=========================================================="
echo " YOLO Model Setup for Hailo-10H (v5.1.1 Compatible)      "
echo "=========================================================="

mkdir -p "$MODEL_DIR"

# --- Step 1: Python virtual environment ---

log "[1/3] Checking Python virtual environment..."
if [[ -d "$VENV_DIR" ]]; then
    log " -> Virtual environment already exists: $VENV_DIR"
else
    log " -> Creating virtual environment with system site-packages..."
    python3 -m venv --system-site-packages "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# --- Step 2: Install Python dependencies ---

log "[2/3] Checking Python dependencies..."
if python3 -c "import ultralytics" &>/dev/null; then
    log " -> Ultralytics is already installed."
else
    log " -> Installing Ultralytics..."
    pip install --upgrade pip
    pip install ultralytics
fi

# --- Step 3: Download pre-compiled HEF ---

log "[3/3] Checking HEF model..."
if [[ -f "$HEF_FILE" ]]; then
    log " -> HEF model already exists: $HEF_FILE"
else
    log " -> Downloading pre-compiled ${MODEL_NAME} HEF for Hailo-10H..."
    log "    Source: Hailo Model Zoo (compiled with DFC v5.1.0, compatible with HailoRT 5.1.1)"

    python3 - <<PYEOF
from urllib.request import urlretrieve, Request, urlopen

url = "${HEF_URL}"
out = "${HEF_FILE}"

print(f"Downloading {url}")
req = Request(url, method="HEAD")
resp = urlopen(req, timeout=15)
size_mb = int(resp.headers.get("Content-Length", 0)) / 1024 / 1024
if size_mb > 0:
    print(f"Size: {size_mb:.1f} MB")
urlretrieve(url, out)
print(f"Saved to {out}")
PYEOF

    if [[ ! -f "$HEF_FILE" ]]; then
        error_exit "Failed to download HEF model from Hailo Model Zoo."
    fi
    log " -> Downloaded: $HEF_FILE"
fi

echo ""
echo "=========================================================="
echo " YOLO setup complete!                                     "
echo "                                                          "
echo " Model: $HEF_FILE                                        "
echo "                                                          "
echo " Run inference:                                           "
echo "   python run_yolo12.py --model $HEF_FILE                "
echo "=========================================================="
