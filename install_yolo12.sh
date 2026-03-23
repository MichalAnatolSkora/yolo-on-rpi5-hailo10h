#!/bin/bash
set -euo pipefail

# Installs Ultralytics, exports YOLOv12n to ONNX, and compiles it to a Hailo-10H
# HEF model. Idempotent — safe to run multiple times.

MODEL_DIR="${HOME}/hailo_models"
MODEL_NAME="yolov12n"
ONNX_FILE="${MODEL_DIR}/${MODEL_NAME}.onnx"
HEF_FILE="${MODEL_DIR}/${MODEL_NAME}.hef"
HAILO_MODEL_ZOO_DIR="${HOME}/hailo_model_zoo"
VENV_DIR="${HOME}/hailo_venv"
IMGSZ=640

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

if ! dpkg -s hailo-all &>/dev/null; then
    error_exit "hailo-all is not installed. Run ./install_hailo.sh first."
fi

echo "=========================================================="
echo " YOLOv12 Model Setup for Hailo-10H                       "
echo "=========================================================="

mkdir -p "$MODEL_DIR"

# --- Step 1: Python virtual environment ---

log "[1/5] Checking Python virtual environment..."
if [[ -d "$VENV_DIR" ]]; then
    log " -> Virtual environment already exists: $VENV_DIR"
else
    log " -> Creating virtual environment with system site-packages..."
    python3 -m venv --system-site-packages "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# --- Step 2: Install Ultralytics ---

log "[2/5] Checking Ultralytics..."
if python3 -c "import ultralytics" &>/dev/null; then
    log " -> Ultralytics is already installed."
else
    log " -> Installing Ultralytics..."
    pip install --upgrade pip
    pip install ultralytics
fi

# --- Step 3: Export YOLOv12n to ONNX ---

log "[3/5] Checking ONNX model..."
if [[ -f "$ONNX_FILE" ]]; then
    log " -> ONNX model already exists: $ONNX_FILE"
else
    log " -> Exporting ${MODEL_NAME} to ONNX (imgsz=${IMGSZ})..."
    python3 - <<PYEOF
from ultralytics import YOLO

model = YOLO("${MODEL_NAME}.pt")
model.export(format="onnx", imgsz=${IMGSZ}, opset=13, dynamic=False)

import shutil
shutil.move("${MODEL_NAME}.onnx", "${ONNX_FILE}")
PYEOF
    log " -> Exported: $ONNX_FILE"
fi

# --- Step 4: Clone Hailo Model Zoo ---

log "[4/5] Checking Hailo Model Zoo..."
if [[ -d "$HAILO_MODEL_ZOO_DIR" ]]; then
    log " -> Hailo Model Zoo already cloned: $HAILO_MODEL_ZOO_DIR"
else
    log " -> Cloning Hailo Model Zoo..."
    git clone https://github.com/hailo-ai/hailo_model_zoo.git "$HAILO_MODEL_ZOO_DIR"
fi

# --- Step 5: Compile ONNX to HEF ---

log "[5/5] Checking HEF model..."
if [[ -f "$HEF_FILE" ]]; then
    log " -> HEF model already exists: $HEF_FILE"
else
    log " -> Compiling ${MODEL_NAME} ONNX to HEF for Hailo-10H..."
    log "    This may take a while (10-30+ minutes)."

    # Install Hailo Model Zoo Python package if not already installed
    if ! python3 -c "import hailo_model_zoo" &>/dev/null; then
        log " -> Installing Hailo Model Zoo Python package..."
        pip install -e "${HAILO_MODEL_ZOO_DIR}"
    fi

    # Compile using hailomz if a config exists, otherwise use the DFC directly
    YAML_PATH=$(find "$HAILO_MODEL_ZOO_DIR" -name "${MODEL_NAME}.yaml" -type f 2>/dev/null | head -1 || true)

    if [[ -n "$YAML_PATH" ]]; then
        log " -> Found Model Zoo config: $YAML_PATH"
        hailomz compile "${MODEL_NAME}" \
            --ckpt "$ONNX_FILE" \
            --hw-arch hailo10h \
            --classes 80

        # hailomz outputs to a nested directory — find and move it
        COMPILED_HEF=$(find . -name "${MODEL_NAME}.hef" -newer "$ONNX_FILE" -type f 2>/dev/null | head -1 || true)
        if [[ -n "$COMPILED_HEF" ]]; then
            mv "$COMPILED_HEF" "$HEF_FILE"
        else
            error_exit "Compilation finished but HEF file not found."
        fi
    else
        log " -> No Model Zoo YAML found for ${MODEL_NAME}. Using Dataflow Compiler directly..."
        python3 - <<PYEOF
from hailo_sdk_client import ClientRunner

runner = ClientRunner(hw_arch="hailo10h")
hn, npz = runner.translate_onnx_model(
    "${ONNX_FILE}",
    "${MODEL_NAME}",
    start_node_names=["images"],
    end_node_names=["output0"],
    net_input_shapes={"images": [1, 3, ${IMGSZ}, ${IMGSZ}]},
)
runner.save_har("${MODEL_DIR}/${MODEL_NAME}.har")
runner.optimize(runner.get_default_optimization_config())
hef = runner.compile()
with open("${HEF_FILE}", "wb") as f:
    f.write(hef)
PYEOF
    fi
    log " -> Compiled: $HEF_FILE"
fi

echo ""
echo "=========================================================="
echo " YOLOv12 setup complete!                                  "
echo "                                                          "
echo " Model: $HEF_FILE                                        "
echo "                                                          "
echo " Run inference:                                           "
echo "   python run_yolo.py --model $HEF_FILE                  "
echo "=========================================================="
