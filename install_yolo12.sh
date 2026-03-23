#!/bin/bash
set -euo pipefail

# Installs Ultralytics, exports YOLOv12n to ONNX, and compiles it to a Hailo-10H
# HEF model. Idempotent — safe to run multiple times.

MODEL_DIR="${HOME}/hailo_models"
MODEL_NAME="yolov12n"
ONNX_FILE="${MODEL_DIR}/${MODEL_NAME}.onnx"
HEF_FILE="${MODEL_DIR}/${MODEL_NAME}.hef"
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

log "[1/4] Checking Python virtual environment..."
if [[ -d "$VENV_DIR" ]]; then
    log " -> Virtual environment already exists: $VENV_DIR"
else
    log " -> Creating virtual environment with system site-packages..."
    python3 -m venv --system-site-packages "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# --- Step 2: Install Ultralytics ---

log "[2/4] Checking Ultralytics..."
if python3 -c "import ultralytics" &>/dev/null; then
    log " -> Ultralytics is already installed."
else
    log " -> Installing Ultralytics..."
    pip install --upgrade pip
    pip install ultralytics
fi

# --- Step 3: Export YOLOv12n to ONNX ---

log "[3/4] Checking ONNX model..."
PT_FILE="${MODEL_DIR}/${MODEL_NAME}.pt"
if [[ -f "$ONNX_FILE" ]]; then
    log " -> ONNX model already exists: $ONNX_FILE"
else
    # Download weights if not cached
    if [[ ! -f "$PT_FILE" ]]; then
        log " -> Downloading ${MODEL_NAME} weights..."
        python3 - <<DLEOF
import os
from urllib.request import urlretrieve

pt_path = "${PT_FILE}"
# YOLOv12 weights are hosted on GitHub releases
url = "https://github.com/sunsmarterjie/yolov12/releases/download/v1.0/${MODEL_NAME}.pt"
print(f"Downloading {url}")
urlretrieve(url, pt_path)
print(f"Saved to {pt_path}")
DLEOF
        if [[ ! -f "$PT_FILE" ]]; then
            error_exit "Failed to download ${MODEL_NAME}.pt weights."
        fi
    fi

    log " -> Exporting ${MODEL_NAME} to ONNX (imgsz=${IMGSZ})..."
    python3 - <<PYEOF
import os, shutil
from ultralytics import YOLO

pt_path = "${PT_FILE}"
model = YOLO(pt_path)
model.export(format="onnx", imgsz=${IMGSZ}, opset=13, dynamic=False)

# The exported ONNX lands next to the .pt file
onnx_candidates = [
    pt_path.replace(".pt", ".onnx"),
    "${MODEL_NAME}.onnx",
]
for candidate in onnx_candidates:
    if os.path.isfile(candidate):
        shutil.move(candidate, "${ONNX_FILE}")
        break
PYEOF
    if [[ ! -f "$ONNX_FILE" ]]; then
        error_exit "ONNX export failed. Check the output above."
    fi
    log " -> Exported: $ONNX_FILE"
fi

# --- Step 4: Compile ONNX to HEF ---

log "[4/4] Checking HEF model..."
if [[ -f "$HEF_FILE" ]]; then
    log " -> HEF model already exists: $HEF_FILE"
else
    log " -> Compiling ${MODEL_NAME} ONNX to HEF for Hailo-10H..."
    log "    This may take a while (10-30+ minutes)."

    # Use Hailo Dataflow Compiler directly (hailo_model_zoo Python package
    # requires Python <3.13 due to numba, so we avoid it entirely).
    log " -> Compiling with Hailo Dataflow Compiler..."

    # Prepare calibration images for INT8 quantization.
    # The Hailo DFC needs representative images to measure activation ranges.
    CALIB_DIR="${MODEL_DIR}/calibration_${MODEL_NAME}"
    if [[ ! -d "$CALIB_DIR" ]] || [[ $(find "$CALIB_DIR" -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" \) 2>/dev/null | wc -l) -lt 8 ]]; then
        log " -> Generating calibration images from COCO validation set..."
        mkdir -p "$CALIB_DIR"
        python3 - <<CALPYEOF
import os
import urllib.request
import zipfile

calib_dir = "${CALIB_DIR}"
zip_url = "https://ultralytics.com/assets/coco128.zip"
zip_path = os.path.join(calib_dir, "coco128.zip")

if not os.path.exists(zip_path):
    print("Downloading COCO128 calibration subset...")
    urllib.request.urlretrieve(zip_url, zip_path)

print("Extracting calibration images...")
with zipfile.ZipFile(zip_path, "r") as zf:
    extracted = 0
    for member in zf.namelist():
        if member.endswith((".jpg", ".png", ".jpeg")) and extracted < 64:
            data = zf.read(member)
            filename = os.path.basename(member)
            with open(os.path.join(calib_dir, filename), "wb") as f:
                f.write(data)
            extracted += 1
    print(f"Extracted {extracted} calibration images to {calib_dir}")

os.remove(zip_path)
CALPYEOF
    else
        log " -> Calibration images already exist: $CALIB_DIR"
    fi

    python3 - <<PYEOF
import glob
import numpy as np
from PIL import Image
from hailo_sdk_client import ClientRunner

IMGSZ = ${IMGSZ}
CALIB_DIR = "${CALIB_DIR}"
NUM_CALIB = 64

# --- Translate ONNX to Hailo representation ---
runner = ClientRunner(hw_arch="hailo10h")
hn, npz = runner.translate_onnx_model(
    "${ONNX_FILE}",
    "${MODEL_NAME}",
    start_node_names=["images"],
    end_node_names=["output0"],
    net_input_shapes={"images": [1, 3, IMGSZ, IMGSZ]},
)
runner.save_har("${MODEL_DIR}/${MODEL_NAME}_translated.har")

# Add normalization to map 0-255 inputs to 0.0-1.0 internally
runner.load_model_script("normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])\\n")

# --- Build calibration dataset ---
calib_images = sorted(glob.glob(f"{CALIB_DIR}/*.jpg") + glob.glob(f"{CALIB_DIR}/*.png"))
if len(calib_images) == 0:
    raise RuntimeError(f"No calibration images found in {CALIB_DIR}")

calib_images = calib_images[:NUM_CALIB]
print(f"Using {len(calib_images)} calibration images for INT8 quantization")

calib_dataset = np.zeros((len(calib_images), IMGSZ, IMGSZ, 3), dtype=np.float32)
for i, img_path in enumerate(calib_images):
    img = Image.open(img_path).convert("RGB").resize((IMGSZ, IMGSZ))
    calib_dataset[i] = np.array(img, dtype=np.float32)

# --- Optimize with calibration data ---
print("Running INT8 quantization with calibration data (this may take a while)...")
runner.optimize(runner.get_default_optimization_config(), calib_dataset)
runner.save_har("${MODEL_DIR}/${MODEL_NAME}_quantized.har")

# --- Compile to HEF ---
print("Compiling to HEF...")
hef = runner.compile()
with open("${HEF_FILE}", "wb") as f:
    f.write(hef)
print(f"Done: ${HEF_FILE}")
PYEOF
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
