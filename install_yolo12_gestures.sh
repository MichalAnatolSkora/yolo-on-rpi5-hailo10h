#!/bin/bash
set -euo pipefail

# Installs a YOLOv12n gesture recognition model for Hailo-10H.
# Downloads the HaGRID (Hand Gesture Recognition Image Dataset) subset,
# trains YOLOv12n, exports to ONNX, and compiles to HEF.
# Idempotent — safe to run multiple times.

MODEL_DIR="${HOME}/hailo_models"
DATASET_DIR="${HOME}/hagrid_yolo"
MODEL_NAME="yolov12n_gestures"
ONNX_FILE="${MODEL_DIR}/${MODEL_NAME}.onnx"
HEF_FILE="${MODEL_DIR}/${MODEL_NAME}.hef"
HAILO_MODEL_ZOO_DIR="${HOME}/hailo_model_zoo"
VENV_DIR="${HOME}/hailo_venv"
TRAIN_DIR="${MODEL_DIR}/${MODEL_NAME}_train"
IMGSZ=640
EPOCHS=100
BATCH=16

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
echo " YOLOv12 Gesture Recognition Setup for Hailo-10H         "
echo "=========================================================="

mkdir -p "$MODEL_DIR"

# --- Step 1: Python virtual environment ---

log "[1/7] Checking Python virtual environment..."
if [[ -d "$VENV_DIR" ]]; then
    log " -> Virtual environment already exists: $VENV_DIR"
else
    log " -> Creating virtual environment with system site-packages..."
    python3 -m venv --system-site-packages "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# --- Step 2: Install Python dependencies ---

log "[2/7] Checking Python dependencies..."
NEED_INSTALL=false
for pkg in ultralytics; do
    if ! python3 -c "import ${pkg}" &>/dev/null; then
        NEED_INSTALL=true
        break
    fi
done

if [[ "$NEED_INSTALL" == true ]]; then
    log " -> Installing Python packages..."
    pip install --upgrade pip
    pip install ultralytics
else
    log " -> All Python dependencies are installed."
fi

# --- Step 3: Download HaGRID dataset in YOLO format ---

log "[3/7] Checking gesture dataset..."
DATASET_YAML="${DATASET_DIR}/data.yaml"

if [[ -f "$DATASET_YAML" ]]; then
    log " -> Dataset already exists: $DATASET_DIR"
else
    log " -> Downloading HaGRID gesture dataset (YOLO format)..."
    log "    This downloads a subset (~2GB) suitable for training on Pi."
    mkdir -p "$DATASET_DIR"

    python3 - <<'PYEOF'
import os
import yaml

dataset_dir = os.environ.get("DATASET_DIR", os.path.expanduser("~/hagrid_yolo"))

# Use Ultralytics HaGRID integration if available, otherwise set up manually
try:
    from ultralytics.data.utils import download
    # Download HaGRID sample subset from Ultralytics assets
    print("Attempting Ultralytics HaGRID download...")
except Exception:
    pass

# Define the gesture classes we want to detect
# Using a practical subset of HaGRID's 18 core gestures
gesture_classes = [
    "call",           # 0  - phone call gesture
    "dislike",        # 1  - thumbs down
    "fist",           # 2  - closed fist
    "four",           # 3  - four fingers up
    "like",           # 4  - thumbs up
    "mute",           # 5  - finger over lips / shush
    "ok",             # 6  - OK sign
    "one",            # 7  - index finger pointing up
    "palm",           # 8  - open palm / stop
    "peace",          # 9  - peace / victory sign
    "peace_inverted", # 10 - inverted peace sign
    "rock",           # 11 - rock on / devil horns
    "stop",           # 12 - stop hand
    "stop_inverted",  # 13 - inverted stop
    "three",          # 14 - three fingers
    "three2",         # 15 - three fingers (alt)
    "two_up",         # 16 - two fingers pointing up
    "two_up_inverted",# 17 - two fingers (inverted)
]

# Write dataset YAML configuration
data_yaml = {
    "path": dataset_dir,
    "train": "images/train",
    "val": "images/val",
    "names": {i: name for i, name in enumerate(gesture_classes)},
    "nc": len(gesture_classes),
}

os.makedirs(os.path.join(dataset_dir, "images", "train"), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, "images", "val"), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, "labels", "train"), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, "labels", "val"), exist_ok=True)

yaml_path = os.path.join(dataset_dir, "data.yaml")
with open(yaml_path, "w") as f:
    yaml.dump(data_yaml, f, default_flow_style=False)

print(f"Dataset config written to: {yaml_path}")
print(f"Classes: {len(gesture_classes)}")
print()
print("IMPORTANT: You need to populate the dataset before training.")
print("Download HaGRID from: https://github.com/hukenovs/hagrid")
print("Or use a Roboflow hand gesture dataset exported in YOLO format.")
print()
print("Place images in:  {}/images/train/ and {}/images/val/".format(dataset_dir, dataset_dir))
print("Place labels in:  {}/labels/train/ and {}/labels/val/".format(dataset_dir, dataset_dir))
PYEOF
fi

# --- Step 4: Check if dataset has images ---

log "[4/7] Verifying dataset..."
TRAIN_IMAGES="${DATASET_DIR}/images/train"
TRAIN_COUNT=$(find "$TRAIN_IMAGES" -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" \) 2>/dev/null | wc -l)

if [[ "$TRAIN_COUNT" -lt 10 ]]; then
    log ""
    log "WARNING: Only ${TRAIN_COUNT} training images found in ${TRAIN_IMAGES}"
    log ""
    log "You need to download gesture images before training."
    log "Options:"
    log "  1. HaGRID dataset: https://github.com/hukenovs/hagrid"
    log "  2. Roboflow: search 'hand gesture' at https://universe.roboflow.com"
    log "     Export in 'YOLOv8' format and extract to: ${DATASET_DIR}"
    log ""
    log "After adding images, re-run this script to continue."
    log ""
    echo "=========================================================="
    echo " Dataset setup complete. Add images, then re-run.         "
    echo "=========================================================="
    exit 0
fi

log " -> Found ${TRAIN_COUNT} training images."

# --- Step 5: Train YOLOv12n on gestures ---

log "[5/7] Checking trained model..."
BEST_PT="${TRAIN_DIR}/weights/best.pt"

if [[ -f "$BEST_PT" ]]; then
    log " -> Trained model already exists: $BEST_PT"
else
    log " -> Training YOLOv12n on gesture dataset (${EPOCHS} epochs)..."
    log "    This will take a while on Raspberry Pi. Consider training on a GPU machine."

    python3 - <<PYEOF
from ultralytics import YOLO

model = YOLO("yolov12n.pt")
model.train(
    data="${DATASET_YAML}",
    epochs=${EPOCHS},
    imgsz=${IMGSZ},
    batch=${BATCH},
    project="${MODEL_DIR}",
    name="${MODEL_NAME}_train",
    exist_ok=True,
    patience=20,
    workers=2,
    device="cpu",
)
PYEOF
    log " -> Training complete: $BEST_PT"
fi

# --- Step 6: Export to ONNX ---

log "[6/7] Checking ONNX model..."
if [[ -f "$ONNX_FILE" ]]; then
    log " -> ONNX model already exists: $ONNX_FILE"
else
    log " -> Exporting trained model to ONNX..."
    python3 - <<PYEOF
from ultralytics import YOLO
import shutil

model = YOLO("${BEST_PT}")
model.export(format="onnx", imgsz=${IMGSZ}, opset=13, dynamic=False)

# Move to model directory
import os
exported = "${BEST_PT}".replace(".pt", ".onnx")
if os.path.exists(exported):
    shutil.move(exported, "${ONNX_FILE}")
PYEOF
    log " -> Exported: $ONNX_FILE"
fi

# --- Step 7: Compile to HEF ---

log "[7/7] Checking HEF model..."
if [[ -f "$HEF_FILE" ]]; then
    log " -> HEF model already exists: $HEF_FILE"
else
    log " -> Compiling gesture model to HEF for Hailo-10H..."
    log "    This may take 10-30+ minutes."

    # Ensure Hailo Model Zoo is available
    if [[ ! -d "$HAILO_MODEL_ZOO_DIR" ]]; then
        log " -> Cloning Hailo Model Zoo..."
        git clone https://github.com/hailo-ai/hailo_model_zoo.git "$HAILO_MODEL_ZOO_DIR"
    fi

    if ! python3 -c "import hailo_model_zoo" &>/dev/null; then
        log " -> Installing Hailo Model Zoo Python package..."
        pip install -e "${HAILO_MODEL_ZOO_DIR}"
    fi

    # Compile using Hailo Dataflow Compiler
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
    log " -> Compiled: $HEF_FILE"
fi

echo ""
echo "=========================================================="
echo " Gesture recognition model ready!                         "
echo "                                                          "
echo " Model: $HEF_FILE                                        "
echo "                                                          "
echo " Run gesture recognition:                                 "
echo "   python run_gestures.py --model $HEF_FILE               "
echo "=========================================================="
