#!/bin/bash
set -euo pipefail

CONFIG_FILE="/boot/firmware/config.txt"
PCIE_PARAM="dtparam=pciex1_gen=3"

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

echo "=========================================================="
echo " Hailo-10H and YOLO Setup on Raspberry Pi 5              "
echo "=========================================================="

CHANGES_MADE=false

# --- Step 1: Update system (skip if updated within the last hour) ---

log "[1/4] Checking system packages..."
APT_LISTS_DIR="/var/lib/apt/lists"
LAST_UPDATE=$(stat -c %Y "$APT_LISTS_DIR" 2>/dev/null || echo 0)
NOW=$(date +%s)
SECONDS_SINCE_UPDATE=$(( NOW - LAST_UPDATE ))

if [[ $SECONDS_SINCE_UPDATE -gt 3600 ]]; then
    log " -> Package lists are stale ($(( SECONDS_SINCE_UPDATE / 3600 ))h old). Updating..."
    sudo apt-get update
    sudo apt-get full-upgrade -y
    CHANGES_MADE=true
else
    log " -> Package lists are fresh (updated $(( SECONDS_SINCE_UPDATE / 60 ))m ago). Skipping update."
fi

# --- Step 2: Install Hailo software suite (skip if already installed) ---

log "[2/4] Checking Hailo software suite..."
if dpkg -s hailo-all &>/dev/null; then
    log " -> hailo-all is already installed."
else
    log " -> Installing hailo-all..."
    sudo apt-get install hailo-all -y
    CHANGES_MADE=true
fi

# --- Step 3: Enable PCIe Gen 3 ---

log "[3/4] Checking PCIe Gen 3 configuration..."
if [[ ! -f "$CONFIG_FILE" ]]; then
    error_exit "Boot config not found at $CONFIG_FILE. Is this Raspberry Pi OS Bookworm?"
fi

if grep -q "^${PCIE_PARAM}" "$CONFIG_FILE"; then
    log " -> PCIe Gen 3 is already enabled."
else
    if grep -q "^#.*${PCIE_PARAM}" "$CONFIG_FILE"; then
        sudo sed -i "s/^#.*${PCIE_PARAM}/${PCIE_PARAM}/" "$CONFIG_FILE"
        log " -> Uncommented existing PCIe Gen 3 entry."
    else
        echo "${PCIE_PARAM}" | sudo tee -a "$CONFIG_FILE" > /dev/null
        log " -> Added PCIe Gen 3 to $CONFIG_FILE."
    fi
    CHANGES_MADE=true
fi

# --- Step 4: Create model directory ---

log "[4/4] Checking model directory..."
MODEL_DIR="${HOME}/hailo_models"
if [[ -d "$MODEL_DIR" ]]; then
    log " -> Model directory already exists: $MODEL_DIR"
else
    mkdir -p "$MODEL_DIR"
    log " -> Created model directory: $MODEL_DIR"
    CHANGES_MADE=true
fi

# --- Summary ---

echo ""
echo "=========================================================="
if [[ "$CHANGES_MADE" == true ]]; then
    echo " Setup complete — changes were applied.                   "
    echo "                                                          "
    echo " Next steps:                                              "
    echo "  1. Reboot:  sudo reboot                                 "
    echo "  2. Verify:  hailortcli fw-control identify              "
    echo "  3. Place .hef models in: $MODEL_DIR                     "
else
    echo " Everything is already configured. No changes needed.     "
fi
echo "=========================================================="
