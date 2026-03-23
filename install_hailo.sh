#!/bin/bash
set -e

echo "=========================================================="
echo " Automating Hailo-10H and YOLO Setup on Raspberry Pi 5 "
echo "=========================================================="

echo "[1/4] Updating System Packages..."
sudo apt-get update
sudo apt-get full-upgrade -y

echo "[2/4] Installing dependencies and Hailo Software Suite..."
# hailo-all installs the driver, hailort, and rpicam-apps integrations.
sudo apt-get install hailo-all -y

echo "[3/4] Enabling PCIe Gen 3 for maximum bandwidth..."
CONFIG_FILE="/boot/firmware/config.txt"

if grep -q "^dtparam=pciex1_gen=3" "$CONFIG_FILE"; then
    echo " -> PCIe Gen 3 is already enabled in $CONFIG_FILE."
else
    echo " -> Adding 'dtparam=pciex1_gen=3' to $CONFIG_FILE"
    # Ensure it's added cleanly at the end or under [all]
    echo "dtparam=pciex1_gen=3" | sudo tee -a "$CONFIG_FILE" > /dev/null
fi

echo "[4/4] Finalizing Configurations..."
# Create a directory to store models for convenience
HOME_DIR=$(eval echo ~$USER)
MODEL_DIR="$HOME_DIR/hailo_models"
mkdir -p "$MODEL_DIR"
echo " -> Created model directory at $MODEL_DIR"

echo "=========================================================="
echo " Installation Complete! "
echo " Please acquire YOLO .hef models specifically compiled "
echo " for Hailo-10H and place them in $MODEL_DIR."
echo " "
echo " IMPORTANT: You MUST REBOOT your Raspberry Pi now."
echo " Run: sudo reboot"
echo "=========================================================="
