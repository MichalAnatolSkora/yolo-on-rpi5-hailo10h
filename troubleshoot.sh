#!/bin/bash

# Hailo-10H diagnostic script for Raspberry Pi 5.
# Checks hardware, driver, firmware, and runtime status.

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PASS=0
WARN=0
FAIL=0

pass() {
    echo -e "  ${GREEN}[PASS]${NC} $1"
    ((PASS++))
}

warn() {
    echo -e "  ${YELLOW}[WARN]${NC} $1"
    ((WARN++))
}

fail() {
    echo -e "  ${RED}[FAIL]${NC} $1"
    ((FAIL++))
}

info() {
    echo -e "  [INFO] $1"
}

echo "=========================================================="
echo " Hailo-10H Diagnostic Report"
echo " $(date)"
echo "=========================================================="
echo ""

# --- 1. Platform ---
echo "--- Platform ---"

ARCH=$(uname -m)
if [[ "$ARCH" == "aarch64" ]]; then
    pass "Architecture: $ARCH"
else
    fail "Architecture: $ARCH (expected aarch64)"
fi

if [[ -f /proc/device-tree/model ]]; then
    MODEL=$(tr -d '\0' < /proc/device-tree/model)
    info "Board: $MODEL"
else
    info "Board: unknown"
fi

KERNEL=$(uname -r)
info "Kernel: $KERNEL"

PYTHON_VER=$(python3 --version 2>/dev/null || echo "not found")
info "Python: $PYTHON_VER"

echo ""

# --- 2. PCIe ---
echo "--- PCIe ---"

PCIE_CONFIG="/boot/firmware/config.txt"
if [[ -f "$PCIE_CONFIG" ]]; then
    if grep -q "^dtparam=pciex1_gen=3" "$PCIE_CONFIG" 2>/dev/null; then
        pass "PCIe Gen 3 enabled in $PCIE_CONFIG"
    elif grep -q "dtparam=pciex1_gen=3" "$PCIE_CONFIG" 2>/dev/null; then
        warn "PCIe Gen 3 line found but may be commented out in $PCIE_CONFIG"
    else
        warn "PCIe Gen 3 not configured in $PCIE_CONFIG (using Gen 2 — lower bandwidth)"
        info "Add 'dtparam=pciex1_gen=3' under [all] and reboot"
    fi
else
    info "Config file not found at $PCIE_CONFIG"
fi

HAILO_PCI=$(lspci 2>/dev/null | grep -i hailo)
if [[ -n "$HAILO_PCI" ]]; then
    pass "Hailo device found on PCIe bus"
    info "$HAILO_PCI"

    # Check link speed
    LINK_SPEED=$(sudo lspci -vv 2>/dev/null | grep -i hailo -A 30 | grep "LnkSta:" | head -1)
    if [[ -n "$LINK_SPEED" ]]; then
        if echo "$LINK_SPEED" | grep -q "8GT/s"; then
            pass "PCIe link speed: Gen 3 (8GT/s)"
        elif echo "$LINK_SPEED" | grep -q "5GT/s"; then
            warn "PCIe link speed: Gen 2 (5GT/s) — Gen 3 recommended for best performance"
        else
            info "PCIe link: $LINK_SPEED"
        fi
    fi
else
    fail "No Hailo device found on PCIe bus"
    info "Check M.2 module seating, ribbon cable, and Hat+ power"
    info "Try: power off (not reboot), reseat hardware, power on"
fi

echo ""

# --- 3. Kernel driver ---
echo "--- Kernel Driver ---"

if lsmod 2>/dev/null | grep -q hailo_pci; then
    pass "hailo_pci kernel module is loaded"
else
    fail "hailo_pci kernel module is NOT loaded"
    info "Try: sudo modprobe hailo_pci"
    info "If that fails, reinstall: sudo apt install --reinstall hailo-all"
fi

if [[ -e /dev/hailo0 ]]; then
    pass "/dev/hailo0 device node exists"

    # Check permissions
    if [[ -r /dev/hailo0 ]] && [[ -w /dev/hailo0 ]]; then
        pass "/dev/hailo0 is readable and writable by current user"
    else
        fail "/dev/hailo0 permission denied for current user ($(whoami))"
        HAILO_GROUP=$(stat -c '%G' /dev/hailo0 2>/dev/null || stat -f '%Sg' /dev/hailo0 2>/dev/null)
        info "Device group: $HAILO_GROUP"
        info "Try: sudo usermod -aG $HAILO_GROUP $(whoami) && logout/login"
    fi
else
    fail "/dev/hailo0 does NOT exist"
    info "The kernel driver did not create the device node"
    info "Check dmesg for errors: dmesg | grep -i hailo"
fi

# Check dmesg for Hailo messages
DMESG_HAILO=$(dmesg 2>/dev/null | grep -i hailo | tail -5)
if [[ -n "$DMESG_HAILO" ]]; then
    info "Recent dmesg messages:"
    while IFS= read -r line; do
        if echo "$line" | grep -qi "error\|fail\|fault"; then
            echo -e "    ${RED}$line${NC}"
        else
            echo "    $line"
        fi
    done <<< "$DMESG_HAILO"
else
    info "No Hailo messages in dmesg (may need sudo: sudo dmesg | grep -i hailo)"
fi

echo ""

# --- 4. Hailo software packages ---
echo "--- Software Packages ---"

if dpkg -s hailo-all &>/dev/null; then
    HAILO_VER=$(dpkg -s hailo-all 2>/dev/null | grep "^Version:" | awk '{print $2}')
    pass "hailo-all is installed (version: ${HAILO_VER:-unknown})"
else
    fail "hailo-all is NOT installed"
    info "Install with: sudo apt install hailo-all"
fi

if command -v hailortcli &>/dev/null; then
    pass "hailortcli is available"
else
    fail "hailortcli not found in PATH"
fi

# Check HailoRT Python bindings
if python3 -c "import hailo_platform" &>/dev/null; then
    pass "hailo_platform Python module is importable"
else
    fail "hailo_platform Python module cannot be imported"
    info "Ensure your venv uses --system-site-packages"
fi

echo ""

# --- 5. Firmware ---
echo "--- Firmware ---"

if [[ -e /dev/hailo0 ]] && command -v hailortcli &>/dev/null; then
    FW_OUTPUT=$(timeout 10 hailortcli fw-control identify 2>&1)
    FW_EXIT=$?
    if [[ $FW_EXIT -eq 0 ]] && [[ -n "$FW_OUTPUT" ]]; then
        pass "Firmware responded"
        # Extract firmware version
        FW_VER=$(echo "$FW_OUTPUT" | grep -i "firmware version" | head -1)
        if [[ -n "$FW_VER" ]]; then
            info "$FW_VER"
        else
            echo "$FW_OUTPUT" | head -5 | while IFS= read -r line; do
                info "$line"
            done
        fi
    elif [[ $FW_EXIT -eq 124 ]]; then
        fail "Firmware identify timed out (10s)"
        info "Device may be hung — try power cycling the Pi"
    else
        fail "Firmware identify failed (exit code: $FW_EXIT)"
        if [[ -n "$FW_OUTPUT" ]]; then
            info "$FW_OUTPUT" | head -3
        fi
        info "Try power cycling (not just rebooting) the Pi"
    fi
else
    if [[ ! -e /dev/hailo0 ]]; then
        fail "Cannot check firmware — /dev/hailo0 missing"
    else
        fail "Cannot check firmware — hailortcli not found"
    fi
fi

echo ""

# --- 6. Device availability ---
echo "--- Device Availability ---"

if [[ -e /dev/hailo0 ]] && python3 -c "import hailo_platform" &>/dev/null; then
    DEVICE_CHECK=$(timeout 10 python3 -c "
from hailo_platform import VDevice, HailoSchedulingAlgorithm
try:
    params = VDevice.create_params()
    params.scheduling_algorithm = HailoSchedulingAlgorithm.NONE
    with VDevice(params) as vd:
        print('OK')
except Exception as e:
    print(f'ERROR: {e}')
" 2>&1)
    if [[ "$DEVICE_CHECK" == "OK" ]]; then
        pass "VDevice opens successfully — ready for inference"
    elif echo "$DEVICE_CHECK" | grep -qi "OUT_OF_PHYSICAL_DEVICES\|status=74"; then
        fail "HAILO_OUT_OF_PHYSICAL_DEVICES — device is busy or unavailable"
        info "Check if another process is using the Hailo device:"
        HAILO_PROCS=$(sudo lsof /dev/hailo0 2>/dev/null | tail -n +2)
        if [[ -n "$HAILO_PROCS" ]]; then
            info "Processes using /dev/hailo0:"
            echo "$HAILO_PROCS" | while IFS= read -r line; do
                echo "    $line"
            done
            info "Stop those processes or reboot to free the device"
        else
            info "No processes found holding /dev/hailo0"
            info "Try: sudo modprobe -r hailo_pci && sudo modprobe hailo_pci"
            info "Or power cycle the Pi"
        fi
    else
        fail "VDevice failed: $DEVICE_CHECK"
    fi
else
    fail "Cannot test VDevice — prerequisites missing (see above)"
fi

echo ""

# --- 7. Model files ---
echo "--- Model Files ---"

MODEL_DIR="${HOME}/hailo_models"
if [[ -d "$MODEL_DIR" ]]; then
    HEF_COUNT=$(find "$MODEL_DIR" -name "*.hef" -type f 2>/dev/null | wc -l)
    if [[ "$HEF_COUNT" -gt 0 ]]; then
        pass "Found $HEF_COUNT HEF model(s) in $MODEL_DIR:"
        find "$MODEL_DIR" -name "*.hef" -type f 2>/dev/null | while IFS= read -r hef; do
            SIZE=$(du -h "$hef" 2>/dev/null | cut -f1)
            echo "    $hef ($SIZE)"
        done
    else
        warn "No .hef models found in $MODEL_DIR"
        info "Run ./install_yolo12.sh to download a model"
    fi
else
    warn "Model directory $MODEL_DIR does not exist"
fi

echo ""

# --- 8. Camera ---
echo "--- Camera ---"

if [[ -e /dev/video0 ]]; then
    pass "Video device /dev/video0 exists"
else
    info "No /dev/video0 — USB camera not connected (optional)"
fi

if command -v rpicam-hello &>/dev/null; then
    pass "rpicam-hello is available (Pi Camera support)"
else
    info "rpicam-hello not found (Pi Camera tools not installed, optional)"
fi

echo ""

# --- Summary ---
echo "=========================================================="
echo -e " Results:  ${GREEN}${PASS} passed${NC}  ${YELLOW}${WARN} warnings${NC}  ${RED}${FAIL} failures${NC}"
echo "=========================================================="

if [[ $FAIL -eq 0 ]]; then
    echo ""
    echo " System looks ready for Hailo-10H inference!"
elif [[ -n "$HAILO_PCI" ]] && [[ ! -e /dev/hailo0 ]]; then
    echo ""
    echo " Hardware detected but driver not working."
    echo " Try: sudo modprobe hailo_pci"
    echo " If that fails: sudo apt install --reinstall hailo-all && sudo reboot"
elif [[ -z "$HAILO_PCI" ]]; then
    echo ""
    echo " Hardware not detected. Check physical connections,"
    echo " then power off and back on (not just reboot)."
fi
