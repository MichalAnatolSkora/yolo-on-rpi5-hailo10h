#!/bin/bash

# Hailo-10H diagnostic script for Raspberry Pi 5.
# Checks hardware, driver, firmware, and runtime status.
# Offers to run fix commands interactively.

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m' # No Color

PASS=0
WARN=0
FAIL=0

# Collect suggested fixes to offer at the end
FIXES=()
FIX_CMDS=()

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

# Register a fix: description and command
add_fix() {
    FIXES+=("$1")
    FIX_CMDS+=("$2")
}

# Prompt user to run a command
ask_and_run() {
    local description="$1"
    local cmd="$2"
    echo ""
    echo -e "  ${BOLD}Fix:${NC} $description"
    echo -e "  ${BOLD}Command:${NC} $cmd"
    echo -n "  Run this command? [y/N] "
    read -r answer
    if [[ "$answer" =~ ^[Yy]$ ]]; then
        echo "  Running: $cmd"
        eval "$cmd"
        local exit_code=$?
        if [[ $exit_code -eq 0 ]]; then
            echo -e "  ${GREEN}Done.${NC}"
        else
            echo -e "  ${RED}Command exited with code $exit_code${NC}"
        fi
        return $exit_code
    else
        echo "  Skipped."
        return 1
    fi
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
        add_fix "Enable PCIe Gen 3 in boot config (requires reboot)" \
                "sudo sh -c 'echo \"dtparam=pciex1_gen=3\" >> $PCIE_CONFIG'"
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
    info "Power off (not reboot), reseat hardware, power on"
fi

echo ""

# --- 3. Kernel driver ---
echo "--- Kernel Driver ---"

# Determine which Hailo device is present: Hailo-8 uses hailo_pci, Hailo-10H uses hailo1x_pci
HAILO_PCI_ID=$(lspci -nn 2>/dev/null | grep -i hailo | grep -o '\[[0-9a-fA-F]\{4\}:[0-9a-fA-F]\{4\}\]' | head -1 | tr -d '[]')
HAILO_BDF=$(lspci 2>/dev/null | grep -i hailo | awk '{print $1}' | head -1)
if [[ -n "$HAILO_PCI_ID" ]]; then
    VENDOR_ID=$(echo "$HAILO_PCI_ID" | cut -d: -f1)
    DEVICE_ID=$(echo "$HAILO_PCI_ID" | cut -d: -f2)
fi

# Detect Hailo-10H (device ID 45c4) vs Hailo-8 (device ID 2864)
IS_HAILO10H=false
if [[ "$DEVICE_ID" == "45c4" ]]; then
    IS_HAILO10H=true
    EXPECTED_DRIVER="hailo1x_pci"
    EXPECTED_PKG="hailo-h10-all"
    WRONG_DRIVER="hailo_pci"
    WRONG_PKG="hailo-all"
    info "Detected Hailo-10H (PCI ID $VENDOR_ID:$DEVICE_ID)"
else
    EXPECTED_DRIVER="hailo_pci"
    EXPECTED_PKG="hailo-all"
    WRONG_DRIVER="hailo1x_pci"
    WRONG_PKG="hailo-h10-all"
    info "Detected Hailo-8 (PCI ID ${VENDOR_ID:-unknown}:${DEVICE_ID:-unknown})"
fi

# Check for wrong driver conflict (e.g. hailo_pci loaded for Hailo-10H)
WRONG_DRIVER_LOADED=false
if lsmod 2>/dev/null | grep -q "^${WRONG_DRIVER} "; then
    WRONG_DRIVER_LOADED=true
    fail "Wrong driver loaded: $WRONG_DRIVER (this device needs $EXPECTED_DRIVER)"
    info "The $WRONG_DRIVER driver is for $(if $IS_HAILO10H; then echo 'Hailo-8'; else echo 'Hailo-10H'; fi), not your device."
    add_fix "Blacklist wrong driver ($WRONG_DRIVER) and install correct package ($EXPECTED_PKG)" \
            "echo 'blacklist $WRONG_DRIVER' | sudo tee /etc/modprobe.d/hailo-blacklist.conf >/dev/null && sudo modprobe -r $WRONG_DRIVER 2>/dev/null; sudo apt update && sudo apt install -y $EXPECTED_PKG"
fi

# Check if expected driver is loaded
DRIVER_LOADED=false
if lsmod 2>/dev/null | grep -q "^${EXPECTED_DRIVER} "; then
    pass "$EXPECTED_DRIVER kernel module is loaded"
    DRIVER_LOADED=true
else
    if [[ "$WRONG_DRIVER_LOADED" == false ]]; then
        fail "$EXPECTED_DRIVER kernel module is NOT loaded"
    fi

    # Check if the correct package is installed
    if dpkg -s "$EXPECTED_PKG" &>/dev/null; then
        add_fix "Load $EXPECTED_DRIVER kernel module" \
                "sudo modprobe $EXPECTED_DRIVER || (echo 'Failed to load. Reinstalling driver...' && sudo apt install --reinstall -y $EXPECTED_PKG && sudo modprobe $EXPECTED_DRIVER)"
    else
        fail "Package $EXPECTED_PKG is NOT installed (required for your device)"
        if dpkg -s "$WRONG_PKG" &>/dev/null; then
            warn "Package $WRONG_PKG is installed instead — this is for $(if $IS_HAILO10H; then echo 'Hailo-8'; else echo 'Hailo-10H'; fi)"
        fi
        add_fix "Install correct Hailo package ($EXPECTED_PKG)" \
                "sudo apt update && sudo apt install -y $EXPECTED_PKG && echo 'blacklist $WRONG_DRIVER' | sudo tee /etc/modprobe.d/hailo-blacklist.conf >/dev/null"
    fi
fi

# DKMS status check
if command -v dkms &>/dev/null; then
    DKMS_STATUS=$(dkms status 2>/dev/null | grep -i hailo)
    if [[ -n "$DKMS_STATUS" ]]; then
        info "DKMS: $DKMS_STATUS"
    fi
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
        add_fix "Add current user to $HAILO_GROUP group (requires logout/login)" \
                "sudo usermod -aG $HAILO_GROUP $(whoami)"
    fi
else
    fail "/dev/hailo0 does NOT exist"
    if [[ "$DRIVER_LOADED" == true ]]; then
        info "Driver $EXPECTED_DRIVER is loaded but /dev/hailo0 was not created"
        if [[ -n "$HAILO_BDF" ]] && [[ -d "/sys/bus/pci/drivers/$EXPECTED_DRIVER" ]]; then
            if [[ ! -e "/sys/bus/pci/drivers/$EXPECTED_DRIVER/$HAILO_BDF" ]]; then
                add_fix "Manually bind PCIe device $HAILO_BDF to $EXPECTED_DRIVER" \
                        "echo '$HAILO_BDF' | sudo tee /sys/bus/pci/drivers/$EXPECTED_DRIVER/bind >/dev/null"
            fi
        fi
    elif [[ "$WRONG_DRIVER_LOADED" == true ]] || ! dpkg -s "$EXPECTED_PKG" &>/dev/null; then
        info "Install $EXPECTED_PKG and reboot to create the device node (see fixes above)"
    else
        info "Load the driver first (see fixes above)"
    fi
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

if dpkg -s "$EXPECTED_PKG" &>/dev/null; then
    HAILO_VER=$(dpkg -s "$EXPECTED_PKG" 2>/dev/null | grep "^Version:" | awk '{print $2}')
    pass "$EXPECTED_PKG is installed (version: ${HAILO_VER:-unknown})"
else
    fail "$EXPECTED_PKG is NOT installed"
    add_fix "Install $EXPECTED_PKG package" \
            "sudo apt update && sudo apt install -y $EXPECTED_PKG"
fi
# Warn if the wrong meta-package is installed
if dpkg -s "$WRONG_PKG" &>/dev/null; then
    WRONG_VER=$(dpkg -s "$WRONG_PKG" 2>/dev/null | grep "^Version:" | awk '{print $2}')
    warn "$WRONG_PKG is installed (version: ${WRONG_VER:-unknown}) — this is for $(if $IS_HAILO10H; then echo 'Hailo-8'; else echo 'Hailo-10H'; fi)"
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
    # Check if it exists in system but venv blocks it
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        info "You are in a virtual environment: $VIRTUAL_ENV"
        add_fix "Recreate venv with system site-packages" \
                "python3 -m venv --system-site-packages \"$VIRTUAL_ENV\" --clear"
    else
        info "Ensure hailo-all is installed (provides hailo_platform)"
    fi
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
        info "Device may be hung"
        add_fix "Reload Hailo kernel driver" \
                "sudo modprobe -r $EXPECTED_DRIVER && sudo modprobe $EXPECTED_DRIVER"
    else
        fail "Firmware identify failed (exit code: $FW_EXIT)"
        if [[ -n "$FW_OUTPUT" ]]; then
            echo "$FW_OUTPUT" | head -3 | while IFS= read -r line; do
                info "$line"
            done
        fi
        add_fix "Reload Hailo kernel driver" \
                "sudo modprobe -r $EXPECTED_DRIVER && sudo modprobe $EXPECTED_DRIVER"
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
        info "Checking for processes using /dev/hailo0..."
        HAILO_PROCS=$(sudo lsof /dev/hailo0 2>/dev/null | tail -n +2)
        if [[ -n "$HAILO_PROCS" ]]; then
            info "Processes using /dev/hailo0:"
            echo "$HAILO_PROCS" | while IFS= read -r line; do
                echo "    $line"
            done
            # Extract PIDs
            HAILO_PIDS=$(echo "$HAILO_PROCS" | awk '{print $2}' | sort -u | tr '\n' ' ')
            add_fix "Kill processes holding Hailo device (PIDs: $HAILO_PIDS)" \
                    "sudo kill $HAILO_PIDS"
        else
            info "No processes found holding /dev/hailo0"
            add_fix "Reload Hailo kernel driver to reset device" \
                    "sudo modprobe -r $EXPECTED_DRIVER && sudo modprobe $EXPECTED_DRIVER"
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
DETECTED_ARCH=$(if $IS_HAILO10H; then echo "hailo10h"; else echo "hailo8"; fi)
if [[ -d "$MODEL_DIR" ]]; then
    HEF_COUNT=$(find "$MODEL_DIR" -name "*.hef" -type f 2>/dev/null | wc -l)
    if [[ "$HEF_COUNT" -gt 0 ]]; then
        pass "Found $HEF_COUNT HEF model(s) in $MODEL_DIR:"
        find "$MODEL_DIR" -name "*.hef" -type f 2>/dev/null | while IFS= read -r hef; do
            SIZE=$(du -h "$hef" 2>/dev/null | cut -f1)
            echo "    $hef ($SIZE)"
        done
        # Check HEF architecture compatibility
        if python3 -c "import hailo_platform" &>/dev/null; then
            find "$MODEL_DIR" -name "*.hef" -type f 2>/dev/null | while IFS= read -r hef; do
                HEF_CHECK=$(python3 -c "
from hailo_platform import HEF
try:
    h = HEF('$hef')
    # Try to get target device from HEF metadata
    print('ok')
except Exception as e:
    if 'HAILO_NOT_IMPLEMENTED' in str(e) or 'error: 7' in str(e):
        print('wrong_arch')
    else:
        print('ok')
" 2>&1)
                if [[ "$HEF_CHECK" == "wrong_arch" ]]; then
                    warn "$(basename "$hef") may be compiled for the wrong architecture (not $DETECTED_ARCH)"
                    add_fix "Re-download $(basename "$hef") for $DETECTED_ARCH" \
                            "./install_yolo11.sh"
                fi
            done
        fi
    else
        warn "No .hef models found in $MODEL_DIR"
        add_fix "Download YOLO model for $DETECTED_ARCH" \
                "./install_yolo11.sh"
    fi
else
    warn "Model directory $MODEL_DIR does not exist"
    add_fix "Download YOLO model for $DETECTED_ARCH" \
            "./install_yolo11.sh"
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

if [[ $FAIL -eq 0 ]] && [[ ${#FIXES[@]} -eq 0 ]]; then
    echo ""
    echo " System looks ready for Hailo-10H inference!"
    exit 0
fi

# --- Offer fixes ---
if [[ ${#FIXES[@]} -gt 0 ]]; then
    echo ""
    echo -e "${BOLD}Found ${#FIXES[@]} issue(s) that can be fixed automatically:${NC}"

    for i in "${!FIXES[@]}"; do
        echo ""
        echo -e "  ${BOLD}$((i+1)). ${FIXES[$i]}${NC}"
        echo -e "     Command: ${FIX_CMDS[$i]}"
    done

    echo ""
    echo -n "Apply all fixes? [y/N/select] (enter number to apply one, 'y' for all): "
    read -r answer

    if [[ "$answer" =~ ^[Yy]$ ]]; then
        NEED_REBOOT=false
        for i in "${!FIXES[@]}"; do
            echo ""
            echo -e "  ${BOLD}[$((i+1))/${#FIXES[@]}] ${FIXES[$i]}${NC}"
            echo "  Running: ${FIX_CMDS[$i]}"
            FIX_OUTPUT=$(eval "${FIX_CMDS[$i]}" 2>&1)
            exit_code=$?
            echo "$FIX_OUTPUT"
            if [[ $exit_code -eq 0 ]]; then
                echo -e "  ${GREEN}Done.${NC}"
            else
                echo -e "  ${RED}Failed (exit code: $exit_code)${NC}"
            fi
            # Check if this fix likely needs a reboot
            if echo "${FIX_CMDS[$i]}" | grep -q "config.txt\|usermod\|hailort-pcie-driver"; then
                NEED_REBOOT=true
            fi
            if echo "$FIX_OUTPUT" | grep -qi "reboot"; then
                NEED_REBOOT=true
            fi
        done
        if [[ "$NEED_REBOOT" == true ]]; then
            echo ""
            echo -n "Some changes require a reboot. Reboot now? [y/N] "
            read -r reboot_answer
            if [[ "$reboot_answer" =~ ^[Yy]$ ]]; then
                echo "Rebooting..."
                sudo reboot
            else
                echo "Remember to reboot for changes to take effect."
            fi
        fi
    elif [[ "$answer" =~ ^[0-9]+$ ]]; then
        idx=$((answer - 1))
        if [[ $idx -ge 0 ]] && [[ $idx -lt ${#FIXES[@]} ]]; then
            echo ""
            echo -e "  ${BOLD}${FIXES[$idx]}${NC}"
            echo "  Running: ${FIX_CMDS[$idx]}"
            FIX_OUTPUT=$(eval "${FIX_CMDS[$idx]}" 2>&1)
            exit_code=$?
            echo "$FIX_OUTPUT"
            if [[ $exit_code -eq 0 ]]; then
                echo -e "  ${GREEN}Done.${NC}"
            else
                echo -e "  ${RED}Failed (exit code: $exit_code)${NC}"
            fi
            if echo "${FIX_CMDS[$idx]}" | grep -q "config.txt\|usermod\|hailort-pcie-driver" || echo "$FIX_OUTPUT" | grep -qi "reboot"; then
                echo ""
                echo "This change requires a reboot to take effect."
                echo -n "Reboot now? [y/N] "
                read -r reboot_answer
                if [[ "$reboot_answer" =~ ^[Yy]$ ]]; then
                    sudo reboot
                fi
            fi
        else
            echo "  Invalid number."
        fi
    else
        echo "  No fixes applied."
    fi

    # Suggest re-running after fixes
    echo ""
    echo "Run ./troubleshoot.sh again to verify fixes."
fi
