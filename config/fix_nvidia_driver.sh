#!/bin/bash
# Fix NVIDIA Driver Issues
# Date: October 31, 2025
# Handles: Driver not loaded, version mismatch, configuration issues

echo "=========================================="
echo "NVIDIA Driver Diagnostic & Fix"
echo "=========================================="
echo ""

# Check if nvidia-smi exists
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. NVIDIA drivers may not be installed."
    exit 1
fi

echo "Step 1: Diagnosing the issue..."
echo ""

# Check current nvidia-smi status
echo "Testing nvidia-smi:"
if nvidia-smi &> /dev/null; then
    echo "✓ nvidia-smi is working fine!"
    nvidia-smi
    exit 0
else
    NVIDIA_ERROR=$(nvidia-smi 2>&1)
    echo "✗ nvidia-smi failed with:"
    echo "$NVIDIA_ERROR"
fi

echo ""
echo "Step 2: Checking NVIDIA kernel modules..."
MODULES_LOADED=$(lsmod | grep nvidia)
if [ -z "$MODULES_LOADED" ]; then
    echo "✗ No NVIDIA kernel modules are loaded"
    ISSUE="modules_not_loaded"
else
    echo "✓ NVIDIA modules are loaded:"
    echo "$MODULES_LOADED"
    ISSUE="version_mismatch"
fi

echo ""
echo "Step 3: Checking NVIDIA driver installation..."
DRIVER_STATUS=$(dpkg -l | grep nvidia-driver | head -1)
echo "$DRIVER_STATUS"

if echo "$DRIVER_STATUS" | grep -q "^iU"; then
    echo "⚠ NVIDIA driver is installed but NOT CONFIGURED (status: iU)"
    echo "  This usually happens after a kernel update or incomplete installation"
    ISSUE="driver_not_configured"
elif echo "$DRIVER_STATUS" | grep -q "^ii"; then
    echo "✓ NVIDIA driver package is properly installed (status: ii)"
fi

echo ""
echo "=========================================="
echo "Detected Issue: $ISSUE"
echo "=========================================="
echo ""

case $ISSUE in
    "driver_not_configured")
        echo "SOLUTION: The NVIDIA driver needs to be reconfigured."
        echo ""
        echo "This requires running:"
        echo "  sudo dpkg --configure -a"
        echo "  sudo apt install --reinstall nvidia-driver-570-open"
        echo ""
        read -p "Do you want to attempt this fix? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Reconfiguring packages..."
            sudo dpkg --configure -a
            echo ""
            echo "Reinstalling NVIDIA driver..."
            sudo apt install --reinstall nvidia-driver-570-open -y
            echo ""
            echo "Loading NVIDIA modules..."
            sudo modprobe nvidia
            sudo modprobe nvidia_modeset 2>/dev/null || true
            sudo modprobe nvidia_drm 2>/dev/null || true
            sudo modprobe nvidia_uvm 2>/dev/null || true
        else
            echo "Aborted. Please fix manually or reboot the system."
            exit 0
        fi
        ;;
        
    "modules_not_loaded")
        echo "SOLUTION: The NVIDIA kernel modules need to be loaded."
        echo ""
        read -p "Do you want to load the NVIDIA modules? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Loading NVIDIA kernel modules..."
            sudo modprobe nvidia
            sudo modprobe nvidia_modeset 2>/dev/null || true
            sudo modprobe nvidia_drm 2>/dev/null || true
            sudo modprobe nvidia_uvm 2>/dev/null || true
        else
            echo "Aborted."
            exit 0
        fi
        ;;
        
    "version_mismatch")
        echo "SOLUTION: Reload NVIDIA kernel modules to fix version mismatch."
        echo ""
        read -p "Do you want to reload the modules? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Unloading NVIDIA kernel modules..."
            sudo rmmod nvidia_uvm 2>/dev/null || true
            sudo rmmod nvidia_drm 2>/dev/null || true
            sudo rmmod nvidia_modeset 2>/dev/null || true
            sudo rmmod nvidia 2>/dev/null || true
            echo "Reloading NVIDIA kernel modules..."
            sudo modprobe nvidia
            sudo modprobe nvidia_modeset 2>/dev/null || true
            sudo modprobe nvidia_drm 2>/dev/null || true
            sudo modprobe nvidia_uvm 2>/dev/null || true
        else
            echo "Aborted."
            exit 0
        fi
        ;;
esac

echo ""
echo "=========================================="
echo "Verifying Fix..."
echo "=========================================="
if nvidia-smi &> /dev/null; then
    echo "✓ SUCCESS! NVIDIA driver is now working correctly."
    echo ""
    nvidia-smi
    echo ""
    echo "You can now run: ./setup_conda_env.sh"
else
    echo "✗ FAILED: Driver issue persists."
    echo ""
    echo "Alternative solutions:"
    echo "1. REBOOT the system (recommended - will reload all kernel modules properly)"
    echo "2. Check kernel compatibility:"
    echo "   uname -r  # Your kernel version"
    echo "   modinfo nvidia | grep vermagic  # Driver kernel version"
    echo "3. Reinstall NVIDIA drivers completely:"
    echo "   sudo apt purge nvidia-* -y"
    echo "   sudo apt install nvidia-driver-570-open -y"
    echo "   sudo reboot"
    echo "4. Use CPU-only version (run setup_conda_env.sh, it will auto-detect and use CPU)"
    echo ""
    exit 1
fi
