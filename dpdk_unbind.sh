#!/bin/bash
#
# dpdk_unbind.sh
# Safely unbind Mellanox ConnectX-3 Pro from DPDK and restore kernel networking
#

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PCI_ADDR="0000:01:00.0"
KERNEL_DRIVER="mlx4_core"
DPDK_DRIVER="vfio-pci"

echo "=========================================="
echo "DPDK Unbind Script - ConnectX-3 Pro"
echo "=========================================="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}Error: Please run as root (sudo)${NC}"
    exit 1
fi

# Check if NIC exists
if ! lspci -s $PCI_ADDR > /dev/null 2>&1; then
    echo -e "${RED}Error: NIC $PCI_ADDR not found${NC}"
    lspci | grep -i ethernet
    exit 1
fi

echo -e "${GREEN}Found NIC:${NC}"
lspci -s $PCI_ADDR

# Check current status
echo ""
echo "Current status:"
dpdk-devbind.py --status | grep -A2 "Network devices"

# Check if bound to DPDK
if ! dpdk-devbind.py --status | grep -q "$PCI_ADDR.*drv=$DPDK_DRIVER"; then
    echo ""
    echo -e "${YELLOW}NIC is NOT bound to DPDK${NC}"
    
    # Check if already bound to kernel driver
    if dpdk-devbind.py --status | grep -q "$PCI_ADDR.*drv=$KERNEL_DRIVER"; then
        echo -e "${GREEN}Already using kernel driver ($KERNEL_DRIVER)${NC}"
        exit 0
    fi
fi

# Unbind from DPDK
echo ""
echo "Restoring kernel networking..."
echo "  1. Unbinding from DPDK driver ($DPDK_DRIVER)"
echo "  2. Binding to kernel driver ($KERNEL_DRIVER)"
echo "  3. Restarting NetworkManager"
echo ""

# Bind to kernel driver
dpdk-devbind.py --bind=$KERNEL_DRIVER $PCI_ADDR

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Successfully bound to kernel driver!${NC}"
else
    echo -e "${RED}✗ Failed to bind to kernel driver${NC}"
    exit 1
fi

# Restart NetworkManager to bring up interface
echo ""
echo "Restarting NetworkManager..."
systemctl restart NetworkManager

# Wait a moment for interface to come up
sleep 2

# Show new status
echo ""
echo "New status:"
dpdk-devbind.py --status | grep -A2 "Network devices"

# Check if interface is up
echo ""
echo "Network interfaces:"
ip link show | grep -E "^[0-9]+: enp"

echo ""
echo -e "${GREEN}Done!${NC}"
echo ""
echo "Kernel networking restored on enp1s0"
echo ""
echo "To bind back to DPDK:"
echo "  $ sudo ./dpdk_bind.sh"

