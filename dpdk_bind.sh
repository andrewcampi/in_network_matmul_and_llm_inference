#!/bin/bash
#
# dpdk_bind.sh
# Safely bind Mellanox ConnectX-3 Pro to DPDK
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
echo "DPDK Bind Script - ConnectX-3 Pro"
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

# Check if already bound to DPDK
if dpdk-devbind.py --status | grep -q "$PCI_ADDR.*drv=$DPDK_DRIVER"; then
    echo ""
    echo -e "${YELLOW}NIC is already bound to DPDK!${NC}"
    exit 0
fi

# Warning
echo ""
echo -e "${YELLOW}⚠️  WARNING ⚠️${NC}"
echo "This will:"
echo "  1. Unbind the ConnectX-3 from the kernel driver ($KERNEL_DRIVER)"
echo "  2. Bind it to DPDK driver ($DPDK_DRIVER)"
echo "  3. Disconnect enp1s0 network interface"
echo ""
echo "Make sure you have:"
echo "  ✓ Alternative network access (enp2s0, console, etc.)"
echo "  ✓ Hugepages configured (see /proc/meminfo)"
echo "  ✓ DPDK installed"
echo ""
read -p "Continue? (yes/no): " response

if [ "$response" != "yes" ]; then
    echo "Aborted."
    exit 0
fi

# Load VFIO module
echo ""
echo "Loading VFIO-PCI module..."
if ! lsmod | grep -q vfio_pci; then
    modprobe vfio-pci
    echo "  ✓ vfio-pci loaded"
else
    echo "  ✓ vfio-pci already loaded"
fi

# Bind to DPDK
echo ""
echo "Binding NIC to DPDK..."
dpdk-devbind.py --bind=$DPDK_DRIVER $PCI_ADDR

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Successfully bound to DPDK!${NC}"
else
    echo -e "${RED}✗ Failed to bind to DPDK${NC}"
    exit 1
fi

# Show new status
echo ""
echo "New status:"
dpdk-devbind.py --status | grep -A2 "Network devices"

echo ""
echo -e "${GREEN}Done!${NC}"
echo ""
echo "To unbind and restore kernel networking:"
echo "  $ sudo ./dpdk_unbind.sh"

