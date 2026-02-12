#!/bin/bash
#
# dpdk_check.sh
# Quick diagnostic to check DPDK installation status
#

echo "=========================================="
echo "DPDK Installation Check"
echo "=========================================="
echo ""

# Check for dpdk-devbind.py
echo "1. dpdk-devbind.py:"
if which dpdk-devbind.py > /dev/null 2>&1; then
    echo "   ✓ Found: $(which dpdk-devbind.py)"
else
    echo "   ✗ Not found in PATH"
fi

# Check for dpdk-testpmd
echo ""
echo "2. dpdk-testpmd:"
if which dpdk-testpmd > /dev/null 2>&1; then
    echo "   ✓ Found: $(which dpdk-testpmd)"
else
    echo "   ✗ Not found in PATH"
fi

# Check for DPDK libraries
echo ""
echo "3. DPDK libraries (pkg-config):"
if pkg-config --exists libdpdk 2>/dev/null; then
    echo "   ✓ Found: version $(pkg-config --modversion libdpdk)"
    echo "   Location: $(pkg-config --variable=prefix libdpdk)"
else
    echo "   ✗ Not found via pkg-config"
fi

# Check for DPDK headers
echo ""
echo "4. DPDK headers:"
if [ -f /usr/include/dpdk/rte_eal.h ]; then
    echo "   ✓ Found: /usr/include/dpdk/rte_eal.h"
elif [ -f /usr/include/rte_eal.h ]; then
    echo "   ✓ Found: /usr/include/rte_eal.h"
elif [ -f /usr/local/include/rte_eal.h ]; then
    echo "   ✓ Found: /usr/local/include/rte_eal.h"
else
    echo "   ✗ Not found"
fi

# Check hugepages
echo ""
echo "5. Hugepages:"
if [ -f /proc/meminfo ]; then
    total=$(grep HugePages_Total /proc/meminfo | awk '{print $2}')
    free=$(grep HugePages_Free /proc/meminfo | awk '{print $2}')
    size=$(grep Hugepagesize /proc/meminfo | awk '{print $2}')
    echo "   Total: $total pages × ${size}kB = $((total * size / 1024))MB"
    echo "   Free:  $free pages"
    if [ "$total" -gt 0 ]; then
        echo "   ✓ Hugepages configured"
    else
        echo "   ✗ No hugepages configured"
    fi
else
    echo "   ✗ Cannot check"
fi

# Check NIC binding
echo ""
echo "6. ConnectX-3 Pro NIC (01:00.0):"
if which dpdk-devbind.py > /dev/null 2>&1; then
    if dpdk-devbind.py --status 2>/dev/null | grep -q "0000:01:00.0.*drv=vfio-pci"; then
        echo "   ✓ Bound to DPDK (vfio-pci)"
    elif dpdk-devbind.py --status 2>/dev/null | grep -q "0000:01:00.0.*drv=mlx4_core"; then
        echo "   ⚠ Bound to kernel driver (mlx4_core)"
    else
        echo "   ? Status unclear"
    fi
else
    echo "   ✗ Cannot check (dpdk-devbind.py not found)"
fi

# Check DPDK packages
echo ""
echo "7. Installed DPDK packages:"
if which dpkg > /dev/null 2>&1; then
    dpkg -l | grep dpdk | awk '{printf "   %s (%s)\n", $2, $3}'
fi

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo ""

# Determine overall status
status="UNKNOWN"
if pkg-config --exists libdpdk 2>/dev/null || [ -f /usr/include/rte_eal.h ]; then
    if [ "$total" -gt 0 ]; then
        if dpdk-devbind.py --status 2>/dev/null | grep -q "0000:01:00.0.*drv=vfio-pci"; then
            status="READY"
        else
            status="PARTIAL (NIC not bound)"
        fi
    else
        status="PARTIAL (no hugepages)"
    fi
else
    status="NOT INSTALLED"
fi

echo "DPDK Status: $status"
echo ""

case $status in
    "READY")
        echo "✓ All prerequisites met!"
        echo "  Ready to run DPDK benchmarks"
        ;;
    "PARTIAL"*)
        echo "⚠ DPDK installed but not fully configured"
        echo "  Run: sudo python3 e092_dpdk_speedtest.py"
        echo "  For setup instructions"
        ;;
    "NOT INSTALLED")
        echo "✗ DPDK not installed"
        echo "  Install: sudo apt install dpdk dpdk-dev"
        ;;
esac

