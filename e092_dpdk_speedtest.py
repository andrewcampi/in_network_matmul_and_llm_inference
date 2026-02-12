#!/usr/bin/env python3
"""
e092_dpdk_speedtest.py

DPDK PACKET SENDING SPEEDTEST - ConnectX-3 Pro
===============================================

GOAL: Measure ACTUAL speedup with DPDK (not guessing 20×!)

HARDWARE:
  - Mellanox ConnectX-3 Pro (DPDK supported!)
  - 40 Gbps capable
  - Line rate: 59.5M pps @ 64-byte packets

CURRENT PERFORMANCE:
  - Regular sockets: ~650K pps (1.1% of line rate)
  - Bottleneck: Linux kernel networking stack
  
TARGET WITH DPDK:
  - Bypass kernel completely
  - Direct NIC access via PCIe DMA
  - Theoretical: 10-50M pps
  - Let's measure the ACTUAL number!

EXPERIMENT STRUCTURE:
  Part 1: Setup & Prerequisites
  Part 2: Baseline (regular sockets)
  Part 3: DPDK (kernel bypass)
  Part 4: Comparison & Analysis

DPDK COMPONENTS:
  - EAL (Environment Abstraction Layer): Init
  - PMD (Poll Mode Driver): Zero-copy packet I/O
  - Mempool: Pre-allocated packet buffers
  - Port: Direct NIC control

INSTALLATION:
  See instructions in main() for DPDK setup

USAGE:
  $ sudo python3 e092_dpdk_speedtest.py

Author: Research Phase 001
Date: January 2026
"""

import os
import sys
import time
import socket
import subprocess
from typing import List, Tuple, Optional

# =============================================================================
# CONFIGURATION
# =============================================================================

# Test parameters
NUM_PACKETS = 100_000  # Test with 100K packets for good statistics
PACKET_SIZE = 64       # Minimum Ethernet frame
NUM_RUNS = 3

# Network config
IFACE = "enp1s0"  # Adjust to your ConnectX-3 interface
PCI_ADDR = "01:00.0"  # From lspci (Mellanox ConnectX-3 Pro)

# =============================================================================
# DPDK SETUP & PREREQUISITES
# =============================================================================

def check_dpdk_installed() -> bool:
    """Check if DPDK is installed."""
    # Try multiple ways to detect DPDK
    
    # Method 1: Check for dpdk-testpmd
    try:
        result = subprocess.run(
            ["dpdk-testpmd", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print(f"  ✓ DPDK installed: dpdk-testpmd found")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    # Method 2: Check for dpdk-devbind.py (more common)
    try:
        result = subprocess.run(
            ["dpdk-devbind.py", "--help"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print(f"  ✓ DPDK installed: dpdk-devbind.py found")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    # Method 3: Check for DPDK libraries
    try:
        result = subprocess.run(
            ["pkg-config", "--exists", "libdpdk"],
            capture_output=True,
            timeout=5
        )
        if result.returncode == 0:
            version = subprocess.run(
                ["pkg-config", "--modversion", "libdpdk"],
                capture_output=True,
                text=True,
                timeout=5
            )
            print(f"  ✓ DPDK installed: libdpdk {version.stdout.strip()}")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    # Method 4: Check for DPDK headers
    if os.path.exists("/usr/include/rte_eal.h") or os.path.exists("/usr/local/include/rte_eal.h"):
        print(f"  ✓ DPDK installed: headers found")
        return True
    
    print(f"  ✗ DPDK not found (no binaries, libs, or headers)")
    return False


def check_hugepages() -> Tuple[bool, int]:
    """Check if hugepages are configured."""
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if "HugePages_Total" in line:
                    total = int(line.split()[1])
                    if total > 0:
                        print(f"  ✓ Hugepages configured: {total} pages")
                        return True, total
    except:
        pass
    
    print(f"  ✗ Hugepages not configured")
    return False, 0


def check_nic_bound() -> bool:
    """Check if NIC is bound to DPDK driver."""
    try:
        result = subprocess.run(
            ["dpdk-devbind.py", "--status"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if PCI_ADDR in result.stdout:
            # Accept vfio-pci, uio_pci_generic, OR mlx4_core/mlx5_core (Mellanox native drivers)
            if any(drv in result.stdout for drv in ["drv=uio", "drv=vfio", "drv=mlx4_core", "drv=mlx5_core"]):
                print(f"  ✓ NIC bound to DPDK driver")
                return True
            else:
                print(f"  ⚠ NIC found but not bound to DPDK driver")
                return False
    except:
        pass
    
    print(f"  ✗ Could not check NIC binding")
    return False


def print_dpdk_setup_instructions():
    """Print instructions for DPDK setup."""
    print(f"\n{'='*80}")
    print("DPDK SETUP INSTRUCTIONS")
    print(f"{'='*80}")
    print("""
DPDK is not fully configured. Here's how to set it up:

1. INSTALL DPDK:
   $ sudo apt update
   $ sudo apt install dpdk dpdk-dev python3-pyelftools
   
2. CONFIGURE HUGEPAGES (2GB = 1024 × 2MB pages):
   $ sudo sh -c 'echo 1024 > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages'
   $ mkdir -p /dev/hugepages
   $ sudo mount -t hugetlbfs nodev /dev/hugepages
   
3. BIND NIC TO DPDK DRIVER (choose VFIO or UIO):
   
   Option A - VFIO (recommended, needs IOMMU):
   $ sudo modprobe vfio-pci
   $ sudo dpdk-devbind.py --bind=vfio-pci 01:00.0
   
   Option B - UIO (if VFIO doesn't work):
   $ sudo modprobe uio_pci_generic
   $ sudo dpdk-devbind.py --bind=uio_pci_generic 01:00.0
   
4. VERIFY SETUP:
   $ dpdk-devbind.py --status
   
   Should show ConnectX-3 bound to vfio-pci or uio_pci_generic

5. TO UNBIND (restore normal networking):
   $ sudo dpdk-devbind.py --bind=mlx4_core 01:00.0
   $ sudo systemctl restart NetworkManager

⚠️  WARNING: Binding to DPDK will disconnect network on that interface!
    Make sure you have alternative access (SSH via different NIC, console, etc.)

After setup, re-run this script!
""")


# =============================================================================
# BASELINE: REGULAR SOCKET SENDING
# =============================================================================

def generate_test_packets(count: int) -> List[bytes]:
    """Generate simple test packets."""
    # Simple Ethernet frame
    dst_mac = b'\xff\xff\xff\xff\xff\xff'  # Broadcast
    src_mac = b'\x00\x00\x00\x00\x00\x00'  # Dummy
    eth_type = b'\x08\x00'  # IPv4
    payload = b'\x00' * (PACKET_SIZE - 14)  # Padding to 64 bytes
    
    packet = dst_mac + src_mac + eth_type + payload
    return [packet] * count


def measure_baseline_performance(packets: List[bytes]) -> Tuple[float, float]:
    """Measure baseline socket.send() performance."""
    print(f"\n{'─'*60}")
    print("BASELINE: Regular socket.send()")
    print(f"{'─'*60}")
    
    # Check if interface exists
    try:
        with open(f"/sys/class/net/{IFACE}/operstate", "r") as f:
            pass
    except FileNotFoundError:
        print(f"  ⚠️  Interface {IFACE} not available")
        print(f"  (NIC is bound to DPDK - interface doesn't exist)")
        print(f"\n  Using baseline from e090: ~650K pps")
        print(f"  (This is the typical kernel networking stack performance)")
        
        # Return estimated baseline from e090 results
        estimated_pps = 650_000
        estimated_time = len(packets) / estimated_pps
        
        print(f"\n  Estimated: {estimated_time*1000:.1f}ms = {estimated_pps/1000:.0f}K pps")
        return estimated_time, estimated_pps
    
    sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.ntohs(0x0003))
    sock.bind((IFACE, 0))
    
    # Optimize socket
    BUFFER_SIZE = 32 * 1024 * 1024
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, BUFFER_SIZE)
    
    times = []
    
    for run in range(NUM_RUNS):
        start = time.time()
        for packet in packets:
            sock.send(packet)
        elapsed = time.time() - start
        times.append(elapsed)
        
        pps = len(packets) / elapsed
        print(f"  Run {run+1}/{NUM_RUNS}: {elapsed*1000:.1f}ms ({pps/1000:.0f}K pps)")
    
    sock.close()
    
    best_time = min(times)
    best_pps = len(packets) / best_time
    
    print(f"\n  Best: {best_time*1000:.1f}ms = {best_pps/1000:.0f}K pps")
    
    return best_time, best_pps


# =============================================================================
# DPDK: KERNEL BYPASS SENDING
# =============================================================================

def measure_dpdk_performance(packets: List[bytes]) -> Optional[Tuple[float, float]]:
    """Measure DPDK performance using Python bindings."""
    print(f"\n{'─'*60}")
    print("DPDK: Kernel bypass (if available)")
    print(f"{'─'*60}")
    
    # Check if we can import DPDK Python bindings
    try:
        # Try to import dpdk module
        # Note: This requires python3-pyelftools and proper DPDK Python bindings
        print("  Attempting to use DPDK Python bindings...")
        print("  (This is experimental - DPDK primarily uses C)")
        
        # Since DPDK Python bindings are limited, we'll document the approach
        print("\n  ⚠️  Native DPDK requires C/C++ code")
        print("  Python bindings are experimental and limited")
        
        return None
        
    except ImportError:
        print("  ✗ DPDK Python bindings not available")
        return None


def run_dpdk_c_benchmark() -> Optional[Tuple[float, float]]:
    """
    Run DPDK benchmark using compiled C program.
    
    This is the PROPER way to use DPDK - it's a C library!
    """
    print(f"\n{'─'*60}")
    print("DPDK: Running C benchmark")
    print(f"{'─'*60}")
    
    # Check if we have the compiled DPDK test program
    dpdk_program = "./dpdk_packet_sender"
    
    if not os.path.exists(dpdk_program):
        print(f"  ⚠️  DPDK C program not found: {dpdk_program}")
        print(f"  This script will create the C code for you!")
        create_dpdk_c_program()
        return None
    
    try:
        # Run the DPDK program
        print(f"  Running: {dpdk_program} {NUM_PACKETS}")
        result = subprocess.run(
            [dpdk_program, str(NUM_PACKETS)],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            # Parse output
            for line in result.stdout.split('\n'):
                if "pps" in line.lower():
                    print(f"  {line}")
            
            # Extract timing info
            # (Would need to parse from C program output)
            return None
        else:
            print(f"  ✗ DPDK program failed:")
            print(result.stderr)
            return None
            
    except subprocess.TimeoutExpired:
        print(f"  ✗ DPDK program timed out")
        return None
    except Exception as e:
        print(f"  ✗ Error running DPDK program: {e}")
        return None


def create_dpdk_c_program():
    """Create a simple DPDK C program for packet sending."""
    print(f"\n{'='*80}")
    print("CREATING DPDK C PROGRAM")
    print(f"{'='*80}")
    
    c_code = """
/*
 * dpdk_packet_sender.c
 * Simple DPDK packet sender for benchmarking
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_mbuf.h>
#include <rte_cycles.h>

#define PACKET_SIZE 64
#define BURST_SIZE 32
#define MEMPOOL_CACHE_SIZE 256
#define NUM_MBUFS 8191

static struct rte_mempool *mbuf_pool = NULL;

static void create_test_packet(struct rte_mbuf *mbuf) {
    uint8_t *data = rte_pktmbuf_mtod(mbuf, uint8_t *);
    
    // Ethernet header (dummy)
    memset(data, 0xFF, 6);      // Dst MAC (broadcast)
    memset(data + 6, 0x00, 6);  // Src MAC
    data[12] = 0x08;             // EtherType
    data[13] = 0x00;             // IPv4
    
    // Padding
    memset(data + 14, 0x00, PACKET_SIZE - 14);
    
    mbuf->data_len = PACKET_SIZE;
    mbuf->pkt_len = PACKET_SIZE;
}

int main(int argc, char *argv[]) {
    int ret;
    uint16_t port_id = 0;
    uint32_t num_packets;
    uint64_t total_sent = 0;
    uint64_t start_tsc, end_tsc;
    double elapsed_sec, pps, gbps;
    
    if (argc < 2) {
        printf("Usage: %s <num_packets>\\n", argv[0]);
        return 1;
    }
    
    num_packets = atoi(argv[1]);
    
    // Initialize DPDK EAL
    ret = rte_eal_init(argc, argv);
    if (ret < 0) {
        fprintf(stderr, "Error: EAL initialization failed\\n");
        return 1;
    }
    
    // Create mbuf pool
    mbuf_pool = rte_pktmbuf_pool_create(
        "MBUF_POOL",
        NUM_MBUFS,
        MEMPOOL_CACHE_SIZE,
        0,
        RTE_MBUF_DEFAULT_BUF_SIZE,
        rte_socket_id()
    );
    
    if (mbuf_pool == NULL) {
        fprintf(stderr, "Error: Cannot create mbuf pool\\n");
        return 1;
    }
    
    // Get device info to check capabilities
    struct rte_eth_dev_info dev_info;
    ret = rte_eth_dev_info_get(port_id, &dev_info);
    if (ret != 0) {
        fprintf(stderr, "Error: Cannot get device info\\n");
        return 1;
    }
    
    // Configure port - TX only for mlx4 compatibility (no RX = no flow rules!)
    struct rte_eth_conf port_conf = {0};
    // Disable all multiqueue modes
    port_conf.rxmode.mq_mode = RTE_ETH_MQ_RX_NONE;
    port_conf.txmode.mq_mode = RTE_ETH_MQ_TX_NONE;
    // Disable all offloads
    port_conf.rxmode.offloads = 0;
    port_conf.txmode.offloads = 0;
    
    // Configure with 0 RX queues, 1 TX queue (TX-only mode!)
    ret = rte_eth_dev_configure(port_id, 0, 1, &port_conf);
    if (ret < 0) {
        fprintf(stderr, "Error: Cannot configure device (code %d)\\n", ret);
        return 1;
    }
    
    // NO RX queue setup - this avoids flow rule issues!
    
    // Setup TX queue with minimal config
    struct rte_eth_txconf txconf = {0};
    txconf.offloads = 0;  // No offloads
    ret = rte_eth_tx_queue_setup(port_id, 0, 512, rte_eth_dev_socket_id(port_id), &txconf);
    if (ret < 0) {
        fprintf(stderr, "Error: Cannot setup TX queue (code %d)\\n", ret);
        return 1;
    }
    
    // Start device
    ret = rte_eth_dev_start(port_id);
    if (ret < 0) {
        fprintf(stderr, "Error: Cannot start device (code %d)\\n", ret);
        return 1;
    }
    
    // Enable promiscuous mode (often needed for mlx4)
    ret = rte_eth_promiscuous_enable(port_id);
    if (ret != 0) {
        fprintf(stderr, "Warning: Cannot enable promiscuous mode (code %d)\\n", ret);
        // Don't fail - continue anyway
    }
    
    printf("DPDK initialized successfully!\\n");
    printf("Sending %u packets...\\n", num_packets);
    
    // Allocate packet burst
    struct rte_mbuf *bufs[BURST_SIZE];
    
    start_tsc = rte_rdtsc();
    
    // Send packets
    while (total_sent < num_packets) {
        uint16_t nb_tx = (num_packets - total_sent) < BURST_SIZE ? 
                         (num_packets - total_sent) : BURST_SIZE;
        
        // Allocate mbufs
        if (rte_pktmbuf_alloc_bulk(mbuf_pool, bufs, nb_tx) != 0) {
            fprintf(stderr, "Error: Failed to allocate mbufs\\n");
            break;
        }
        
        // Fill packets
        for (int i = 0; i < nb_tx; i++) {
            create_test_packet(bufs[i]);
        }
        
        // Send burst
        uint16_t nb_sent = rte_eth_tx_burst(port_id, 0, bufs, nb_tx);
        
        // Free unsent packets
        if (nb_sent < nb_tx) {
            for (int i = nb_sent; i < nb_tx; i++) {
                rte_pktmbuf_free(bufs[i]);
            }
        }
        
        total_sent += nb_sent;
    }
    
    end_tsc = rte_rdtsc();
    
    // Calculate performance
    uint64_t tsc_hz = rte_get_tsc_hz();
    elapsed_sec = (double)(end_tsc - start_tsc) / tsc_hz;
    pps = total_sent / elapsed_sec;
    gbps = (pps * PACKET_SIZE * 8) / 1e9;
    
    printf("\\nResults:\\n");
    printf("  Packets sent: %"PRIu64"\\n", total_sent);
    printf("  Time:         %.3f ms\\n", elapsed_sec * 1000);
    printf("  PPS:          %.0f (%.1fM pps)\\n", pps, pps / 1e6);
    printf("  Throughput:   %.2f Gbps\\n", gbps);
    
    // Cleanup
    rte_eth_dev_stop(port_id);
    rte_eal_cleanup();
    
    return 0;
}
"""
    
    # Write C code
    with open("dpdk_packet_sender.c", "w") as f:
        f.write(c_code)
    
    print("  ✓ Created: dpdk_packet_sender.c")
    
    # Write Makefile
    makefile = """
# Makefile for DPDK packet sender (DPDK 23.x)
CC = gcc
CFLAGS = -O3 -Wall -I/usr/include/dpdk
LDFLAGS = -lpthread -lnuma

# DPDK flags from pkg-config (provides all libraries)
PKG_CONFIG_PATH ?= /usr/lib/x86_64-linux-gnu/pkgconfig
CFLAGS += $(shell pkg-config --cflags libdpdk 2>/dev/null || echo "")
LDFLAGS += $(shell pkg-config --libs libdpdk 2>/dev/null || echo "")

TARGET = dpdk_packet_sender

all: $(TARGET)

$(TARGET): dpdk_packet_sender.c
\t$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
\trm -f $(TARGET)

.PHONY: all clean
"""
    
    with open("Makefile.dpdk", "w") as f:
        f.write(makefile)
    
    print("  ✓ Created: Makefile.dpdk")
    
    print(f"\n  To compile:")
    print(f"    $ make -f Makefile.dpdk")
    print(f"\n  To run:")
    print(f"    $ sudo ./dpdk_packet_sender {NUM_PACKETS}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"\n{'='*80}")
    print("E092: DPDK SPEEDTEST - Mellanox ConnectX-3 Pro")
    print(f"{'='*80}")
    print(f"""
GOAL: Measure ACTUAL DPDK speedup (not guessing!)

Hardware:   Mellanox ConnectX-3 Pro @ {PCI_ADDR}
Interface:  {IFACE}
Capability: 40 Gbps (59.5M pps theoretical)

Test:       {NUM_PACKETS:,} packets × {PACKET_SIZE} bytes
Baseline:   ~650K pps (kernel networking stack)
Target:     10-50M pps (kernel bypass via DPDK)

Let's find out the REAL number! 🚀
""")
    
    # Check prerequisites
    print(f"{'='*80}")
    print("CHECKING PREREQUISITES")
    print(f"{'='*80}")
    
    dpdk_installed = check_dpdk_installed()
    hugepages_ok, _ = check_hugepages()
    nic_bound = check_nic_bound()
    
    if not (dpdk_installed and hugepages_ok and nic_bound):
        print(f"\n  ⚠️  DPDK is not fully configured")
        print_dpdk_setup_instructions()
        
        print(f"\n  However, we can still run the BASELINE test!")
        response = input(f"\n  Run baseline test only? (y/n): ")
        if response.lower() != 'y':
            print(f"\n  Exiting. Run again after DPDK setup!")
            return
        
        dpdk_available = False
    else:
        print(f"\n  ✓ All prerequisites met!")
        dpdk_available = True
    
    # Generate test packets
    print(f"\n{'='*80}")
    print("GENERATING TEST PACKETS")
    print(f"{'='*80}")
    print(f"  Generating {NUM_PACKETS:,} test packets...")
    packets = generate_test_packets(NUM_PACKETS)
    print(f"  ✓ Generated {len(packets):,} packets")
    
    # Run baseline
    print(f"\n{'='*80}")
    print("BENCHMARK 1: BASELINE")
    print(f"{'='*80}")
    baseline_time, baseline_pps = measure_baseline_performance(packets)
    
    # Run DPDK (if available)
    if dpdk_available:
        print(f"\n{'='*80}")
        print("BENCHMARK 2: DPDK")
        print(f"{'='*80}")
        
        dpdk_result = run_dpdk_c_benchmark()
        
        if dpdk_result is None:
            print(f"\n  DPDK benchmark requires C program")
            print(f"  See dpdk_packet_sender.c (created above)")
    else:
        print(f"\n{'='*80}")
        print("BENCHMARK 2: DPDK (SKIPPED)")
        print(f"{'='*80}")
        print(f"  DPDK not configured - see instructions above")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    baseline_gbps = (baseline_pps * PACKET_SIZE * 8) / 1e9
    baseline_utilization = (baseline_pps / 59.5e6) * 100
    
    print(f"\n  BASELINE (Linux kernel sockets):")
    print(f"    Packets/sec:  {baseline_pps:,.0f} ({baseline_pps/1e6:.2f}M)")
    print(f"    Throughput:   {baseline_gbps:.2f} Gbps")
    print(f"    Line rate:    {baseline_utilization:.2f}% of 40G")
    
    print(f"\n  DPDK (kernel bypass):")
    print(f"    Status:       {'Configured - compile and run C program!' if dpdk_available else 'Not configured - see setup above'}")
    
    print(f"\n{'='*80}")
    print("NEXT STEPS")
    print(f"{'='*80}")
    
    if dpdk_available:
        print("""
  1. Compile DPDK program:
     $ make -f Makefile.dpdk
  
  2. Run DPDK benchmark:
     $ sudo ./dpdk_packet_sender 100000
  
  3. Compare with baseline!
""")
    else:
        print("""
  1. Follow DPDK setup instructions above
  2. Rerun this script to verify setup
  3. Compile and run DPDK benchmark
  4. Measure the REAL speedup!
""")


if __name__ == "__main__":
    if os.geteuid() != 0:
        print("This script requires root privileges.")
        print("Please run with: sudo python3 e092_dpdk_speedtest.py")
        sys.exit(1)
    
    main()

