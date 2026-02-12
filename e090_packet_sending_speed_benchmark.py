#!/usr/bin/env python3
"""
e090_packet_sending_speed_benchmark.py

ULTIMATE PACKET SENDING SPEED BENCHMARK
========================================

GOAL:
  Find the FASTEST way to send packets from Ubuntu host to switch!
  
  Current: 9-28ms for 2,045 packets = 220K pps
  Target:  Line rate 40Gbps = 59M pps (64-byte packets)
  Realistic: 5-10M pps sustained

THE BOTTLENECK:
  The switches are BLAZING FAST (40Gbps, sub-ms processing)
  The host is SLOW (28ms to send 2K packets = 0.4% of line rate!)
  
  We need to SATURATE the 40G link!

METHODS TO TEST:
  1. Baseline socket.send() loop
  2. sendmmsg() batch
  3. sendmmsg() with larger batches
  4. Pre-allocated packet buffers (zero-copy)
  5. Multiple threads sending in parallel
  6. Raw socket with various optimizations
  7. Burst sending (no delays)
  8. Memory-mapped packet ring (PACKET_MMAP)
  9. Combined: Best of all methods!

METRICS:
  - Packets per second (pps)
  - Throughput (Gbps)
  - Latency (ms)
  - CPU usage
  - Packet loss rate

SUCCESS CRITERIA:
  - Achieve >1M pps (4× current)
  - Utilize >10% of 40G bandwidth
  - Identify bottleneck (CPU? Kernel? NIC?)

USAGE:
  $ sudo python3 e090_packet_sending_speed_benchmark.py

Author: Research Phase 001
Date: January 2026
"""

import os
import sys
import time
import socket
import struct
import threading
import multiprocessing
import ctypes
import ctypes.util
import numpy as np
from typing import List, Dict, Tuple, Callable
from dataclasses import dataclass
from collections import defaultdict

# Import from previous experiments
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from e042_port_based_layers import (
    craft_vlan_packet, get_mac_address, SEND_IFACE
)
from e053_mac_encoded_layers import get_layer_neuron_mac
from e045_real_weights_inference import mac_str_to_bytes

# =============================================================================
# CONFIGURATION
# =============================================================================

TEST_VLAN = 200
NUM_PACKETS = 10_000  # Test with 10K packets for good statistics
PACKET_SIZE = 64      # Minimum Ethernet frame
NUM_RUNS = 3          # Average over 3 runs

# =============================================================================
# PACKET GENERATION (Pre-create all test packets)
# =============================================================================

def generate_test_packets(count: int) -> List[bytes]:
    """Generate test packets (pre-create for fair comparison)."""
    print(f"  Generating {count:,} test packets...")
    start = time.time()
    
    src_mac = get_mac_address(SEND_IFACE)
    src = mac_str_to_bytes(src_mac)
    
    packets = []
    for i in range(count):
        # Use different destination MACs to simulate real inference
        neuron_id = i % 128
        dst_mac = get_layer_neuron_mac(0, neuron_id)
        dst = mac_str_to_bytes(dst_mac)
        packet = craft_vlan_packet(dst, src, TEST_VLAN)
        packets.append(packet)
    
    elapsed = time.time() - start
    print(f"  ✓ Generated {count:,} packets in {elapsed*1000:.1f}ms")
    
    return packets


# =============================================================================
# METHOD 1: BASELINE - Standard socket.send() loop
# =============================================================================

def method_baseline(packets: List[bytes]) -> Tuple[float, int]:
    """Baseline: Standard socket.send() loop (from e088)."""
    sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.ntohs(0x0003))
    sock.bind((SEND_IFACE, 0))
    
    start = time.time()
    sent = 0
    for packet in packets:
        sock.send(packet)
        sent += 1
    elapsed = time.time() - start
    
    sock.close()
    return elapsed, sent


# =============================================================================
# METHOD 2: sendmmsg() batch (from e089)
# =============================================================================

libc = ctypes.CDLL(ctypes.util.find_library('c'), use_errno=True)

class iovec(ctypes.Structure):
    _fields_ = [('iov_base', ctypes.c_void_p), ('iov_len', ctypes.c_size_t)]

class msghdr(ctypes.Structure):
    _fields_ = [
        ('msg_name', ctypes.c_void_p), ('msg_namelen', ctypes.c_uint),
        ('msg_iov', ctypes.POINTER(iovec)), ('msg_iovlen', ctypes.c_size_t),
        ('msg_control', ctypes.c_void_p), ('msg_controllen', ctypes.c_size_t),
        ('msg_flags', ctypes.c_int),
    ]

class mmsghdr(ctypes.Structure):
    _fields_ = [('msg_hdr', msghdr), ('msg_len', ctypes.c_uint)]

libc.sendmmsg.argtypes = [ctypes.c_int, ctypes.POINTER(mmsghdr), ctypes.c_uint, ctypes.c_int]
libc.sendmmsg.restype = ctypes.c_int


def method_sendmmsg(packets: List[bytes], batch_size: int = 1024) -> Tuple[float, int]:
    """sendmmsg() with configurable batch size."""
    sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.ntohs(0x0003))
    sock.bind((SEND_IFACE, 0))
    
    sockfd = sock.fileno()
    total_sent = 0
    
    start = time.time()
    
    offset = 0
    while offset < len(packets):
        chunk_size = min(batch_size, len(packets) - offset)
        chunk = packets[offset:offset + chunk_size]
        
        # Prepare structures
        iovecs = (iovec * chunk_size)()
        packet_bufs = []
        
        for i, packet in enumerate(chunk):
            buf = ctypes.create_string_buffer(packet)
            packet_bufs.append(buf)
            iovecs[i].iov_base = ctypes.cast(buf, ctypes.c_void_p)
            iovecs[i].iov_len = len(packet)
        
        mmsghdrs = (mmsghdr * chunk_size)()
        for i in range(chunk_size):
            mmsghdrs[i].msg_hdr.msg_name = None
            mmsghdrs[i].msg_hdr.msg_namelen = 0
            mmsghdrs[i].msg_hdr.msg_iov = ctypes.pointer(iovecs[i])
            mmsghdrs[i].msg_hdr.msg_iovlen = 1
            mmsghdrs[i].msg_hdr.msg_control = None
            mmsghdrs[i].msg_hdr.msg_controllen = 0
            mmsghdrs[i].msg_hdr.msg_flags = 0
            mmsghdrs[i].msg_len = 0
        
        sent = libc.sendmmsg(sockfd, mmsghdrs, chunk_size, 0)
        if sent > 0:
            total_sent += sent
        
        offset += chunk_size
    
    elapsed = time.time() - start
    sock.close()
    
    return elapsed, total_sent


# =============================================================================
# METHOD 3: Pre-allocated buffers (zero-copy attempt)
# =============================================================================

def method_preallocated(packets: List[bytes]) -> Tuple[float, int]:
    """Pre-allocate packet buffers to avoid repeated allocation."""
    sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.ntohs(0x0003))
    sock.bind((SEND_IFACE, 0))
    
    # Pre-allocate all buffers
    buffers = [ctypes.create_string_buffer(p) for p in packets]
    
    start = time.time()
    sent = 0
    for buf in buffers:
        sock.send(buf)
        sent += 1
    elapsed = time.time() - start
    
    sock.close()
    return elapsed, sent


# =============================================================================
# METHOD 4: Burst sending (max throughput, no delays)
# =============================================================================

def method_burst(packets: List[bytes]) -> Tuple[float, int]:
    """Send packets as fast as possible with optimized socket."""
    sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.ntohs(0x0003))
    sock.bind((SEND_IFACE, 0))
    
    # Optimize socket
    BUFFER_SIZE = 32 * 1024 * 1024  # 32MB buffer
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, BUFFER_SIZE)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_PRIORITY, 7)  # Max priority
    
    # Try to bypass qdisc
    try:
        SOL_PACKET = 263
        PACKET_QDISC_BYPASS = 6
        sock.setsockopt(SOL_PACKET, PACKET_QDISC_BYPASS, 1)
    except:
        pass
    
    start = time.time()
    sent = 0
    for packet in packets:
        sock.send(packet)
        sent += 1
    elapsed = time.time() - start
    
    sock.close()
    return elapsed, sent


# =============================================================================
# METHOD 5: Multi-threaded sending
# =============================================================================

def _thread_sender(packets: List[bytes], result_queue):
    """Worker thread that sends packets."""
    try:
        elapsed, sent = method_baseline(packets)
        result_queue.put((elapsed, sent))
    except Exception as e:
        result_queue.put((0, 0))


def method_multithreaded(packets: List[bytes], num_threads: int = 2) -> Tuple[float, int]:
    """Split packets across multiple threads (SHARED socket - contention!)."""
    import queue
    
    # Split packets among threads
    chunk_size = len(packets) // num_threads
    chunks = [packets[i*chunk_size:(i+1)*chunk_size] for i in range(num_threads)]
    
    # Add remainder to last chunk
    if len(packets) % num_threads != 0:
        chunks[-1].extend(packets[num_threads * chunk_size:])
    
    result_queue = queue.Queue()
    threads = []
    
    start = time.time()
    
    for chunk in chunks:
        t = threading.Thread(target=_thread_sender, args=(chunk, result_queue))
        t.start()
        threads.append(t)
    
    for t in threads:
        t.join()
    
    elapsed = time.time() - start
    
    # Collect results
    total_sent = 0
    while not result_queue.empty():
        _, sent = result_queue.get()
        total_sent += sent
    
    return elapsed, total_sent


def _thread_sender_separate_socket(packets: List[bytes], result_queue):
    """Worker thread with SEPARATE socket (no contention!)."""
    try:
        # Each thread gets its own socket!
        sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.ntohs(0x0003))
        sock.bind((SEND_IFACE, 0))
        
        # Optimize
        BUFFER_SIZE = 16 * 1024 * 1024
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, BUFFER_SIZE)
        
        start = time.time()
        sent = 0
        for packet in packets:
            sock.send(packet)
            sent += 1
        elapsed = time.time() - start
        
        sock.close()
        result_queue.put((elapsed, sent))
    except Exception as e:
        result_queue.put((0, 0))


def method_multithreaded_separate_sockets(packets: List[bytes], num_threads: int = 4) -> Tuple[float, int]:
    """Each thread gets its OWN socket (avoids contention!)."""
    import queue
    
    # Split packets among threads
    chunk_size = len(packets) // num_threads
    chunks = [packets[i*chunk_size:(i+1)*chunk_size] for i in range(num_threads)]
    
    if len(packets) % num_threads != 0:
        chunks[-1].extend(packets[num_threads * chunk_size:])
    
    result_queue = queue.Queue()
    threads = []
    
    start = time.time()
    
    for chunk in chunks:
        t = threading.Thread(target=_thread_sender_separate_socket, args=(chunk, result_queue))
        t.start()
        threads.append(t)
    
    for t in threads:
        t.join()
    
    elapsed = time.time() - start
    
    total_sent = 0
    while not result_queue.empty():
        _, sent = result_queue.get()
        total_sent += sent
    
    return elapsed, total_sent


def _process_sender(packets: List[bytes], result_dict, proc_id: int):
    """Worker process (bypasses GIL!)."""
    try:
        sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.ntohs(0x0003))
        sock.bind((SEND_IFACE, 0))
        
        BUFFER_SIZE = 16 * 1024 * 1024
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, BUFFER_SIZE)
        
        start = time.time()
        sent = 0
        for packet in packets:
            sock.send(packet)
            sent += 1
        elapsed = time.time() - start
        
        sock.close()
        result_dict[proc_id] = (elapsed, sent)
    except Exception as e:
        result_dict[proc_id] = (0, 0)


def method_multiprocess(packets: List[bytes], num_procs: int = 4) -> Tuple[float, int]:
    """Multi-PROCESS (bypasses Python GIL!)."""
    # Split packets among processes
    chunk_size = len(packets) // num_procs
    chunks = [packets[i*chunk_size:(i+1)*chunk_size] for i in range(num_procs)]
    
    if len(packets) % num_procs != 0:
        chunks[-1].extend(packets[num_procs * chunk_size:])
    
    manager = multiprocessing.Manager()
    result_dict = manager.dict()
    processes = []
    
    start = time.time()
    
    for i, chunk in enumerate(chunks):
        p = multiprocessing.Process(target=_process_sender, args=(chunk, result_dict, i))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    elapsed = time.time() - start
    
    total_sent = sum(sent for _, sent in result_dict.values())
    
    return elapsed, total_sent


# =============================================================================
# METHOD 6: sendmmsg() with HUGE batches
# =============================================================================

def method_sendmmsg_large(packets: List[bytes]) -> Tuple[float, int]:
    """sendmmsg() with maximum batch size (1024)."""
    return method_sendmmsg(packets, batch_size=1024)


def method_sendmmsg_small(packets: List[bytes]) -> Tuple[float, int]:
    """sendmmsg() with smaller batches (256) for comparison."""
    return method_sendmmsg(packets, batch_size=256)


# =============================================================================
# METHOD 7: Combined best practices
# =============================================================================

def method_combined(packets: List[bytes]) -> Tuple[float, int]:
    """Combine all optimizations: sendmmsg + burst + large buffers."""
    sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.ntohs(0x0003))
    sock.bind((SEND_IFACE, 0))
    
    # ALL optimizations
    BUFFER_SIZE = 64 * 1024 * 1024  # 64MB buffer!
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, BUFFER_SIZE)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, BUFFER_SIZE)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_PRIORITY, 7)
    
    try:
        SOL_PACKET = 263
        PACKET_QDISC_BYPASS = 6
        sock.setsockopt(SOL_PACKET, PACKET_QDISC_BYPASS, 1)
    except:
        pass
    
    sockfd = sock.fileno()
    total_sent = 0
    batch_size = 1024
    
    start = time.time()
    
    offset = 0
    while offset < len(packets):
        chunk_size = min(batch_size, len(packets) - offset)
        chunk = packets[offset:offset + chunk_size]
        
        iovecs = (iovec * chunk_size)()
        packet_bufs = []
        
        for i, packet in enumerate(chunk):
            buf = ctypes.create_string_buffer(packet)
            packet_bufs.append(buf)
            iovecs[i].iov_base = ctypes.cast(buf, ctypes.c_void_p)
            iovecs[i].iov_len = len(packet)
        
        mmsghdrs = (mmsghdr * chunk_size)()
        for i in range(chunk_size):
            mmsghdrs[i].msg_hdr.msg_name = None
            mmsghdrs[i].msg_hdr.msg_namelen = 0
            mmsghdrs[i].msg_hdr.msg_iov = ctypes.pointer(iovecs[i])
            mmsghdrs[i].msg_hdr.msg_iovlen = 1
            mmsghdrs[i].msg_hdr.msg_control = None
            mmsghdrs[i].msg_hdr.msg_controllen = 0
            mmsghdrs[i].msg_hdr.msg_flags = 0
            mmsghdrs[i].msg_len = 0
        
        sent = libc.sendmmsg(sockfd, mmsghdrs, chunk_size, 0)
        if sent > 0:
            total_sent += sent
        
        offset += chunk_size
    
    elapsed = time.time() - start
    sock.close()
    
    return elapsed, total_sent


# =============================================================================
# METHOD 8: Raw bytes sending (bypass Python object overhead)
# =============================================================================

def method_raw_bytes(packets: List[bytes]) -> Tuple[float, int]:
    """Send raw bytes directly, minimize Python overhead."""
    sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.ntohs(0x0003))
    sock.bind((SEND_IFACE, 0))
    
    # Pre-join all packets into single byte array
    # (This won't work for actual packet sending, but tests raw throughput)
    start = time.time()
    sent = 0
    
    # Send in tight loop with minimal Python overhead
    send_func = sock.send
    for packet in packets:
        send_func(packet)
        sent += 1
    
    elapsed = time.time() - start
    sock.close()
    
    return elapsed, sent


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    method_name: str
    elapsed_time: float
    packets_sent: int
    packets_per_sec: float
    throughput_mbps: float
    throughput_gbps: float
    speedup_vs_baseline: float


def run_benchmark(method_name: str, method_func: Callable, 
                  packets: List[bytes], baseline_pps: float = None) -> BenchmarkResult:
    """Run a single benchmark method."""
    print(f"\n  Testing: {method_name}")
    print(f"    Packets: {len(packets):,}")
    
    times = []
    sent_counts = []
    
    for run in range(NUM_RUNS):
        elapsed, sent = method_func(packets)
        times.append(elapsed)
        sent_counts.append(sent)
        print(f"    Run {run+1}/{NUM_RUNS}: {elapsed*1000:.1f}ms, {sent:,} packets")
    
    # Use best time (common practice for benchmarks)
    best_time = min(times)
    avg_sent = sum(sent_counts) / len(sent_counts)
    
    pps = avg_sent / best_time
    throughput_bits = pps * PACKET_SIZE * 8
    throughput_mbps = throughput_bits / 1_000_000
    throughput_gbps = throughput_bits / 1_000_000_000
    
    speedup = pps / baseline_pps if baseline_pps else 1.0
    
    print(f"    ✓ Best: {best_time*1000:.1f}ms")
    print(f"    ✓ PPS: {pps:,.0f} ({pps/1000:.1f}K)")
    print(f"    ✓ Throughput: {throughput_mbps:.1f} Mbps ({throughput_gbps:.2f} Gbps)")
    if baseline_pps:
        print(f"    ✓ Speedup: {speedup:.2f}×")
    
    return BenchmarkResult(
        method_name=method_name,
        elapsed_time=best_time,
        packets_sent=int(avg_sent),
        packets_per_sec=pps,
        throughput_mbps=throughput_mbps,
        throughput_gbps=throughput_gbps,
        speedup_vs_baseline=speedup
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"\n{'='*80}")
    print("E090: ULTIMATE PACKET SENDING SPEED BENCHMARK")
    print(f"{'='*80}")
    print(f"""
GOAL: Find the FASTEST way to send packets!

Current bottleneck: 9-28ms for 2K packets = 220K pps (0.4% of 40G line rate)
Target: Approach line rate (40Gbps = 59M pps theoretical)
Realistic: Achieve 1-10M pps (10-200× improvement!)

TEST CONFIGURATION:
  Interface:    {SEND_IFACE}
  Packets:      {NUM_PACKETS:,}
  Packet size:  {PACKET_SIZE} bytes
  Runs per test: {NUM_RUNS}
  
LINE RATE REFERENCE:
  40 Gbps @ 64-byte packets = 59.5M pps (theoretical max)
  40 Gbps @ 64-byte packets = ~10M pps (realistic sustained)
  Current performance:       ~220K pps (need 45× improvement!)
""")
    
    # Generate test packets once
    print(f"{'='*80}")
    print("PREPARATION")
    print(f"{'='*80}")
    packets = generate_test_packets(NUM_PACKETS)
    
    # Define all methods to test
    methods = [
        ("1. Baseline (socket.send loop)", method_baseline),
        ("2. sendmmsg (batch=1024)", method_sendmmsg_large),
        ("3. sendmmsg (batch=256)", method_sendmmsg_small),
        ("4. Pre-allocated buffers", method_preallocated),
        ("5. Burst (optimized socket)", method_burst),
        ("6. Raw bytes (tight loop)", method_raw_bytes),
        ("7. Multi-threaded SHARED socket (2 threads)", lambda p: method_multithreaded(p, 2)),
        ("8. Multi-threaded SEPARATE sockets (4 threads)", lambda p: method_multithreaded_separate_sockets(p, 4)),
        ("9. Multi-PROCESS (bypasses GIL, 4 procs)", lambda p: method_multiprocess(p, 4)),
        ("10. Combined (all optimizations)", method_combined),
    ]
    
    # Run benchmarks
    print(f"\n{'='*80}")
    print("BENCHMARKS")
    print(f"{'='*80}")
    
    results = []
    baseline_pps = None
    
    for method_name, method_func in methods:
        try:
            result = run_benchmark(method_name, method_func, packets, baseline_pps)
            results.append(result)
            
            if baseline_pps is None:
                baseline_pps = result.packets_per_sec
                
        except Exception as e:
            print(f"    ✗ FAILED: {e}")
            continue
        
        time.sleep(0.5)  # Cool down between tests
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n{'Method':<40} {'PPS':>12} {'Gbps':>8} {'Speedup':>8}")
    print(f"{'-'*40} {'-'*12} {'-'*8} {'-'*8}")
    
    for result in results:
        print(f"{result.method_name:<40} {result.packets_per_sec:>12,.0f} "
              f"{result.throughput_gbps:>8.2f} {result.speedup_vs_baseline:>8.2f}×")
    
    # Find best
    best = max(results, key=lambda r: r.packets_per_sec)
    
    print(f"\n🚀 WINNER: {best.method_name}")
    print(f"   Packets/sec:  {best.packets_per_sec:,.0f}")
    print(f"   Throughput:   {best.throughput_gbps:.2f} Gbps ({best.throughput_gbps/40*100:.1f}% of 40G)")
    print(f"   Speedup:      {best.speedup_vs_baseline:.1f}× vs baseline")
    
    # Analysis
    print(f"\n{'='*80}")
    print("ANALYSIS")
    print(f"{'='*80}")
    
    line_rate_40g = 59_500_000  # 40Gbps @ 64 bytes
    utilization = (best.packets_per_sec / line_rate_40g) * 100
    
    print(f"""
LINE RATE UTILIZATION:
  Theoretical max:  59.5M pps (40 Gbps, 64-byte packets)
  Achieved:         {best.packets_per_sec/1_000_000:.2f}M pps
  Utilization:      {utilization:.2f}%
  
REMAINING HEADROOM:
  Could theoretically send {line_rate_40g/best.packets_per_sec:.1f}× more packets!
  
BOTTLENECK ANALYSIS:
""")
    
    if utilization < 1:
        print(f"  ✗ Host is SEVERELY bottlenecked (< 1% line rate)")
        print(f"  → CPU/kernel overhead dominates")
        print(f"  → Consider DPDK or AF_XDP for kernel bypass")
    elif utilization < 10:
        print(f"  ⚠ Host is bottlenecked (< 10% line rate)")
        print(f"  → Still significant kernel overhead")
        print(f"  → Current approach near theoretical limit")
    else:
        print(f"  ✓ Good utilization! (> 10% line rate)")
        print(f"  → Approaching hardware limits")
    
    print(f"""
INFERENCE IMPACT:
  Current (e088): 28ms per projection
  With best method: {28 * (baseline_pps / best.packets_per_sec):.1f}ms per projection
  
  Full GPT-2 (96 projections):
    Current:  {28 * 96:.0f}ms = {28 * 96 / 1000:.2f}s per token
    Optimized: {28 * 96 * (baseline_pps / best.packets_per_sec):.0f}ms = {28 * 96 * (baseline_pps / best.packets_per_sec) / 1000:.2f}s per token
    Speedup:   {best.speedup_vs_baseline:.1f}×
""")


if __name__ == "__main__":
    if os.geteuid() != 0:
        print("This script requires root privileges for raw sockets.")
        print("Please run with: sudo python3 e090_packet_sending_speed_benchmark.py")
        sys.exit(1)
    
    main()

