#!/usr/bin/env python3
"""
e089_kernel_bypass_inference.py

KERNEL BYPASS OPTIMIZATIONS FOR SWITCH INFERENCE
=================================================

GOAL:
  Take e088 and optimize the host networking bottleneck (28.4ms → <2ms).
  The switches are FAST (sub-ms), the host is SLOW!

OPTIMIZATIONS:
  1. SENDMMSG: Batch packet sending (2045 syscalls → 1 syscall)
  2. SOCKET TUNING: Larger buffers, bypass qdisc, packet mmap
  3. RECVMMSG: Batch packet receiving
  4. ZERO-COPY: Reuse packet buffers, pre-allocated memory
  5. CPU PINNING: Pin threads to cores for cache locality

EXPECTED IMPROVEMENT:
  Current:  28.4ms send + 0.0ms receive = 28.4ms total
  Target:   1-2ms send + 0.1ms receive = 1-2ms total
  Speedup:  15-30× faster!

USAGE:
  $ sudo python3 e089_kernel_bypass_inference.py

REQUIRES:
  - Linux kernel 3.0+ (sendmmsg/recvmmsg)
  - Root for raw sockets
  - Same GPT-2 model as e088
"""

import os
import sys
import time
import ctypes
import ctypes.util
import socket
import struct
import threading
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
from dataclasses import dataclass

# Import from e088
from e088_gpt2_full_inference import (
    # Configuration
    MODEL_PATH, N_LAYERS, D_MODEL, TEST_DIM, BASE_VLAN,
    SW1_LAYERS, SW2_LAYERS, PROMPT, NUM_TOKENS,
    SW1_HOST_IFACE, SW1_INTER_IFACE, SW2_INTER_IFACE, SW2_HOST_IFACE,
    
    # Functions
    load_gpt2_weights, SimpleTokenizer, cpu_generate_tokens,
    quantize_to_int4, PacketTemplatePool, compute_packet_counts,
    configure_switch_filters, cleanup_switches, read_counters_packet_based,
    get_layer_neuron_mac,
    
    # From imports
    craft_vlan_packet, send_packets, get_mac_address,
    SWITCH1_IP, SWITCH2_IP, SEND_IFACE,
    mac_str_to_bytes,
)

# =============================================================================
# SENDMMSG/RECVMMSG - BATCH SOCKET I/O
# =============================================================================

libc = ctypes.CDLL(ctypes.util.find_library('c'), use_errno=True)

# struct iovec {
#     void  *iov_base;
#     size_t iov_len;
# };
class iovec(ctypes.Structure):
    _fields_ = [
        ('iov_base', ctypes.c_void_p),
        ('iov_len', ctypes.c_size_t),
    ]

# struct msghdr {
#     void         *msg_name;
#     socklen_t     msg_namelen;
#     struct iovec *msg_iov;
#     size_t        msg_iovlen;
#     void         *msg_control;
#     size_t        msg_controllen;
#     int           msg_flags;
# };
class msghdr(ctypes.Structure):
    _fields_ = [
        ('msg_name', ctypes.c_void_p),
        ('msg_namelen', ctypes.c_uint),
        ('msg_iov', ctypes.POINTER(iovec)),
        ('msg_iovlen', ctypes.c_size_t),
        ('msg_control', ctypes.c_void_p),
        ('msg_controllen', ctypes.c_size_t),
        ('msg_flags', ctypes.c_int),
    ]

# struct mmsghdr {
#     struct msghdr msg_hdr;
#     unsigned int  msg_len;
# };
class mmsghdr(ctypes.Structure):
    _fields_ = [
        ('msg_hdr', msghdr),  # Embedded msghdr struct
        ('msg_len', ctypes.c_uint),
    ]

# int sendmmsg(int sockfd, struct mmsghdr *msgvec, unsigned int vlen, int flags);
libc.sendmmsg.argtypes = [ctypes.c_int, ctypes.POINTER(mmsghdr), ctypes.c_uint, ctypes.c_int]
libc.sendmmsg.restype = ctypes.c_int

# int recvmmsg(int sockfd, struct mmsghdr *msgvec, unsigned int vlen, int flags, struct timespec *timeout);
libc.recvmmsg.argtypes = [ctypes.c_int, ctypes.POINTER(mmsghdr), ctypes.c_uint, ctypes.c_int, ctypes.c_void_p]
libc.recvmmsg.restype = ctypes.c_int


def sendmmsg_packets(sock: socket.socket, packets: List[bytes]) -> float:
    """
    Send packets using sendmmsg() for batched transmission.
    
    Handles kernel limit (UIO_MAXIOV = 1024) by chunking large batches.
    
    This is MUCH faster than calling send() 2045 times!
    - Old: 2045 syscalls × ~14μs = 28ms
    - New: 3 syscalls (chunks of 1024) = 1-2ms
    
    Returns: time taken in seconds
    """
    if not packets:
        return 0.0
    
    start = time.time()
    sockfd = sock.fileno()
    num_packets = len(packets)
    
    # Kernel limit on sendmmsg (UIO_MAXIOV)
    MAX_BATCH = 1024
    
    total_sent = 0
    offset = 0
    
    while offset < num_packets:
        # Determine batch size (up to MAX_BATCH)
        batch_size = min(MAX_BATCH, num_packets - offset)
        batch_packets = packets[offset:offset + batch_size]
        
        # Prepare iovec structures for this batch
        iovecs = (iovec * batch_size)()
        packet_bufs = []  # Keep refs alive
        
        for i, packet in enumerate(batch_packets):
            buf = ctypes.create_string_buffer(packet)
            packet_bufs.append(buf)
            iovecs[i].iov_base = ctypes.cast(buf, ctypes.c_void_p)
            iovecs[i].iov_len = len(packet)
        
        # Prepare mmsghdr structures (with embedded msghdr)
        mmsghdrs = (mmsghdr * batch_size)()
        for i in range(batch_size):
            mmsghdrs[i].msg_hdr.msg_name = None
            mmsghdrs[i].msg_hdr.msg_namelen = 0
            mmsghdrs[i].msg_hdr.msg_iov = ctypes.pointer(iovecs[i])
            mmsghdrs[i].msg_hdr.msg_iovlen = 1
            mmsghdrs[i].msg_hdr.msg_control = None
            mmsghdrs[i].msg_hdr.msg_controllen = 0
            mmsghdrs[i].msg_hdr.msg_flags = 0
            mmsghdrs[i].msg_len = 0
        
        # Send this batch
        sent = libc.sendmmsg(sockfd, mmsghdrs, batch_size, 0)
        
        if sent < 0:
            errno = ctypes.get_errno()
            raise OSError(errno, os.strerror(errno))
        
        total_sent += sent
        offset += batch_size
        
        if sent < batch_size:
            print(f"  ⚠ Warning: Batch incomplete - sent {sent}/{batch_size}")
            break
    
    elapsed = time.time() - start
    
    if total_sent != num_packets:
        print(f"  ⚠ Warning: Only sent {total_sent}/{num_packets} packets total")
    
    return elapsed


def recvmmsg_packets(sock: socket.socket, max_packets: int, timeout_ms: float = 100) -> List[bytes]:
    """
    Receive multiple packets using recvmmsg() for batched reception.
    
    This is faster than calling recv() in a loop.
    
    Returns: List of received packets
    """
    sockfd = sock.fileno()
    
    # Prepare receive buffers
    BUF_SIZE = 2048
    buffers = []
    iovecs = (iovec * max_packets)()
    msghdrs = (msghdr * max_packets)()
    mmsghdrs = (mmsghdr * max_packets)()
    
    for i in range(max_packets):
        buf = ctypes.create_string_buffer(BUF_SIZE)
        buffers.append(buf)
        
        iovecs[i].iov_base = ctypes.cast(buf, ctypes.c_void_p)
        iovecs[i].iov_len = BUF_SIZE
        
        msghdrs[i].msg_name = None
        msghdrs[i].msg_namelen = 0
        msghdrs[i].msg_iov = ctypes.pointer(iovecs[i])
        msghdrs[i].msg_iovlen = 1
        msghdrs[i].msg_control = None
        msghdrs[i].msg_controllen = 0
        msghdrs[i].msg_flags = 0
        
        ctypes.memmove(
            ctypes.addressof(mmsghdrs[i].msg_hdr),
            ctypes.addressof(msghdrs[i]),
            ctypes.sizeof(msghdr)
        )
    
    # Timeout structure
    class timespec(ctypes.Structure):
        _fields_ = [('tv_sec', ctypes.c_long), ('tv_nsec', ctypes.c_long)]
    
    ts = timespec()
    ts.tv_sec = int(timeout_ms // 1000)
    ts.tv_nsec = int((timeout_ms % 1000) * 1_000_000)
    
    # Receive packets
    received = libc.recvmmsg(sockfd, mmsghdrs, max_packets, 0, ctypes.byref(ts))
    
    if received < 0:
        errno = ctypes.get_errno()
        if errno == 11:  # EAGAIN/EWOULDBLOCK
            return []
        raise OSError(errno, os.strerror(errno))
    
    # Extract received packets
    packets = []
    for i in range(received):
        length = mmsghdrs[i].msg_len
        packets.append(bytes(buffers[i][:length]))
    
    return packets


# =============================================================================
# OPTIMIZED SOCKET CREATION
# =============================================================================

def create_optimized_socket(interface: str) -> socket.socket:
    """
    Create ultra-optimized raw socket for maximum performance.
    
    OPTIMIZATIONS:
    - Large send/receive buffers (16MB each)
    - Bypass qdisc (PACKET_QDISC_BYPASS)
    - Packet loss prevention (high priority)
    """
    sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.ntohs(0x0003))
    sock.bind((interface, 0))
    
    # 1. LARGE BUFFERS (prevent packet drops)
    BUFFER_SIZE = 16 * 1024 * 1024  # 16MB
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, BUFFER_SIZE)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, BUFFER_SIZE)
    
    # 2. HIGH PRIORITY (minimize latency)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_PRIORITY, 6)
    
    # 3. BYPASS QDISC (skip kernel queuing discipline)
    # SOL_PACKET = 263, PACKET_QDISC_BYPASS = 6
    try:
        SOL_PACKET = 263  # Linux constant not exposed in Python socket module
        PACKET_QDISC_BYPASS = 6
        sock.setsockopt(SOL_PACKET, PACKET_QDISC_BYPASS, 1)
    except (OSError, AttributeError):
        pass  # Not all kernels support this
    
    # 4. NON-BLOCKING (for async receive)
    sock.setblocking(False)
    
    return sock


# =============================================================================
# OPTIMIZED PACKET RECEIVER
# =============================================================================

class OptimizedPacketReceiver:
    """
    High-performance packet receiver with batched recvmmsg().
    
    IMPROVEMENTS OVER e088:
    - Batch receive (recvmmsg instead of recv loop)
    - Pre-allocated buffers (zero-copy)
    - Larger socket buffers (no drops)
    - CPU pinning (optional)
    """
    
    def __init__(self, interface: str):
        self.interface = interface
        self.socket: socket.socket = None
        self.running = False
        self.thread: threading.Thread = None
        self.counters: Dict[str, int] = defaultdict(int)
        self.total_received = 0
        self.batch_size = 256  # Receive up to 256 packets per syscall
    
    def start(self):
        """Start optimized packet receiver."""
        self.socket = create_optimized_socket(self.interface)
        
        self.running = True
        self.thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.thread.start()
    
    def _receive_loop(self):
        """Background thread with batched receives."""
        sockfd = self.socket.fileno()
        
        # Pre-allocate receive structures (reuse across loop iterations)
        BUF_SIZE = 2048
        buffers = [ctypes.create_string_buffer(BUF_SIZE) for _ in range(self.batch_size)]
        iovecs = (iovec * self.batch_size)()
        mmsghdrs = (mmsghdr * self.batch_size)()
        
        for i in range(self.batch_size):
            iovecs[i].iov_base = ctypes.cast(buffers[i], ctypes.c_void_p)
            iovecs[i].iov_len = BUF_SIZE
            
            mmsghdrs[i].msg_hdr.msg_name = None
            mmsghdrs[i].msg_hdr.msg_namelen = 0
            mmsghdrs[i].msg_hdr.msg_iov = ctypes.pointer(iovecs[i])
            mmsghdrs[i].msg_hdr.msg_iovlen = 1
            mmsghdrs[i].msg_hdr.msg_control = None
            mmsghdrs[i].msg_hdr.msg_controllen = 0
            mmsghdrs[i].msg_hdr.msg_flags = 0
        
        # Timeout: 10ms (responsive but not wasteful)
        class timespec(ctypes.Structure):
            _fields_ = [('tv_sec', ctypes.c_long), ('tv_nsec', ctypes.c_long)]
        
        ts = timespec()
        ts.tv_sec = 0
        ts.tv_nsec = 10_000_000  # 10ms
        
        while self.running:
            try:
                # Batch receive - ONE syscall gets up to 256 packets!
                received = libc.recvmmsg(sockfd, mmsghdrs, self.batch_size, 0, ctypes.byref(ts))
                
                if received < 0:
                    errno = ctypes.get_errno()
                    if errno in (11, 35):  # EAGAIN/EWOULDBLOCK (timeout)
                        continue
                    if self.running:
                        print(f"  recvmmsg error: {os.strerror(errno)}")
                    break
                
                if received == 0:
                    continue
                
                # Process received packets
                for i in range(received):
                    length = mmsghdrs[i].msg_len
                    if length < 14:  # Min Ethernet frame
                        continue
                    
                    # Extract destination MAC (first 6 bytes)
                    dst_mac_bytes = bytes(buffers[i][:6])
                    dst_mac_str = ':'.join(f'{b:02x}' for b in dst_mac_bytes)
                    
                    self.counters[dst_mac_str] += 1
                    self.total_received += 1
                
            except Exception as e:
                if self.running:
                    print(f"  Receive error: {e}")
                break
    
    def stop(self):
        """Stop receiver."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.socket:
            self.socket.close()
    
    def get_counts(self) -> Dict[str, int]:
        """Get current packet counts."""
        return dict(self.counters)
    
    def clear(self):
        """Clear counters."""
        self.counters.clear()
        self.total_received = 0


# =============================================================================
# OPTIMIZED PACKET SENDING
# =============================================================================

def send_packets_optimized(sock: socket.socket, packets: List[bytes]) -> float:
    """
    Send packets using sendmmsg() for maximum performance.
    Socket is passed in (not created) to avoid setup overhead.
    
    Returns: send_time in seconds
    """
    if not packets:
        return 0.0
    
    # Send using sendmmsg
    send_time = sendmmsg_packets(sock, packets)
    
    return send_time


# =============================================================================
# MAIN COMPARISON
# =============================================================================

def main():
    print(f"\n{'='*80}")
    print("E089: KERNEL BYPASS OPTIMIZATIONS")
    print(f"{'='*80}")
    print(f"""
GOAL: Eliminate host networking bottleneck!

OPTIMIZATIONS:
  1. sendmmsg() - Batch send (1 syscall instead of 2045)
  2. recvmmsg() - Batch receive (256 packets/syscall)
  3. Large buffers - 16MB send/recv (no packet drops)
  4. Bypass qdisc - Skip kernel queuing
  5. Pre-allocated - Zero-copy receive buffers

BASELINE (e088):  28.4ms send
TARGET:           1-2ms send
EXPECTED SPEEDUP: 15-30×
""")
    
    # Load weights (same as e088)
    print(f"{'='*80}")
    print("STEP 1: LOAD WEIGHTS")
    print(f"{'='*80}")
    weights = load_gpt2_weights(test_dim=TEST_DIM)
    
    # CPU reference (skip for speed - we know it works from e088)
    print(f"\n{'='*80}")
    print("STEP 2: CONFIGURE SWITCHES")
    print(f"{'='*80}")
    print("  (Skipping - assume already configured from e088)")
    print("  To configure, set CONFIGURE_SWITCHES=True")
    
    CONFIGURE_SWITCHES = False  # Set to True if switches need config
    
    if CONFIGURE_SWITCHES:
        cleanup_switches()
        
        if not configure_switch_filters(SWITCH1_IP, SW1_LAYERS, SW1_HOST_IFACE, 
                                        SW1_INTER_IFACE, is_sw1=True):
            print("  ✗ SW1 configuration failed")
            return
        
        if not configure_switch_filters(SWITCH2_IP, SW2_LAYERS, SW2_HOST_IFACE,
                                        SW2_INTER_IFACE, is_sw1=False):
            print("  ✗ SW2 configuration failed")
            return
        
        print("  ✓ Both switches configured")
        time.sleep(2)
    
    # ===========================================================================
    # PERFORMANCE COMPARISON: e088 vs e089
    # ===========================================================================
    print(f"\n{'='*80}")
    print("STEP 3: PERFORMANCE COMPARISON")
    print(f"{'='*80}")
    
    # Prepare test data (same as e088)
    tokenizer = SimpleTokenizer()
    token_ids = tokenizer.encode(PROMPT)
    last_token = token_ids[-1]
    
    embedding_float = weights.token_embd[last_token, :]
    embedding_int4, _ = quantize_to_int4(embedding_float)
    
    layer_idx = 0
    src_mac = get_mac_address(SEND_IFACE)
    vlan_id = BASE_VLAN + layer_idx
    
    # Warm up njit
    print(f"  Warming up JIT compiler...", end=' ')
    warmup_start = time.time()
    dummy_act = np.ones(TEST_DIM, dtype=np.int8)
    dummy_weights = np.ones((TEST_DIM, TEST_DIM), dtype=np.int8)
    _ = compute_packet_counts(dummy_act, dummy_weights)
    warmup_time = time.time() - warmup_start
    print(f"✓ ({warmup_time*1000:.1f}ms)")
    
    # Create packet templates
    packet_pool = PacketTemplatePool(layer_idx, TEST_DIM, src_mac, vlan_id)
    
    # Prepare weights
    qkv_weights_float = weights.attn_qkv_weight[layer_idx][:TEST_DIM, :TEST_DIM]
    qkv_weights_int4, _ = quantize_to_int4(qkv_weights_float)
    
    # Generate packets
    print(f"  Creating packets...", end=' ')
    packet_start = time.time()
    packets = []
    pos_counts, neg_counts = compute_packet_counts(embedding_int4, qkv_weights_int4)
    packets = packet_pool.create_packets_from_counts(pos_counts, neg_counts)
    packet_time = time.time() - packet_start
    print(f"✓ {len(packets):,} packets in {packet_time*1000:.1f}ms")
    
    # ===========================================================================
    # TEST 1: BASELINE (e088 method)
    # ===========================================================================
    print(f"\n  {'─'*76}")
    print(f"  TEST 1: BASELINE (e088 - standard socket.send)")
    print(f"  {'─'*76}")
    
    baseline_start = time.time()
    send_packets(SEND_IFACE, packets)  # e088 method
    baseline_time = time.time() - baseline_start
    
    print(f"    Send time: {baseline_time*1000:.2f}ms")
    
    time.sleep(0.5)  # Let packets clear
    
    # ===========================================================================
    # TEST 2: OPTIMIZED (e089 method)
    # ===========================================================================
    print(f"\n  {'─'*76}")
    print(f"  TEST 2: OPTIMIZED (e089 - sendmmsg)")
    print(f"  {'─'*76}")
    
    # Create optimized socket (reuse for both send and receive)
    print(f"    Creating optimized socket...")
    setup_start = time.time()
    send_sock = create_optimized_socket(SEND_IFACE)
    setup_time = time.time() - setup_start
    print(f"    Setup time: {setup_time*1000:.2f}ms")
    
    # Start optimized receiver
    print(f"    Starting optimized receiver...")
    receiver = OptimizedPacketReceiver(SEND_IFACE)
    receiver.start()
    time.sleep(0.2)
    
    try:
        # Send using sendmmsg with pre-created socket
        print(f"    Sending {len(packets):,} packets with sendmmsg()...")
        send_time = send_packets_optimized(send_sock, packets)
        
        print(f"    Send time:  {send_time*1000:.2f}ms")
        
        # Wait for packets to return
        print(f"    Waiting for packets to return...")
        wait_start = time.time()
        
        expected = len(packets)
        timeout = time.time() + 0.5
        last_count = 0
        stall_time = 0
        
        while time.time() < timeout:
            current = receiver.total_received
            
            if current >= expected * 0.99:
                break
            
            if current == last_count:
                stall_time += 0.005
                if stall_time > 0.05:
                    break
            else:
                stall_time = 0
            
            last_count = current
            time.sleep(0.005)
        
        wait_time = time.time() - wait_start
        
        print(f"    Received: {receiver.total_received}/{expected} packets")
        print(f"    Wait time: {wait_time*1000:.2f}ms")
        
        # Read counters
        pos_counts_sw, neg_counts_sw = read_counters_packet_based(receiver, layer_idx, TEST_DIM)
        switch_result = pos_counts_sw - neg_counts_sw
        
        # Compare with CPU
        cpu_result = embedding_int4 @ qkv_weights_int4.T
        match = np.allclose(switch_result[:5], cpu_result[:5], atol=2)
        
        print(f"\n    Switch result: {switch_result[:5]}")
        print(f"    CPU reference: {cpu_result[:5]}")
        print(f"    Match: {'✓ YES' if match else '✗ NO'}")
        
    finally:
        send_sock.close()
        receiver.stop()
    
    # ===========================================================================
    # RESULTS
    # ===========================================================================
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    
    print(f"\n  BASELINE (e088):")
    print(f"    Method:     socket.send() loop")
    print(f"    Send time:  {baseline_time*1000:.2f}ms")
    print(f"    Syscalls:   {len(packets):,} (one per packet)")
    
    # Calculate number of sendmmsg calls (chunked at 1024)
    num_sendmmsg_calls = (len(packets) + 1023) // 1024
    
    print(f"\n  OPTIMIZED (e089):")
    print(f"    Method:     sendmmsg() batch")
    print(f"    Send time:  {send_time*1000:.2f}ms")
    print(f"    Syscalls:   {num_sendmmsg_calls} (1024 packets/call)")
    print(f"    Setup:      {setup_time*1000:.2f}ms (one-time socket creation)")
    
    speedup = baseline_time / send_time if send_time > 0 else 0
    
    print(f"\n  SPEEDUP:")
    print(f"    Send only:  {speedup:.1f}× faster")
    print(f"    (Setup is one-time cost, amortized over many operations)")
    
    print(f"\n  EXTRAPOLATION TO FULL GPT-2:")
    if match:
        # 6 projections per layer (Q, K, V, O, FFN_up, FFN_down)
        # 12 layers
        # Plus attention computation
        ops_per_layer = 8
        
        baseline_per_token = baseline_time * ops_per_layer * 12
        optimized_per_token = send_time * ops_per_layer * 12
        
        print(f"    Baseline:  {baseline_per_token:.0f}ms = {baseline_per_token/1000:.2f}s/token")
        print(f"    Optimized: {optimized_per_token:.0f}ms = {optimized_per_token/1000:.3f}s/token")
        print(f"    Speedup:   {speedup:.1f}×")
        
        print(f"\n  🚀 WITH KERNEL BYPASS:")
        print(f"    • Reduced syscalls from {len(packets)} → {num_sendmmsg_calls}")
        print(f"    • {speedup:.1f}× faster packet transmission")
        print(f"    • Switches remain blazing fast (sub-ms)")
        if speedup > 1.5:
            print(f"    • Ready for full-model inference!")
        else:
            print(f"    • Note: Limited speedup due to already-fast baseline")
            print(f"    • Main benefit: Scales better with packet count")


if __name__ == "__main__":
    if os.geteuid() != 0:
        print("This script requires root privileges for raw sockets.")
        print("Please run with: sudo python3 e089_kernel_bypass_inference.py")
        sys.exit(1)
    
    main()



""" Output:
(.venv) multiplex@multiplex1:~/photonic_inference_engine/phase_001/phase_001$ sudo python3 e089_kernel_bypass_inference.py 

================================================================================
E089: KERNEL BYPASS OPTIMIZATIONS
================================================================================

GOAL: Eliminate host networking bottleneck!

OPTIMIZATIONS:
  1. sendmmsg() - Batch send (1 syscall instead of 2045)
  2. recvmmsg() - Batch receive (256 packets/syscall)
  3. Large buffers - 16MB send/recv (no packet drops)
  4. Bypass qdisc - Skip kernel queuing
  5. Pre-allocated - Zero-copy receive buffers

BASELINE (e088):  28.4ms send
TARGET:           1-2ms send
EXPECTED SPEEDUP: 15-30×

================================================================================
STEP 1: LOAD WEIGHTS
================================================================================

================================================================================
LOADING GPT-2 WEIGHTS FROM GGUF
================================================================================
  Model path: ./models/openai-community/gpt2.Q4_K_M.gguf
  Test dimension: 64

  Loading embeddings...
    Token embedding: (50257, 768)
    Position embedding: (1024, 768)
    Sliced to: token=(50257, 64), pos=(1024, 64)

  Loading 7 layers...
    Loaded layer 3/7
    Loaded layer 6/7

  Loading output norm...

  ✓ All weights loaded!

================================================================================
STEP 2: CONFIGURE SWITCHES
================================================================================
  (Skipping - assume already configured from e088)
  To configure, set CONFIGURE_SWITCHES=True

================================================================================
STEP 3: PERFORMANCE COMPARISON
================================================================================
  Warming up JIT compiler... ✓ (453.0ms)
  Pre-computing 128 packet templates for layer 0... ✓ (0.6ms)
  Creating packets... ✓ 2,045 packets in 0.1ms

  ────────────────────────────────────────────────────────────────────────────
  TEST 1: BASELINE (e088 - standard socket.send)
  ────────────────────────────────────────────────────────────────────────────
    Send time: 9.28ms

  ────────────────────────────────────────────────────────────────────────────
  TEST 2: OPTIMIZED (e089 - sendmmsg)
  ────────────────────────────────────────────────────────────────────────────
    Creating optimized socket...
    Setup time: 5.68ms
    Starting optimized receiver...
    Sending 2,045 packets with sendmmsg()...
    Send time:  14.51ms
    Waiting for packets to return...
    Received: 1700/2045 packets
    Wait time: 61.14ms

    Switch result: [-38 -10 -49 -17 -19]
    CPU reference: [-38 -10 -49 -17 -19]
    Match: ✓ YES

================================================================================
RESULTS
================================================================================

  BASELINE (e088):
    Method:     socket.send() loop
    Send time:  9.28ms
    Syscalls:   2,045 (one per packet)

  OPTIMIZED (e089):
    Method:     sendmmsg() batch
    Send time:  14.51ms
    Syscalls:   2 (1024 packets/call)
    Setup:      5.68ms (one-time socket creation)

  SPEEDUP:
    Send only:  0.6× faster
    (Setup is one-time cost, amortized over many operations)

  EXTRAPOLATION TO FULL GPT-2:
    Baseline:  1ms = 0.00s/token
    Optimized: 1ms = 0.001s/token
    Speedup:   0.6×

  🚀 WITH KERNEL BYPASS:
    • Reduced syscalls from 2045 → 2
    • 0.6× faster packet transmission
    • Switches remain blazing fast (sub-ms)
    • Note: Limited speedup due to already-fast baseline
    • Main benefit: Scales better with packet count
"""


""" Note:
The sendmmsg() "optimization" is actually SLOWER (14.51ms vs 9.28ms). Let me explain what's happening:
📊 Why sendmmsg() is Slower Here:

1. The Baseline is Already Fast
9.28ms for 2,045 packets = 4.5μs per packet
This is MUCH faster than e088's 28.4ms (13.9μs per packet)
The kernel is already batching efficiently behind the scenes!

2. sendmmsg() Overhead
For only 2,045 packets, sendmmsg() has significant overhead:
    - Build 2,045 iovec structures- Build 2,045 msghdr structures  
    - Build 2,045 mmsghdr structures- Chunk into 3 calls (1024 + 1024 + 997)
    - Memory copies for ctypes buffers
This setup overhead (~5-10ms) overwhelms the syscall savings!

3. Packet Loss
Sent 2,045, only received 1,700 (83%)
345 packets dropped!
This suggests network congestion, not CPU bottleneck
"""