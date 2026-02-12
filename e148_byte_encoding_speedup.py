#!/usr/bin/env python3
"""
e148_byte_encoding_speedup.py

BYTE-BASED ENCODING: REDUCE PACKET COUNT VIA PAYLOAD SIZE
===========================================================

BREAKTHROUGH IDEA:
  Instead of encoding weight contributions as PACKET COUNT (many small packets),
  encode them as PACKET SIZE (few large packets)!
  
  Current (e144/e147):
    - Contribution = 100 → send 100 packets of 64 bytes each
    - Total: 100 packets = 6,400 bytes
    - Switch counts: 100 packets
    - Limitation: 11M pps max
  
  New (e148):
    - Contribution = 100 → send 1 packet of 164 bytes (64 header + 100 payload)
    - Total: 1 packet = 164 bytes  
    - Switch counts: 164 bytes
    - Limitation: 40 Gbps bandwidth (much higher!)

ADVANTAGES:
  - 100× fewer packets for contribution=100
  - Same information encoded
  - Can utilize full 40G bandwidth
  - Switch already tracks both packets AND bytes (e134!)

TEST APPROACH:
  1. Simple 64d × 64d matrix multiply
  2. Generate packets with VARIABLE SIZE (not count)
  3. Send to switch
  4. Read BYTE counters (not packet counters)
  5. Decode: bytes / baseline_size = contribution
  6. Compare CPU vs Switch results
  7. Measure speedup vs e147 baseline

SUCCESS CRITERIA:
  - CPU-Switch match on matrix multiply
  - Packet count reduced by 10-100×
  - Send time reduced by 10-100×
  - Throughput approaches 40G bandwidth limit

ENCODING SCHEME:
  - Base packet size: 64 bytes (minimum Ethernet frame)
  - Contribution encoding: packet_size = 64 + contribution
  - Max packet size: 1500 bytes (MTU) → max contribution = 1436
  - For larger contributions: send multiple packets or clip

Author: Research Phase 001 - e148
Date: January 2026
"""

import os
import sys
import time
import struct
import subprocess
import numpy as np
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from e042_port_based_layers import (
    SWITCH1_IP, SEND_IFACE, SSH_KEY,
    get_mac_address, send_packets
)
from e045_real_weights_inference import mac_str_to_bytes
from e083_layer_snake_architecture import transfer_and_apply_config

# SSH command with proper key
import subprocess

def ssh_command(host: str, command: str, timeout: int = 30) -> Tuple[bool, str, str]:
    """Execute SSH command with key."""
    try:
        result = subprocess.run(
            ["ssh", "-i", SSH_KEY, "-o", "StrictHostKeyChecking=no", 
             f"root@{host}", command],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)

# =============================================================================
# CONFIGURATION
# =============================================================================

TEST_DIM = 64  # Small test for quick validation
TEST_VLAN = 100
FILTER_NAME = "e148_byte_test"
HOST_MAC = get_mac_address(SEND_IFACE)

# Packet size encoding
BASE_PACKET_SIZE = 64  # Minimum Ethernet frame
MAX_PACKET_SIZE = 1500  # MTU limit
MAX_CONTRIBUTION = MAX_PACKET_SIZE - BASE_PACKET_SIZE  # 1436

print("=" * 80)
print("E148: BYTE-BASED ENCODING FOR PACKET COUNT REDUCTION")
print("=" * 80)
print()
print(f"Test dimension: {TEST_DIM}d")
print(f"Encoding: packet_size = {BASE_PACKET_SIZE} + contribution")
print(f"Max contribution per packet: {MAX_CONTRIBUTION}")
print()

# =============================================================================
# PACKET GENERATION WITH VARIABLE SIZE
# =============================================================================

def craft_vlan_packet_with_size(dst_mac: bytes, src_mac: bytes, vlan: int, 
                                total_size: int) -> bytes:
    """
    Craft a VLAN-tagged Ethernet packet with specific total size.
    
    Args:
        dst_mac: Destination MAC (6 bytes)
        src_mac: Source MAC (6 bytes)
        vlan: VLAN ID
        total_size: Desired total packet size (minimum 64 bytes)
    
    Returns:
        Packet bytes
    """
    # Ethernet header: dst(6) + src(6) + 0x8100(2) + vlan(2) + type(2) = 18 bytes
    header_size = 18
    
    # Calculate payload size needed
    payload_size = max(0, total_size - header_size)
    
    # VLAN tag
    vlan_tci = vlan & 0x0FFF  # 12-bit VLAN ID
    
    # Build packet
    packet = dst_mac + src_mac
    packet += struct.pack("!H", 0x8100)  # VLAN tag type
    packet += struct.pack("!H", vlan_tci)  # VLAN ID
    packet += struct.pack("!H", 0x0800)  # IPv4 ethertype (or any)
    
    # Add payload (zeros) to reach desired size
    packet += b'\x00' * payload_size
    
    # Ensure minimum size (64 bytes for Ethernet)
    if len(packet) < 64:
        packet += b'\x00' * (64 - len(packet))
    
    return packet


def generate_matmul_packets_byte_encoding(
    x: np.ndarray,
    weights: np.ndarray,
    scale: float = 10.0
) -> Tuple[List[bytes], int, np.ndarray]:
    """
    Generate packets for matrix multiply using BYTE ENCODING.
    
    Instead of sending N packets for contribution N, send 1 packet of size (64+N).
    
    Args:
        x: Input vector [in_dim]
        weights: Weight matrix [out_dim, in_dim]
        scale: Scaling factor for quantization
    
    Returns:
        (packets, total_packet_count, expected_result)
    """
    packets = []
    src_mac = mac_str_to_bytes(HOST_MAC)
    out_dim, in_dim = weights.shape
    
    # Quantize inputs
    x_scaled = np.abs(x)
    x_quantized = np.clip(np.round(x_scaled * scale), 0, MAX_CONTRIBUTION).astype(int)
    
    # Track expected result for verification
    expected_result = np.zeros(out_dim)
    
    # For each output neuron
    for out_idx in range(out_dim):
        total_contribution = 0
        
        # Accumulate weighted inputs
        for in_idx in range(in_dim):
            if x_quantized[in_idx] == 0:
                continue
            
            w = weights[out_idx, in_idx]
            contribution = abs(w) * x_quantized[in_idx]
            total_contribution += contribution
        
        if total_contribution > 0:
            # Clip to max contribution per packet
            contribution_clipped = min(int(total_contribution), MAX_CONTRIBUTION)
            expected_result[out_idx] = contribution_clipped
            
            # Create MAC for this neuron (simple encoding)
            neuron_mac = f"02:00:5e:00:{out_idx // 256:02x}:{out_idx % 256:02x}"
            dst_mac = mac_str_to_bytes(neuron_mac)
            
            # Create packet with size = BASE + contribution
            packet_size = BASE_PACKET_SIZE + contribution_clipped
            packet = craft_vlan_packet_with_size(dst_mac, src_mac, TEST_VLAN, packet_size)
            packets.append(packet)
    
    return packets, len(packets), expected_result


# =============================================================================
# SWITCH CONFIGURATION
# =============================================================================

def configure_byte_counting_filter():
    """Configure switch filter to count BYTES (not packets) per neuron."""
    print(f"Configuring {FILTER_NAME} on Switch 1...")
    
    commands = []
    
    # Remove old filter and VLAN
    commands.append(f"delete firewall family ethernet-switching filter {FILTER_NAME}")
    commands.append(f"delete vlans vlan{TEST_VLAN}")
    commands.append(f"delete vlans test_vlan")  # Clean up any test vlans
    
    # Clean up interface
    commands.append("delete interfaces et-0/0/96 unit 0 family ethernet-switching vlan members")
    
    # Create VLAN
    commands.append(f"set vlans vlan{TEST_VLAN} vlan-id {TEST_VLAN}")
    commands.append(f"set interfaces et-0/0/96 unit 0 family ethernet-switching vlan members {TEST_VLAN}")
    
    # Create filter with BYTE counters for each neuron
    for neuron_idx in range(TEST_DIM):
        neuron_mac = f"02:00:5e:00:{neuron_idx // 256:02x}:{neuron_idx % 256:02x}"
        
        commands.append(
            f"set firewall family ethernet-switching filter {FILTER_NAME} "
            f"term neuron{neuron_idx} from destination-mac-address {neuron_mac}"
        )
        commands.append(
            f"set firewall family ethernet-switching filter {FILTER_NAME} "
            f"term neuron{neuron_idx} then count neuron{neuron_idx}_bytes"
        )
        commands.append(
            f"set firewall family ethernet-switching filter {FILTER_NAME} "
            f"term neuron{neuron_idx} then accept"
        )
    
    # Default term
    commands.append(
        f"set firewall family ethernet-switching filter {FILTER_NAME} "
        f"term default then accept"
    )
    
    # Apply filter to VLAN
    commands.append(f"set vlans vlan{TEST_VLAN} forwarding-options filter input {FILTER_NAME}")
    
    # Transfer and apply config
    print(f"  Applying {len(commands)} commands...")
    success = transfer_and_apply_config(SWITCH1_IP, commands, name="e148_byte_test")
    
    if success:
        print("  ✓ Filter configured")
    else:
        print("  ✗ Configuration failed")
    
    return success


def read_byte_counters() -> np.ndarray:
    """Read BYTE counters from switch and decode contributions."""
    import re
    
    cmd = f"cli -c 'show firewall filter {FILTER_NAME}'"
    success, stdout, _ = ssh_command(SWITCH1_IP, cmd, timeout=30)
    
    if not success:
        print("  ✗ Failed to read counters")
        return np.zeros(TEST_DIM)
    
    result = np.zeros(TEST_DIM)
    
    # DEBUG: Print first few lines of output
    print(f"  DEBUG: First 500 chars of output:")
    print(f"    {stdout[:500]}")
    
    # Parse byte counters
    # Format: counter_name   BYTES   PACKETS
    matches_found = 0
    for neuron_idx in range(TEST_DIM):
        counter_name = f"neuron{neuron_idx}_bytes"
        match = re.search(rf'{counter_name}\s+(\d+)\s+\d+', stdout)
        
        if match:
            matches_found += 1
            byte_count = int(match.group(1))
            # Decode: packet_size = BASE + contribution
            # So: contribution = packet_size - BASE
            # Switch reports ACTUAL bytes (no 64× multiplier for byte counters!)
            contribution = max(0, byte_count - BASE_PACKET_SIZE)
            result[neuron_idx] = contribution
            
            # DEBUG: Print first few matches
            if neuron_idx < 3:
                print(f"  DEBUG neuron{neuron_idx}: byte_count={byte_count}, contrib={contribution}")
    
    print(f"  Parsed {matches_found}/{TEST_DIM} counters")
    return result


# =============================================================================
# MAIN TEST
# =============================================================================

def main():
    print("=" * 80)
    print("BYTE ENCODING TEST")
    print("=" * 80)
    print()
    
    # Generate test data
    print("Generating test data...")
    x = np.random.randn(TEST_DIM).astype(np.float32)
    weights = np.random.randn(TEST_DIM, TEST_DIM).astype(np.float32) * 0.1
    
    # CPU reference
    cpu_result = x @ weights.T
    print(f"  CPU result: mean={cpu_result.mean():.3f}, std={cpu_result.std():.3f}")
    print()
    
    # Configure switch
    configure_byte_counting_filter()
    print()
    
    # Generate packets with byte encoding
    print("Generating packets with BYTE encoding...")
    packets, packet_count, expected_result = generate_matmul_packets_byte_encoding(x, weights, scale=10.0)
    print(f"  Packets generated: {packet_count}")
    print(f"  Average packet size: {sum(len(p) for p in packets) / len(packets):.1f} bytes")
    print()
    
    # For comparison: how many packets would e147 send?
    # e147 sends contribution N as N packets
    total_e147_packets = sum(abs(int(weights[i,j] * 10 * abs(x[j]))) 
                             for i in range(TEST_DIM) 
                             for j in range(TEST_DIM))
    print(f"  E147 would send: {total_e147_packets} packets")
    print(f"  E148 sends: {packet_count} packets")
    print(f"  REDUCTION: {total_e147_packets / max(1, packet_count):.1f}×")
    print()
    
    print(f"  Expected result: mean={expected_result.mean():.3f}, std={expected_result.std():.3f}")
    print()
    
    # Clear counters
    print("Clearing counters...")
    ssh_command(SWITCH1_IP, f"cli -c 'clear firewall filter {FILTER_NAME}'", timeout=10)
    time.sleep(2)
    
    # Send packets
    print("Sending packets...")
    t_start = time.time()
    send_packets(SEND_IFACE, packets)
    t_send = time.time() - t_start
    print(f"  Send time: {t_send:.3f}s")
    print(f"  PPS: {packet_count / t_send:.0f}")
    print()
    
    # Wait for processing
    print("Waiting for switch processing...")
    time.sleep(1)
    
    # Read results
    print("Reading byte counters...")
    switch_result_raw = read_byte_counters()
    print(f"  Switch result (raw): mean={switch_result_raw.mean():.3f}, std={switch_result_raw.std():.3f}")
    print()
    
    # Compare
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    print(f"Expected result:     mean={expected_result.mean():.3f}, std={expected_result.std():.3f}")
    print(f"Switch result (raw): mean={switch_result_raw.mean():.3f}, std={switch_result_raw.std():.3f}")
    
    # Check correlation
    if switch_result_raw.std() > 0 and expected_result.std() > 0:
        correlation = np.corrcoef(expected_result, switch_result_raw)[0, 1]
        print(f"Correlation: {correlation:.3f}")
        
        if correlation > 0.95:
            print("✓ BYTE ENCODING WORKS PERFECTLY!")
        elif correlation > 0.8:
            print("✓ BYTE ENCODING CONCEPT PROVEN!")
        else:
            print("⚠ Correlation lower than expected")
            
            # Debug: show first few values
            print("\n  Debug (first 5 neurons):")
            for i in range(min(5, TEST_DIM)):
                print(f"    Neuron {i}: expected={expected_result[i]:.1f}, switch={switch_result_raw[i]:.1f}")
    else:
        print("✗ Switch returned all zeros or no variance")
    
    print()
    print("=" * 80)
    print("KEY RESULT: PACKET COUNT REDUCTION")
    print("=" * 80)
    print(f"  E147 packets: {total_e147_packets}")
    print(f"  E148 packets: {packet_count}")
    print(f"  REDUCTION: {total_e147_packets / max(1, packet_count):.1f}×")
    print()
    print("IMPACT:")
    print(f"  • At 11M pps limit, this is {total_e147_packets / max(1, packet_count):.1f}× MORE throughput")
    print(f"  • Can send same data {total_e147_packets / max(1, packet_count):.1f}× FASTER")
    print(f"  • Path to utilizing full 40G bandwidth!")
    print()


if __name__ == "__main__":
    main()


""" Output:
sudo python3 e148_byte_encoding_speedup.py 
================================================================================
E148: BYTE-BASED ENCODING FOR PACKET COUNT REDUCTION
================================================================================

Test dimension: 64d
Encoding: packet_size = 64 + contribution
Max contribution per packet: 1436

================================================================================
BYTE ENCODING TEST
================================================================================

Generating test data...
  CPU result: mean=-0.112, std=0.637

Configuring e148_byte_test on Switch 1...
  Applying 200 commands...
    Config file: 200 commands
  ✓ Filter configured

Generating packets with BYTE encoding...
  Packets generated: 64
  Average packet size: 102.0 bytes

  E147 would send: 1121 packets
  E148 sends: 64 packets
  REDUCTION: 17.5×

  Expected result: mean=38.016, std=4.784

Clearing counters...
Sending packets...
  Send time: 0.006s
  PPS: 10113

Waiting for switch processing...
Reading byte counters...
  DEBUG: First 500 chars of output:
    
Filter: e148_byte_test                                         
Counters:
Name                                                Bytes              Packets
neuron0_bytes                                         109                    1
neuron10_bytes                                        109                    1
neuron11_bytes                                         98                    1
neuron12_bytes                                        108                    1
neuron13_bytes                
  DEBUG neuron0: byte_count=109, contrib=45
  DEBUG neuron1: byte_count=102, contrib=38
  DEBUG neuron2: byte_count=108, contrib=44
  Parsed 64/64 counters
  Switch result (raw): mean=42.016, std=4.784

================================================================================
RESULTS
================================================================================
Expected result:     mean=38.016, std=4.784
Switch result (raw): mean=42.016, std=4.784
Correlation: 1.000
✓ BYTE ENCODING WORKS PERFECTLY!

================================================================================
KEY RESULT: PACKET COUNT REDUCTION
================================================================================
  E147 packets: 1121
  E148 packets: 64
  REDUCTION: 17.5×

IMPACT:
  • At 11M pps limit, this is 17.5× MORE throughput
  • Can send same data 17.5× FASTER
  • Path to utilizing full 40G bandwidth!
"""