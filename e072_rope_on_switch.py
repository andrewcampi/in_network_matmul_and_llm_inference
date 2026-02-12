#!/usr/bin/env python3
"""
e072_rope_on_switch.py

ROTARY POSITION EMBEDDINGS (RoPE) ON SWITCHES
==============================================

THE CHALLENGE:
  RoPE applies position-dependent rotations to Q and K vectors:
  
  For each dimension pair (2i, 2i+1) at position `pos`:
    q_rot[2i]   = q[2i] * cos(θ) - q[2i+1] * sin(θ)
    q_rot[2i+1] = q[2i] * sin(θ) + q[2i+1] * cos(θ)
  
  Where θ = pos * (1 / base^(2i/d))
  
  This seems hard because:
    1. Position-dependent (different θ for each position)
    2. Transcendental functions (sin/cos)
    3. Mixing dimension pairs

THE INSIGHT:
  For a FIXED position, sin(θ) and cos(θ) are CONSTANTS!
  
  The rotation is just a 2×2 matrix multiply per pair:
    [q_rot[2i]  ]   [cos(θ)  -sin(θ)] [q[2i]  ]
    [q_rot[2i+1]] = [sin(θ)   cos(θ)] [q[2i+1]]
  
  With quantized inputs (4-bit), this becomes weighted packet sums!
  
  For each dimension pair:
    - q[2i] * cos(θ)  →  packets to output[2i] (positive)
    - q[2i+1] * sin(θ) → packets to output[2i] (NEGATIVE - use neg counter)
    - q[2i] * sin(θ)  →  packets to output[2i+1] (positive)
    - q[2i+1] * cos(θ) → packets to output[2i+1] (positive)

THE SOLUTION:
  1. Pre-compute sin/cos lookup tables for each (position, dimension)
  2. Quantize sin/cos values to integers (scale factor)
  3. For each position, treat as position-specific "weights"
  4. Use dual counters (pos/neg) for the subtraction in cos rotation

Author: Research Phase 001
Date: December 2025
"""

import time
import os
import sys
import re
import subprocess
import numpy as np
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from previous experiments
from e053_mac_encoded_layers import get_layer_neuron_mac
from e045_real_weights_inference import mac_str_to_bytes
from e042_port_based_layers import (
    ssh_command, run_config_commands,
    craft_vlan_packet, send_packets, get_mac_address,
    SWITCH1_IP, SWITCH2_IP, SEND_IFACE,
)

# Architecture (simplified for proof)
DIM = 8           # Number of dimensions (must be even, pairs = DIM/2)
NUM_PAIRS = DIM // 2  # 4 dimension pairs
MAX_POSITIONS = 16    # Test positions 0-15
VALUE_RANGE = (-8, 8)  # Input value range (4-bit)

# Qwen3-0.6B uses base=1000000, but for testing we use a smaller base
# so the rotations are more visible in small position ranges
ROPE_BASE = 10.0  # For testing (real: 1000000)

FILTER_NAME = "rope_proof"
TEST_VLAN = 100
SCALE = 10  # Quantization scale for sin/cos


def ssh_command_long(switch_ip: str, command: str, timeout: int = 60) -> Tuple[bool, str, str]:
    """SSH command with configurable timeout."""
    ssh_key = "/home/multiplex/.ssh/id_rsa"
    cmd = [
        'ssh', '-i', ssh_key,
        '-o', 'StrictHostKeyChecking=no',
        '-o', 'UserKnownHostsFile=/dev/null',
        '-o', 'LogLevel=ERROR',
        f'root@{switch_ip}',
        command
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return True, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, '', f'Timeout after {timeout}s'
    except Exception as e:
        return False, '', str(e)


# =============================================================================
# RoPE LOOKUP TABLES
# =============================================================================

def compute_rope_freqs(dim: int, base: float = 10000.0) -> np.ndarray:
    """
    Compute the frequency for each dimension pair.
    
    θ_i = 1 / (base^(2i/dim))
    """
    # For each pair i, compute 1 / base^(2i/dim)
    pair_indices = np.arange(dim // 2)
    freqs = 1.0 / (base ** (2 * pair_indices / dim))
    return freqs


def build_rope_lut(max_pos: int, dim: int, base: float, scale: int) -> Dict:
    """
    Build lookup tables for sin and cos at each (position, dimension_pair).
    
    Returns dict with:
      'cos': [max_pos, dim//2] quantized cos values
      'sin': [max_pos, dim//2] quantized sin values
      'cos_float': [max_pos, dim//2] exact cos values
      'sin_float': [max_pos, dim//2] exact sin values
    """
    freqs = compute_rope_freqs(dim, base)
    
    cos_table = np.zeros((max_pos, dim // 2), dtype=np.int32)
    sin_table = np.zeros((max_pos, dim // 2), dtype=np.int32)
    cos_float = np.zeros((max_pos, dim // 2))
    sin_float = np.zeros((max_pos, dim // 2))
    
    for pos in range(max_pos):
        for i, freq in enumerate(freqs):
            theta = pos * freq
            c = np.cos(theta)
            s = np.sin(theta)
            
            cos_float[pos, i] = c
            sin_float[pos, i] = s
            
            # Quantize to integers
            cos_table[pos, i] = int(round(c * scale))
            sin_table[pos, i] = int(round(s * scale))
    
    return {
        'cos': cos_table,
        'sin': sin_table,
        'cos_float': cos_float,
        'sin_float': sin_float,
        'freqs': freqs,
        'scale': scale,
    }


def print_rope_table(lut: Dict, positions: List[int] = None):
    """Print the RoPE lookup table for inspection."""
    if positions is None:
        positions = [0, 1, 2, 4, 8]
    
    dim_pairs = lut['cos'].shape[1]
    
    print(f"\n  RoPE Lookup Tables (scale={lut['scale']}):")
    print(f"  Frequencies per pair: {lut['freqs']}")
    print("  " + "-" * 70)
    print(f"  {'Pos':>4} │ {'Pair':>4} │ {'cos(θ)':>10} │ {'sin(θ)':>10} │ {'cos_q':>6} │ {'sin_q':>6}")
    print("  " + "-" * 70)
    
    for pos in positions:
        if pos >= lut['cos'].shape[0]:
            continue
        for pair in range(min(2, dim_pairs)):  # Just show first 2 pairs
            c = lut['cos_float'][pos, pair]
            s = lut['sin_float'][pos, pair]
            cq = lut['cos'][pos, pair]
            sq = lut['sin'][pos, pair]
            print(f"  {pos:>4} │ {pair:>4} │ {c:>10.4f} │ {s:>10.4f} │ {cq:>6} │ {sq:>6}")
        if pos != positions[-1]:
            print("  " + "-" * 70)
    print("  " + "-" * 70)


# =============================================================================
# CPU REFERENCE
# =============================================================================

def cpu_rope_exact(x: np.ndarray, pos: int, lut: Dict) -> np.ndarray:
    """
    Apply RoPE rotation using exact floating-point math.
    
    x: [dim] input vector
    pos: position index
    Returns: [dim] rotated vector
    """
    dim = len(x)
    result = np.zeros(dim)
    
    cos_vals = lut['cos_float'][pos]
    sin_vals = lut['sin_float'][pos]
    
    for i in range(dim // 2):
        x0 = x[2 * i]
        x1 = x[2 * i + 1]
        c = cos_vals[i]
        s = sin_vals[i]
        
        # Rotation matrix:
        # [x0'] = [cos  -sin] [x0]
        # [x1'] = [sin   cos] [x1]
        result[2 * i] = x0 * c - x1 * s
        result[2 * i + 1] = x0 * s + x1 * c
    
    return result


def cpu_rope_quantized(x: np.ndarray, pos: int, lut: Dict) -> np.ndarray:
    """
    Apply RoPE rotation using quantized sin/cos values.
    
    This matches what the switch will compute.
    """
    dim = len(x)
    result = np.zeros(dim, dtype=np.int32)
    
    cos_vals = lut['cos'][pos]  # Quantized cos
    sin_vals = lut['sin'][pos]  # Quantized sin
    scale = lut['scale']
    
    for i in range(dim // 2):
        x0 = int(x[2 * i])
        x1 = int(x[2 * i + 1])
        c = cos_vals[i]
        s = sin_vals[i]
        
        # Rotation with quantized values
        # Result needs to be divided by scale later, but for counting we keep raw
        result[2 * i] = x0 * c - x1 * s  # Note: subtraction!
        result[2 * i + 1] = x0 * s + x1 * c
    
    return result


# =============================================================================
# SWITCH CONFIGURATION
# =============================================================================

def full_cleanup():
    """Clean switch configuration thoroughly."""
    print("\n  Cleanup...")
    
    thorough_cleanup_cmds = [
        "delete vlans",
        "delete firewall family ethernet-switching filter",
        "delete interfaces et-0/0/96 unit 0 family ethernet-switching",
        "delete interfaces et-0/0/100 unit 0 family ethernet-switching",
    ]
    
    cleanup_config = "; ".join(thorough_cleanup_cmds)
    ssh_command_long(
        SWITCH1_IP,
        f"cli -c 'configure; {cleanup_config}; commit'",
        timeout=30
    )
    
    time.sleep(1)
    print("  ✓ Done")


def configure_filters(dim: int):
    """
    Configure filters for RoPE computation.
    
    We need dual counters (pos/neg) for each output dimension
    because the rotation involves subtraction.
    """
    print(f"\n  Configuring filters...")
    
    all_cmds = []
    
    all_cmds.append("set forwarding-options storm-control-profiles default all")
    all_cmds.append(f"set firewall family ethernet-switching filter {FILTER_NAME}")
    
    # Positive counters for each output dimension
    for d in range(dim):
        mac = get_layer_neuron_mac(0, d)  # Layer 0, neuron d (pos counter)
        term = f"pos{d}"
        all_cmds.extend([
            f"set firewall family ethernet-switching filter {FILTER_NAME} term {term} from destination-mac-address {mac}/48",
            f"set firewall family ethernet-switching filter {FILTER_NAME} term {term} then count {term}",
            f"set firewall family ethernet-switching filter {FILTER_NAME} term {term} then accept",
        ])
    
    # Negative counters for each output dimension
    for d in range(dim):
        mac = get_layer_neuron_mac(1, d)  # Layer 1, neuron d (neg counter)
        term = f"neg{d}"
        all_cmds.extend([
            f"set firewall family ethernet-switching filter {FILTER_NAME} term {term} from destination-mac-address {mac}/48",
            f"set firewall family ethernet-switching filter {FILTER_NAME} term {term} then count {term}",
            f"set firewall family ethernet-switching filter {FILTER_NAME} term {term} then accept",
        ])
    
    all_cmds.extend([
        f"set firewall family ethernet-switching filter {FILTER_NAME} term default then count default_pkts",
        f"set firewall family ethernet-switching filter {FILTER_NAME} term default then accept",
    ])
    
    all_cmds.extend([
        "delete interfaces et-0/0/100 unit 0 family ethernet-switching",
        f"set vlans test_vlan vlan-id {TEST_VLAN}",
        "set interfaces et-0/0/96 unit 0 family ethernet-switching interface-mode access",
        f"set interfaces et-0/0/96 unit 0 family ethernet-switching vlan members test_vlan",
        f"set interfaces et-0/0/96 unit 0 family ethernet-switching filter input {FILTER_NAME}",
    ])
    
    print(f"    Positive counters (layer 0): {dim}")
    print(f"    Negative counters (layer 1): {dim}")
    print(f"    Total counters: {dim * 2}")
    
    config_file = "/tmp/e072_config.txt"
    with open(config_file, 'w') as f:
        f.write('\n'.join(all_cmds))
    
    ssh_key = "/home/multiplex/.ssh/id_rsa"
    ssh_cmd = [
        'ssh', '-i', ssh_key,
        '-o', 'StrictHostKeyChecking=no',
        '-o', 'UserKnownHostsFile=/dev/null',
        '-o', 'LogLevel=ERROR',
        f'root@{SWITCH1_IP}',
        'cat > /var/tmp/config.txt'
    ]
    with open(config_file, 'rb') as f:
        result = subprocess.run(ssh_cmd, stdin=f, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"    ✗ Transfer failed: {result.stderr}")
        return False
    
    load_cmd = "cli -c 'configure; load set /var/tmp/config.txt; commit'"
    success, stdout, stderr = ssh_command_long(SWITCH1_IP, load_cmd, timeout=60)
    
    if not success or 'error' in stdout.lower():
        print(f"    ✗ Config failed: {stdout[:200]}")
        return False
    
    print("  ✓ Configuration complete")
    time.sleep(1)
    return True


def clear_counters():
    """Clear all firewall counters."""
    ssh_command_long(SWITCH1_IP, f"cli -c 'clear firewall filter {FILTER_NAME}'", timeout=30)
    time.sleep(0.2)


def read_dual_counters(dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """Read positive and negative counters."""
    success, stdout, _ = ssh_command_long(
        SWITCH1_IP,
        f"cli -c 'show firewall filter {FILTER_NAME}'",
        timeout=30
    )
    
    pos_vals = np.zeros(dim, dtype=np.int32)
    neg_vals = np.zeros(dim, dtype=np.int32)
    
    if not success or not stdout:
        return pos_vals, neg_vals
    
    for d in range(dim):
        # Positive counter
        pattern = rf'pos{d}\s+\d+\s+(\d+)'
        match = re.search(pattern, stdout)
        if match:
            pos_vals[d] = int(match.group(1))
        
        # Negative counter
        pattern = rf'neg{d}\s+\d+\s+(\d+)'
        match = re.search(pattern, stdout)
        if match:
            neg_vals[d] = int(match.group(1))
    
    return pos_vals, neg_vals


# =============================================================================
# PACKET CREATION
# =============================================================================

def create_rope_packets(x: np.ndarray, pos: int, lut: Dict, src_mac: str) -> Tuple[List[bytes], np.ndarray]:
    """
    Create packets for RoPE rotation on switch.
    
    For each dimension pair (2i, 2i+1):
      Output[2i]   = x[2i]*cos - x[2i+1]*sin
      Output[2i+1] = x[2i]*sin + x[2i+1]*cos
    
    We use:
      - Positive counter: accumulates positive contributions
      - Negative counter: accumulates negative contributions
      - Final value = pos - neg
    
    Packets:
      x[2i] * cos   → pos counter for output 2i   (if cos > 0, else neg)
      x[2i+1] * sin → neg counter for output 2i   (if sin > 0, else pos - inverted!)
      x[2i] * sin   → pos counter for output 2i+1 (if sin > 0, else neg)
      x[2i+1] * cos → pos counter for output 2i+1 (if cos > 0, else neg)
    """
    packets = []
    src = mac_str_to_bytes(src_mac)
    dim = len(x)
    
    cos_vals = lut['cos'][pos]
    sin_vals = lut['sin'][pos]
    
    # Track expected outputs (pos - neg)
    expected_pos = np.zeros(dim, dtype=np.int32)
    expected_neg = np.zeros(dim, dtype=np.int32)
    
    for i in range(dim // 2):
        x0 = int(x[2 * i])
        x1 = int(x[2 * i + 1])
        c = cos_vals[i]
        s = sin_vals[i]
        
        # Output dimension indices
        out0 = 2 * i
        out1 = 2 * i + 1
        
        # Calculate contributions for output[2i] = x0*c - x1*s
        contrib_x0_c = x0 * c  # Goes to output[2i]
        contrib_x1_s = x1 * s  # SUBTRACTED from output[2i]
        
        # Calculate contributions for output[2i+1] = x0*s + x1*c
        contrib_x0_s = x0 * s  # Goes to output[2i+1]
        contrib_x1_c = x1 * c  # Goes to output[2i+1]
        
        # Route contributions to pos/neg counters
        def add_packets(value: int, out_dim: int, is_positive: bool):
            """Add packets for a value to the appropriate counter."""
            if value == 0:
                return
            
            # Determine which counter based on sign and is_positive
            if (value > 0 and is_positive) or (value < 0 and not is_positive):
                # Positive contribution: pos counter
                mac = get_layer_neuron_mac(0, out_dim)
            else:
                # Negative contribution: neg counter  
                mac = get_layer_neuron_mac(1, out_dim)
            
            dst = mac_str_to_bytes(mac)
            abs_val = abs(value)
            
            for _ in range(abs_val):
                packets.append(craft_vlan_packet(dst, src, TEST_VLAN))
            
            # Track expected
            if (value > 0 and is_positive) or (value < 0 and not is_positive):
                expected_pos[out_dim] += abs_val
            else:
                expected_neg[out_dim] += abs_val
        
        # Output[2i] = x0*c - x1*s
        add_packets(contrib_x0_c, out0, True)   # +x0*c
        add_packets(contrib_x1_s, out0, False)  # -x1*s
        
        # Output[2i+1] = x0*s + x1*c
        add_packets(contrib_x0_s, out1, True)   # +x0*s
        add_packets(contrib_x1_c, out1, True)   # +x1*c
    
    expected_output = expected_pos - expected_neg
    
    return packets, expected_output


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_rope_experiment():
    """Run the RoPE experiment."""
    print("="*80)
    print("E072: RoPE (ROTARY POSITION EMBEDDINGS) ON SWITCHES")
    print("="*80)
    print(f"""
  GOAL: Prove RoPE can work on switches!
  
  RoPE FORMULA:
    For each dimension pair (2i, 2i+1):
      q_rot[2i]   = q[2i] × cos(θ) - q[2i+1] × sin(θ)
      q_rot[2i+1] = q[2i] × sin(θ) + q[2i+1] × cos(θ)
    
    Where θ = position × freq_i (freq depends on dimension)
  
  THE CHALLENGE:
    1. Position-dependent (different θ for each position)
    2. Transcendental functions (sin/cos)
    3. Involves subtraction (the -sin term)
  
  THE SOLUTION:
    1. Pre-compute sin/cos LUT for each (position, dimension)
    2. Quantize sin/cos to integers
    3. Use DUAL COUNTERS (pos/neg) for subtraction
    4. Final result = pos_counter - neg_counter
    
    This is just a position-specific "weighted sum" - exactly
    what we already do for matrix multiplies!
""")
    
    # Build lookup tables
    print("\n" + "="*60)
    print("STEP 1: BUILD RoPE LOOKUP TABLES")
    print("="*60)
    
    lut = build_rope_lut(MAX_POSITIONS, DIM, ROPE_BASE, SCALE)
    print_rope_table(lut)
    
    # Cleanup
    full_cleanup()
    
    # Configure
    if not configure_filters(DIM):
        print("  ✗ Configuration failed!")
        return False
    
    src_mac = get_mac_address(SEND_IFACE)
    print(f"\n  Source MAC: {src_mac}")
    
    # Test 1: Single position rotation
    print("\n" + "="*60)
    print("TEST 1: SINGLE POSITION RoPE")
    print("="*60)
    
    np.random.seed(42)
    x = np.random.randint(VALUE_RANGE[0], VALUE_RANGE[1] + 1, DIM)
    pos = 3
    
    print(f"\n  Input vector: {x}")
    print(f"  Position: {pos}")
    print(f"  cos values: {lut['cos'][pos]}")
    print(f"  sin values: {lut['sin'][pos]}")
    
    # CPU references
    cpu_exact = cpu_rope_exact(x, pos, lut)
    cpu_quant = cpu_rope_quantized(x, pos, lut)
    
    print(f"\n  CPU exact (float): {cpu_exact}")
    print(f"  CPU quantized (int): {cpu_quant}")
    
    # Switch computation
    clear_counters()
    
    packets, expected_output = create_rope_packets(x, pos, lut, src_mac)
    
    print(f"\n  Sending packets:")
    print(f"    Total packets: {len(packets)}")
    
    if packets:
        send_packets(SEND_IFACE, packets)
        print(f"    ✓ Sent {len(packets)} packets")
    
    time.sleep(0.3)
    
    # Read counters
    pos_counters, neg_counters = read_dual_counters(DIM)
    switch_output = pos_counters - neg_counters
    
    print(f"\n  Switch results:")
    print(f"    Positive counters: {pos_counters}")
    print(f"    Negative counters: {neg_counters}")
    print(f"    Result (pos-neg):  {switch_output}")
    print(f"    Expected:          {expected_output}")
    print(f"    CPU quantized:     {cpu_quant}")
    
    # Compare
    match_expected = np.array_equal(switch_output, expected_output)
    match_cpu = np.array_equal(switch_output, cpu_quant)
    
    print(f"\n  Verification:")
    print(f"    Match expected packets: {'✓' if match_expected else '✗'}")
    print(f"    Match CPU quantized:    {'✓' if match_cpu else '✗'}")
    
    # Test 2: Multiple positions
    print("\n" + "="*60)
    print("TEST 2: RoPE ACROSS MULTIPLE POSITIONS")
    print("="*60)
    
    test_positions = [0, 1, 4, 8, 15]
    all_match = True
    
    for test_pos in test_positions:
        # Same input vector, different positions
        clear_counters()
        
        # CPU reference
        cpu_quant = cpu_rope_quantized(x, test_pos, lut)
        
        # Switch
        packets, _ = create_rope_packets(x, test_pos, lut, src_mac)
        if packets:
            send_packets(SEND_IFACE, packets)
        time.sleep(0.2)
        
        pos_counters, neg_counters = read_dual_counters(DIM)
        switch_output = pos_counters - neg_counters
        
        match = np.array_equal(switch_output, cpu_quant)
        all_match = all_match and match
        
        status = "✓" if match else "✗"
        print(f"    Position {test_pos:2d}: CPU={cpu_quant[:4]}... Switch={switch_output[:4]}... {status}")
    
    # Test 3: Different input vectors
    print("\n" + "="*60)
    print("TEST 3: RoPE WITH DIFFERENT INPUT VECTORS")
    print("="*60)
    
    n_vectors = 5
    fixed_pos = 5
    vectors_match = 0
    
    for v in range(n_vectors):
        test_x = np.random.randint(VALUE_RANGE[0], VALUE_RANGE[1] + 1, DIM)
        
        clear_counters()
        
        cpu_quant = cpu_rope_quantized(test_x, fixed_pos, lut)
        
        packets, _ = create_rope_packets(test_x, fixed_pos, lut, src_mac)
        if packets:
            send_packets(SEND_IFACE, packets)
        time.sleep(0.2)
        
        pos_counters, neg_counters = read_dual_counters(DIM)
        switch_output = pos_counters - neg_counters
        
        match = np.array_equal(switch_output, cpu_quant)
        if match:
            vectors_match += 1
        
        status = "✓" if match else "✗"
        print(f"    Vector {v+1}: x={test_x[:4]}... CPU={cpu_quant[:4]}... Switch={switch_output[:4]}... {status}")
    
    # Summary
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    test1_pass = match_expected and match_cpu
    test2_pass = all_match
    test3_pass = (vectors_match == n_vectors)
    
    print(f"""
  Test 1 (Single position):
    Packet counting correct: {'✓' if match_expected else '✗'}
    Matches CPU quantized:   {'✓' if match_cpu else '✗'}
  
  Test 2 (Multiple positions): {'✓ All passed' if test2_pass else '✗ Some failed'}
    Tested positions: {test_positions}
  
  Test 3 (Different vectors): {vectors_match}/{n_vectors} = {100*vectors_match/n_vectors:.0f}%
""")
    
    all_pass = test1_pass and test2_pass and test3_pass
    
    if all_pass:
        print(f"""
  🎉 RoPE ON SWITCHES PROVEN! 🎉
  
  What we demonstrated:
    1. Position-dependent sin/cos via pre-computed LUT
    2. Rotation as weighted sums (like matrix multiply)
    3. Subtraction via dual counters (pos - neg)
    4. 100% match with CPU quantized reference!
  
  Key insight:
    For a FIXED position, RoPE is just a sparse 2×2 block-diagonal
    matrix multiply where the "weights" are sin/cos values.
    
    The switch already does matrix multiply - we just use
    position-specific weights from the LUT!
  
  For real inference:
    1. Pre-compute sin/cos LUT for all positions needed
    2. When processing token at position P:
       - Look up sin_P and cos_P for each dimension pair
       - These become the "weights" for the rotation
       - Switch computes the weighted sum via packet counting
  
  COMPLETE TRANSFORMER OPERATIONS ON SWITCH:
    ✓ Matrix multiply (e056)
    ✓ Element-wise multiply (e066)
    ✓ SiLU activation (e067)
    ✓ RMSNorm (e068)
    ✓ Residual connection (e070)
    ✓ Softmax / Argmax (e071)
    ✓ RoPE position encoding (e072) ← NEW!
""")
    else:
        print(f"""
  Some tests failed. Debugging info:
    Test 1: {'PASS' if test1_pass else 'FAIL'}
    Test 2: {'PASS' if test2_pass else 'FAIL'}  
    Test 3: {'PASS' if test3_pass else 'FAIL'}
""")
    
    full_cleanup()
    
    return all_pass


def print_rope_math_explanation():
    """Print detailed explanation of RoPE math for documentation."""
    print("\n" + "="*80)
    print("APPENDIX: HOW RoPE WORKS ON SWITCH")
    print("="*80)
    print("""
  ROPE ROTATION MATRIX:
    
    For dimension pair (2i, 2i+1) at position p:
    
    θ = p × freq_i   where freq_i = 1 / (base^(2i/dim))
    
    [q'[2i]  ]   [cos(θ)  -sin(θ)] [q[2i]  ]
    [q'[2i+1]] = [sin(θ)   cos(θ)] [q[2i+1]]
    
    Expanding:
      q'[2i]   = q[2i]×cos(θ) - q[2i+1]×sin(θ)
      q'[2i+1] = q[2i]×sin(θ) + q[2i+1]×cos(θ)
    
  SWITCH IMPLEMENTATION:
    
    1. LUT CREATION (once per model):
       For each position p and dimension pair i:
         cos_lut[p][i] = quantize(cos(p × freq_i))
         sin_lut[p][i] = quantize(sin(p × freq_i))
    
    2. PACKET CREATION (per inference):
       For position p and input q:
         - Look up cos_p[i] = cos_lut[p][i]
         - Look up sin_p[i] = sin_lut[p][i]
         
       For output[2i] = q[2i]×cos - q[2i+1]×sin:
         - Send |q[2i] × cos_p[i]| packets to pos_counter[2i] (if positive)
         - Send |q[2i+1] × sin_p[i]| packets to neg_counter[2i] (if positive)
         (Handle signs appropriately)
       
       For output[2i+1] = q[2i]×sin + q[2i+1]×cos:
         - Send |q[2i] × sin_p[i]| packets to pos_counter[2i+1]
         - Send |q[2i+1] × cos_p[i]| packets to pos_counter[2i+1]
    
    3. COUNTER READING:
       final[d] = pos_counter[d] - neg_counter[d]
    
  WHY THIS WORKS:
    - sin/cos are deterministic for each (position, dimension)
    - We pre-compute them, so they're just constants at inference time
    - Multiplication and addition are what switches do (via packet counting)
    - Subtraction uses dual counters (same as signed weights in e054)
    
  SCALING CONSIDERATIONS:
    - Real models use base=10000 or 1000000
    - This makes θ change slowly with position
    - For long sequences, sin/cos cycle through their range
    - Quantization error is small when using sufficient scale factor
""")


if __name__ == '__main__':
    success = run_rope_experiment()
    if success:
        print_rope_math_explanation()


""" Output:
sudo python3 e072_rope_on_switch.py 
================================================================================
E072: RoPE (ROTARY POSITION EMBEDDINGS) ON SWITCHES
================================================================================

  GOAL: Prove RoPE can work on switches!
  
  RoPE FORMULA:
    For each dimension pair (2i, 2i+1):
      q_rot[2i]   = q[2i] × cos(θ) - q[2i+1] × sin(θ)
      q_rot[2i+1] = q[2i] × sin(θ) + q[2i+1] × cos(θ)
    
    Where θ = position × freq_i (freq depends on dimension)
  
  THE CHALLENGE:
    1. Position-dependent (different θ for each position)
    2. Transcendental functions (sin/cos)
    3. Involves subtraction (the -sin term)
  
  THE SOLUTION:
    1. Pre-compute sin/cos LUT for each (position, dimension)
    2. Quantize sin/cos to integers
    3. Use DUAL COUNTERS (pos/neg) for subtraction
    4. Final result = pos_counter - neg_counter
    
    This is just a position-specific "weighted sum" - exactly
    what we already do for matrix multiplies!


============================================================
STEP 1: BUILD RoPE LOOKUP TABLES
============================================================

  RoPE Lookup Tables (scale=10):
  Frequencies per pair: [1.         0.56234133 0.31622777 0.17782794]
  ----------------------------------------------------------------------
   Pos │ Pair │     cos(θ) │     sin(θ) │  cos_q │  sin_q
  ----------------------------------------------------------------------
     0 │    0 │     1.0000 │     0.0000 │     10 │      0
     0 │    1 │     1.0000 │     0.0000 │     10 │      0
  ----------------------------------------------------------------------
     1 │    0 │     0.5403 │     0.8415 │      5 │      8
     1 │    1 │     0.8460 │     0.5332 │      8 │      5
  ----------------------------------------------------------------------
     2 │    0 │    -0.4161 │     0.9093 │     -4 │      9
     2 │    1 │     0.4315 │     0.9021 │      4 │      9
  ----------------------------------------------------------------------
     4 │    0 │    -0.6536 │    -0.7568 │     -7 │     -8
     4 │    1 │    -0.6277 │     0.7785 │     -6 │      8
  ----------------------------------------------------------------------
     8 │    0 │    -0.1455 │     0.9894 │     -1 │     10
     8 │    1 │    -0.2120 │    -0.9773 │     -2 │    -10
  ----------------------------------------------------------------------

  Cleanup...
  ✓ Done

  Configuring filters...
    Positive counters (layer 0): 8
    Negative counters (layer 1): 8
    Total counters: 16
  ✓ Configuration complete

  Source MAC: 7c:fe:90:9d:2a:f0

============================================================
TEST 1: SINGLE POSITION RoPE
============================================================

  Input vector: [-2  6  2 -1 -2  2  2 -5]
  Position: 3
  cos values: [-10  -1   6   9]
  sin values: [ 1 10  8  5]

  CPU exact (float): [ 1.13326494 -6.222195    0.76132088  2.10247248 -2.79080501 -0.45979057
  4.26476197 -3.28813098]
  CPU quantized (int): [ 14 -62   8  21 -28  -4  43 -35]

  Sending packets:
    Total packets: 275
    ✓ Sent 275 packets

  Switch results:
    Positive counters: [20  0 10 21  0 12 43 10]
    Negative counters: [ 6 62  2  0 28 16  0 45]
    Result (pos-neg):  [ 14 -62   8  21 -28  -4  43 -35]
    Expected:          [ 14 -62   8  21 -28  -4  43 -35]
    CPU quantized:     [ 14 -62   8  21 -28  -4  43 -35]

  Verification:
    Match expected packets: ✓
    Match CPU quantized:    ✓

============================================================
TEST 2: RoPE ACROSS MULTIPLE POSITIONS
============================================================
    Position  0: CPU=[-20  60  20 -10]... Switch=[-20  60  20 -10]... ✓
    Position  1: CPU=[-58  14  21   2]... Switch=[-58  14  21   2]... ✓
    Position  4: CPU=[ 62 -26  -4  22]... Switch=[ 62 -26  -4  22]... ✓
    Position  8: CPU=[-58 -26 -14 -18]... Switch=[-58 -26 -14 -18]... ✓
    Position 15: CPU=[-26 -62  -2  21]... Switch=[-26 -62  -2  21]... ✓

============================================================
TEST 3: RoPE WITH DIFFERENT INPUT VECTORS
============================================================
    Vector 1: x=[-1 -6 -7  3]... CPU=[-63  -8  54 -48]... Switch=[-63  -8  54 -48]... ✓
    Vector 2: x=[3 8 1 7]... CPU=[ 89  -6 -30 -60]... Switch=[ 89  -6 -30 -60]... ✓
    Vector 3: x=[-4 -2  0 -2]... CPU=[-32  34   6  18]... Switch=[-32  34   6  18]... ✓
    Vector 4: x=[ 6 -2  3 -1]... CPU=[ -2 -66 -24  18]... Switch=[ -2 -66 -24  18]... ✓
    Vector 5: x=[-5 -1 -5 -7]... CPU=[-25  47  66  48]... Switch=[-25  47  66  48]... ✓

================================================================================
RESULTS
================================================================================

  Test 1 (Single position):
    Packet counting correct: ✓
    Matches CPU quantized:   ✓
  
  Test 2 (Multiple positions): ✓ All passed
    Tested positions: [0, 1, 4, 8, 15]
  
  Test 3 (Different vectors): 5/5 = 100%


  🎉 RoPE ON SWITCHES PROVEN! 🎉
  
  What we demonstrated:
    1. Position-dependent sin/cos via pre-computed LUT
    2. Rotation as weighted sums (like matrix multiply)
    3. Subtraction via dual counters (pos - neg)
    4. 100% match with CPU quantized reference!
  
  Key insight:
    For a FIXED position, RoPE is just a sparse 2×2 block-diagonal
    matrix multiply where the "weights" are sin/cos values.
    
    The switch already does matrix multiply - we just use
    position-specific weights from the LUT!
  
  For real inference:
    1. Pre-compute sin/cos LUT for all positions needed
    2. When processing token at position P:
       - Look up sin_P and cos_P for each dimension pair
       - These become the "weights" for the rotation
       - Switch computes the weighted sum via packet counting
  
  COMPLETE TRANSFORMER OPERATIONS ON SWITCH:
    ✓ Matrix multiply (e056)
    ✓ Element-wise multiply (e066)
    ✓ SiLU activation (e067)
    ✓ RMSNorm (e068)
    ✓ Residual connection (e070)
    ✓ Softmax / Argmax (e071)
    ✓ RoPE position encoding (e072) ← NEW!


  Cleanup...
  ✓ Done

================================================================================
APPENDIX: HOW RoPE WORKS ON SWITCH
================================================================================

  ROPE ROTATION MATRIX:
    
    For dimension pair (2i, 2i+1) at position p:
    
    θ = p × freq_i   where freq_i = 1 / (base^(2i/dim))
    
    [q'[2i]  ]   [cos(θ)  -sin(θ)] [q[2i]  ]
    [q'[2i+1]] = [sin(θ)   cos(θ)] [q[2i+1]]
    
    Expanding:
      q'[2i]   = q[2i]×cos(θ) - q[2i+1]×sin(θ)
      q'[2i+1] = q[2i]×sin(θ) + q[2i+1]×cos(θ)
    
  SWITCH IMPLEMENTATION:
    
    1. LUT CREATION (once per model):
       For each position p and dimension pair i:
         cos_lut[p][i] = quantize(cos(p × freq_i))
         sin_lut[p][i] = quantize(sin(p × freq_i))
    
    2. PACKET CREATION (per inference):
       For position p and input q:
         - Look up cos_p[i] = cos_lut[p][i]
         - Look up sin_p[i] = sin_lut[p][i]
         
       For output[2i] = q[2i]×cos - q[2i+1]×sin:
         - Send |q[2i] × cos_p[i]| packets to pos_counter[2i] (if positive)
         - Send |q[2i+1] × sin_p[i]| packets to neg_counter[2i] (if positive)
         (Handle signs appropriately)
       
       For output[2i+1] = q[2i]×sin + q[2i+1]×cos:
         - Send |q[2i] × sin_p[i]| packets to pos_counter[2i+1]
         - Send |q[2i+1] × cos_p[i]| packets to pos_counter[2i+1]
    
    3. COUNTER READING:
       final[d] = pos_counter[d] - neg_counter[d]
    
  WHY THIS WORKS:
    - sin/cos are deterministic for each (position, dimension)
    - We pre-compute them, so they're just constants at inference time
    - Multiplication and addition are what switches do (via packet counting)
    - Subtraction uses dual counters (same as signed weights in e054)
    
  SCALING CONSIDERATIONS:
    - Real models use base=10000 or 1000000
    - This makes θ change slowly with position
    - For long sequences, sin/cos cycle through their range
    - Quantization error is small when using sufficient scale factor
"""

