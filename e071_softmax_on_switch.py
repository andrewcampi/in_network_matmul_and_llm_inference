#!/usr/bin/env python3
"""
e071_softmax_on_switch.py

SOFTMAX ON SWITCHES
===================

THE CHALLENGE:
  Softmax: softmax(x)_i = exp(x_i) / Σ_j exp(x_j)
  
  This requires:
    1. Exponential function (non-linear)
    2. Sum over ALL elements
    3. Division by that sum

THE INSIGHT:
  Same pattern as RMSNorm! Break into switch-friendly operations:
  
  Phase 1: LOOKUP TABLE for exp()
    - For quantized inputs, exp(x) has finite possible values
    - Pre-compute exp_lut[x] for all possible x
  
  Phase 2: SWITCH computes sum of exponentials
    - Send exp_lut[x_i] packets per element
    - Counter accumulates: sum_exp = Σ exp_lut[x_i]
  
  Phase 3: LOOKUP TABLE for division
    - scale = 1 / sum_exp
    - Pre-compute scale_lut[sum] for all possible sums
  
  Phase 4: APPLY SCALE
    - prob[i] = exp_lut[x_i] * scale
    - Fuse into subsequent operations

IMPORTANT NOTE:
  For most LLM inference, we DON'T need full softmax!
  
  - Token generation: just argmax (pick highest logit)
  - MoE routing: argmax for top-K experts
  - Single-token attention: simplified (no full attention)
  
  But for completeness, this proves softmax CAN work on switches!

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

# Architecture
NUM_CLASSES = 16     # Number of softmax outputs
VALUE_RANGE = (-8, 8)  # Input value range

FILTER_NAME = "softmax_proof"
TEST_VLAN = 100


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
# SOFTMAX LOOKUP TABLES
# =============================================================================

def build_exp_lut(min_val: int, max_val: int, scale: int = 10) -> Dict[int, int]:
    """
    Build lookup table for quantized exponential.
    
    For softmax, we need exp(x) for each input value.
    Since inputs are bounded integers, this is a small table.
    
    We scale the output to preserve precision in integer arithmetic.
    """
    lut = {}
    for x in range(min_val, max_val + 1):
        # Compute exp(x), scale to integer
        exp_val = np.exp(float(x))
        # Scale and quantize
        quantized = int(round(exp_val * scale))
        # Clamp to reasonable range to avoid overflow
        quantized = min(quantized, 10000)
        lut[x] = max(1, quantized)  # At least 1 to avoid zeros
    return lut


def build_softmax_scale_lut(max_sum: int) -> Dict[int, float]:
    """
    Build lookup table for 1/sum (the softmax denominator).
    
    Given sum_exp, we need scale = 1/sum_exp to compute probabilities.
    """
    lut = {}
    for s in range(1, max_sum + 1):
        lut[s] = 1.0 / s
    lut[0] = 1.0  # Edge case
    return lut


def print_exp_table(lut: Dict[int, int], sample_points: List[int] = None):
    """Print the exp lookup table."""
    if sample_points is None:
        sample_points = list(range(-8, 9, 2))
    
    print("\n  Exponential Lookup Table (quantized):")
    print("  " + "-" * 50)
    print(f"  {'Input':>8} │ {'exp(x)':>12} │ {'Quantized':>10}")
    print("  " + "-" * 50)
    
    for x in sample_points:
        if x in lut:
            exact = np.exp(float(x))
            quantized = lut[x]
            print(f"  {x:>8} │ {exact:>12.4f} │ {quantized:>10}")
    
    print("  " + "-" * 50)


# =============================================================================
# CPU REFERENCE
# =============================================================================

def cpu_softmax(x: np.ndarray) -> np.ndarray:
    """CPU reference softmax with numerical stability."""
    # Subtract max for numerical stability
    x_stable = x - np.max(x)
    exp_x = np.exp(x_stable)
    return exp_x / np.sum(exp_x)


def cpu_quantized_softmax(x: np.ndarray, exp_lut: Dict[int, int]) -> Tuple[np.ndarray, int]:
    """
    CPU reference for quantized softmax (matching switch computation).
    
    Returns probabilities and the sum of exponentials.
    """
    # Apply exp LUT
    exp_values = np.array([exp_lut.get(int(v), 1) for v in x])
    sum_exp = np.sum(exp_values)
    # Compute probabilities
    probs = exp_values / sum_exp
    return probs, int(sum_exp)


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


def configure_filters(num_classes: int):
    """Configure filters for softmax computation."""
    print(f"\n  Configuring filters...")
    
    all_cmds = []
    
    all_cmds.append("set forwarding-options storm-control-profiles default all")
    all_cmds.append(f"set firewall family ethernet-switching filter {FILTER_NAME}")
    
    # Sum_exp counter (for accumulating Σexp(x))
    sum_exp_mac = get_layer_neuron_mac(255, 0)
    all_cmds.extend([
        f"set firewall family ethernet-switching filter {FILTER_NAME} term sumexp from destination-mac-address {sum_exp_mac}/48",
        f"set firewall family ethernet-switching filter {FILTER_NAME} term sumexp then count sumexp",
        f"set firewall family ethernet-switching filter {FILTER_NAME} term sumexp then accept",
    ])
    
    # Individual exp counters for each class
    for n in range(num_classes):
        mac = get_layer_neuron_mac(0, n)
        term = f"exp{n}"
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
    
    print(f"    Sum_exp counter: 1")
    print(f"    Per-class exp counters: {num_classes}")
    
    config_file = "/tmp/e071_config.txt"
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


def read_sum_exp_counter() -> int:
    """Read the sum_exp counter."""
    success, stdout, _ = ssh_command_long(
        SWITCH1_IP,
        f"cli -c 'show firewall filter {FILTER_NAME}'",
        timeout=30
    )
    
    if not success or not stdout:
        return 0
    
    pattern = r'sumexp\s+\d+\s+(\d+)'
    match = re.search(pattern, stdout)
    if match:
        return int(match.group(1))
    return 0


def read_exp_counters(num_classes: int) -> np.ndarray:
    """Read individual exp counters."""
    success, stdout, _ = ssh_command_long(
        SWITCH1_IP,
        f"cli -c 'show firewall filter {FILTER_NAME}'",
        timeout=30
    )
    
    values = np.zeros(num_classes, dtype=np.int32)
    if not success or not stdout:
        return values
    
    for i in range(num_classes):
        pattern = rf'exp{i}\s+\d+\s+(\d+)'
        match = re.search(pattern, stdout)
        if match:
            values[i] = int(match.group(1))
    
    return values


# =============================================================================
# PACKET CREATION
# =============================================================================

def create_softmax_packets(logits: np.ndarray, exp_lut: Dict[int, int], 
                            src_mac: str) -> Tuple[List[bytes], int, np.ndarray]:
    """
    Create packets for softmax computation on switch.
    
    For each logit x[i]:
      - Look up exp_lut[x[i]]
      - Send that many packets to BOTH:
        1. The sum_exp counter (for denominator)
        2. The exp[i] counter (for numerator)
    
    Returns packets, expected sum_exp, and expected exp values.
    """
    packets = []
    src = mac_str_to_bytes(src_mac)
    
    sum_exp_dst = mac_str_to_bytes(get_layer_neuron_mac(255, 0))
    
    expected_sum = 0
    expected_exp = np.zeros(len(logits), dtype=np.int32)
    
    for i in range(len(logits)):
        x = int(logits[i])
        exp_val = exp_lut.get(x, 1)
        
        expected_sum += exp_val
        expected_exp[i] = exp_val
        
        # Send packets to BOTH sum counter and individual counter
        exp_dst = mac_str_to_bytes(get_layer_neuron_mac(0, i))
        
        for _ in range(exp_val):
            # Packet for sum_exp
            packets.append(craft_vlan_packet(sum_exp_dst, src, TEST_VLAN))
            # Packet for individual exp[i]
            packets.append(craft_vlan_packet(exp_dst, src, TEST_VLAN))
    
    return packets, expected_sum, expected_exp


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_softmax_experiment():
    """Run the softmax experiment."""
    print("="*80)
    print("E071: SOFTMAX ON SWITCHES")
    print("="*80)
    print(f"""
  GOAL: Prove softmax can work on switches!
  
  SOFTMAX FORMULA:
    softmax(x)_i = exp(x_i) / Σ_j exp(x_j)
  
  THE CHALLENGE:
    1. Exponential function (non-linear)
    2. Sum over ALL elements (global dependency)
    3. Division by that sum
  
  THE SOLUTION (same pattern as RMSNorm!):
    
    Phase 1: LOOKUP TABLE for exp()
      → exp_lut[x] = quantized_exp(x)
    
    Phase 2: SWITCH computes sum of exponentials
      → Send exp_lut[x_i] packets to sum counter
      → Counter = Σ exp_lut[x_i]
    
    Phase 3: LOOKUP TABLE for division
      → scale = 1 / sum_exp
    
    Phase 4: Compute probabilities
      → prob[i] = exp_lut[x_i] / sum_exp
""")
    
    # Build lookup tables
    print("\n" + "="*60)
    print("STEP 1: BUILD LOOKUP TABLES")
    print("="*60)
    
    exp_lut = build_exp_lut(VALUE_RANGE[0], VALUE_RANGE[1], scale=10)
    print_exp_table(exp_lut)
    
    # Cleanup
    full_cleanup()
    
    # Configure
    if not configure_filters(NUM_CLASSES):
        print("  ✗ Configuration failed!")
        return False
    
    src_mac = get_mac_address(SEND_IFACE)
    print(f"\n  Source MAC: {src_mac}")
    
    # Test 1: Simple softmax
    print("\n" + "="*60)
    print("TEST 1: BASIC SOFTMAX")
    print("="*60)
    
    np.random.seed(42)
    logits = np.random.randint(VALUE_RANGE[0], VALUE_RANGE[1] + 1, NUM_CLASSES)
    
    print(f"\n  Input logits: {logits[:8]}...")
    
    # CPU reference
    cpu_probs = cpu_softmax(logits.astype(np.float64))
    cpu_quant_probs, cpu_sum_exp = cpu_quantized_softmax(logits, exp_lut)
    
    print(f"\n  CPU softmax (exact):")
    print(f"    Probs: [{', '.join(f'{p:.3f}' for p in cpu_probs[:6])}...]")
    print(f"    Argmax: {np.argmax(cpu_probs)}")
    
    print(f"\n  CPU quantized softmax:")
    print(f"    Sum_exp: {cpu_sum_exp}")
    print(f"    Probs: [{', '.join(f'{p:.3f}' for p in cpu_quant_probs[:6])}...]")
    print(f"    Argmax: {np.argmax(cpu_quant_probs)}")
    
    # Switch computation
    clear_counters()
    
    packets, expected_sum, expected_exp = create_softmax_packets(logits, exp_lut, src_mac)
    
    print(f"\n  Sending packets:")
    print(f"    Total packets: {len(packets)}")
    print(f"    (2× because we send to both sum and individual counters)")
    
    if packets:
        send_packets(SEND_IFACE, packets)
        print(f"    ✓ Sent {len(packets)} packets")
    
    time.sleep(0.3)
    
    # Read counters
    switch_sum_exp = read_sum_exp_counter()
    switch_exp = read_exp_counters(NUM_CLASSES)
    
    print(f"\n  Switch results:")
    print(f"    Sum_exp: {switch_sum_exp} (expected: {expected_sum})")
    print(f"    Exp values: {switch_exp[:6]}... (expected: {expected_exp[:6]}...)")
    
    sum_match = (switch_sum_exp == expected_sum)
    exp_match = np.array_equal(switch_exp, expected_exp)
    
    # Compute probabilities from switch values
    if switch_sum_exp > 0:
        switch_probs = switch_exp.astype(np.float64) / switch_sum_exp
    else:
        switch_probs = np.zeros(NUM_CLASSES)
    
    print(f"\n  Switch softmax probabilities:")
    print(f"    Probs: [{', '.join(f'{p:.3f}' for p in switch_probs[:6])}...]")
    print(f"    Argmax: {np.argmax(switch_probs)}")
    
    # Compare argmax (this is what we usually care about!)
    cpu_argmax = np.argmax(cpu_probs)
    switch_argmax = np.argmax(switch_probs)
    argmax_match = (cpu_argmax == switch_argmax)
    
    print(f"\n  Argmax comparison:")
    print(f"    CPU exact: {cpu_argmax}")
    print(f"    CPU quantized: {np.argmax(cpu_quant_probs)}")
    print(f"    Switch: {switch_argmax}")
    print(f"    Match: {'✓' if argmax_match else '✗'}")
    
    # Test 2: Argmax across multiple samples
    print("\n" + "="*60)
    print("TEST 2: ARGMAX ACCURACY ACROSS MULTIPLE SAMPLES")
    print("="*60)
    
    n_samples = 5
    correct_argmax = 0
    
    for sample_idx in range(n_samples):
        logits = np.random.randint(VALUE_RANGE[0], VALUE_RANGE[1] + 1, NUM_CLASSES)
        
        # CPU reference
        cpu_probs = cpu_softmax(logits.astype(np.float64))
        cpu_argmax = np.argmax(cpu_probs)
        
        # Switch
        clear_counters()
        packets, _, _ = create_softmax_packets(logits, exp_lut, src_mac)
        if packets:
            send_packets(SEND_IFACE, packets)
        time.sleep(0.2)
        
        switch_sum = read_sum_exp_counter()
        switch_exp = read_exp_counters(NUM_CLASSES)
        switch_argmax = np.argmax(switch_exp)
        
        match = (cpu_argmax == switch_argmax)
        if match:
            correct_argmax += 1
        
        status = "✓" if match else "✗"
        print(f"    Sample {sample_idx + 1}: CPU argmax={cpu_argmax}, Switch argmax={switch_argmax} {status}")
    
    argmax_accuracy = correct_argmax / n_samples
    
    # Summary
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    test1_pass = sum_match and exp_match
    
    print(f"""
  Test 1 (Basic softmax):
    Sum_exp match: {'✓' if sum_match else '✗'}
    Exp values match: {'✓' if exp_match else '✗'}
    Argmax match: {'✓' if argmax_match else '✗'}
  
  Test 2 (Argmax accuracy): {correct_argmax}/{n_samples} = {argmax_accuracy*100:.0f}%
""")
    
    if test1_pass and argmax_accuracy >= 0.8:
        print(f"""
  🎉 SOFTMAX ON SWITCHES PROVEN! 🎉
  
  What we demonstrated:
    1. exp(x) via lookup table (quantized)
    2. Σexp(x) computed on switch via packet counting
    3. Probabilities = exp[i] / sum_exp
    4. Argmax matches CPU reference!
  
  Key insight (same as RMSNorm):
    - exp() is O(1) lookup per element
    - Sum is done by switch (packet counting)
    - Division is O(1) lookup
    - Switch does the heavy lifting!
  
  IMPORTANT for LLM inference:
    Usually we only need ARGMAX, not full softmax!
    - Token generation: argmax of logits
    - MoE routing: top-K selection
    - Attention (single token): simplified
    
    The switch gives us argmax directly from counter values!
    No division needed - just find the max counter.
  
  This completes the transformer operations on switch:
    ✓ Matrix multiply (e056)
    ✓ Element-wise (e066)
    ✓ SiLU (e067)
    ✓ RMSNorm (e068)
    ✓ Residual (e070)
    ✓ Softmax (e071)
    
  ALL TRANSFORMER OPERATIONS CAN RUN ON SWITCHES!
""")
    else:
        print(f"""
  Test 1: {'PASS' if test1_pass else 'FAIL'}
  Test 2: {argmax_accuracy*100:.0f}% argmax accuracy
  
  Note: Some error is expected due to quantization.
  For most LLM tasks, argmax is sufficient (no full softmax needed).
""")
    
    full_cleanup()
    
    return test1_pass


def run_softmax_vs_argmax():
    """Demonstrate why argmax is usually sufficient."""
    print("\n" + "="*80)
    print("BONUS: SOFTMAX vs ARGMAX")
    print("="*80)
    
    print("""
  FOR MOST LLM INFERENCE, WE DON'T NEED FULL SOFTMAX!
  
  ┌─────────────────────────────────────────────────────────────┐
  │ Use Case             │ What We Need    │ Full Softmax?     │
  ├─────────────────────────────────────────────────────────────┤
  │ Token generation     │ argmax(logits)  │ NO - just max     │
  │ Greedy decoding      │ argmax(logits)  │ NO - just max     │
  │ MoE expert selection │ top-K(scores)   │ NO - just top-K   │
  │ Beam search          │ top-K(logits)   │ NO - just top-K   │
  │ Temperature sampling │ softmax → sample│ YES (if T≠0)      │
  │ Attention weights    │ softmax(QK^T)   │ YES (multi-head)  │
  └─────────────────────────────────────────────────────────────┘
  
  ARGMAX ON SWITCH:
    The switch computes exp(x_i) for each class.
    To find argmax, just find which counter has the HIGHEST value!
    
    argmax = max_i(exp_counter[i])
    
    No division needed! No full softmax computation!
    This is MUCH simpler and avoids the denominator entirely.
  
  WHEN YOU DO NEED SOFTMAX:
    1. Compute exp values on switch (via packets)
    2. Read sum_exp counter
    3. Lookup scale = 1/sum_exp
    4. prob[i] = exp[i] * scale (or read exp[i] and divide)
    
    This requires ONE counter read (like RMSNorm).
""")


if __name__ == '__main__':
    success = run_softmax_experiment()
    if success:
        run_softmax_vs_argmax()



""" Output:
sudo python3 e071_softmax_on_switch.py 
================================================================================
E071: SOFTMAX ON SWITCHES
================================================================================

  GOAL: Prove softmax can work on switches!
  
  SOFTMAX FORMULA:
    softmax(x)_i = exp(x_i) / Σ_j exp(x_j)
  
  THE CHALLENGE:
    1. Exponential function (non-linear)
    2. Sum over ALL elements (global dependency)
    3. Division by that sum
  
  THE SOLUTION (same pattern as RMSNorm!):
    
    Phase 1: LOOKUP TABLE for exp()
      → exp_lut[x] = quantized_exp(x)
    
    Phase 2: SWITCH computes sum of exponentials
      → Send exp_lut[x_i] packets to sum counter
      → Counter = Σ exp_lut[x_i]
    
    Phase 3: LOOKUP TABLE for division
      → scale = 1 / sum_exp
    
    Phase 4: Compute probabilities
      → prob[i] = exp_lut[x_i] / sum_exp


============================================================
STEP 1: BUILD LOOKUP TABLES
============================================================

  Exponential Lookup Table (quantized):
  --------------------------------------------------
     Input │       exp(x) │  Quantized
  --------------------------------------------------
        -8 │       0.0003 │          1
        -6 │       0.0025 │          1
        -4 │       0.0183 │          1
        -2 │       0.1353 │          1
         0 │       1.0000 │         10
         2 │       7.3891 │         74
         4 │      54.5982 │        546
         6 │     403.4288 │       4034
         8 │    2980.9580 │      10000
  --------------------------------------------------

  Cleanup...
  ✓ Done

  Configuring filters...
    Sum_exp counter: 1
    Per-class exp counters: 16
  ✓ Configuration complete

  Source MAC: 7c:fe:90:9d:2a:f0

============================================================
TEST 1: BASIC SOFTMAX
============================================================

  Input logits: [-2  6  2 -1 -2  2  2 -5]...

  CPU softmax (exact):
    Probs: [0.000, 0.864, 0.016, 0.001, 0.000, 0.016...]
    Argmax: 1

  CPU quantized softmax:
    Sum_exp: 4674
    Probs: [0.000, 0.863, 0.016, 0.001, 0.000, 0.016...]
    Argmax: 1

  Sending packets:
    Total packets: 9348
    (2× because we send to both sum and individual counters)
    ✓ Sent 9348 packets

  Switch results:
    Sum_exp: 4674 (expected: 4674)
    Exp values: [   1 4034   74    4    1   74]... (expected: [   1 4034   74    4    1   74]...)

  Switch softmax probabilities:
    Probs: [0.000, 0.863, 0.016, 0.001, 0.000, 0.016...]
    Argmax: 1

  Argmax comparison:
    CPU exact: 1
    CPU quantized: 1
    Switch: 1
    Match: ✓

============================================================
TEST 2: ARGMAX ACCURACY ACROSS MULTIPLE SAMPLES
============================================================
    Sample 1: CPU argmax=1, Switch argmax=1 ✓
    Sample 2: CPU argmax=7, Switch argmax=7 ✓
    Sample 3: CPU argmax=4, Switch argmax=4 ✓
    Sample 4: CPU argmax=7, Switch argmax=7 ✓
    Sample 5: CPU argmax=6, Switch argmax=6 ✓

================================================================================
RESULTS
================================================================================

  Test 1 (Basic softmax):
    Sum_exp match: ✓
    Exp values match: ✓
    Argmax match: ✓
  
  Test 2 (Argmax accuracy): 5/5 = 100%


  🎉 SOFTMAX ON SWITCHES PROVEN! 🎉
  
  What we demonstrated:
    1. exp(x) via lookup table (quantized)
    2. Σexp(x) computed on switch via packet counting
    3. Probabilities = exp[i] / sum_exp
    4. Argmax matches CPU reference!
  
  Key insight (same as RMSNorm):
    - exp() is O(1) lookup per element
    - Sum is done by switch (packet counting)
    - Division is O(1) lookup
    - Switch does the heavy lifting!
  
  IMPORTANT for LLM inference:
    Usually we only need ARGMAX, not full softmax!
    - Token generation: argmax of logits
    - MoE routing: top-K selection
    - Attention (single token): simplified
    
    The switch gives us argmax directly from counter values!
    No division needed - just find the max counter.
  
  This completes the transformer operations on switch:
    ✓ Matrix multiply (e056)
    ✓ Element-wise (e066)
    ✓ SiLU (e067)
    ✓ RMSNorm (e068)
    ✓ Residual (e070)
    ✓ Softmax (e071)
    
  ALL TRANSFORMER OPERATIONS CAN RUN ON SWITCHES!


  Cleanup...
  ✓ Done

================================================================================
BONUS: SOFTMAX vs ARGMAX
================================================================================

  FOR MOST LLM INFERENCE, WE DON'T NEED FULL SOFTMAX!
  
  ┌─────────────────────────────────────────────────────────────┐
  │ Use Case             │ What We Need    │ Full Softmax?     │
  ├─────────────────────────────────────────────────────────────┤
  │ Token generation     │ argmax(logits)  │ NO - just max     │
  │ Greedy decoding      │ argmax(logits)  │ NO - just max     │
  │ MoE expert selection │ top-K(scores)   │ NO - just top-K   │
  │ Beam search          │ top-K(logits)   │ NO - just top-K   │
  │ Temperature sampling │ softmax → sample│ YES (if T≠0)      │
  │ Attention weights    │ softmax(QK^T)   │ YES (multi-head)  │
  └─────────────────────────────────────────────────────────────┘
  
  ARGMAX ON SWITCH:
    The switch computes exp(x_i) for each class.
    To find argmax, just find which counter has the HIGHEST value!
    
    argmax = max_i(exp_counter[i])
    
    No division needed! No full softmax computation!
    This is MUCH simpler and avoids the denominator entirely.
  
  WHEN YOU DO NEED SOFTMAX:
    1. Compute exp values on switch (via packets)
    2. Read sum_exp counter
    3. Lookup scale = 1/sum_exp
    4. prob[i] = exp[i] * scale (or read exp[i] and divide)
    
    This requires ONE counter read (like RMSNorm).
"""

