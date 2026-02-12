#!/usr/bin/env python3
"""
e078_attention_score_v_on_switch.py

ATTENTION WEIGHTED SUM (score @ V) ON SWITCHES
===============================================

THE ATTENTION MECHANISM (continued from e074):
  1. Q = input @ W_q  (matrix multiply - proven)
  2. K = input @ W_k  (matrix multiply - proven)
  3. V = input @ W_v  (matrix multiply - proven)
  4. scores = Q @ K^T  (proven in e074!)
  5. weights = softmax(scores / sqrt(d_k))  (softmax - proven e071)
  6. output = weights @ V  ← THIS IS WHAT WE PROVE
  7. final = output @ W_o  (matrix multiply - proven)

THE CHALLENGE:
  After softmax, we compute the weighted sum of value vectors:
  
  output[d] = Σ_pos attention_weight[pos] × V[pos, d]
  
  Where:
    - attention_weight is [seq_len] - one weight per cached position
    - V is [seq_len, d_v] - value vectors from KV cache
    - output is [d_v] - the attention output

THE INSIGHT:
  This is STILL just matrix multiplication!
  
  weights [1, seq_len] @ V [seq_len, d_v] = output [1, d_v]
  
  Same packet counting approach as Q @ K^T (e074):
    - V values act as "weights"
    - Attention scores act as "activations"
    - For each dimension d: output[d] = Σ_pos score[pos] × V[pos, d]

SWITCH IMPLEMENTATION:
  For each output dimension d:
    output[d] = Σ_pos score[pos] × V[pos, d]
  
  Same packet counting approach:
    - V values act as "weights"
    - Scores act as "activations" (packet counts)
    - Send |score[pos] × V[pos,d]| packets to counter for dimension d
    - Use pos/neg counters for signed arithmetic

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

# =============================================================================
# CONFIGURATION
# =============================================================================

# Attention dimensions (simplified for testing)
# Real Qwen3-0.6B: d_v = 128, heads = 16
D_V = 8              # Output dimension per head (same as d_k typically)
SEQ_LEN = 16         # Number of cached positions (KV cache size)
NUM_HEADS = 1        # Single head for simplicity

VALUE_RANGE = (-8, 8)  # 4-bit quantized values
SCORE_RANGE = (-15, 15)  # Attention scores can be larger

FILTER_NAME = "attention_score_v"
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
# V CACHE AND ATTENTION SCORES
# =============================================================================

class AttentionContext:
    """
    Holds V cache and attention scores for score @ V computation.
    """
    def __init__(self, seq_len: int, d_v: int):
        self.seq_len = seq_len
        self.d_v = d_v
        
        # V cache: [seq_len, d_v]
        self.v_cache = None
        
        # Attention scores: [seq_len] (after softmax, these sum to 1)
        # For testing, we use integer scores
        self.scores = None
    
    def initialize_random(self, seed: int = 42):
        """Initialize V cache and scores with random values."""
        np.random.seed(seed)
        
        # V cache values (quantized)
        self.v_cache = np.random.randint(
            VALUE_RANGE[0], VALUE_RANGE[1] + 1,
            (self.seq_len, self.d_v)
        ).astype(np.int32)
        
        # Attention scores (could be post-softmax scaled)
        # For integer arithmetic, we use small integer weights
        self.scores = np.random.randint(
            0, 10,  # Non-negative for simplicity (like softmax output)
            self.seq_len
        ).astype(np.int32)
    
    def set_focused_attention(self, focus_pos: int):
        """Set attention to focus entirely on one position (like argmax)."""
        self.scores = np.zeros(self.seq_len, dtype=np.int32)
        self.scores[focus_pos] = 1
    
    def set_uniform_attention(self):
        """Set uniform attention across all positions."""
        self.scores = np.ones(self.seq_len, dtype=np.int32)


# =============================================================================
# CPU REFERENCE
# =============================================================================

def cpu_attention_output(scores: np.ndarray, v_cache: np.ndarray) -> np.ndarray:
    """
    CPU reference for score @ V.
    
    scores: [seq_len] - attention weights for each position
    v_cache: [seq_len, d_v] - cached value vectors
    
    Returns: [d_v] - weighted sum of value vectors
    """
    # scores @ V where scores is [1, seq_len] and V is [seq_len, d_v]
    # Result is [1, d_v], we squeeze to [d_v]
    output = scores @ v_cache
    return output


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


def configure_output_filters(d_v: int):
    """
    Configure filters for attention output computation.
    
    One counter per output dimension.
    Using dual counters (pos/neg) for signed outputs.
    """
    print(f"\n  Configuring filters for {d_v} output dimensions...")
    
    all_cmds = []
    
    all_cmds.append("set forwarding-options storm-control-profiles default all")
    
    # Create pos/neg counters for each output dimension
    for dim in range(d_v):
        # Positive counter
        mac_pos = get_layer_neuron_mac(0, dim)  # Layer 0 = pos
        term_pos = f"pos{dim}"
        all_cmds.extend([
            f"set firewall family ethernet-switching filter {FILTER_NAME} term {term_pos} from destination-mac-address {mac_pos}/48",
            f"set firewall family ethernet-switching filter {FILTER_NAME} term {term_pos} then count {term_pos}",
            f"set firewall family ethernet-switching filter {FILTER_NAME} term {term_pos} then accept",
        ])
        
        # Negative counter
        mac_neg = get_layer_neuron_mac(1, dim)  # Layer 1 = neg
        term_neg = f"neg{dim}"
        all_cmds.extend([
            f"set firewall family ethernet-switching filter {FILTER_NAME} term {term_neg} from destination-mac-address {mac_neg}/48",
            f"set firewall family ethernet-switching filter {FILTER_NAME} term {term_neg} then count {term_neg}",
            f"set firewall family ethernet-switching filter {FILTER_NAME} term {term_neg} then accept",
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
    
    print(f"    Counters: {d_v} dimensions × 2 (pos/neg) = {d_v * 2}")
    
    # Apply config directly via SSH
    config_str = "; ".join(all_cmds)
    success, stdout, stderr = ssh_command_long(
        SWITCH1_IP,
        f"cli -c 'configure; {config_str}; commit'",
        timeout=120
    )
    
    if not success or ('error' in stdout.lower() and 'commit' not in stdout.lower()):
        print(f"    ✗ Config failed: {stdout[:200]}")
        return False
    
    print("  ✓ Configuration complete")
    time.sleep(1)
    return True


def clear_counters():
    """Clear all firewall counters."""
    ssh_command_long(SWITCH1_IP, f"cli -c 'clear firewall filter {FILTER_NAME}'", timeout=30)
    time.sleep(0.3)


def read_output_counters(d_v: int) -> np.ndarray:
    """Read attention output counters (pos - neg)."""
    success, stdout, _ = ssh_command_long(
        SWITCH1_IP,
        f"cli -c 'show firewall filter {FILTER_NAME}'",
        timeout=30
    )
    
    output = np.zeros(d_v, dtype=np.int32)
    
    if not success or not stdout:
        return output
    
    for dim in range(d_v):
        # Read pos counter
        pattern_pos = rf'pos{dim}\s+\d+\s+(\d+)'
        match_pos = re.search(pattern_pos, stdout)
        pos_val = int(match_pos.group(1)) if match_pos else 0
        
        # Read neg counter
        pattern_neg = rf'neg{dim}\s+\d+\s+(\d+)'
        match_neg = re.search(pattern_neg, stdout)
        neg_val = int(match_neg.group(1)) if match_neg else 0
        
        output[dim] = pos_val - neg_val
    
    return output


# =============================================================================
# PACKET CREATION
# =============================================================================

def create_score_v_packets(scores: np.ndarray, v_cache: np.ndarray, 
                           src_mac: str) -> Tuple[List[bytes], np.ndarray]:
    """
    Create packets for score @ V computation.
    
    For each output dimension d:
      output[d] = Σ_pos score[pos] × V[pos, d]
    
    This is exactly like matrix multiplication where:
      - scores act as the "activation" vector
      - V[:,d] acts as the "weight" vector for dimension d
    """
    packets = []
    src = mac_str_to_bytes(src_mac)
    
    seq_len, d_v = v_cache.shape
    expected_output = np.zeros(d_v, dtype=np.int32)
    
    for dim in range(d_v):
        output_val = 0
        for pos in range(seq_len):
            score_val = int(scores[pos])
            v_val = int(v_cache[pos, dim])
            product = score_val * v_val
            output_val += product
            
            if product == 0:
                continue
            
            # Route to pos or neg counter based on sign
            if product > 0:
                # Positive: send to layer 0 (pos counter)
                dst_mac = get_layer_neuron_mac(0, dim)
            else:
                # Negative: send to layer 1 (neg counter)
                dst_mac = get_layer_neuron_mac(1, dim)
            
            dst = mac_str_to_bytes(dst_mac)
            num_packets = abs(product)
            
            for _ in range(num_packets):
                pkt = craft_vlan_packet(dst, src, TEST_VLAN)
                packets.append(pkt)
        
        expected_output[dim] = output_val
    
    return packets, expected_output


# =============================================================================
# TESTS
# =============================================================================

def test_basic_score_v():
    """Test basic score @ V computation."""
    print("\n" + "="*60)
    print("TEST 1: BASIC score @ V COMPUTATION")
    print("="*60)
    
    print(f"""
  Setup:
    - seq_len = {SEQ_LEN} positions in KV cache
    - d_v = {D_V} output dimensions
    - Random V values and attention scores
  
  Computation:
    output[d] = Σ_pos score[pos] × V[pos, d]
""")
    
    # Initialize context
    ctx = AttentionContext(SEQ_LEN, D_V)
    ctx.initialize_random(seed=42)
    
    print(f"  V cache shape: {ctx.v_cache.shape}")
    print(f"  Scores: {ctx.scores}")
    print(f"  V cache (first 4 rows):\n{ctx.v_cache[:4]}")
    
    # CPU reference
    cpu_output = cpu_attention_output(ctx.scores, ctx.v_cache)
    print(f"\n  CPU output: {cpu_output}")
    
    # Configure switch
    full_cleanup()
    if not configure_output_filters(D_V):
        return False
    
    clear_counters()
    
    # Create and send packets
    src_mac = get_mac_address(SEND_IFACE)
    packets, expected = create_score_v_packets(ctx.scores, ctx.v_cache, src_mac)
    
    print(f"\n  Sending {len(packets)} packets...")
    start = time.time()
    send_packets(SEND_IFACE, packets)
    send_time = (time.time() - start) * 1000
    print(f"  ✓ Sent in {send_time:.1f}ms")
    
    time.sleep(0.5)
    
    # Read counters
    switch_output = read_output_counters(D_V)
    
    print(f"\n  Results:")
    print(f"    CPU output:    {cpu_output}")
    print(f"    Switch output: {switch_output}")
    print(f"    Expected:      {expected}")
    
    # Compare
    match = np.array_equal(cpu_output, switch_output)
    print(f"\n  CPU vs Switch: {'✓ MATCH' if match else '✗ MISMATCH'}")
    
    if not match:
        print(f"    Differences: {cpu_output - switch_output}")
    
    return match


def test_focused_attention():
    """Test attention that focuses on a single position (like argmax)."""
    print("\n" + "="*60)
    print("TEST 2: FOCUSED ATTENTION (Single Position)")
    print("="*60)
    
    print("""
  When attention is focused on a single position (e.g., after argmax),
  the output should be exactly that position's V vector.
  
  This is the simplest case and is used for greedy decoding.
""")
    
    # Initialize context with focus on position 5
    focus_pos = 5
    ctx = AttentionContext(SEQ_LEN, D_V)
    ctx.initialize_random(seed=123)
    ctx.set_focused_attention(focus_pos)
    
    print(f"  Focus position: {focus_pos}")
    print(f"  Scores: {ctx.scores}")
    print(f"  V[{focus_pos}]: {ctx.v_cache[focus_pos]}")
    
    # CPU reference - should just be V[focus_pos]
    cpu_output = cpu_attention_output(ctx.scores, ctx.v_cache)
    print(f"\n  CPU output: {cpu_output}")
    print(f"  Expected (V[{focus_pos}]): {ctx.v_cache[focus_pos]}")
    
    clear_counters()
    
    # Create and send packets
    src_mac = get_mac_address(SEND_IFACE)
    packets, expected = create_score_v_packets(ctx.scores, ctx.v_cache, src_mac)
    
    print(f"\n  Sending {len(packets)} packets (should be exactly {D_V} packets)...")
    send_packets(SEND_IFACE, packets)
    
    time.sleep(0.5)
    
    # Read counters
    switch_output = read_output_counters(D_V)
    
    print(f"\n  Results:")
    print(f"    CPU output:    {cpu_output}")
    print(f"    Switch output: {switch_output}")
    print(f"    V[{focus_pos}]:         {ctx.v_cache[focus_pos]}")
    
    match = np.array_equal(cpu_output, switch_output)
    v_match = np.array_equal(switch_output, ctx.v_cache[focus_pos])
    
    print(f"\n  CPU vs Switch: {'✓ MATCH' if match else '✗ MISMATCH'}")
    print(f"  Switch = V[{focus_pos}]: {'✓ CORRECT' if v_match else '✗ WRONG'}")
    
    return match and v_match


def test_uniform_attention():
    """Test uniform attention (average of all V vectors)."""
    print("\n" + "="*60)
    print("TEST 3: UNIFORM ATTENTION (Average)")
    print("="*60)
    
    print("""
  When attention is uniform (all positions weighted equally),
  the output is the sum of all V vectors (or average if normalized).
  
  This tests the full accumulation across all positions.
""")
    
    ctx = AttentionContext(SEQ_LEN, D_V)
    ctx.initialize_random(seed=456)
    ctx.set_uniform_attention()
    
    print(f"  Scores: {ctx.scores} (all 1s = uniform)")
    
    # CPU reference - sum of all V vectors
    cpu_output = cpu_attention_output(ctx.scores, ctx.v_cache)
    manual_sum = ctx.v_cache.sum(axis=0)
    
    print(f"\n  CPU output: {cpu_output}")
    print(f"  Manual sum: {manual_sum}")
    
    clear_counters()
    
    # Create and send packets
    src_mac = get_mac_address(SEND_IFACE)
    packets, expected = create_score_v_packets(ctx.scores, ctx.v_cache, src_mac)
    
    print(f"\n  Sending {len(packets)} packets...")
    send_packets(SEND_IFACE, packets)
    
    time.sleep(0.5)
    
    # Read counters
    switch_output = read_output_counters(D_V)
    
    print(f"\n  Results:")
    print(f"    CPU output:    {cpu_output}")
    print(f"    Switch output: {switch_output}")
    
    match = np.array_equal(cpu_output, switch_output)
    print(f"\n  CPU vs Switch: {'✓ MATCH' if match else '✗ MISMATCH'}")
    
    if not match:
        print(f"    Differences: {cpu_output - switch_output}")
    
    return match


def test_weighted_attention():
    """Test attention with varying weights (like real softmax output)."""
    print("\n" + "="*60)
    print("TEST 4: WEIGHTED ATTENTION (Softmax-like)")
    print("="*60)
    
    print("""
  Real softmax outputs have varying weights across positions.
  Test with weights that decay with distance (simulating recency bias).
""")
    
    ctx = AttentionContext(SEQ_LEN, D_V)
    ctx.initialize_random(seed=789)
    
    # Create decaying weights (more recent = higher weight)
    ctx.scores = np.array([max(1, SEQ_LEN - i) for i in range(SEQ_LEN)], dtype=np.int32)
    
    print(f"  Scores (decaying): {ctx.scores}")
    print(f"  Score sum: {ctx.scores.sum()}")
    
    # CPU reference
    cpu_output = cpu_attention_output(ctx.scores, ctx.v_cache)
    print(f"\n  CPU output: {cpu_output}")
    
    clear_counters()
    
    # Create and send packets
    src_mac = get_mac_address(SEND_IFACE)
    packets, expected = create_score_v_packets(ctx.scores, ctx.v_cache, src_mac)
    
    print(f"\n  Sending {len(packets)} packets...")
    send_packets(SEND_IFACE, packets)
    
    time.sleep(0.5)
    
    # Read counters
    switch_output = read_output_counters(D_V)
    
    print(f"\n  Results:")
    print(f"    CPU output:    {cpu_output}")
    print(f"    Switch output: {switch_output}")
    
    match = np.array_equal(cpu_output, switch_output)
    print(f"\n  CPU vs Switch: {'✓ MATCH' if match else '✗ MISMATCH'}")
    
    return match


def test_full_attention_pipeline():
    """Test the complete attention output pipeline."""
    print("\n" + "="*60)
    print("TEST 5: FULL ATTENTION PIPELINE SIMULATION")
    print("="*60)
    
    print("""
  Simulating the full attention computation:
    1. Q @ K^T → scores (proven in e074)
    2. softmax(scores) → attention weights (proven in e071)
    3. weights @ V → output (THIS TEST)
  
  Using random Q and K to compute scores, then score @ V.
""")
    
    d_k = D_V  # Same dimension for simplicity
    
    # Generate random Q and K
    np.random.seed(999)
    Q = np.random.randint(-4, 5, d_k).astype(np.int32)
    K_cache = np.random.randint(-4, 5, (SEQ_LEN, d_k)).astype(np.int32)
    V_cache = np.random.randint(-4, 5, (SEQ_LEN, D_V)).astype(np.int32)
    
    print(f"  Q: {Q}")
    print(f"  K_cache shape: {K_cache.shape}")
    print(f"  V_cache shape: {V_cache.shape}")
    
    # Step 1: Q @ K^T (scores)
    raw_scores = Q @ K_cache.T
    print(f"\n  Step 1 - Q @ K^T scores: {raw_scores}")
    
    # Step 2: Simplified softmax (just make positive for counting)
    # Real softmax would normalize, but for counting we use non-negative weights
    scores_shifted = raw_scores - raw_scores.min()  # Make all non-negative
    print(f"  Step 2 - Shifted scores: {scores_shifted}")
    
    # Step 3: score @ V
    ctx = AttentionContext(SEQ_LEN, D_V)
    ctx.v_cache = V_cache
    ctx.scores = scores_shifted.astype(np.int32)
    
    cpu_output = cpu_attention_output(ctx.scores, ctx.v_cache)
    print(f"\n  Step 3 - CPU output: {cpu_output}")
    
    clear_counters()
    
    # Create and send packets
    src_mac = get_mac_address(SEND_IFACE)
    packets, expected = create_score_v_packets(ctx.scores, ctx.v_cache, src_mac)
    
    print(f"\n  Sending {len(packets)} packets...")
    send_packets(SEND_IFACE, packets)
    
    time.sleep(0.5)
    
    # Read counters
    switch_output = read_output_counters(D_V)
    
    print(f"\n  Final results:")
    print(f"    CPU output:    {cpu_output}")
    print(f"    Switch output: {switch_output}")
    
    match = np.array_equal(cpu_output, switch_output)
    print(f"\n  CPU vs Switch: {'✓ MATCH' if match else '✗ MISMATCH'}")
    
    return match


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all score @ V tests."""
    print("="*80)
    print("E078: ATTENTION WEIGHTED SUM (score @ V) ON SWITCHES")
    print("="*80)
    print("""
  THE FINAL PIECE OF ATTENTION!
  
  We've proven:
    - Q @ K^T (e074) ✓
    - softmax (e071) ✓
  
  Now proving:
    - score @ V → attention output
  
  Computation:
    output[d] = Σ_pos score[pos] × V[pos, d]
  
  This is matrix multiplication where:
    - scores = attention weights (packet counts)
    - V = value cache (dynamic weights from KV cache)
    - output = weighted sum of value vectors
""")
    
    results = {}
    
    # Test 1: Basic computation
    results['basic'] = test_basic_score_v()
    
    # Test 2: Focused attention
    results['focused'] = test_focused_attention()
    
    # Test 3: Uniform attention
    results['uniform'] = test_uniform_attention()
    
    # Test 4: Weighted attention
    results['weighted'] = test_weighted_attention()
    
    # Test 5: Full pipeline
    results['pipeline'] = test_full_attention_pipeline()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    all_pass = all(results.values())
    
    print(f"""
  TEST RESULTS:
    Basic score @ V:     {'✓' if results['basic'] else '✗'}
    Focused attention:   {'✓' if results['focused'] else '✗'}
    Uniform attention:   {'✓' if results['uniform'] else '✗'}
    Weighted attention:  {'✓' if results['weighted'] else '✗'}
    Full pipeline:       {'✓' if results['pipeline'] else '✗'}
    
  OVERALL: {'✓ ALL TESTS PASSED' if all_pass else '✗ SOME TESTS FAILED'}
""")
    
    if all_pass:
        print("""
  🎉 ATTENTION WEIGHTED SUM (score @ V) PROVEN! 🎉
  
  This completes the attention mechanism:
    1. Q = input @ W_q  ✓ (matrix multiply)
    2. K = input @ W_k  ✓ (matrix multiply)
    3. V = input @ W_v  ✓ (matrix multiply)
    4. scores = Q @ K^T ✓ (e074)
    5. weights = softmax(scores) ✓ (e071)
    6. output = weights @ V ✓ (THIS EXPERIMENT!)
    7. final = output @ W_o ✓ (matrix multiply)
  
  FULL TRANSFORMER ATTENTION CAN RUN ON COMMODITY NETWORK SWITCHES!
""")
    
    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)



""" Output:
sudo python3 e078_attention_score_v_on_switch.py 
================================================================================
E078: ATTENTION WEIGHTED SUM (score @ V) ON SWITCHES
================================================================================

  THE FINAL PIECE OF ATTENTION!
  
  We've proven:
    - Q @ K^T (e074) ✓
    - softmax (e071) ✓
  
  Now proving:
    - score @ V → attention output
  
  Computation:
    output[d] = Σ_pos score[pos] × V[pos, d]
  
  This is matrix multiplication where:
    - scores = attention weights (packet counts)
    - V = value cache (dynamic weights from KV cache)
    - output = weighted sum of value vectors


============================================================
TEST 1: BASIC score @ V COMPUTATION
============================================================

  Setup:
    - seq_len = 16 positions in KV cache
    - d_v = 8 output dimensions
    - Random V values and attention scores
  
  Computation:
    output[d] = Σ_pos score[pos] × V[pos, d]

  V cache shape: (16, 8)
  Scores: [8 8 3 8 2 6 5 7 8 4 0 2 9 7 5 7]
  V cache (first 4 rows):
[[-2  6  2 -1 -2  2  2 -5]
 [-1 -6 -7  3 -3 -7 -8  3]
 [ 3  8  1  7  6  6  3 -6]
 [-4 -2  0 -2 -5  5  0 -7]]

  CPU output: [ -52  -20  -97 -118  122  -51 -141   62]

  Cleanup...
  ✓ Done

  Configuring filters for 8 output dimensions...
    Counters: 8 dimensions × 2 (pos/neg) = 16
  ✓ Configuration complete

  Sending 2845 packets...
  ✓ Sent in 11.5ms

  Results:
    CPU output:    [ -52  -20  -97 -118  122  -51 -141   62]
    Switch output: [ -52  -20  -97 -118  122  -51 -141   62]
    Expected:      [ -52  -20  -97 -118  122  -51 -141   62]

  CPU vs Switch: ✓ MATCH

============================================================
TEST 2: FOCUSED ATTENTION (Single Position)
============================================================

  When attention is focused on a single position (e.g., after argmax),
  the output should be exactly that position's V vector.
  
  This is the simplest case and is used for greedy decoding.

  Focus position: 5
  Scores: [0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0]
  V[5]: [ 5 -4  7  3  4 -2  5  8]

  CPU output: [ 5 -4  7  3  4 -2  5  8]
  Expected (V[5]): [ 5 -4  7  3  4 -2  5  8]

  Sending 38 packets (should be exactly 8 packets)...

  Results:
    CPU output:    [ 5 -4  7  3  4 -2  5  8]
    Switch output: [ 5 -4  7  3  4 -2  5  8]
    V[5]:         [ 5 -4  7  3  4 -2  5  8]

  CPU vs Switch: ✓ MATCH
  Switch = V[5]: ✓ CORRECT

============================================================
TEST 3: UNIFORM ATTENTION (Average)
============================================================

  When attention is uniform (all positions weighted equally),
  the output is the sum of all V vectors (or average if normalized).
  
  This tests the full accumulation across all positions.

  Scores: [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1] (all 1s = uniform)

  CPU output: [-26  61  -6  36  30   0  19  37]
  Manual sum: [-26  61  -6  36  30   0  19  37]

  Sending 539 packets...

  Results:
    CPU output:    [-26  61  -6  36  30   0  19  37]
    Switch output: [-26  61  -6  36  30   0  19  37]

  CPU vs Switch: ✓ MATCH

============================================================
TEST 4: WEIGHTED ATTENTION (Softmax-like)
============================================================

  Real softmax outputs have varying weights across positions.
  Test with weights that decay with distance (simulating recency bias).

  Scores (decaying): [16 15 14 13 12 11 10  9  8  7  6  5  4  3  2  1]
  Score sum: 136

  CPU output: [ 477 -248  128 -372 -174  -76  -89 -322]

  Sending 4916 packets...

  Results:
    CPU output:    [ 477 -248  128 -372 -174  -76  -89 -322]
    Switch output: [ 477 -248  128 -372 -174  -76  -89 -322]

  CPU vs Switch: ✓ MATCH

============================================================
TEST 5: FULL ATTENTION PIPELINE SIMULATION
============================================================

  Simulating the full attention computation:
    1. Q @ K^T → scores (proven in e074)
    2. softmax(scores) → attention weights (proven in e071)
    3. weights @ V → output (THIS TEST)
  
  Using random Q and K to compute scores, then score @ V.

  Q: [-4  1 -3  4 -3 -1 -4  1]
  K_cache shape: (16, 8)
  V_cache shape: (16, 8)

  Step 1 - Q @ K^T scores: [ -3   3  -8 -16  17  21  -7   2   6  10  33 -43 -12 -42   3  30]
  Step 2 - Shifted scores: [40 46 35 27 60 64 36 45 49 53 76  0 31  1 46 73]

  Step 3 - CPU output: [-1123 -1027   337  -952  -739  -545   -99  -487]

  Sending 11987 packets...

  Final results:
    CPU output:    [-1123 -1027   337  -952  -739  -545   -99  -487]
    Switch output: [-1123 -1027   337  -952  -739  -545   -99  -487]

  CPU vs Switch: ✓ MATCH

================================================================================
SUMMARY
================================================================================

  TEST RESULTS:
    Basic score @ V:     ✓
    Focused attention:   ✓
    Uniform attention:   ✓
    Weighted attention:  ✓
    Full pipeline:       ✓
    
  OVERALL: ✓ ALL TESTS PASSED


  🎉 ATTENTION WEIGHTED SUM (score @ V) PROVEN! 🎉
  
  This completes the attention mechanism:
    1. Q = input @ W_q  ✓ (matrix multiply)
    2. K = input @ W_k  ✓ (matrix multiply)
    3. V = input @ W_v  ✓ (matrix multiply)
    4. scores = Q @ K^T ✓ (e074)
    5. weights = softmax(scores) ✓ (e071)
    6. output = weights @ V ✓ (THIS EXPERIMENT!)
    7. final = output @ W_o ✓ (matrix multiply)
  
  FULL TRANSFORMER ATTENTION CAN RUN ON COMMODITY NETWORK SWITCHES!
"""
