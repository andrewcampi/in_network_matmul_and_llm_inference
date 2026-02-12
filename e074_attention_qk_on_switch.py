#!/usr/bin/env python3
"""
e074_attention_qk_on_switch.py

ATTENTION Q@K^T ON SWITCHES
============================

THE ATTENTION MECHANISM:
  1. Q = input @ W_q  (matrix multiply - proven)
  2. K = input @ W_k  (matrix multiply - proven)
  3. V = input @ W_v  (matrix multiply - proven)
  4. scores = Q @ K^T  ← THIS IS WHAT WE PROVE
  5. weights = softmax(scores / sqrt(d_k))  (softmax - proven)
  6. output = weights @ V  (matrix multiply - proven)
  7. final = output @ W_o  (matrix multiply - proven)

THE CHALLENGE:
  For single-token generation (autoregressive inference):
    - Q is [1, d_k] - query for current token
    - K is [seq_len, d_k] - keys from KV cache (all previous tokens)
    - Q @ K^T gives [1, seq_len] - attention scores
  
  This looks different because K comes from cache, not model weights.

THE INSIGHT:
  Q @ K^T is STILL just matrix multiplication!
  
  Q [1, d_k] @ K^T [d_k, seq_len] = scores [1, seq_len]
  
  This is exactly like:
    activation [1, d_k] @ weights [d_k, seq_len] = output [1, seq_len]
  
  The only difference is that the "weights" (K^T) come from the 
  KV cache instead of model weights. The computation is identical!

SWITCH IMPLEMENTATION:
  For each cached position i:
    score[i] = Σ_j Q[j] × K[i,j]
  
  Same packet counting approach:
    - K values act as "weights"
    - Q values act as "activations"
    - Send |Q[j] × K[i,j]| packets to counter for position i
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
# Real Qwen3-0.6B: d_k = 128, heads = 16
D_K = 8              # Dimension per head
SEQ_LEN = 16         # Number of cached positions (KV cache size)
NUM_HEADS = 1        # Single head for simplicity

VALUE_RANGE = (-8, 8)  # 4-bit quantized values

FILTER_NAME = "attention_qk"
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
# KV CACHE SIMULATION
# =============================================================================

class KVCache:
    """
    Simulates the KV cache that stores Key and Value vectors
    from all previous tokens in the sequence.
    """
    def __init__(self, seq_len: int, d_k: int, num_heads: int = 1):
        self.seq_len = seq_len
        self.d_k = d_k
        self.num_heads = num_heads
        
        # K cache: [seq_len, d_k] for single head
        # In real impl: [num_heads, seq_len, d_k]
        self.k_cache = None
        self.v_cache = None
    
    def initialize_random(self, seed: int = 42):
        """Initialize cache with random values (simulating previous tokens)."""
        np.random.seed(seed)
        self.k_cache = np.random.randint(
            VALUE_RANGE[0], VALUE_RANGE[1] + 1,
            (self.seq_len, self.d_k)
        ).astype(np.int32)
        self.v_cache = np.random.randint(
            VALUE_RANGE[0], VALUE_RANGE[1] + 1,
            (self.seq_len, self.d_k)
        ).astype(np.int32)
    
    def get_k_transpose(self) -> np.ndarray:
        """Return K^T for Q @ K^T computation."""
        return self.k_cache.T  # [d_k, seq_len]


# =============================================================================
# CPU REFERENCE
# =============================================================================

def cpu_attention_scores(q: np.ndarray, k_cache: np.ndarray) -> np.ndarray:
    """
    CPU reference for Q @ K^T.
    
    q: [d_k] - query vector for current token
    k_cache: [seq_len, d_k] - cached key vectors
    
    Returns: [seq_len] - attention scores for each position
    """
    # Q @ K^T where Q is [1, d_k] and K^T is [d_k, seq_len]
    # Result is [1, seq_len], we squeeze to [seq_len]
    scores = q @ k_cache.T
    return scores


def cpu_attention_scores_scaled(q: np.ndarray, k_cache: np.ndarray, scale: float = None) -> np.ndarray:
    """
    CPU reference with scaling (1/sqrt(d_k)).
    
    In real attention: scores = (Q @ K^T) / sqrt(d_k)
    """
    if scale is None:
        scale = 1.0 / np.sqrt(len(q))
    
    scores = q @ k_cache.T
    return scores * scale


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


def configure_attention_filters(seq_len: int):
    """
    Configure filters for attention score computation.
    
    One counter per sequence position (for the attention score).
    Using dual counters (pos/neg) for signed scores.
    """
    print(f"\n  Configuring filters for {seq_len} positions...")
    
    all_cmds = []
    
    all_cmds.append("set forwarding-options storm-control-profiles default all")
    all_cmds.append(f"set firewall family ethernet-switching filter {FILTER_NAME}")
    
    # Create pos/neg counters for each sequence position
    for pos in range(seq_len):
        # Positive counter
        mac_pos = get_layer_neuron_mac(0, pos)  # Layer 0 = pos
        term_pos = f"pos{pos}"
        all_cmds.extend([
            f"set firewall family ethernet-switching filter {FILTER_NAME} term {term_pos} from destination-mac-address {mac_pos}/48",
            f"set firewall family ethernet-switching filter {FILTER_NAME} term {term_pos} then count {term_pos}",
            f"set firewall family ethernet-switching filter {FILTER_NAME} term {term_pos} then accept",
        ])
        
        # Negative counter
        mac_neg = get_layer_neuron_mac(1, pos)  # Layer 1 = neg
        term_neg = f"neg{pos}"
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
    
    print(f"    Counters: {seq_len} positions × 2 (pos/neg) = {seq_len * 2}")
    
    config_file = "/tmp/e074_config.txt"
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


def read_attention_counters(seq_len: int) -> np.ndarray:
    """Read attention score counters (pos - neg)."""
    success, stdout, _ = ssh_command_long(
        SWITCH1_IP,
        f"cli -c 'show firewall filter {FILTER_NAME}'",
        timeout=30
    )
    
    scores = np.zeros(seq_len, dtype=np.int32)
    
    if not success or not stdout:
        return scores
    
    for pos in range(seq_len):
        # Read pos counter
        pattern_pos = rf'pos{pos}\s+\d+\s+(\d+)'
        match_pos = re.search(pattern_pos, stdout)
        pos_val = int(match_pos.group(1)) if match_pos else 0
        
        # Read neg counter
        pattern_neg = rf'neg{pos}\s+\d+\s+(\d+)'
        match_neg = re.search(pattern_neg, stdout)
        neg_val = int(match_neg.group(1)) if match_neg else 0
        
        scores[pos] = pos_val - neg_val
    
    return scores


# =============================================================================
# PACKET CREATION
# =============================================================================

def create_attention_packets(q: np.ndarray, k_cache: np.ndarray, 
                             src_mac: str) -> Tuple[List[bytes], np.ndarray]:
    """
    Create packets for Q @ K^T computation.
    
    For each position i in the sequence:
      score[i] = Σ_j Q[j] × K[i,j]
    
    This is exactly like matrix multiplication where:
      - Q acts as the "activation" vector
      - K[i,:] acts as the "weight" vector for position i
    """
    packets = []
    src = mac_str_to_bytes(src_mac)
    
    seq_len, d_k = k_cache.shape
    expected_scores = np.zeros(seq_len, dtype=np.int32)
    
    for pos in range(seq_len):
        score = 0
        for j in range(d_k):
            q_val = int(q[j])
            k_val = int(k_cache[pos, j])
            product = q_val * k_val
            score += product
            
            if product == 0:
                continue
            
            # Route to pos or neg counter based on sign
            if product > 0:
                mac = get_layer_neuron_mac(0, pos)  # pos counter
            else:
                mac = get_layer_neuron_mac(1, pos)  # neg counter
            
            dst = mac_str_to_bytes(mac)
            for _ in range(abs(product)):
                packets.append(craft_vlan_packet(dst, src, TEST_VLAN))
        
        expected_scores[pos] = score
    
    return packets, expected_scores


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_attention_qk_experiment():
    """Run the attention Q@K^T experiment."""
    print("="*80)
    print("E074: ATTENTION Q@K^T ON SWITCHES")
    print("="*80)
    print(f"""
  GOAL: Prove Q @ K^T (attention scores) works on switches!
  
  ATTENTION FORMULA:
    For current token with query Q and cached keys K:
      score[i] = Q · K[i] = Σ_j Q[j] × K[i,j]
    
    This is the dot product of Q with each cached K vector.
  
  THE INSIGHT:
    Q @ K^T is just matrix multiplication where:
      - Q [1, d_k] is the "activation" 
      - K^T [d_k, seq_len] is the "weight matrix"
      - Result [1, seq_len] is attention scores
    
    The switch already does matrix multiply - K values just
    come from cache instead of model weights!
  
  TEST CONFIGURATION:
    d_k (dimension per head): {D_K}
    seq_len (KV cache size): {SEQ_LEN}
    Value range: {VALUE_RANGE}
""")
    
    # Create KV cache
    print("\n" + "="*60)
    print("STEP 1: INITIALIZE KV CACHE")
    print("="*60)
    
    cache = KVCache(SEQ_LEN, D_K)
    cache.initialize_random(seed=42)
    
    print(f"\n  K cache shape: [{SEQ_LEN}, {D_K}]")
    print(f"  K cache (first 4 positions):")
    for i in range(min(4, SEQ_LEN)):
        print(f"    K[{i}]: {cache.k_cache[i]}")
    
    # Cleanup and configure
    full_cleanup()
    
    if not configure_attention_filters(SEQ_LEN):
        print("  ✗ Configuration failed!")
        return False
    
    src_mac = get_mac_address(SEND_IFACE)
    print(f"\n  Source MAC: {src_mac}")
    
    # Clear counters and wait
    clear_counters()
    time.sleep(0.5)
    
    # Test 1: Single query attention
    print("\n" + "="*60)
    print("TEST 1: SINGLE QUERY ATTENTION SCORES")
    print("="*60)
    
    np.random.seed(123)
    q = np.random.randint(VALUE_RANGE[0], VALUE_RANGE[1] + 1, D_K)
    
    print(f"\n  Query Q: {q}")
    
    # CPU reference
    cpu_scores = cpu_attention_scores(q, cache.k_cache)
    cpu_max_pos = np.argmax(cpu_scores)
    
    print(f"\n  CPU attention scores: {cpu_scores}")
    print(f"  CPU max attention position: {cpu_max_pos} (score: {cpu_scores[cpu_max_pos]})")
    
    # Switch computation
    clear_counters()
    
    packets, expected_scores = create_attention_packets(q, cache.k_cache, src_mac)
    
    print(f"\n  Sending packets:")
    print(f"    Total packets: {len(packets)}")
    print(f"    (one dot product per cached position)")
    
    if packets:
        send_packets(SEND_IFACE, packets)
        print(f"    ✓ Sent {len(packets)} packets")
    
    time.sleep(0.5)
    
    # Read counters
    switch_scores = read_attention_counters(SEQ_LEN)
    switch_max_pos = np.argmax(switch_scores)
    
    print(f"\n  Switch attention scores: {switch_scores}")
    print(f"  Switch max attention position: {switch_max_pos} (score: {switch_scores[switch_max_pos]})")
    
    # Compare
    scores_match = np.array_equal(switch_scores, expected_scores)
    max_pos_match = (cpu_max_pos == switch_max_pos)
    
    print(f"\n  Verification:")
    print(f"    Scores match expected: {'✓' if scores_match else '✗'}")
    print(f"    Scores match CPU: {'✓' if np.array_equal(switch_scores, cpu_scores) else '✗'}")
    print(f"    Max position match: {'✓' if max_pos_match else '✗'}")
    
    # Test 2: Multiple queries
    print("\n" + "="*60)
    print("TEST 2: MULTIPLE QUERIES (DIFFERENT TOKENS)")
    print("="*60)
    
    n_queries = 5
    correct_max = 0
    all_scores_match = True
    
    for i in range(n_queries):
        q = np.random.randint(VALUE_RANGE[0], VALUE_RANGE[1] + 1, D_K)
        
        cpu_scores = cpu_attention_scores(q, cache.k_cache)
        cpu_max = np.argmax(cpu_scores)
        
        clear_counters()
        packets, expected = create_attention_packets(q, cache.k_cache, src_mac)
        if packets:
            send_packets(SEND_IFACE, packets)
        time.sleep(0.3)
        
        switch_scores = read_attention_counters(SEQ_LEN)
        switch_max = np.argmax(switch_scores)
        
        if not np.array_equal(switch_scores, expected):
            all_scores_match = False
        
        if cpu_max == switch_max:
            correct_max += 1
        
        status = "✓" if cpu_max == switch_max else "✗"
        print(f"    Query {i+1}: CPU max={cpu_max}, Switch max={switch_max} {status}")
    
    # Test 3: Verify attention pattern makes sense
    print("\n" + "="*60)
    print("TEST 3: ATTENTION PATTERN ANALYSIS")
    print("="*60)
    
    # Create a query that should strongly attend to a specific position
    # by making it similar to one of the cached K vectors
    target_pos = 5
    q_targeted = cache.k_cache[target_pos].copy()  # Q = K[5], should max attention at pos 5
    
    clear_counters()
    packets, _ = create_attention_packets(q_targeted, cache.k_cache, src_mac)
    if packets:
        send_packets(SEND_IFACE, packets)
    time.sleep(0.3)
    
    switch_scores = read_attention_counters(SEQ_LEN)
    switch_max = np.argmax(switch_scores)
    
    # When Q = K[i], the score at position i should be highest (Q·K[i] = ||K[i]||²)
    print(f"\n  Targeted query test:")
    print(f"    Q = K[{target_pos}] (query identical to cached key at position {target_pos})")
    print(f"    Expected max attention: position {target_pos}")
    print(f"    Switch max attention: position {switch_max}")
    print(f"    Match: {'✓' if switch_max == target_pos else '✗'}")
    
    targeted_pass = (switch_max == target_pos)
    
    # Summary
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    test1_pass = scores_match and max_pos_match
    test2_pass = all_scores_match and (correct_max == n_queries)
    test3_pass = targeted_pass
    
    print(f"""
  Test 1 (Single query):
    All scores match: {'✓' if scores_match else '✗'}
    Max position match: {'✓' if max_pos_match else '✗'}
  
  Test 2 (Multiple queries): {correct_max}/{n_queries} max positions correct
    All scores exact: {'✓' if all_scores_match else '✗'}
  
  Test 3 (Targeted attention): {'✓' if targeted_pass else '✗'}
""")
    
    all_pass = test1_pass and test2_pass and test3_pass
    
    if all_pass:
        print(f"""
  🎉 ATTENTION Q@K^T ON SWITCHES PROVEN! 🎉
  
  What we demonstrated:
    1. Q @ K^T computed as dot products via packet counting
    2. KV cache values act as "dynamic weights"
    3. Attention scores match CPU reference exactly
    4. Max attention position (argmax) correct for all queries
  
  Key insight:
    Q @ K^T is just matrix multiplication where the "weights"
    come from the KV cache instead of model parameters.
    The switch computation is IDENTICAL!
  
  For real inference:
    1. Compute Q for current token (Q projection on switch)
    2. For each cached K at position i:
       - Send Q[j] × K[i,j] packets to counter i
    3. Read counters → attention scores
    4. Apply softmax (proven in e071)
    5. Compute weighted sum with V (matrix multiply)
  
  SINGLE-TOKEN ATTENTION COMPLEXITY:
    - Query: 1 vector of d_k dimensions
    - Cache: seq_len vectors of d_k dimensions
    - Packets: O(d_k × seq_len) per attention head
    - For Qwen3-0.6B (d_k=128, 16 heads): 2048 × seq_len packets
  
  COMPLETE TRANSFORMER OPERATIONS ON SWITCH:
    ✓ Matrix multiply (e056)
    ✓ Element-wise multiply (e066)
    ✓ SiLU activation (e067)
    ✓ RMSNorm (e068)
    ✓ Residual connection (e070)
    ✓ Softmax / Argmax (e071)
    ✓ RoPE position encoding (e072)
    ✓ LM Head sharding (e073)
    ✓ Attention Q@K^T (e074) ← NEW!
    
  🎊 ALL TRANSFORMER OPERATIONS NOW PROVEN! 🎊
""")
    else:
        print(f"""
  Some tests failed:
    Test 1: {'PASS' if test1_pass else 'FAIL'}
    Test 2: {'PASS' if test2_pass else 'FAIL'}
    Test 3: {'PASS' if test3_pass else 'FAIL'}
""")
    
    full_cleanup()
    
    return all_pass


def print_attention_scaling():
    """Print analysis of attention scaling for real models."""
    print("\n" + "="*80)
    print("APPENDIX: ATTENTION SCALING ANALYSIS")
    print("="*80)
    
    print("""
  ATTENTION COMPUTATION BREAKDOWN:
    
    For single-token inference at position T:
    
    1. Q PROJECTION:
       Q = hidden @ W_q
       Packets: hidden_dim × (num_heads × d_k)
       Qwen3-0.6B: 1024 × 2048 = 2M products
    
    2. Q @ K^T (what we just proved):
       For each head h:
         scores[h] = Q[h] @ K_cache[h]^T
       Packets per head: d_k × T
       Total: num_heads × d_k × T
       Qwen3-0.6B at T=1000: 16 × 128 × 1000 = 2M products
    
    3. SOFTMAX:
       Already proven in e071
       O(T) per head
    
    4. SCORES @ V:
       out = softmax(scores) @ V_cache
       Packets per head: T × d_k  
       Total: num_heads × T × d_k
       Same as Q @ K^T
    
    5. O PROJECTION:
       final = concat(out) @ W_o
       Packets: (num_heads × d_k) × hidden_dim
       Same as Q projection
  
  TOTAL ATTENTION PACKETS (single token, single layer):
    Q proj + Q@K^T + scores@V + O proj
    = 2 × hidden × (h × d_k) + 2 × h × d_k × T
    = 2 × 1024 × 2048 + 2 × 16 × 128 × T
    = 4.2M + 4K × T
    
    At T=1000: ~8.2M packets per layer
    28 layers: ~230M packets
    
  PARALLELIZATION:
    - Heads can be computed in parallel (different counters)
    - Layers can pipeline (switch 1 does layer 0, switch 2 does layer 1)
    - Multiple switches can share KV cache computation
  
  KV CACHE SIZE:
    For Qwen3-0.6B at context T:
      K cache: 28 layers × 8 KV heads × T × 128 dims × 4 bytes
      V cache: same
      Total: 28 × 8 × T × 128 × 4 × 2 = 230KB × T
      
      At T=4096: ~940MB KV cache
""")


if __name__ == '__main__':
    success = run_attention_qk_experiment()
    if success:
        print_attention_scaling()


""" Output:
sudo python3 e074_attention_qk_on_switch.py 
================================================================================
E074: ATTENTION Q@K^T ON SWITCHES
================================================================================

  GOAL: Prove Q @ K^T (attention scores) works on switches!
  
  ATTENTION FORMULA:
    For current token with query Q and cached keys K:
      score[i] = Q · K[i] = Σ_j Q[j] × K[i,j]
    
    This is the dot product of Q with each cached K vector.
  
  THE INSIGHT:
    Q @ K^T is just matrix multiplication where:
      - Q [1, d_k] is the "activation" 
      - K^T [d_k, seq_len] is the "weight matrix"
      - Result [1, seq_len] is attention scores
    
    The switch already does matrix multiply - K values just
    come from cache instead of model weights!
  
  TEST CONFIGURATION:
    d_k (dimension per head): 8
    seq_len (KV cache size): 16
    Value range: (-8, 8)


============================================================
STEP 1: INITIALIZE KV CACHE
============================================================

  K cache shape: [16, 8]
  K cache (first 4 positions):
    K[0]: [-2  6  2 -1 -2  2  2 -5]
    K[1]: [-1 -6 -7  3 -3 -7 -8  3]
    K[2]: [ 3  8  1  7  6  6  3 -6]
    K[3]: [-4 -2  0 -2 -5  5  0 -7]

  Cleanup...
  ✓ Done

  Configuring filters for 16 positions...
    Counters: 16 positions × 2 (pos/neg) = 32
  ✓ Configuration complete

  Source MAC: 7c:fe:90:9d:2a:f0

============================================================
TEST 1: SINGLE QUERY ATTENTION SCORES
============================================================

  Query Q: [ 5 -6 -6 -2  2 -7 -8  7]

  CPU attention scores: [-125  195 -149  -98   96   73   -6  -95   41   84  -52   70  183   73
   67   40]
  CPU max attention position: 1 (score: 195)

  Sending packets:
    Total packets: 2757
    (one dot product per cached position)
    ✓ Sent 2757 packets

  Switch attention scores: [-125  195 -149  -98   96   73   -6  -95   41   84  -52   70  183   73
   67   40]
  Switch max attention position: 1 (score: 195)

  Verification:
    Scores match expected: ✓
    Scores match CPU: ✓
    Max position match: ✓

============================================================
TEST 2: MULTIPLE QUERIES (DIFFERENT TOKENS)
============================================================
    Query 1: CPU max=15, Switch max=15 ✓
    Query 2: CPU max=12, Switch max=12 ✓
    Query 3: CPU max=11, Switch max=11 ✓
    Query 4: CPU max=9, Switch max=9 ✓
    Query 5: CPU max=4, Switch max=4 ✓

============================================================
TEST 3: ATTENTION PATTERN ANALYSIS
============================================================

  Targeted query test:
    Q = K[5] (query identical to cached key at position 5)
    Expected max attention: position 5
    Switch max attention: position 5
    Match: ✓

================================================================================
RESULTS
================================================================================

  Test 1 (Single query):
    All scores match: ✓
    Max position match: ✓
  
  Test 2 (Multiple queries): 5/5 max positions correct
    All scores exact: ✓
  
  Test 3 (Targeted attention): ✓


  🎉 ATTENTION Q@K^T ON SWITCHES PROVEN! 🎉
  
  What we demonstrated:
    1. Q @ K^T computed as dot products via packet counting
    2. KV cache values act as "dynamic weights"
    3. Attention scores match CPU reference exactly
    4. Max attention position (argmax) correct for all queries
  
  Key insight:
    Q @ K^T is just matrix multiplication where the "weights"
    come from the KV cache instead of model parameters.
    The switch computation is IDENTICAL!
  
  For real inference:
    1. Compute Q for current token (Q projection on switch)
    2. For each cached K at position i:
       - Send Q[j] × K[i,j] packets to counter i
    3. Read counters → attention scores
    4. Apply softmax (proven in e071)
    5. Compute weighted sum with V (matrix multiply)
  
  SINGLE-TOKEN ATTENTION COMPLEXITY:
    - Query: 1 vector of d_k dimensions
    - Cache: seq_len vectors of d_k dimensions
    - Packets: O(d_k × seq_len) per attention head
    - For Qwen3-0.6B (d_k=128, 16 heads): 2048 × seq_len packets
  
  COMPLETE TRANSFORMER OPERATIONS ON SWITCH:
    ✓ Matrix multiply (e056)
    ✓ Element-wise multiply (e066)
    ✓ SiLU activation (e067)
    ✓ RMSNorm (e068)
    ✓ Residual connection (e070)
    ✓ Softmax / Argmax (e071)
    ✓ RoPE position encoding (e072)
    ✓ LM Head sharding (e073)
    ✓ Attention Q@K^T (e074) ← NEW!
    
  🎊 ALL TRANSFORMER OPERATIONS NOW PROVEN! 🎊


  Cleanup...
  ✓ Done

================================================================================
APPENDIX: ATTENTION SCALING ANALYSIS
================================================================================

  ATTENTION COMPUTATION BREAKDOWN:
    
    For single-token inference at position T:
    
    1. Q PROJECTION:
       Q = hidden @ W_q
       Packets: hidden_dim × (num_heads × d_k)
       Qwen3-0.6B: 1024 × 2048 = 2M products
    
    2. Q @ K^T (what we just proved):
       For each head h:
         scores[h] = Q[h] @ K_cache[h]^T
       Packets per head: d_k × T
       Total: num_heads × d_k × T
       Qwen3-0.6B at T=1000: 16 × 128 × 1000 = 2M products
    
    3. SOFTMAX:
       Already proven in e071
       O(T) per head
    
    4. SCORES @ V:
       out = softmax(scores) @ V_cache
       Packets per head: T × d_k  
       Total: num_heads × T × d_k
       Same as Q @ K^T
    
    5. O PROJECTION:
       final = concat(out) @ W_o
       Packets: (num_heads × d_k) × hidden_dim
       Same as Q projection
  
  TOTAL ATTENTION PACKETS (single token, single layer):
    Q proj + Q@K^T + scores@V + O proj
    = 2 × hidden × (h × d_k) + 2 × h × d_k × T
    = 2 × 1024 × 2048 + 2 × 16 × 128 × T
    = 4.2M + 4K × T
    
    At T=1000: ~8.2M packets per layer
    28 layers: ~230M packets
    
  PARALLELIZATION:
    - Heads can be computed in parallel (different counters)
    - Layers can pipeline (switch 1 does layer 0, switch 2 does layer 1)
    - Multiple switches can share KV cache computation
  
  KV CACHE SIZE:
    For Qwen3-0.6B at context T:
      K cache: 28 layers × 8 KV heads × T × 128 dims × 4 bytes
      V cache: same
      Total: 28 × 8 × T × 128 × 4 × 2 = 230KB × T
      
      At T=4096: ~940MB KV cache
"""
