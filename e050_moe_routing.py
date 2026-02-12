#!/usr/bin/env python3
"""
e050_moe_routing.py

MIXTURE OF EXPERTS (MoE) ROUTING ON SWITCHES
=============================================

GOAL: Prove MoE routing works on switches.

From Qwen3-Next-80B specs:
  - expert_count: 512       (total experts)
  - expert_used_count: 10   (top-10 selected per token)

MoE ROUTING:
  1. Router projection: hidden → 512 expert scores (matrix multiply)
  2. Top-K selection: Find 10 highest scores (argmax on counters!)
  3. Route to selected experts

KEY INSIGHT: 
  - Router is a matrix multiply → SWITCH
  - Top-K = find highest counters → CPU reads, sorts
  - No softmax needed for routing!
"""

import time
import os
import sys
import numpy as np
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import proven infrastructure
from e046_full_layer_real_weights import (
    configure_split_weights_parallel,
    run_full_layer_inference,
)

from e045_real_weights_inference import (
    parse_gguf_header, find_weight_tensors, extract_small_weight_sample,
    weights_to_binary,
    MODEL_PATH
)

from e044_full_layer_mirror import (
    configure_port_mirroring,
    configure_sw2_port_mirroring_split,
    read_counters_split,
    FILTER_NAME, BATCH_SIZE,
)

from e042_port_based_layers import (
    ssh_command, run_config_commands,
    SWITCH1_IP, SWITCH2_IP,
)

# MoE parameters from model specs
NUM_EXPERTS = 512
TOP_K = 10

# Test parameters (reduced for speed)
TEST_NUM_EXPERTS = 64    # Reduced from 512
TEST_HIDDEN_DIM = 32     # Input dimension
TEST_TOP_K = 5           # Reduced from 10


def full_cleanup():
    """Thorough cleanup of both switches."""
    print("\n  Full cleanup...")
    for sw_ip in [SWITCH1_IP, SWITCH2_IP]:
        cleanup_cmds = [
            "delete firewall family ethernet-switching filter",
            "delete interfaces et-0/0/96 unit 0 family ethernet-switching",
            "delete interfaces et-0/0/100 unit 0 family ethernet-switching",
            "delete forwarding-options port-mirroring",
            "delete forwarding-options analyzer",
            "delete vlans mirror_test",
            "delete vlans test_vlan",
        ]
        for cmd in cleanup_cmds:
            run_config_commands(sw_ip, [cmd], debug=False)
    time.sleep(0.5)
    print("  ✓ Cleanup complete")


def run_router_projection(router_weights: np.ndarray, 
                          hidden_state: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Run router projection on switches.
    Router: hidden_dim → num_experts (matrix multiply)
    
    Returns: (expert_scores, success)
    """
    num_experts = router_weights.shape[0]
    hidden_dim = router_weights.shape[1]
    
    print(f"\n  Router projection: {hidden_dim} → {num_experts}")
    
    # Cleanup
    full_cleanup()
    
    # Configure port mirroring
    if not configure_port_mirroring(SWITCH1_IP, debug=False):
        return np.zeros(num_experts), False
    if not configure_sw2_port_mirroring_split(debug=False):
        return np.zeros(num_experts), False
    
    # Configure split filters
    success, config_time = configure_split_weights_parallel(router_weights, BATCH_SIZE)
    if not success:
        return np.zeros(num_experts), False
    
    print(f"    Configured in {config_time:.1f}s")
    
    # Clear counters
    ssh_command(SWITCH1_IP, f"cli -c 'clear firewall filter {FILTER_NAME}'")
    ssh_command(SWITCH2_IP, f"cli -c 'clear firewall filter {FILTER_NAME}'")
    time.sleep(0.3)
    
    # Run inference
    expected_counts = run_full_layer_inference(router_weights, hidden_state)
    time.sleep(0.5)
    
    # Read counters
    split_point = num_experts // 2
    neuron_indices = list(range(num_experts))
    actual_counts = read_counters_split(neuron_indices, split_point)
    
    # Convert to array
    scores = np.zeros(num_experts, dtype=np.int32)
    for idx, count in actual_counts.items():
        if idx < num_experts:
            scores[idx] = count
    
    # Expected array for verification
    expected = np.zeros(num_experts, dtype=np.int32)
    for idx, count in expected_counts.items():
        if idx < num_experts:
            expected[idx] = count
    
    matches = np.array_equal(scores, expected)
    
    active = np.count_nonzero(scores)
    print(f"    Scores: {active} non-zero experts")
    print(f"    {'✓ CORRECT' if matches else '✗ MISMATCH'}")
    
    return scores, matches


def select_top_k(scores: np.ndarray, k: int) -> List[int]:
    """
    Select top-K experts based on scores.
    This is just argmax/argsort - done on CPU after reading counters.
    """
    # Get indices of top-K scores
    top_k_indices = np.argsort(scores)[-k:][::-1]  # Descending order
    return list(top_k_indices)


def run_moe_routing_experiment():
    """
    Test MoE routing on switches.
    """
    print("="*80)
    print("E050: MoE ROUTING ON SWITCHES")
    print("="*80)
    
    print(f"""
  MoE Parameters (from Qwen3-Next-80B):
    expert_count: {NUM_EXPERTS}
    expert_used_count: {TOP_K}
    
  Test Parameters (reduced):
    num_experts: {TEST_NUM_EXPERTS}
    hidden_dim: {TEST_HIDDEN_DIM}
    top_k: {TEST_TOP_K}
""")
    
    # Step 1: Create router weights
    print("="*80)
    print("STEP 1: CREATE ROUTER WEIGHTS")
    print("="*80)
    
    # Try to extract real router weights
    try:
        metadata, tensors, data_offset = parse_gguf_header(MODEL_PATH)
        
        router_tensor = None
        for name, tensor in tensors.items():
            if 'gate' in name.lower() or 'router' in name.lower():
                if 'blk.0' in name:
                    router_tensor = tensor
                    print(f"  Found router: {name} {tensor.dims}")
                    break
        
        if router_tensor:
            max_elements = TEST_NUM_EXPERTS * TEST_HIDDEN_DIM
            raw = extract_small_weight_sample(MODEL_PATH, router_tensor, data_offset, max_elements)
            router_weights = weights_to_binary(raw.astype(float))
            router_weights = router_weights[:max_elements].reshape((TEST_NUM_EXPERTS, TEST_HIDDEN_DIM))
            print(f"  Using real router weights")
        else:
            raise Exception("No router tensor found")
            
    except Exception as e:
        print(f"  No real router found, using synthetic: {e}")
        router_weights = np.random.choice([-1, 0, 1], 
                                          size=(TEST_NUM_EXPERTS, TEST_HIDDEN_DIM))
        router_weights = weights_to_binary(router_weights.astype(float)).reshape(
            (TEST_NUM_EXPERTS, TEST_HIDDEN_DIM))
    
    positive = np.sum(router_weights > 0)
    print(f"  Router: {TEST_HIDDEN_DIM} → {TEST_NUM_EXPERTS}")
    print(f"  Positive weights: {positive} ({100*positive/router_weights.size:.1f}%)")
    
    # Step 2: Create test hidden state
    print("\n" + "="*80)
    print("STEP 2: CREATE HIDDEN STATE")
    print("="*80)
    
    hidden_state = np.zeros(TEST_HIDDEN_DIM, dtype=np.int32)
    active_indices = np.random.choice(TEST_HIDDEN_DIM, 5, replace=False)
    for idx in active_indices:
        hidden_state[idx] = np.random.randint(1, 4)
    
    print(f"  Hidden state: {np.count_nonzero(hidden_state)} active dimensions")
    print(f"  Values: {hidden_state[hidden_state > 0]}")
    
    # Step 3: Run router on switches
    print("\n" + "="*80)
    print("STEP 3: RUN ROUTER PROJECTION")
    print("="*80)
    
    scores, success = run_router_projection(router_weights, hidden_state)
    
    if not success:
        print("  ✗ Router projection failed")
        return
    
    # Step 4: Select top-K experts
    print("\n" + "="*80)
    print("STEP 4: SELECT TOP-K EXPERTS")
    print("="*80)
    
    top_k_experts = select_top_k(scores, TEST_TOP_K)
    
    print(f"  Top-{TEST_TOP_K} experts selected:")
    for rank, expert_id in enumerate(top_k_experts):
        print(f"    #{rank+1}: Expert {expert_id} (score: {scores[expert_id]})")
    
    # Verify against NumPy reference
    expected_scores = np.zeros(TEST_NUM_EXPERTS, dtype=np.int32)
    for i in range(TEST_NUM_EXPERTS):
        for j in range(TEST_HIDDEN_DIM):
            if router_weights[i, j] > 0 and hidden_state[j] > 0:
                expected_scores[i] += int(hidden_state[j])
    
    expected_top_k = select_top_k(expected_scores, TEST_TOP_K)
    
    routing_matches = (top_k_experts == expected_top_k)
    
    print(f"\n  Expected top-{TEST_TOP_K}: {expected_top_k}")
    print(f"  Actual top-{TEST_TOP_K}:   {top_k_experts}")
    print(f"  Routing match: {'✓ YES' if routing_matches else '✗ NO'}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if success and routing_matches:
        print(f"""
  🎉🎉🎉 MoE ROUTING WORKS ON SWITCHES! 🎉🎉🎉
  
  What we proved:
    - Router projection (matrix multiply) → 100% correct
    - Top-K expert selection → Matches expected
    - No softmax needed for routing!
    
  For full model:
    - Run router: hidden → 512 expert scores
    - Select top-10 from counter values
    - Route to selected expert FFNs
    
  MoE is fully compatible with photonic inference!
""")
    else:
        print("  ⚠ Some issues detected")
    
    # Cleanup
    print("  Final cleanup...")
    full_cleanup()
    print("  ✓ Done")


if __name__ == '__main__':
    run_moe_routing_experiment()

