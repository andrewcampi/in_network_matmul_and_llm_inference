#!/usr/bin/env python3
"""
e048_multi_layer_loop.py

MULTI-LAYER INFERENCE BY LOOPING e046's PROVEN ARCHITECTURE
============================================================

Simple approach: Call the EXACT same code that works in e046, just in a loop
for multiple layers. No reinventing - just reuse what works.

For each layer:
  1. Extract weights for this layer from the model
  2. Run e046's proven split-switch configuration
  3. Send packets, read counters
  4. Use output as input for next layer
"""

import time
import os
import sys
import numpy as np
from typing import Dict, Tuple, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from e046/e044/e045 - the proven working code
from e046_full_layer_real_weights import (
    configure_split_weights_parallel,
    run_full_layer_inference,
    MIRROR_VLAN,
)

from e045_real_weights_inference import (
    parse_gguf_header, find_weight_tensors, extract_small_weight_sample,
    weights_to_binary, mac_str_to_bytes,
    MODEL_PATH
)

from e044_full_layer_mirror import (
    configure_port_mirroring,
    configure_sw2_port_mirroring_split,
    read_counters_split,
    cleanup_switch,
    get_neuron_mac,
    FILTER_NAME, BATCH_SIZE,
)

from e042_port_based_layers import (
    ssh_command, run_config_commands,
    craft_vlan_packet, send_packets, get_mac_address,
    SWITCH1_IP, SWITCH2_IP, SEND_IFACE,
)

# Configuration
NUM_LAYERS = 4
NUM_OUTPUTS = 256
NUM_INPUTS = 64
TEST_INPUTS = 5


def run_single_layer(layer_idx: int, 
                     weight_matrix: np.ndarray, 
                     input_vector: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Run a single layer using e046's EXACT proven architecture.
    
    Returns: (output_counts, success)
    """
    num_outputs = weight_matrix.shape[0]
    num_inputs = weight_matrix.shape[1]
    
    print(f"\n    ─── Layer {layer_idx} ───")
    print(f"    Configuring split architecture...")
    
    # Step 1: Clean up both switches (like e046 does)
    for sw_ip in [SWITCH1_IP, SWITCH2_IP]:
        cleanup_cmds = [
            "delete firewall family ethernet-switching filter",
            "delete interfaces et-0/0/96 unit 0 family ethernet-switching",
            "delete interfaces et-0/0/100 unit 0 family ethernet-switching",
            "delete forwarding-options port-mirroring",
            "delete forwarding-options analyzer",
            "delete vlans mirror_test",
        ]
        run_config_commands(sw_ip, cleanup_cmds, debug=False)
    time.sleep(0.5)
    
    # Step 2: Configure port mirroring (like e046)
    if not configure_port_mirroring(SWITCH1_IP, debug=False):
        print("    ✗ SW1 port mirroring failed")
        return np.zeros(num_outputs, dtype=np.int32), False
    
    if not configure_sw2_port_mirroring_split(debug=False):
        print("    ✗ SW2 port mirroring failed")
        return np.zeros(num_outputs, dtype=np.int32), False
    
    time.sleep(0.5)
    
    # Step 3: Configure split filters with weights (like e046)
    success, config_time = configure_split_weights_parallel(weight_matrix, BATCH_SIZE)
    
    if not success:
        print("    ✗ Filter configuration failed")
        return np.zeros(num_outputs, dtype=np.int32), False
    
    print(f"    ✓ Configured in {config_time:.1f}s")
    
    # Step 4: Clear counters
    ssh_command(SWITCH1_IP, f"cli -c 'clear firewall filter {FILTER_NAME}'")
    ssh_command(SWITCH2_IP, f"cli -c 'clear firewall filter {FILTER_NAME}'")
    time.sleep(0.3)
    
    # Step 5: Run inference (send packets) - like e046
    expected_counts = run_full_layer_inference(weight_matrix, input_vector)
    time.sleep(0.5)
    
    # Step 6: Read counters from both switches - like e046
    split_point = num_outputs // 2  # SW1 has 0 to split-1, SW2 has split to num_outputs-1
    neuron_indices = list(range(num_outputs))
    actual_counts = read_counters_split(neuron_indices, split_point)
    
    # Convert to numpy array
    output = np.zeros(num_outputs, dtype=np.int32)
    for neuron_idx, count in actual_counts.items():
        if neuron_idx < num_outputs:
            output[neuron_idx] = count
    
    # Compare with expected
    expected_array = np.zeros(num_outputs, dtype=np.int32)
    for neuron_idx, count in expected_counts.items():
        if neuron_idx < num_outputs:
            expected_array[neuron_idx] = count
    
    matches = np.array_equal(output, expected_array)
    
    active = np.count_nonzero(output)
    total = int(np.sum(output))
    print(f"    Output: {active} active neurons, {total} total packets")
    
    if matches:
        print(f"    ✓ CORRECT! Matches expected.")
    else:
        mismatches = int(np.sum(output != expected_array))
        print(f"    ✗ {mismatches} mismatches")
        # Show first few
        for i in range(min(3, num_outputs)):
            if output[i] != expected_array[i]:
                print(f"      n{i}: got {output[i]}, expected {expected_array[i]}")
    
    return output, matches


def run_multi_layer_experiment(num_layers: int = NUM_LAYERS,
                                num_outputs: int = NUM_OUTPUTS,
                                num_inputs: int = NUM_INPUTS,
                                test_inputs: int = TEST_INPUTS):
    """
    Run multi-layer inference by looping through layers.
    Uses e046's proven architecture for each layer.
    """
    print("="*80)
    print("E048: MULTI-LAYER INFERENCE VIA LOOP")
    print("="*80)
    print(f"""
  Configuration:
    Layers: {num_layers}
    Outputs per layer: {num_outputs}
    Inputs: {num_inputs}
    Test inputs: {test_inputs}
    
  Strategy:
    Run e046's proven split-architecture for each layer.
    Output of layer N becomes input to layer N+1.
""")
    
    # Step 1: Load model and extract weights for all layers
    print("="*80)
    print("STEP 1: EXTRACT WEIGHTS")
    print("="*80)
    
    if not os.path.exists(MODEL_PATH):
        print(f"  ✗ Model not found: {MODEL_PATH}")
        return
    
    print(f"  Model: {MODEL_PATH}")
    
    try:
        metadata, tensors, data_offset = parse_gguf_header(MODEL_PATH)
        weight_matrices = []
        
        for layer_idx in range(num_layers):
            # Try different blocks
            block_tensors = find_weight_tensors(tensors, f"blk.{layer_idx}")
            
            target_tensor = None
            for t in block_tensors:
                if 'ffn_gate_inp.weight' in t.name or 'ffn_up' in t.name:
                    target_tensor = t
                    break
            
            if target_tensor is None and block_tensors:
                target_tensor = block_tensors[0]
            
            if target_tensor is None:
                print(f"  Layer {layer_idx}: Using random weights")
                weights = np.random.choice([-1, 0, 1], size=num_outputs * num_inputs)
            else:
                print(f"  Layer {layer_idx}: {target_tensor.name}")
                weights = extract_small_weight_sample(
                    MODEL_PATH, target_tensor, data_offset, 
                    max_elements=num_outputs * num_inputs
                )
            
            # Convert to binary and reshape
            binary = weights_to_binary(weights.astype(float))
            matrix = binary[:num_outputs * num_inputs].reshape((num_outputs, num_inputs))
            weight_matrices.append(matrix)
        
        print(f"\n  ✓ Extracted {len(weight_matrices)} weight matrices")
        
    except Exception as e:
        print(f"  ✗ Failed to extract weights: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 2: Create initial input
    print("\n" + "="*80)
    print("STEP 2: CREATE INPUT")
    print("="*80)
    
    input_vector = np.zeros(num_inputs, dtype=np.int32)
    active_indices = np.random.choice(num_inputs, min(test_inputs, num_inputs), replace=False)
    for idx in active_indices:
        input_vector[idx] = np.random.randint(1, 4)
    
    print(f"  Initial input:")
    print(f"    Active: {len(active_indices)}")
    print(f"    Indices: {sorted(active_indices)}")
    print(f"    Values: {input_vector[active_indices]}")
    
    # Step 3: Run each layer
    print("\n" + "="*80)
    print("STEP 3: RUN LAYERS")
    print("="*80)
    
    current_input = input_vector.copy()
    results = []
    
    for layer_idx in range(num_layers):
        weights = weight_matrices[layer_idx]
        
        # Show input stats
        active_in = np.count_nonzero(current_input)
        print(f"\n  Layer {layer_idx} input: {active_in} active neurons")
        
        # Run layer
        output, success = run_single_layer(layer_idx, weights, current_input)
        results.append(success)
        
        # Use output as next input (cap to prevent explosion)
        current_input = np.minimum(output, 10).astype(np.int32)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    correct = sum(results)
    print(f"\n  Layers correct: {correct}/{num_layers}")
    
    if all(results):
        print(f"""
  🎉🎉🎉 SUCCESS! ALL {num_layers} LAYERS COMPUTED CORRECTLY! 🎉🎉🎉
  
  This proves:
    - Multi-layer inference works with real Qwen3 weights
    - Layer outputs correctly chain as inputs
    - Photonic inference engine can run MULTI-LAYER neural networks!
    
  Next: When the second inter-switch cable arrives, we can run
  all layers in a SINGLE PASS without host intervention!
""")
    else:
        print(f"\n  ⚠ {num_layers - correct} layers had mismatches")
    
    # Cleanup
    print("\n  Cleaning up switches...")
    cleanup_switch(SWITCH1_IP)
    cleanup_switch(SWITCH2_IP)
    print("  ✓ Done")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Multi-layer inference by looping e046's proven architecture"
    )
    parser.add_argument('--layers', '-l', type=int, default=NUM_LAYERS)
    parser.add_argument('--outputs', '-o', type=int, default=NUM_OUTPUTS)
    parser.add_argument('--inputs', '-i', type=int, default=NUM_INPUTS)
    parser.add_argument('--test-inputs', '-t', type=int, default=TEST_INPUTS)
    
    args = parser.parse_args()
    
    run_multi_layer_experiment(
        num_layers=args.layers,
        num_outputs=args.outputs,
        num_inputs=args.inputs,
        test_inputs=args.test_inputs
    )

""" Output:
sudo python3 e048_multi_layer_loop.py --layers 4 --outputs 128 --inputs 32 --test-inputs 3
================================================================================
E048: MULTI-LAYER INFERENCE VIA LOOP
================================================================================

  Configuration:
    Layers: 4
    Outputs per layer: 128
    Inputs: 32
    Test inputs: 3
    
  Strategy:
    Run e046's proven split-architecture for each layer.
    Output of layer N becomes input to layer N+1.

================================================================================
STEP 1: EXTRACT WEIGHTS
================================================================================
  Model: ./models/Qwen3-Next-80B-A3B-Thinking-UD-TQ1_0.gguf

  Parsing GGUF file: ./models/Qwen3-Next-80B-A3B-Thinking-UD-TQ1_0.gguf
    GGUF version: 3
    Tensor count: 807
    KV count: 53
    Data starts at: 5985152
  Layer 0: blk.0.ffn_gate_inp.weight

  Extracting weights from: blk.0.ffn_gate_inp.weight
    Dims: [2048, 512]
    Type ID: 0
    Total elements: 1048576
  Layer 1: blk.1.ffn_gate_inp.weight

  Extracting weights from: blk.1.ffn_gate_inp.weight
    Dims: [2048, 512]
    Type ID: 0
    Total elements: 1048576
  Layer 2: blk.2.ffn_gate_inp.weight

  Extracting weights from: blk.2.ffn_gate_inp.weight
    Dims: [2048, 512]
    Type ID: 0
    Total elements: 1048576
  Layer 3: blk.3.ffn_gate_inp.weight

  Extracting weights from: blk.3.ffn_gate_inp.weight
    Dims: [2048, 512]
    Type ID: 0
    Total elements: 1048576

  ✓ Extracted 4 weight matrices

================================================================================
STEP 2: CREATE INPUT
================================================================================
  Initial input:
    Active: 3
    Indices: [21, 29, 30]
    Values: [2 2 2]

================================================================================
STEP 3: RUN LAYERS
================================================================================

  Layer 0 input: 3 active neurons

    ─── Layer 0 ───
    Configuring split architecture...

  Configuring port-mirroring on 10.10.10.55...
    ✓ Port-mirroring instance 'packet_mirror' configured
    → Mirrored packets will go to et-0/0/100

  Configuring port-mirroring on SW2 for SPLIT MODE...
    Input port: et-0/0/100 (from SW1)
    Mirror output: et-0/0/96 (to host)
    ✓ SW2 port-mirroring configured for split mode
    → Mirrored packets go to et-0/0/96 (host)

  PARALLEL SPLIT CONFIGURATION WITH REAL WEIGHTS
    Weight matrix: 128 x 32
    SW1 (10.10.10.55): outputs 0-63
    SW2 (10.10.10.56): outputs 64-127

    Configuring 10.10.10.55 for outputs 0-63...

    Configuring 10.10.10.56 for outputs 64-127...
      Positive weights (rules): 1004

  Configuring neurons 64-127 (64 terms) on 10.10.10.56...
  Input port: et-0/0/100
  Batch size: 50 terms per commit
      Positive weights (rules): 1030

  Configuring neurons 0-63 (64 terms) on 10.10.10.55...
  Input port: et-0/0/96
  Batch size: 50 terms per commit
    [DEBUG] stdout: Entering configuration mode
configuration check succeeds
commit complete

    Batch 1/2: neurons 64-113...
    Batch 2/2: neurons 114-127...
    [DEBUG] stdout: Entering configuration mode
configuration check succeeds
commit complete

    Batch 1/2: neurons 0-49...
    Batch 2/2: neurons 50-63...
    [DEBUG] stdout: Entering configuration mode
configuration check succeeds
commit complete

    ✓ 64 neurons configured on 10.10.10.56
    [DEBUG] stdout: Entering configuration mode
configuration check succeeds
commit complete

    ✓ 64 neurons configured on 10.10.10.55

  ✓ Both switches configured in 29.4s
    Total rules: 2034
    Rate: 4.4 neurons/second
    ✓ Configured in 29.4s

  Running full-layer inference...
    Outputs: 128
    Inputs: 32
    Active inputs: 3
    Sent 374 packets
    Outputs with traffic: 110
    Output: 110 active neurons, 374 total packets
    ✓ CORRECT! Matches expected.

  Layer 1 input: 110 active neurons

    ─── Layer 1 ───
    Configuring split architecture...

  Configuring port-mirroring on 10.10.10.55...
    ✓ Port-mirroring instance 'packet_mirror' configured
    → Mirrored packets will go to et-0/0/100

  Configuring port-mirroring on SW2 for SPLIT MODE...
    Input port: et-0/0/100 (from SW1)
    Mirror output: et-0/0/96 (to host)
    ✓ SW2 port-mirroring configured for split mode
    → Mirrored packets go to et-0/0/96 (host)

  PARALLEL SPLIT CONFIGURATION WITH REAL WEIGHTS
    Weight matrix: 128 x 32
    SW1 (10.10.10.55): outputs 0-63
    SW2 (10.10.10.56): outputs 64-127

    Configuring 10.10.10.55 for outputs 0-63...

    Configuring 10.10.10.56 for outputs 64-127...
      Positive weights (rules): 972

  Configuring neurons 0-63 (64 terms) on 10.10.10.55...
  Input port: et-0/0/96
  Batch size: 50 terms per commit
      Positive weights (rules): 1001

  Configuring neurons 64-127 (64 terms) on 10.10.10.56...
  Input port: et-0/0/100
  Batch size: 50 terms per commit
    [DEBUG] stdout: Entering configuration mode
configuration check succeeds
commit complete

    Batch 1/2: neurons 64-113...
    Batch 2/2: neurons 114-127...
    [DEBUG] stdout: Entering configuration mode
configuration check succeeds
commit complete

    Batch 1/2: neurons 0-49...
    Batch 2/2: neurons 50-63...
    [DEBUG] stdout: Entering configuration mode
configuration check succeeds
commit complete

    ✓ 64 neurons configured on 10.10.10.56
    [DEBUG] stdout: Entering configuration mode
configuration check succeeds
commit complete

    ✓ 64 neurons configured on 10.10.10.55

  ✓ Both switches configured in 29.3s
    Total rules: 1973
    Rate: 4.4 neurons/second
    ✓ Configured in 29.3s

  Running full-layer inference...
    Outputs: 128
    Inputs: 128
    Active inputs: 110
    Sent 4982 packets
    Outputs with traffic: 128
    Output: 128 active neurons, 4982 total packets
    ✓ CORRECT! Matches expected.

  Layer 2 input: 128 active neurons

    ─── Layer 2 ───
    Configuring split architecture...

  Configuring port-mirroring on 10.10.10.55...
    ✓ Port-mirroring instance 'packet_mirror' configured
    → Mirrored packets will go to et-0/0/100

  Configuring port-mirroring on SW2 for SPLIT MODE...
    Input port: et-0/0/100 (from SW1)
    Mirror output: et-0/0/96 (to host)
    ✓ SW2 port-mirroring configured for split mode
    → Mirrored packets go to et-0/0/96 (host)

  PARALLEL SPLIT CONFIGURATION WITH REAL WEIGHTS
    Weight matrix: 128 x 32
    SW1 (10.10.10.55): outputs 0-63
    SW2 (10.10.10.56): outputs 64-127

    Configuring 10.10.10.55 for outputs 0-63...
      Positive weights (rules): 1006

  Configuring neurons 0-63 (64 terms) on 10.10.10.55...
  Input port: et-0/0/96
  Batch size: 50 terms per commit

    Configuring 10.10.10.56 for outputs 64-127...
      Positive weights (rules): 983

  Configuring neurons 64-127 (64 terms) on 10.10.10.56...
  Input port: et-0/0/100
  Batch size: 50 terms per commit
    [DEBUG] stdout: Entering configuration mode
configuration check succeeds
commit complete

    Batch 1/2: neurons 64-113...
    Batch 2/2: neurons 114-127...
    [DEBUG] stdout: Entering configuration mode
configuration check succeeds
commit complete

    Batch 1/2: neurons 0-49...
    Batch 2/2: neurons 50-63...
    [DEBUG] stdout: Entering configuration mode
configuration check succeeds
commit complete

    ✓ 64 neurons configured on 10.10.10.56
    [DEBUG] stdout: Entering configuration mode
configuration check succeeds
commit complete

    ✓ 64 neurons configured on 10.10.10.55

  ✓ Both switches configured in 29.3s
    Total rules: 1989
    Rate: 4.4 neurons/second
    ✓ Configured in 29.3s

  Running full-layer inference...
    Outputs: 128
    Inputs: 128
    Active inputs: 128
    Sent 19890 packets
    Outputs with traffic: 128
    Output: 128 active neurons, 19890 total packets
    ✓ CORRECT! Matches expected.

  Layer 3 input: 128 active neurons

    ─── Layer 3 ───
    Configuring split architecture...

  Configuring port-mirroring on 10.10.10.55...
    ✓ Port-mirroring instance 'packet_mirror' configured
    → Mirrored packets will go to et-0/0/100

  Configuring port-mirroring on SW2 for SPLIT MODE...
    Input port: et-0/0/100 (from SW1)
    Mirror output: et-0/0/96 (to host)
    ✓ SW2 port-mirroring configured for split mode
    → Mirrored packets go to et-0/0/96 (host)

  PARALLEL SPLIT CONFIGURATION WITH REAL WEIGHTS
    Weight matrix: 128 x 32
    SW1 (10.10.10.55): outputs 0-63
    SW2 (10.10.10.56): outputs 64-127

    Configuring 10.10.10.55 for outputs 0-63...

    Configuring 10.10.10.56 for outputs 64-127...
      Positive weights (rules): 1009

  Configuring neurons 64-127 (64 terms) on 10.10.10.56...
  Input port: et-0/0/100
  Batch size: 50 terms per commit
      Positive weights (rules): 1020

  Configuring neurons 0-63 (64 terms) on 10.10.10.55...
  Input port: et-0/0/96
  Batch size: 50 terms per commit
    [DEBUG] stdout: Entering configuration mode
configuration check succeeds
commit complete

    Batch 1/2: neurons 64-113...
    Batch 2/2: neurons 114-127...
    [DEBUG] stdout: Entering configuration mode
configuration check succeeds
commit complete

    Batch 1/2: neurons 0-49...
    Batch 2/2: neurons 50-63...
    [DEBUG] stdout: Entering configuration mode
configuration check succeeds
commit complete

    ✓ 64 neurons configured on 10.10.10.56
    [DEBUG] stdout: Entering configuration mode
configuration check succeeds
commit complete

    ✓ 64 neurons configured on 10.10.10.55

  ✓ Both switches configured in 29.5s
    Total rules: 2029
    Rate: 4.3 neurons/second
    ✓ Configured in 29.5s

  Running full-layer inference...
    Outputs: 128
    Inputs: 128
    Active inputs: 128
    Sent 20290 packets
    Outputs with traffic: 128
    Output: 128 active neurons, 20290 total packets
    ✓ CORRECT! Matches expected.

================================================================================
SUMMARY
================================================================================

  Layers correct: 4/4

  🎉🎉🎉 SUCCESS! ALL 4 LAYERS COMPUTED CORRECTLY! 🎉🎉🎉
  
  This proves:
    - Multi-layer inference works with real Qwen3 weights
    - Layer outputs correctly chain as inputs
    - Photonic inference engine can run MULTI-LAYER neural networks!
    
  Next: When the second inter-switch cable arrives, we can run
  all layers in a SINGLE PASS without host intervention!


  Cleaning up switches...

  Cleaning up 10.10.10.55...
    Found 3 VLANs: ['default', 'mirror_test', 'test_vlan']
    Deleting 2 VLANs...
    ✓ Cleanup complete

  Cleaning up 10.10.10.56...
    Found 3 VLANs: ['default', 'mirror_test', 'test_vlan']
    Deleting 2 VLANs...
    ✓ Cleanup complete
  ✓ Done
"""