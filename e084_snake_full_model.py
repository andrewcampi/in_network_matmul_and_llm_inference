#!/usr/bin/env python3
"""
e084_snake_full_model.py

EXPERIMENT:
Run a minimal Qwen‑3 0.6B inference using the *snake* architecture
(described in e083_layer_snake_architecture.py) on the current topology
(topology.md). The goal is to generate the next three tokens after the
prompt "The " using the switches for the heavy linear algebra.

HARDWARE LIMITS & SCOPE
-----------------------
* Each switch port can carry at most 8 VLANs.
* Our topology (two switches) therefore supports at most 8 layers per hop.
* The full 28‑layer model would need 4 hops (7 layers per hop) with
  additional inter‑switch links; this script demonstrates the first 8
  layers as a proof‑of‑concept.  Extending to the full model only requires
  repeating the snake‑configuration for the extra hops.
* FAST_TEST mode limits each projection to 64 neurons (fits the TCAM rule
  limit of 1152 entries per filter).

USAGE
-----
$ python3 e084_snake_full_model.py
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import sys
import time
import re
import numpy as np
from typing import List, Dict, Tuple

# Add the repository root to the import path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, repo_root)

# -------------------------------------------------------------------------
# Weight loading (from e082_full_model_real_weights.py)
# -------------------------------------------------------------------------
from e082_full_model_real_weights import (
    load_gguf_weights,
    Qwen3Weights,
    SimpleTokenizer,  # fallback tokenizer defined in e082
    MODEL_PATH,
    D_MODEL,
    N_HEADS,
    D_HEAD,
    N_KV_HEADS,
    D_FFN,
    VOCAB_SIZE,
    cpu_reference_generate,
)

# -------------------------------------------------------------------------
# Snake‑architecture utilities (from e083_layer_snake_architecture.py)
# -------------------------------------------------------------------------
from e083_layer_snake_architecture import (
    full_cleanup,
    read_all_counters,
    clear_all_counters,
    SWITCH1_IP,
    SWITCH2_IP,
    SEND_IFACE,
    get_mac_address,
    ssh_command_long,
    transfer_and_apply_config,
    BASE_VLAN as E083_BASE_VLAN,
)
from e042_port_based_layers import (
    craft_vlan_packet,
    send_packets,
)
from e053_mac_encoded_layers import get_layer_neuron_mac
from e045_real_weights_inference import mac_str_to_bytes

# =============================================================================
# CONFIGURATION
# =============================================================================
# We only configure the first 8 layers (0‑7) – 4 on each switch.
# This respects the 8‑VLAN‑per‑port limit and keeps the experiment fast.
SW1_LAYERS = list(range(0, 4))   # layers 0‑3 on Switch 1
SW2_LAYERS = list(range(4, 8))   # layers 4‑7 on Switch 2

# Fast test mode: limit neurons per projection to fit TCAM limits
# Each projection needs 2 counters (pos/neg) per neuron
# TCAM limit is ~1152 terms per filter
# For 7 projections per layer: 1152 / 7 / 2 = ~82 neurons max
# We use 64 neurons for safety
FAST_TEST_NEURONS = 64

# VLAN configuration
BASE_VLAN = 200  # Start VLANs from 200 (matches e083)

# Interfaces
SW1_HOST_IFACE = "et-0/0/96"    # SW1 receives from host
SW1_INTER_IFACE = "et-0/0/100"  # SW1 to SW2
SW2_INTER_IFACE = "et-0/0/100"  # SW2 from SW1
SW2_HOST_IFACE = "et-0/0/96"    # SW2 to host (for return path)

SSH_KEY = "/home/multiplex/.ssh/id_rsa"

# Prompt we want to continue
PROMPT = "The "
NUM_TOKENS_TO_GENERATE = 3

# =============================================================================
# SNAKE ARCHITECTURE FUNCTIONS (adapted from e083)
# =============================================================================

def configure_sw1_snake_adapted(layers: List[int], num_neurons: int) -> bool:
    """Configure SW1 for snake architecture with custom neuron count."""
    print(f"\n  Configuring SW1 (layers {layers}, {num_neurons} neurons)...")
    
    all_cmds = []
    all_cmds.append("set forwarding-options storm-control-profiles default all")
    
    # Create VLANs for SW1's layers AND SW2's layers (for forwarding)
    all_vlans_sw1 = []
    all_vlans_sw2 = []
    
    # SW1's layers (0-3): these get filters on et-0/0/96
    for layer in layers:
        vlan_id = BASE_VLAN + layer
        vlan_name = f"layer{layer}_vlan"
        all_vlans_sw1.append(vlan_name)
        all_cmds.append(f"set vlans {vlan_name} vlan-id {vlan_id}")
    
    # SW2's layers (4-7): just VLANs, no filters (pass-through to SW2)
    for layer in SW2_LAYERS:
        vlan_id = BASE_VLAN + layer
        vlan_name = f"layer{layer}_vlan"
        all_vlans_sw2.append(vlan_name)
        all_cmds.append(f"set vlans {vlan_name} vlan-id {vlan_id}")
    
    # Configure et-0/0/96 (from host) as trunk with all VLANs
    all_cmds.append(f"delete interfaces {SW1_HOST_IFACE} unit 0 family ethernet-switching")
    all_cmds.append(f"set interfaces {SW1_HOST_IFACE} unit 0 family ethernet-switching interface-mode trunk")
    for vlan_name in all_vlans_sw1 + all_vlans_sw2:
        all_cmds.append(f"set interfaces {SW1_HOST_IFACE} unit 0 family ethernet-switching vlan members {vlan_name}")
    
    # Configure et-0/0/100 (to SW2) as trunk with SW2's VLANs
    all_cmds.append(f"delete interfaces {SW1_INTER_IFACE} unit 0 family ethernet-switching")
    all_cmds.append(f"set interfaces {SW1_INTER_IFACE} unit 0 family ethernet-switching interface-mode trunk")
    for vlan_name in all_vlans_sw2:
        all_cmds.append(f"set interfaces {SW1_INTER_IFACE} unit 0 family ethernet-switching vlan members {vlan_name}")
    
    # Configure filters for SW1's layers (on the VLAN, not the interface)
    for layer in layers:
        filter_name = f"layer{layer}_filter"
        vlan_name = f"layer{layer}_vlan"
        
        # Create filter with neuron counters
        for neuron in range(num_neurons):
            # Positive counter
            mac_pos = get_layer_neuron_mac(layer * 2, neuron)
            term_pos = f"l{layer}_n{neuron}_p"
            all_cmds.extend([
                f"set firewall family ethernet-switching filter {filter_name} term {term_pos} from destination-mac-address {mac_pos}/48",
                f"set firewall family ethernet-switching filter {filter_name} term {term_pos} then count {term_pos}",
                f"set firewall family ethernet-switching filter {filter_name} term {term_pos} then accept",
            ])
            
            # Negative counter
            mac_neg = get_layer_neuron_mac(layer * 2 + 1, neuron)
            term_neg = f"l{layer}_n{neuron}_n"
            all_cmds.extend([
                f"set firewall family ethernet-switching filter {filter_name} term {term_neg} from destination-mac-address {mac_neg}/48",
                f"set firewall family ethernet-switching filter {filter_name} term {term_neg} then count {term_neg}",
                f"set firewall family ethernet-switching filter {filter_name} term {term_neg} then accept",
            ])
        
        # Default term
        all_cmds.append(f"set firewall family ethernet-switching filter {filter_name} term default then accept")
        
        # Bind filter to VLAN
        all_cmds.append(f"set vlans {vlan_name} forwarding-options filter input {filter_name}")
    
    # Apply config
    return transfer_and_apply_config(SWITCH1_IP, all_cmds, "sw1_snake")


def configure_sw2_snake_adapted(layers: List[int], num_neurons: int) -> bool:
    """Configure SW2 for snake architecture with custom neuron count."""
    print(f"\n  Configuring SW2 (layers {layers}, {num_neurons} neurons)...")
    
    all_cmds = []
    all_cmds.append("set forwarding-options storm-control-profiles default all")
    
    # Create VLANs for SW2's layers
    all_vlans = []
    for layer in layers:
        vlan_id = BASE_VLAN + layer
        vlan_name = f"layer{layer}_vlan"
        all_vlans.append(vlan_name)
        all_cmds.append(f"set vlans {vlan_name} vlan-id {vlan_id}")
    
    # Configure et-0/0/100 (from SW1) as trunk with SW2's VLANs
    all_cmds.append(f"delete interfaces {SW2_INTER_IFACE} unit 0 family ethernet-switching")
    all_cmds.append(f"set interfaces {SW2_INTER_IFACE} unit 0 family ethernet-switching interface-mode trunk")
    for vlan_name in all_vlans:
        all_cmds.append(f"set interfaces {SW2_INTER_IFACE} unit 0 family ethernet-switching vlan members {vlan_name}")
    
    # Configure filters for SW2's layers
    for layer in layers:
        filter_name = f"layer{layer}_filter"
        vlan_name = f"layer{layer}_vlan"
        
        for neuron in range(num_neurons):
            # Positive counter
            mac_pos = get_layer_neuron_mac(layer * 2, neuron)
            term_pos = f"l{layer}_n{neuron}_p"
            all_cmds.extend([
                f"set firewall family ethernet-switching filter {filter_name} term {term_pos} from destination-mac-address {mac_pos}/48",
                f"set firewall family ethernet-switching filter {filter_name} term {term_pos} then count {term_pos}",
                f"set firewall family ethernet-switching filter {filter_name} term {term_pos} then accept",
            ])
            
            # Negative counter
            mac_neg = get_layer_neuron_mac(layer * 2 + 1, neuron)
            term_neg = f"l{layer}_n{neuron}_n"
            all_cmds.extend([
                f"set firewall family ethernet-switching filter {filter_name} term {term_neg} from destination-mac-address {mac_neg}/48",
                f"set firewall family ethernet-switching filter {filter_name} term {term_neg} then count {term_neg}",
                f"set firewall family ethernet-switching filter {filter_name} term {term_neg} then accept",
            ])
        
        all_cmds.append(f"set firewall family ethernet-switching filter {filter_name} term default then accept")
        all_cmds.append(f"set vlans {vlan_name} forwarding-options filter input {filter_name}")
    
    return transfer_and_apply_config(SWITCH2_IP, all_cmds, "sw2_snake")


def create_snake_packets_adapted(activations: np.ndarray, weights_per_layer: Dict[int, np.ndarray],
                                src_mac: str) -> Tuple[List[bytes], Dict[int, np.ndarray]]:
    """
    Create packets for ALL layers in the snake.
    
    Args:
        activations: Input activation vector [in_dim]
        weights_per_layer: Dict mapping layer -> weight matrix [out_dim, in_dim]
        src_mac: Source MAC address string
    
    Returns:
        packets: List of all packets to send
        expected: Dict of expected output per layer
    """
    packets = []
    expected = {}
    
    src = mac_str_to_bytes(src_mac)
    
    for layer, weights in weights_per_layer.items():
        vlan_id = BASE_VLAN + layer
        
        # Transpose weights to (in_dim, out_dim) for matrix multiply
        # weights is (out_dim, in_dim), we need (in_dim, out_dim)
        weights_T = weights.T  # Now (in_dim, out_dim)
        
        # Compute expected output: activations @ weights_T = [out_dim]
        output = activations @ weights_T
        expected[layer] = output
        
        # Create packets for each neuron
        for neuron in range(len(output)):
            value = int(output[neuron])
            
            if value >= 0:
                mac = get_layer_neuron_mac(layer * 2, neuron)
                count = value
            else:
                mac = get_layer_neuron_mac(layer * 2 + 1, neuron)
                count = -value
            
            dst = mac_str_to_bytes(mac)
            
            for _ in range(count):
                packets.append(craft_vlan_packet(dst, src, vlan_id))
    
    return packets, expected


def read_all_counters_adapted(num_neurons: int) -> Dict[int, Dict[int, int]]:
    """Read counters from all layers on both switches."""
    all_counters = {}
    
    # SW1 layers (0-3)
    for layer in SW1_LAYERS:
        filter_name = f"layer{layer}_filter"
        success, stdout, _ = ssh_command_long(
            SWITCH1_IP,
            f"cli -c 'show firewall filter {filter_name}'",
            timeout=30
        )
        
        if not success:
            all_counters[layer] = {}
            continue
        
        counters = {}
        for neuron in range(num_neurons):
            pos_val = 0
            neg_val = 0
            
            pos_match = re.search(rf"l{layer}_n{neuron}_p\s+\d+\s+(\d+)", stdout)
            if pos_match:
                pos_val = int(pos_match.group(1))
            
            neg_match = re.search(rf"l{layer}_n{neuron}_n\s+\d+\s+(\d+)", stdout)
            if neg_match:
                neg_val = int(neg_match.group(1))
            
            counters[neuron] = pos_val - neg_val
        
        all_counters[layer] = counters
    
    # SW2 layers (4-7)
    for layer in SW2_LAYERS:
        filter_name = f"layer{layer}_filter"
        success, stdout, _ = ssh_command_long(
            SWITCH2_IP,
            f"cli -c 'show firewall filter {filter_name}'",
            timeout=30
        )
        
        if not success:
            all_counters[layer] = {}
            continue
        
        counters = {}
        for neuron in range(num_neurons):
            pos_val = 0
            neg_val = 0
            
            pos_match = re.search(rf"l{layer}_n{neuron}_p\s+\d+\s+(\d+)", stdout)
            if pos_match:
                pos_val = int(pos_match.group(1))
            
            neg_match = re.search(rf"l{layer}_n{neuron}_n\s+\d+\s+(\d+)", stdout)
            if neg_match:
                neg_val = int(neg_match.group(1))
            
            counters[neuron] = pos_val - neg_val
        
        all_counters[layer] = counters
    
    return all_counters


def clear_all_counters_adapted():
    """Clear counters on both switches."""
    for layer in SW1_LAYERS:
        ssh_command_long(SWITCH1_IP, f"cli -c 'clear firewall filter layer{layer}_filter'", timeout=10)
    
    for layer in SW2_LAYERS:
        ssh_command_long(SWITCH2_IP, f"cli -c 'clear firewall filter layer{layer}_filter'", timeout=10)
    
    time.sleep(0.3)

# =============================================================================
# MAIN EXPERIMENT
# =============================================================================
def main() -> None:
    print("=" * 80)
    print("E084: Qwen‑3 0.6B INFERENCE WITH SNAKE ARCHITECTURE")
    print("=" * 80)

    # -----------------------------------------------------------------
    # 1️⃣ Load model weights (int4, fast‑test configuration)
    # -----------------------------------------------------------------
    print("\nSTEP 1: LOAD WEIGHTS")
    weights: Qwen3Weights | None = load_gguf_weights()
    if weights is None:
        print("❌ Failed to load weights – aborting.")
        sys.exit(1)

    # -----------------------------------------------------------------
    # 2️⃣ Clean up any previous configuration on both switches
    # -----------------------------------------------------------------
    print("\nSTEP 2: CLEANUP SWITCHES")
    full_cleanup(SWITCH1_IP, "SW1")
    full_cleanup(SWITCH2_IP, "SW2")
    time.sleep(1)
    print("  ✓ Switches cleaned")

    # -----------------------------------------------------------------
    # 3️⃣ Configure the snake architecture for the selected layers
    # -----------------------------------------------------------------
    print("\nSTEP 3: CONFIGURE SNAKE ARCHITECTURE")
    if not configure_sw1_snake_adapted(SW1_LAYERS, FAST_TEST_NEURONS):
        print("❌ Switch 1 configuration failed")
        sys.exit(1)
    if not configure_sw2_snake_adapted(SW2_LAYERS, FAST_TEST_NEURONS):
        print("❌ Switch 2 configuration failed")
        sys.exit(1)
    print("  ✓ Both switches configured for layers 0‑7")

    # -----------------------------------------------------------------
    # 4️⃣ Tokenise the prompt and fetch the embedding for the last token
    # -----------------------------------------------------------------
    print("\nSTEP 4: TOKENISE PROMPT")
    tokenizer = SimpleTokenizer()
    token_ids = tokenizer.encode(PROMPT)
    print(f"  Prompt tokens: {token_ids}")

    if not token_ids:
        print("❌ No tokens produced by tokenizer – aborting.")
        sys.exit(1)

    last_token = token_ids[-1]
    embedding = weights.token_embd[last_token].astype(np.int32)   # shape (1024,)
    print(f"  Embedding shape: {embedding.shape}")

    # -----------------------------------------------------------------
    # 5️⃣ Build a *single burst* of packets that traverses all 8 layers.
    #     For brevity we only demonstrate the **Q‑projection** of each
    #     layer (the heaviest linear operation).  The remaining transformer
    #     steps (K, V, O, FFN, Norm, etc.) are performed on the CPU using
    #     the same weights – this still proves that the switches can handle
    #     the core matrix‑multiply workload.
    # -----------------------------------------------------------------
    print("\nSTEP 5: CREATE & SEND PACKETS (single burst)")
    src_mac = get_mac_address(SEND_IFACE)
    print(f"  Source MAC: {src_mac}")

    # Clear counters before the burst
    clear_all_counters_adapted()
    time.sleep(0.5)

    # Build a dict: layer → Q‑weight matrix (output_dim × input_dim)
    # In the real model Q‑weight shape is (n_heads * d_head, d_model) = (2048, 1024)
    # But e082 stores it as (d_model, n_heads * d_head) = (1024, 2048) after transpose
    # Actually, looking at e082 line 355: attn_q is (N_HEADS * D_HEAD, D_MODEL) = (2048, 1024)
    # So weights.attn_q[layer] is (2048, 1024) = (out_dim, in_dim)
    # For create_snake_packets_adapted, we pass it as-is (it will transpose internally)
    q_weights_per_layer: Dict[int, np.ndarray] = {}
    for layer in SW1_LAYERS + SW2_LAYERS:
        w_q = weights.attn_q[layer]          # shape (2048, 1024) or (1024, 1024) in test mode
        # Trim to the fast‑test size: take first 64 outputs and first 64 inputs
        if w_q.shape[0] > FAST_TEST_NEURONS:
            w_q = w_q[:FAST_TEST_NEURONS, :FAST_TEST_NEURONS]
        elif w_q.shape[1] > FAST_TEST_NEURONS:
            w_q = w_q[:, :FAST_TEST_NEURONS]
        q_weights_per_layer[layer] = w_q.astype(np.int32)

    # Also trim embedding to match
    embedding_trimmed = embedding[:FAST_TEST_NEURONS]

    # Create packets for *all* layers in one call
    packets, expected_q = create_snake_packets_adapted(
        activations=embedding_trimmed,       # input activation vector (trimmed)
        weights_per_layer=q_weights_per_layer,
        src_mac=src_mac,
    )
    print(f"  Total packets to send: {len(packets):,}")

    # Send the burst
    start = time.time()
    send_packets(SEND_IFACE, packets)
    send_time = time.time() - start
    print(f"  ✓ Burst sent in {send_time*1000:.1f} ms")
    time.sleep(1)   # give the switches a moment to finish counting

    # -----------------------------------------------------------------
    # 6️⃣ Read back the Q‑vectors from the switches
    # -----------------------------------------------------------------
    print("\nSTEP 6: READ COUNTERS")
    start = time.time()
    all_counters = read_all_counters_adapted(FAST_TEST_NEURONS)
    read_time = time.time() - start
    print(f"  ✓ Counters read in {read_time*1000:.1f} ms")

    # Reconstruct the Q‑vectors (output_dim = 64 in fast‑test)
    q_vectors: List[np.ndarray] = []
    for layer in SW1_LAYERS + SW2_LAYERS:
        switch_ip = SWITCH1_IP if layer in SW1_LAYERS else SWITCH2_IP
        filter_name = f"layer{layer}_filter"
        # The read_all_counters helper already returns a dict[layer][neuron]
        layer_counts = all_counters.get(layer, {})
        # Convert dict → ordered numpy array
        vec = np.array([layer_counts.get(n, 0) for n in range(FAST_TEST_NEURONS)], dtype=np.int32)
        q_vectors.append(vec)

    # -----------------------------------------------------------------
    # 7️⃣ Finish the transformer computation on the CPU
    # -----------------------------------------------------------------
    print("\nSTEP 7: COMPLETE TRANSFORMER ON CPU")
    print("  Note: This is a simplified version - only Q projection runs on switch")
    print("        Other operations (K, V, O, FFN) run on CPU for now")
    
    # For this demo, we'll use the switch-computed Q vectors but complete
    # the rest of the transformer on CPU. This proves the switch can handle
    # the heavy matrix multiply workload.
    
    # Verify switch results match expected
    print("\n  Verifying switch results...")
    all_match = True
    for idx, layer in enumerate(SW1_LAYERS + SW2_LAYERS):
        switch_q = q_vectors[idx]
        expected_q_layer = expected_q.get(layer, np.zeros(FAST_TEST_NEURONS))
        match = np.allclose(switch_q, expected_q_layer, atol=2)
        if not match:
            print(f"    Layer {layer}: ⚠ Mismatch (max diff: {np.abs(switch_q - expected_q_layer).max()})")
            all_match = False
        else:
            print(f"    Layer {layer}: ✓ Match")
    
    if all_match:
        print("  ✓ All switch results match expected values!")
    else:
        print("  ⚠ Some mismatches detected, but continuing...")
    
    # For token generation, we'll use a simplified approach:
    # Use the switch-computed Q vectors as a proof-of-concept,
    # but complete the full transformer pipeline on CPU for accuracy
    print("\n  Running full transformer pipeline on CPU (using switch Q as proof)...")
    
    # Use CPU reference for actual token generation since we only did Q on switch
    # This demonstrates the concept - full implementation would move more ops to switch
    # Dequantize embedding: int4 values need to be scaled
    QUANT_SCALE = 4.0  # From e082
    activation = (embedding.astype(np.float32) / QUANT_SCALE)
    
    # Run through all 8 layers (simplified transformer)
    for layer in SW1_LAYERS + SW2_LAYERS:
        # Use switch Q if available, otherwise compute on CPU
        if layer < len(q_vectors):
            q_switch = q_vectors[layer].astype(np.float32) / QUANT_SCALE
            # Compute full Q on CPU (we only computed first 64 neurons on switch)
            w_q = (weights.attn_q[layer].astype(np.float32) / QUANT_SCALE)
            q_full = (activation @ w_q.T)  # (1024,)
            # Replace first 64 elements with switch result as proof-of-concept
            q_full[:len(q_switch)] = q_switch
            q = q_full
        else:
            # Fallback: compute Q on CPU
            w_q = (weights.attn_q[layer].astype(np.float32) / QUANT_SCALE)
            q = (activation @ w_q.T)
        
        # K, V projections (CPU) - dequantize weights
        w_k = (weights.attn_k[layer].astype(np.float32) / QUANT_SCALE)
        w_v = (weights.attn_v[layer].astype(np.float32) / QUANT_SCALE)
        k = (activation @ w_k.T)  # (512,) = N_KV_HEADS * D_HEAD
        v = (activation @ w_v.T)  # (512,)
        
        # GQA attention: Q has 16 heads, K/V have 8 heads
        # Reshape Q to (N_HEADS, D_HEAD) = (16, 64)
        # Reshape K/V to (N_KV_HEADS, D_HEAD) = (8, 64)
        q_reshaped = q.reshape(N_HEADS, D_HEAD)  # (16, 64)
        k_reshaped = k.reshape(N_KV_HEADS, D_HEAD)  # (8, 64)
        v_reshaped = v.reshape(N_KV_HEADS, D_HEAD)  # (8, 64)
        
        # For GQA: each KV head is shared by 2 Q heads (16 Q / 8 KV = 2)
        # For single-token attention, compute context directly:
        # For each Q head i, use KV head i//2
        context_heads = np.zeros((N_HEADS, D_HEAD), dtype=np.float32)
        for q_idx in range(N_HEADS):
            kv_idx = q_idx // 2  # Each KV head shared by 2 Q heads
            # Compute attention score (scaled dot-product)
            # Clip to prevent overflow
            score = np.clip(
                (q_reshaped[q_idx] @ k_reshaped[kv_idx]) / np.sqrt(D_HEAD),
                -50.0, 50.0  # Clip to reasonable range
            )
            # For single-token, no softmax needed - just use score directly
            # (In multi-token, we'd softmax over sequence positions)
            context_heads[q_idx] = score * v_reshaped[kv_idx]
        
        # Flatten context back to (N_HEADS * D_HEAD,) = (1024,)
        context = context_heads.flatten()
        
        # Output projection - dequantize weights
        w_o = (weights.attn_o[layer].astype(np.float32) / QUANT_SCALE)
        o = (context @ w_o.T)
        
        # Residual + simplified norm
        activation = np.clip(activation + o * 0.1, -100.0, 100.0)  # Clip to prevent overflow
        
        # FFN (simplified) - dequantize weights first to prevent overflow
        # Note: weights are int8 values in range [-8, 7], scale to float first
        w_gate = weights.ffn_gate[layer].astype(np.float32) / QUANT_SCALE
        w_up = weights.ffn_up[layer].astype(np.float32) / QUANT_SCALE
        w_down = weights.ffn_down[layer].astype(np.float32) / QUANT_SCALE
        
        # Clip activation to prevent overflow in matmul
        activation_clipped = np.clip(activation, -10.0, 10.0)
        
        gate = activation_clipped @ w_gate.T
        up = activation_clipped @ w_up.T
        # Clip intermediate results
        gate = np.clip(gate, -100.0, 100.0)
        up = np.clip(up, -100.0, 100.0)
        ffn = gate * up  # Simplified SiLU approximation
        ffn = np.clip(ffn, -100.0, 100.0)
        down = ffn @ w_down.T
        down = np.clip(down, -100.0, 100.0)
        
        activation = np.clip(activation + down * 0.1, -100.0, 100.0)  # Clip final result

    # -----------------------------------------------------------------
    # 8️⃣ Final linear head + argmax → next token(s)
    # -----------------------------------------------------------------
    print("\nSTEP 8: DECODE NEXT TOKENS")
    
    if weights.output is None:
        print("  ⚠ Output weights not available - using CPU reference only")
        next_token_ids = []
        generated_text = "[output weights not loaded]"
    else:
        # Final projection to vocabulary
        w_output = weights.output.astype(np.float32)
        logits = (activation @ w_output.T).astype(np.float32)
        
        # Greedy argmax for each token
        next_token_ids = []
        current_activation = activation.copy()
        
        for _ in range(NUM_TOKENS_TO_GENERATE):
            token_id = int(np.argmax(logits))
            next_token_ids.append(token_id)
            
            # Feed token back (simplified)
            if weights.token_embd is not None:
                current_activation = weights.token_embd[token_id].astype(np.float32)
                logits = (current_activation @ w_output.T).astype(np.float32)
        
        generated_text = tokenizer.decode(next_token_ids)
    
    print(f"\nGenerated tokens: {next_token_ids}")
    print(f"Generated text : '{generated_text}'")

    # -----------------------------------------------------------------
    # 9️⃣ Reference generation (CPU‑only) for sanity check
    # -----------------------------------------------------------------
    print("\nSTEP 9: CPU REFERENCE (for comparison)")
    ref_tokens, ref_text = cpu_reference_generate(PROMPT, NUM_TOKENS_TO_GENERATE)
    print(f"CPU reference tokens: {ref_tokens}")
    print(f"CPU reference text   : '{ref_text}'")

    # -----------------------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("-" * 80)
    print(f"Prompt                : '{PROMPT}'")
    print(f"Switch‑computed Q‑vectors (8 layers) : {len(q_vectors)}")
    print(f"Generated (switch‑CPU hybrid) tokens : {next_token_ids}")
    print(f"Generated text                     : '{generated_text}'")
    print(f"CPU‑only reference                 : '{ref_text}'")
    print("=" * 80)


if __name__ == "__main__":
    main()



""" Output:
    Note: blk.27.ffn_up.weight has type 12, attempting Q4_K decode
    Note: blk.27.ffn_down.weight has type 14, attempting Q4_K decode
  Loading output weights...
    Warning: output.weight not found
    Note: output.weight not found, using tied weights (token_embd transposed)
  ✓ Weights loaded successfully!

STEP 2: CLEANUP SWITCHES
  Cleaning up SW1...
  Cleaning up SW2...
  ✓ Switches cleaned

STEP 3: CONFIGURE SNAKE ARCHITECTURE

  Configuring SW1 (layers [0, 1, 2, 3], 64 neurons)...
    Config file: 1569 commands

  Configuring SW2 (layers [4, 5, 6, 7], 64 neurons)...
    Config file: 1555 commands
  ✓ Both switches configured for layers 0‑7

STEP 4: TOKENISE PROMPT
llama_context: n_ctx_per_seq (512) < n_ctx_train (40960) -- the full capacity of the model will not be utilized
  ✓ Using llama-cpp tokenizer
  Prompt tokens: [785, 220]
  Embedding shape: (1024,)

STEP 5: CREATE & SEND PACKETS (single burst)
  Source MAC: 7c:fe:90:9d:2a:f0
  Total packets to send: 78,629
  ✓ Burst sent in 123.6 ms

STEP 6: READ COUNTERS
  ✓ Counters read in 9065.7 ms

STEP 7: COMPLETE TRANSFORMER ON CPU
  Note: This is a simplified version - only Q projection runs on switch
        Other operations (K, V, O, FFN) run on CPU for now

  Verifying switch results...
    Layer 0: ✓ Match
    Layer 1: ✓ Match
    Layer 2: ✓ Match
    Layer 3: ✓ Match
    Layer 4: ✓ Match
    Layer 5: ✓ Match
    Layer 6: ✓ Match
    Layer 7: ✓ Match
  ✓ All switch results match expected values!

  Running full transformer pipeline on CPU (using switch Q as proof)...

STEP 8: DECODE NEXT TOKENS

Generated tokens: [73093, 73093, 73093]
Generated text : 'udenceudenceudence'

STEP 9: CPU REFERENCE (for comparison)
  Loading llama-cpp model for CPU reference...
llama_context: n_ctx_per_seq (512) < n_ctx_train (40960) -- the full capacity of the model will not be utilized
  Generating 3 tokens from: 'The '
  Input tokens: [785, 220]
  Generated tokens: [16, 24, 339]
  Generated text: '19th'
CPU reference tokens: [16, 24, 339]
CPU reference text   : 'The 19th'

================================================================================
EXPERIMENT SUMMARY
--------------------------------------------------------------------------------
Prompt                : 'The '
Switch‑computed Q‑vectors (8 layers) : 8
Generated (switch‑CPU hybrid) tokens : [73093, 73093, 73093]
Generated text                     : 'udenceudenceudence'
CPU‑only reference                 : 'The 19th'
================================================================================
"""


""" Note:
The switch-based inference is not producing accurate tokens compared to CPU. 
This is because we over-simplified the transformer pipeline to demonstrate the Snake Layer architecture.
"""