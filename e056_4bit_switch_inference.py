#!/usr/bin/env python3
"""
e056_4bit_switch_inference.py

4-BIT SWITCH INFERENCE WITH QWEN3-0.6B
======================================

GOAL: Run Qwen3-0.6B with 4-bit weights on switches for coherent output!

KEY INSIGHT:
  - e054 used ternary weights (1-bit) → garbage output
  - e055 showed Q4_K_M (4-bit) produces coherent text on CPU
  - This experiment: encode 4-bit weights via packet counts!

4-BIT WEIGHT ENCODING:
  For a weight W (range: -8 to +7 after quantization):
    - W > 0: Send W packets to POSITIVE counter
    - W < 0: Send |W| packets to NEGATIVE counter  
    - W = 0: Send 0 packets
  
  Final activation = pos_count - neg_count = weighted sum!

MODEL: Qwen3-0.6B (Q4_K_M quantization)
  - 28 layers, 1024 hidden dim, 3072 FFN dim
  - We use a slice: first N layers, reduced hidden dim

COMPARISON:
  - e054 (1-bit ternary): Fast but garbage output
  - e056 (4-bit int): Slower but should produce coherent output!

Author: Research Phase 001
Date: December 2025
"""

import time
import os
import sys
import re
import numpy as np
import gguf
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from e053 (MAC-encoded layers)
from e053_mac_encoded_layers import get_layer_neuron_mac

# Import from e045 (utilities only)
from e045_real_weights_inference import mac_str_to_bytes

# Import from e042 (utilities)
from e042_port_based_layers import (
    ssh_command, run_config_commands,
    craft_vlan_packet, send_packets, get_mac_address,
    SWITCH1_IP, SWITCH2_IP, SEND_IFACE,
)

# Model path - use 0.6B Q4 model!
MODEL_PATH = "./models/Qwen3-0.6B-Q4_K_M.gguf"

# Configuration - start small to prove concept
NUM_LAYERS = 2           # Start with 2 layers  
HIDDEN_DIM = 32          # Reduced for TCAM limits (32x32 = 1K weights per layer)
VOCAB_SIZE = 64          # Output vocabulary size
NUM_TOKENS = 5           # Tokens to generate
FILTER_NAME = "q4_inference"
TEST_VLAN = 100

# 4-bit weight scaling - need higher scale since weights are small floats
WEIGHT_SCALE = 30        # Scale factor: map weights to fuller 4-bit range [-8, +7]


def full_cleanup():
    """Clean both switches - AGGRESSIVE cleanup."""
    print("\n  Full cleanup...")
    for sw_ip in [SWITCH1_IP, SWITCH2_IP]:
        # First, create default storm-control profile to fix commit errors
        run_config_commands(sw_ip, [
            "set forwarding-options storm-control-profiles default all"
        ], debug=False)
        
        # Delete SPECIFIC filter by name first
        run_config_commands(sw_ip, [
            f"delete firewall family ethernet-switching filter {FILTER_NAME}"
        ], debug=False)
        
        # Then general cleanup
        cleanup_cmds = [
            "delete firewall family ethernet-switching filter",
            "delete interfaces et-0/0/96 unit 0 family ethernet-switching filter",
            "delete interfaces et-0/0/96 unit 0 family ethernet-switching",
            "delete interfaces et-0/0/100 unit 0 family ethernet-switching",
            "delete vlans",
        ]
        for cmd in cleanup_cmds:
            run_config_commands(sw_ip, [cmd], debug=False)
    
    time.sleep(1)
    print("  ✓ Cleanup complete")


def load_model() -> gguf.GGUFReader:
    """Load Qwen3-0.6B model."""
    print(f"\n  Loading model: {MODEL_PATH}")
    reader = gguf.GGUFReader(MODEL_PATH)
    print(f"    Loaded {len(reader.tensors)} tensors")
    return reader


def get_tensor_by_name(reader: gguf.GGUFReader, name_pattern: str):
    """Find tensor matching pattern."""
    for tensor in reader.tensors:
        if name_pattern in tensor.name:
            return tensor
    return None


def dequantize_tensor(tensor, max_rows: int = None) -> np.ndarray:
    """Dequantize tensor using gguf library."""
    data = tensor.data
    dequant = gguf.dequantize(data, tensor.tensor_type)
    
    if max_rows and len(dequant) > max_rows:
        dequant = dequant[:max_rows]
    
    return dequant


def weights_to_4bit(weights: np.ndarray, scale: float = WEIGHT_SCALE) -> np.ndarray:
    """
    Convert floating point weights to 4-bit signed integers.
    
    Maps weights to range [-8, +7] for 4-bit representation.
    """
    # Scale and clip to 4-bit range
    scaled = weights * scale
    clipped = np.clip(scaled, -8, 7)
    return np.round(clipped).astype(np.int8)


def extract_tokenizer(reader: gguf.GGUFReader) -> Tuple[List[str], Dict[str, int]]:
    """Extract tokenizer vocabulary from GGUF."""
    print("\n  Extracting tokenizer...")
    
    # For Qwen3, tokens 0-127 are ASCII characters
    # This is a simplified tokenizer for demonstration
    tokens = []
    
    # First 128: ASCII characters
    for i in range(128):
        if 32 <= i <= 126:  # Printable ASCII
            tokens.append(chr(i))
        else:
            tokens.append(f"<{i}>")
    
    # Common Qwen tokens (approximate positions)
    # Extend to cover more tokens
    while len(tokens) < 152000:
        tokens.append(f"<tok_{len(tokens)}>")
    
    # Add some known tokens
    tokens[791] = "The"
    tokens[1575] = "The"
    tokens[220] = " "
    tokens[264] = "the"
    tokens[374] = "and"
    tokens[304] = "is"
    
    token_to_id = {tok: i for i, tok in enumerate(tokens)}
    print(f"    Vocabulary size: {len(tokens)}")
    print(f"    Sample tokens: 0='{tokens[0]}', 32='{tokens[32]}', 65='{tokens[65]}'")
    
    return tokens, token_to_id


def extract_weights_4bit(reader: gguf.GGUFReader,
                         num_layers: int, 
                         hidden_dim: int,
                         vocab_size: int) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    """
    Extract weights as 4-bit integers.
    
    Returns: (embeddings_float, layer_weights_4bit, output_proj_4bit)
    """
    print(f"\n  Extracting 4-bit weights for {num_layers} layers...")
    
    # Embeddings (keep as float for lookup)
    print("    Extracting embeddings...")
    emb_tensor = get_tensor_by_name(reader, 'token_embd.weight')
    if emb_tensor:
        emb_dequant = dequantize_tensor(emb_tensor, max_rows=10000)
        embeddings = emb_dequant[:10000, :hidden_dim]
        print(f"      Shape: {embeddings.shape}")
    else:
        print("      WARNING: Using random embeddings")
        embeddings = np.random.randn(10000, hidden_dim).astype(np.float32) * 0.1
    
    # Layer weights as 4-bit
    layer_weights = []
    for layer_idx in range(num_layers):
        print(f"    Extracting layer {layer_idx}...")
        
        # Try FFN gate weights (main dense layer)
        layer_tensor = get_tensor_by_name(reader, f'blk.{layer_idx}.ffn_gate.weight')
        
        if layer_tensor is None:
            # Try ffn_up as fallback
            layer_tensor = get_tensor_by_name(reader, f'blk.{layer_idx}.ffn_up.weight')
        
        if layer_tensor:
            layer_dequant = dequantize_tensor(layer_tensor)
            # Shape: [ffn_dim, hidden_dim] = [3072, 1024]
            # Take slice: [hidden_dim, hidden_dim]
            weights = layer_dequant[:hidden_dim, :hidden_dim]
            print(f"      Float range: [{weights.min():.4f}, {weights.max():.4f}]")
            
            # Convert to 4-bit
            weights_4bit = weights_to_4bit(weights)
            unique, counts = np.unique(weights_4bit, return_counts=True)
            print(f"      4-bit range: [{weights_4bit.min()}, {weights_4bit.max()}]")
            print(f"      Distribution: {dict(zip(unique[:5], counts[:5]))}...")
        else:
            print(f"      WARNING: Using random 4-bit weights")
            weights_4bit = np.random.randint(-8, 8, size=(hidden_dim, hidden_dim), dtype=np.int8)
        
        layer_weights.append(weights_4bit)
    
    # Output projection as 4-bit
    print("    Extracting output projection...")
    out_tensor = get_tensor_by_name(reader, 'output.weight')
    if out_tensor:
        out_dequant = dequantize_tensor(out_tensor, max_rows=vocab_size)
        output_proj = out_dequant[:vocab_size, :hidden_dim]
        print(f"      Float range: [{output_proj.min():.4f}, {output_proj.max():.4f}]")
        output_proj_4bit = weights_to_4bit(output_proj)
        print(f"      4-bit range: [{output_proj_4bit.min()}, {output_proj_4bit.max()}]")
    else:
        print("      WARNING: Using random 4-bit output projection")
        output_proj_4bit = np.random.randint(-8, 8, size=(vocab_size, hidden_dim), dtype=np.int8)
    
    return embeddings, layer_weights, output_proj_4bit


def configure_4bit_filters(layer_weights: List[np.ndarray], output_proj: np.ndarray):
    """
    Configure switch filters for 4-bit weighted inference.
    
    We need one rule per (layer, output_neuron, polarity).
    The weight magnitude is encoded in how many packets we SEND, not in the rules.
    
    Rules:
      - Each output neuron has 2 MACs: one for positive, one for negative accumulation
      - Packets to pos MAC increment pos counter
      - Packets to neg MAC increment neg counter
      - Final activation = pos_count - neg_count
    """
    print("\n  Configuring 4-bit inference filters...")
    
    all_cmds = []
    rule_count = 0
    
    # Create base filter
    all_cmds.append(f"set firewall family ethernet-switching filter {FILTER_NAME}")
    
    # Layer weights - one rule per (layer, output_neuron, polarity)
    for layer_idx, weights in enumerate(layer_weights):
        print(f"    Layer {layer_idx}: {weights.shape[0]} output neurons")
        
        for out_idx in range(weights.shape[0]):
            # Positive MAC: layer L, neuron N*2
            mac_pos = get_layer_neuron_mac(layer_idx, out_idx * 2)
            # Negative MAC: layer L, neuron N*2+1  
            mac_neg = get_layer_neuron_mac(layer_idx, out_idx * 2 + 1)
            
            term_pos = f"L{layer_idx}_n{out_idx}_pos"
            term_neg = f"L{layer_idx}_n{out_idx}_neg"
            
            all_cmds.extend([
                f"set firewall family ethernet-switching filter {FILTER_NAME} "
                f"term {term_pos} from destination-mac-address {mac_pos}/48",
                f"set firewall family ethernet-switching filter {FILTER_NAME} "
                f"term {term_pos} then count l{layer_idx}_n{out_idx}_pos",
                f"set firewall family ethernet-switching filter {FILTER_NAME} "
                f"term {term_pos} then accept",
                
                f"set firewall family ethernet-switching filter {FILTER_NAME} "
                f"term {term_neg} from destination-mac-address {mac_neg}/48",
                f"set firewall family ethernet-switching filter {FILTER_NAME} "
                f"term {term_neg} then count l{layer_idx}_n{out_idx}_neg",
                f"set firewall family ethernet-switching filter {FILTER_NAME} "
                f"term {term_neg} then accept",
            ])
            rule_count += 2
    
    # Output projection - same pattern
    print(f"    Output: {output_proj.shape[0]} output neurons")
    for out_idx in range(output_proj.shape[0]):
        mac_pos = get_layer_neuron_mac(NUM_LAYERS, out_idx * 2)
        mac_neg = get_layer_neuron_mac(NUM_LAYERS, out_idx * 2 + 1)
        
        term_pos = f"out_n{out_idx}_pos"
        term_neg = f"out_n{out_idx}_neg"
        
        all_cmds.extend([
            f"set firewall family ethernet-switching filter {FILTER_NAME} "
            f"term {term_pos} from destination-mac-address {mac_pos}/48",
            f"set firewall family ethernet-switching filter {FILTER_NAME} "
            f"term {term_pos} then count out_n{out_idx}_pos",
            f"set firewall family ethernet-switching filter {FILTER_NAME} "
            f"term {term_pos} then accept",
            
            f"set firewall family ethernet-switching filter {FILTER_NAME} "
            f"term {term_neg} from destination-mac-address {mac_neg}/48",
            f"set firewall family ethernet-switching filter {FILTER_NAME} "
            f"term {term_neg} then count out_n{out_idx}_neg",
            f"set firewall family ethernet-switching filter {FILTER_NAME} "
            f"term {term_neg} then accept",
        ])
        rule_count += 2
    
    # Default accept term
    all_cmds.extend([
        f"set firewall family ethernet-switching filter {FILTER_NAME} term default then count default_pkts",
        f"set firewall family ethernet-switching filter {FILTER_NAME} term default then accept",
    ])
    
    print(f"    Total rules: {rule_count}")
    
    # Apply to switch
    print("    Applying to switch 1...")
    batch_size = 50
    for i in range(0, len(all_cmds), batch_size):
        batch = all_cmds[i:i+batch_size]
        success = run_config_commands(SWITCH1_IP, batch, debug=False)
        if not success:
            print(f"    ✗ Failed at batch {i // batch_size}")
            return False
    
    # Apply filter to interface
    apply_cmds = [
        f"set vlans test_vlan vlan-id {TEST_VLAN}",
        f"set interfaces et-0/0/96 unit 0 family ethernet-switching vlan members test_vlan",
        f"set interfaces et-0/0/96 unit 0 family ethernet-switching filter input {FILTER_NAME}",
    ]
    run_config_commands(SWITCH1_IP, apply_cmds, debug=False)
    
    print("  ✓ Filter configuration complete")
    return True


def run_4bit_layer_inference(layer_idx: int, 
                              hidden_state: np.ndarray,
                              weights: np.ndarray,
                              src_mac: str) -> List:
    """
    Create packets for 4-bit weighted inference.
    
    For each input neuron i with value h[i]:
      For each output neuron o:
        weight = weights[o, i]
        if weight > 0: send (|weight| * |h[i]|) packets to positive counter
        if weight < 0: send (|weight| * |h[i]|) packets to negative counter
    
    Returns list of packets.
    """
    packets = []
    
    # Binarize: treat any non-zero input as active
    # Use absolute value > small threshold
    threshold = 0.01 if np.any(hidden_state != 0) else 0
    hidden_binary = (np.abs(hidden_state) > threshold).astype(int)
    
    for out_idx in range(weights.shape[0]):
        pos_packets = 0
        neg_packets = 0
        
        for in_idx in range(weights.shape[1]):
            if hidden_binary[in_idx] == 0:
                continue  # Input is inactive
            
            w = weights[out_idx, in_idx]
            if w > 0:
                pos_packets += abs(w)
            elif w < 0:
                neg_packets += abs(w)
        
        # Create packets for this output neuron
        mac_pos = get_layer_neuron_mac(layer_idx, out_idx * 2)
        mac_neg = get_layer_neuron_mac(layer_idx, out_idx * 2 + 1)
        
        for _ in range(pos_packets):
            pkt = craft_vlan_packet(mac_str_to_bytes(mac_pos), 
                                    mac_str_to_bytes(src_mac),
                                    TEST_VLAN)
            packets.append(pkt)
        
        for _ in range(neg_packets):
            pkt = craft_vlan_packet(mac_str_to_bytes(mac_neg),
                                    mac_str_to_bytes(src_mac), 
                                    TEST_VLAN)
            packets.append(pkt)
    
    return packets


def read_layer_counters(layer_idx: int, num_neurons: int, debug: bool = False) -> np.ndarray:
    """Read counters and compute signed activations."""
    success, stdout, _ = ssh_command(
        SWITCH1_IP,
        f"cli -c 'show firewall filter {FILTER_NAME}'"
    )
    
    activations = np.zeros(num_neurons, dtype=np.int32)
    
    if not success or not stdout:
        return activations
    
    # Debug: show sample counters
    if debug:
        lines = [l for l in stdout.split('\n') if f'l{layer_idx}_n' in l][:6]
        if lines:
            print(f"\n      [DEBUG] Sample counters:\n        " + "\n        ".join(lines))
    
    for neuron in range(num_neurons):
        # Use regex with word boundary to match correctly
        pos_pattern = rf'l{layer_idx}_n{neuron}_pos\s+\d+\s+(\d+)'
        pos_match = re.search(pos_pattern, stdout)
        
        neg_pattern = rf'l{layer_idx}_n{neuron}_neg\s+\d+\s+(\d+)'
        neg_match = re.search(neg_pattern, stdout)
        
        if pos_match or neg_match:
            pos_count = int(pos_match.group(1)) if pos_match else 0
            neg_count = int(neg_match.group(1)) if neg_match else 0
            activations[neuron] = pos_count - neg_count
    
    return activations


def read_output_counters(vocab_size: int, debug: bool = False) -> np.ndarray:
    """Read output layer counters."""
    success, stdout, _ = ssh_command(
        SWITCH1_IP,
        f"cli -c 'show firewall filter {FILTER_NAME}'"
    )
    
    logits = np.zeros(vocab_size, dtype=np.int32)
    
    if not success or not stdout:
        return logits
    
    # Debug: show sample counters
    if debug:
        lines = [l for l in stdout.split('\n') if 'out_n' in l][:6]
        if lines:
            print(f"\n      [DEBUG] Output counters:\n        " + "\n        ".join(lines))
    
    for neuron in range(vocab_size):
        # Use regex with word boundary
        pos_pattern = rf'out_n{neuron}_pos\s+\d+\s+(\d+)'
        pos_match = re.search(pos_pattern, stdout)
        
        neg_pattern = rf'out_n{neuron}_neg\s+\d+\s+(\d+)'
        neg_match = re.search(neg_pattern, stdout)
        
        if pos_match or neg_match:
            pos_count = int(pos_match.group(1)) if pos_match else 0
            neg_count = int(neg_match.group(1)) if neg_match else 0
            logits[neuron] = pos_count - neg_count
    
    return logits


def cpu_4bit_inference(hidden: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """CPU reference for 4-bit inference."""
    # Binarize: treat any non-zero input as active (same as switch)
    threshold = 0.01 if np.any(hidden != 0) else 0
    hidden_binary = (np.abs(hidden) > threshold).astype(np.float32)
    
    # Matrix multiply with 4-bit weights
    result = weights.astype(np.float32) @ hidden_binary
    return result


def run_4bit_inference():
    """Run full 4-bit inference on switches."""
    print("="*80)
    print("E056: 4-BIT SWITCH INFERENCE (Qwen3-0.6B)")
    print("="*80)
    print("""
  Testing 4-bit weight encoding for coherent output!
  
  Key difference from e054:
    - e054: 1-bit ternary weights → garbage
    - e056: 4-bit integer weights → should be coherent!
  
  Weight encoding: Packet count = |weight magnitude|
""")
    
    # Cleanup
    full_cleanup()
    
    # Load model
    reader = load_model()
    tokens_list, token_to_id = extract_tokenizer(reader)
    
    # Extract 4-bit weights
    embeddings, layer_weights, output_proj = extract_weights_4bit(
        reader, NUM_LAYERS, HIDDEN_DIM, VOCAB_SIZE
    )
    
    # Configure filters
    if not configure_4bit_filters(layer_weights, output_proj):
        print("  ✗ Filter configuration failed!")
        return
    
    # Get source MAC
    src_mac = get_mac_address(SEND_IFACE)
    print(f"\n  Source MAC: {src_mac}")
    
    # Token generation
    print("\n" + "="*60)
    print("Generating Tokens with 4-bit Weights")
    print("="*60)
    
    prompt = "The "
    prompt_token = token_to_id.get(prompt.strip(), token_to_id.get("The", 791))
    generated_ids = [prompt_token]
    
    print(f"\n  Prompt: '{prompt}'")
    print(f"  Prompt token: {prompt_token}")
    
    for step in range(NUM_TOKENS):
        print(f"\n  Step {step + 1}/{NUM_TOKENS}:")
        t0 = time.time()
        
        # Clear counters before each token generation
        ssh_command(SWITCH1_IP, f"cli -c 'clear firewall filter {FILTER_NAME}'")
        time.sleep(0.1)  # Wait for clear
        
        # Get embedding
        last_token = generated_ids[-1]
        if last_token >= len(embeddings):
            last_token = last_token % len(embeddings)
        hidden = embeddings[last_token, :HIDDEN_DIM].copy()
        
        # Forward through layers
        for layer_idx in range(NUM_LAYERS):
            print(f"    Layer {layer_idx}...", end=" ", flush=True)
            
            # Debug: show hidden state
            active_inputs = np.sum(hidden != 0)
            
            # Create and send packets
            packets = run_4bit_layer_inference(layer_idx, hidden, 
                                                layer_weights[layer_idx], src_mac)
            
            # Debug: packet count
            pkt_count = len(packets) if packets else 0
            
            if packets:
                send_packets(SEND_IFACE, packets)
                time.sleep(0.1)  # Wait for counters to update
            
            # Read counters (debug on first step)
            debug = (step == 0 and layer_idx == 0)
            activations = read_layer_counters(layer_idx, HIDDEN_DIM, debug=debug)
            
            # CPU reference
            cpu_ref = cpu_4bit_inference(hidden, layer_weights[layer_idx])
            
            print(f"done (pkts={pkt_count}, switch={activations.sum():.0f}, cpu={cpu_ref.sum():.0f}, active_in={active_inputs})")
            
            # Update hidden state - use CPU for now to debug switch
            hidden = cpu_ref.copy()
            
            # Binarize for next layer - use sign, not magnitude
            if np.any(hidden != 0):
                # Keep top 50% of activations
                threshold = np.percentile(np.abs(hidden), 50)
                hidden = np.where(np.abs(hidden) > threshold, np.sign(hidden), 0)
        
        # Output projection
        print("    Output projection...", end=" ", flush=True)
        
        out_packets = run_4bit_layer_inference(NUM_LAYERS, hidden, 
                                                output_proj, src_mac)
        if out_packets:
            send_packets(SEND_IFACE, out_packets)
        
        logits = read_output_counters(VOCAB_SIZE, debug=(step == 0))
        cpu_logits = cpu_4bit_inference(hidden, output_proj)
        
        print(f"done")
        
        # Get next token
        next_token = int(np.argmax(logits))
        cpu_token = int(np.argmax(cpu_logits))
        
        generated_ids.append(next_token)
        
        # Get token string
        if next_token < len(tokens_list):
            tok_str = tokens_list[next_token]
        else:
            tok_str = f"<{next_token}>"
        
        match = "✓" if next_token == cpu_token else "✗"
        print(f"    Token: {next_token} = '{tok_str}' (CPU: {cpu_token}) {match}")
        print(f"    Logits: switch [{logits[:5]}...] cpu [{cpu_logits[:5]}...]")
        print(f"    Time: {time.time() - t0:.2f}s")
    
    # Final output
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    generated_text = prompt
    for tid in generated_ids[1:]:
        if tid < len(tokens_list):
            tok = tokens_list[tid]
            if tok:
                generated_text += tok
    
    print(f"\n  Prompt: '{prompt}'")
    print(f"  Generated: '{generated_text}'")
    print(f"  Token IDs: {generated_ids}")
    
    # Cleanup
    full_cleanup()


if __name__ == '__main__':
    run_4bit_inference()


""" Output:
sudo python3 e056_4bit_switch_inference.py
================================================================================
E056: 4-BIT SWITCH INFERENCE (Qwen3-0.6B)
================================================================================

  Testing 4-bit weight encoding for coherent output!
  
  Key difference from e054:
    - e054: 1-bit ternary weights → garbage
    - e056: 4-bit integer weights → should be coherent!
  
  Weight encoding: Packet count = |weight magnitude|


  Full cleanup...
  ✓ Cleanup complete

  Loading model: ./models/Qwen3-0.6B-Q4_K_M.gguf
    Loaded 310 tensors

  Extracting tokenizer...
    Vocabulary size: 152000
    Sample tokens: 0='<0>', 32=' ', 65='A'

  Extracting 4-bit weights for 2 layers...
    Extracting embeddings...
      Shape: (10000, 32)
    Extracting layer 0...
      Float range: [-0.2973, 0.2943]
      4-bit range: [-8, 7]
      Distribution: {-8: 2, -7: 1, -4: 4, -3: 14, -2: 78}...
    Extracting layer 1...
      Float range: [-0.1912, 0.1433]
      4-bit range: [-6, 4]
      Distribution: {-6: 1, -5: 1, -4: 4, -3: 14, -2: 43}...
    Extracting output projection...
      Float range: [-0.1363, 0.1446]
      4-bit range: [-4, 4]

  Configuring 4-bit inference filters...
    Layer 0: 32 output neurons
    Layer 1: 32 output neurons
    Output: 64 output neurons
    Total rules: 256
    Applying to switch 1...
  ✓ Filter configuration complete

  Source MAC: 7c:fe:90:9d:2a:f0

============================================================
Generating Tokens with 4-bit Weights
============================================================

  Prompt: 'The '
  Prompt token: 1575

  Step 1/5:
    Layer 0... 
      [DEBUG] Sample counters:
        l0_n0_neg                                            1216                   19
        l0_n0_pos                                            1024                   16
        l0_n10_neg                                            576                    9
        l0_n10_pos                                            384                    6
        l0_n11_neg                                            704                   11
        l0_n11_pos                                            768                   12
done (pkts=701, switch=103, cpu=103, active_in=32)
    Layer 1... done (pkts=416, switch=36, cpu=36, active_in=16)
    Output projection... 
      [DEBUG] Output counters:
        out_n0_neg                                            128                    2
        out_n0_pos                                             64                    1
        out_n10_neg                                           128                    2
        out_n10_pos                                            64                    1
        out_n11_neg                                           128                    2
        out_n11_pos                                           192                    3
done
    Token: 9 = '<9>' (CPU: 9) ✓
    Logits: switch [[-1  0  2  2  2]...] cpu [[-1.  0.  2.  2.  2.]...]
    Time: 5.57s

  Step 2/5:
    Layer 0... done (pkts=704, switch=22, cpu=22, active_in=32)
    Layer 1... done (pkts=345, switch=39, cpu=39, active_in=12)
    Output projection... done
    Token: 32 = ' ' (CPU: 32) ✓
    Logits: switch [[-1  1  4  2 -1]...] cpu [[-1.  1.  4.  2. -1.]...]
    Time: 5.49s

  Step 3/5:
    Layer 0... done (pkts=768, switch=104, cpu=104, active_in=30)
    Layer 1... done (pkts=412, switch=100, cpu=100, active_in=14)
    Output projection... done
    Token: 41 = ')' (CPU: 41) ✓
    Logits: switch [[-2  1  3  2  2]...] cpu [[-2.  1.  3.  2.  2.]...]
    Time: 5.40s

  Step 4/5:
    Layer 0... done (pkts=697, switch=91, cpu=91, active_in=31)
    Layer 1... done (pkts=442, switch=96, cpu=96, active_in=16)
    Output projection... done
    Token: 9 = '<9>' (CPU: 9) ✓
    Logits: switch [[-1  1  4  2  2]...] cpu [[-1.  1.  4.  2.  2.]...]
    Time: 5.53s

  Step 5/5:
    Layer 0... done (pkts=704, switch=22, cpu=22, active_in=32)
    Layer 1... done (pkts=345, switch=39, cpu=39, active_in=12)
    Output projection... done
    Token: 32 = ' ' (CPU: 32) ✓
    Logits: switch [[-1  1  4  2 -1]...] cpu [[-1.  1.  4.  2. -1.]...]
    Time: 5.57s

================================================================================
RESULTS
================================================================================

  Prompt: 'The '
  Generated: 'The <9> )<9> '
  Token IDs: [1575, 9, 32, 41, 9, 32]

  Full cleanup...
  ✓ Cleanup complete
"""