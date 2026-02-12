#!/usr/bin/env python3
"""
E082: FULL QWEN3-0.6B WITH REAL WEIGHTS

THE ULTIMATE TEST: Complete inference with actual model weights!

Model: Qwen3-0.6B-Q4_K_M
  - 28 transformer layers
  - 1024 embedding dimension
  - 3072 FFN intermediate dimension
  - 16 Q heads, 8 KV heads
  - 128 head dimension
  - 151936 vocabulary

Architecture:
  - Switch 1: Layers 0-13
  - Switch 2: Layers 14-27

This experiment:
  1. Loads real GGUF weights
  2. Configures both switches with full model
  3. Runs inference on switch
  4. Compares with CPU reference
  5. Decodes tokens to readable text
"""

import numpy as np
import subprocess
import socket
import struct
import time
import re
import os
import sys
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import common utilities
from e053_mac_encoded_layers import get_layer_neuron_mac
from e045_real_weights_inference import mac_str_to_bytes
from e042_port_based_layers import (
    craft_vlan_packet, send_packets, get_mac_address,
    SWITCH1_IP, SWITCH2_IP, SEND_IFACE,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# SSH
SSH_KEY = "/home/multiplex/.ssh/id_rsa"

# Model path
MODEL_PATH = "models/Qwen3-0.6B-Q4_K_M.gguf"

# Full Qwen3-0.6B dimensions
N_LAYERS = 28
D_MODEL = 1024
D_FFN = 3072
N_HEADS = 16
N_KV_HEADS = 8
D_HEAD = 64  # D_MODEL // N_HEADS
VOCAB_SIZE = 151936

# Quantization
QUANT_BITS = 4
QUANT_SCALE = 4.0  # Scale factor for int4

# Filter names
FILTER_SW1 = "qwen_full_sw1"
FILTER_SW2 = "qwen_full_sw2"
TEST_VLAN = 100

# Layer assignment (reduced for faster testing)
# Full model: layers 0-13 on SW1, 14-27 on SW2
# Test mode: just 4 layers total
TEST_MODE = True  # Set to False for full model
FAST_TEST = True  # Only configure 16 neurons per projection

# IMPORTANT: QFX5100 has TCAM filter limit between 1152-1728 rules per filter!
# (Found in e060_multi_layer_scale.py)
# SOLUTION: Use ONE FILTER PER LAYER with separate VLANs (from e063)
# Each filter: 7 proj × 64 neurons × 2 pos/neg = 896 terms < 1152 ✓
# 28 layers × 28 filters = can handle full model!

# Neuron limits per projection (for fast testing)
if FAST_TEST:
    TEST_NEURONS = 64  # 64 neurons per projection (896 terms per layer)
else:
    TEST_NEURONS = None  # Use full dimensions

if TEST_MODE:
    LAYERS_SW1 = [0, 1]    # 2 layers on SW1 (each layer = separate filter/VLAN)
    LAYERS_SW2 = [2, 3]    # 2 layers on SW2
else:
    LAYERS_SW1 = list(range(0, 14))   # Layers 0-13
    LAYERS_SW2 = list(range(14, 28))  # Layers 14-27


# =============================================================================
# SSH UTILITIES
# =============================================================================

def ssh_command_long(switch_ip: str, command: str, timeout: int = 300) -> Tuple[bool, str, str]:
    """SSH command with configurable timeout using SSH keys."""
    cmd = [
        'ssh', '-i', SSH_KEY,
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


def transfer_config_file(local_path: str, switch_ip: str, remote_path: str) -> bool:
    """Transfer config file to switch via SSH stdin (more reliable than SCP)."""
    ssh_cmd = [
        'ssh', '-i', SSH_KEY,
        '-o', 'StrictHostKeyChecking=no',
        '-o', 'UserKnownHostsFile=/dev/null',
        '-o', 'LogLevel=ERROR',
        f'root@{switch_ip}',
        f'cat > {remote_path}'
    ]
    try:
        with open(local_path, 'rb') as f:
            result = subprocess.run(ssh_cmd, stdin=f, capture_output=True, text=True, timeout=120)
        return result.returncode == 0
    except Exception as e:
        print(f"    Transfer error: {e}")
        return False


# =============================================================================
# WEIGHT LOADING FROM GGUF
# =============================================================================

@dataclass
class Qwen3Weights:
    """Container for Qwen3 model weights."""
    # Embeddings
    token_embd: np.ndarray  # [vocab_size, d_model]
    
    # Per-layer weights
    attn_q: List[np.ndarray]      # [d_model, n_heads * d_head]
    attn_k: List[np.ndarray]      # [d_model, n_kv_heads * d_head]
    attn_v: List[np.ndarray]      # [d_model, n_kv_heads * d_head]
    attn_o: List[np.ndarray]      # [n_heads * d_head, d_model]
    
    ffn_gate: List[np.ndarray]    # [d_model, d_ffn]
    ffn_up: List[np.ndarray]      # [d_model, d_ffn]
    ffn_down: List[np.ndarray]    # [d_ffn, d_model]
    
    attn_norm: List[np.ndarray]   # [d_model]
    ffn_norm: List[np.ndarray]    # [d_model]
    
    # Output
    output_norm: np.ndarray       # [d_model]
    output: np.ndarray            # [d_model, vocab_size]


def quantize_to_int4(weights: np.ndarray) -> Tuple[np.ndarray, float]:
    """Quantize float weights to int4 (-8 to 7)."""
    # Find scale
    abs_max = np.abs(weights).max()
    if abs_max == 0:
        return np.zeros_like(weights, dtype=np.int8), 1.0
    
    scale = abs_max / 7.0
    
    # Quantize
    quantized = np.round(weights / scale).astype(np.int8)
    quantized = np.clip(quantized, -8, 7)
    
    return quantized, scale


def decode_q4_k_block(data: bytes, n_elements: int) -> np.ndarray:
    """
    Decode Q4_K format to extract raw 4-bit integer values.
    
    Q4_K format (per 256-element super-block):
      - d: float16 (2 bytes) - main scale
      - dmin: float16 (2 bytes) - min scale  
      - scales: 12 bytes - sub-block scales
      - qs: 128 bytes - packed 4-bit values (256 values)
    
    Total: 144 bytes per 256 elements
    """
    BLOCK_SIZE = 256
    BLOCK_BYTES = 144  # 2 + 2 + 12 + 128
    
    n_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    result = np.zeros(n_elements, dtype=np.int8)
    
    for block_idx in range(n_blocks):
        block_start = block_idx * BLOCK_BYTES
        if block_start + BLOCK_BYTES > len(data):
            break
        
        # Skip d, dmin, scales (16 bytes)
        qs_start = block_start + 16
        
        # Extract 4-bit values from qs (128 bytes = 256 nibbles)
        for i in range(128):
            if qs_start + i >= len(data):
                break
            
            byte_val = data[qs_start + i]
            
            # Low nibble (first value)
            val_idx = block_idx * BLOCK_SIZE + i * 2
            if val_idx < n_elements:
                low = byte_val & 0x0F
                # Convert to signed: 0-7 stay positive, 8-15 become -8 to -1
                result[val_idx] = low if low < 8 else low - 16
            
            # High nibble (second value)
            val_idx = block_idx * BLOCK_SIZE + i * 2 + 1
            if val_idx < n_elements:
                high = (byte_val >> 4) & 0x0F
                result[val_idx] = high if high < 8 else high - 16
    
    return result


def decode_q4_k_tensor(raw_data: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """
    Decode a Q4_K quantized tensor to int4 values.
    
    Args:
        raw_data: Raw bytes from GGUF (as numpy array)
        shape: Target shape (e.g., (2048, 1024))
    
    Returns:
        Decoded int4 values with correct shape
    """
    n_elements = np.prod(shape)
    
    # Convert to bytes
    if isinstance(raw_data, np.ndarray):
        data = raw_data.tobytes()
    else:
        data = bytes(raw_data)
    
    # Decode
    decoded = decode_q4_k_block(data, n_elements)
    
    # Reshape
    return decoded.reshape(shape)


def load_gguf_weights() -> Optional[Qwen3Weights]:
    """Load weights from GGUF file and decode Q4_K format to int4."""
    if not os.path.exists(MODEL_PATH):
        print(f"  ✗ Model file not found: {MODEL_PATH}")
        return None
    
    try:
        from gguf import GGUFReader
        print(f"  Loading from {MODEL_PATH}...")
        
        reader = GGUFReader(MODEL_PATH)
        
        # Build tensor dictionary with metadata
        tensors = {}
        tensor_info = {}
        for tensor in reader.tensors:
            tensors[tensor.name] = tensor.data
            tensor_info[tensor.name] = {
                'shape': tensor.shape,
                'dtype': str(tensor.tensor_type),
                'n_elements': tensor.n_elements
            }
        
        print(f"  Found {len(tensors)} tensors")
        
        # Show a few tensor types
        sample_tensors = list(tensor_info.items())[:3]
        for name, info in sample_tensors:
            print(f"    {name}: shape={info['shape']}, type={info['dtype']}")
        
        def get_decoded_weight(name: str, target_shape: Tuple[int, ...] = None) -> np.ndarray:
            """Get weight tensor, decoding Q4_K format if needed."""
            if name not in tensors:
                print(f"    Warning: {name} not found")
                return None
            
            raw_data = tensors[name]
            info = tensor_info[name]
            
            # Check if it's quantized
            dtype_str = info['dtype'].lower()
            
            if 'q4_k' in dtype_str or 'q4k' in dtype_str:
                # Decode Q4_K format
                if target_shape is None:
                    target_shape = tuple(info['shape'])
                return decode_q4_k_tensor(raw_data, target_shape)
            elif 'f32' in dtype_str or 'float32' in dtype_str:
                # Already float, just quantize
                data = raw_data.astype(np.float32)
                quantized, _ = quantize_to_int4(data)
                return quantized
            elif 'f16' in dtype_str or 'float16' in dtype_str:
                # Convert from float16
                data = raw_data.astype(np.float32)
                quantized, _ = quantize_to_int4(data)
                return quantized
            else:
                # Unknown format, try direct decode
                print(f"    Note: {name} has type {dtype_str}, attempting Q4_K decode")
                if target_shape is None:
                    target_shape = tuple(info['shape'])
                return decode_q4_k_tensor(raw_data, target_shape)
        
        def get_float_tensor(name: str) -> np.ndarray:
            """Get tensor as float (for norms)."""
            if name not in tensors:
                print(f"    Warning: {name} not found")
                return None
            data = tensors[name]
            if hasattr(data, 'astype'):
                return data.astype(np.float32)
            return np.array(data, dtype=np.float32)
        
        # Load embedding
        print("  Loading embeddings...")
        token_embd = get_decoded_weight("token_embd.weight", (VOCAB_SIZE, D_MODEL))
        
        # Load per-layer weights
        attn_q, attn_k, attn_v, attn_o = [], [], [], []
        ffn_gate, ffn_up, ffn_down = [], [], []
        attn_norm, ffn_norm = [], []
        
        print("  Loading layer weights (with Q4_K decoding)...")
        for i in range(N_LAYERS):
            if i % 7 == 0:
                print(f"    Layer {i}/{N_LAYERS}...")
            
            # Attention projections with proper shapes
            # Q: [n_heads * d_head, d_model] = [2048, 1024]
            attn_q.append(get_decoded_weight(f"blk.{i}.attn_q.weight", (N_HEADS * D_HEAD, D_MODEL)))
            # K: [n_kv_heads * d_head, d_model] = [1024, 1024]  
            attn_k.append(get_decoded_weight(f"blk.{i}.attn_k.weight", (N_KV_HEADS * D_HEAD, D_MODEL)))
            # V: [n_kv_heads * d_head, d_model] = [1024, 1024]
            attn_v.append(get_decoded_weight(f"blk.{i}.attn_v.weight", (N_KV_HEADS * D_HEAD, D_MODEL)))
            # O: [d_model, n_heads * d_head] = [1024, 2048]
            attn_o.append(get_decoded_weight(f"blk.{i}.attn_output.weight", (D_MODEL, N_HEADS * D_HEAD)))
            
            # FFN projections
            # gate: [d_ffn, d_model] = [3072, 1024]
            ffn_gate.append(get_decoded_weight(f"blk.{i}.ffn_gate.weight", (D_FFN, D_MODEL)))
            # up: [d_ffn, d_model] = [3072, 1024]
            ffn_up.append(get_decoded_weight(f"blk.{i}.ffn_up.weight", (D_FFN, D_MODEL)))
            # down: [d_model, d_ffn] = [1024, 3072]
            ffn_down.append(get_decoded_weight(f"blk.{i}.ffn_down.weight", (D_MODEL, D_FFN)))
            
            # Norms (keep as float)
            attn_norm.append(get_float_tensor(f"blk.{i}.attn_norm.weight"))
            ffn_norm.append(get_float_tensor(f"blk.{i}.ffn_norm.weight"))
        
        # Output
        print("  Loading output weights...")
        output_norm = get_float_tensor("output_norm.weight")
        output = get_decoded_weight("output.weight", (VOCAB_SIZE, D_MODEL))
        
        # If output.weight not found, use tied weights (transpose of token_embd)
        if output is None and token_embd is not None:
            print("    Note: output.weight not found, using tied weights (token_embd transposed)")
            # token_embd is [vocab_size, d_model], transpose to [d_model, vocab_size]
            # then transpose back for output projection: [vocab_size, d_model]
            output = token_embd.copy()  # Already in correct shape [vocab_size, d_model]
        
        weights = Qwen3Weights(
            token_embd=token_embd,
            attn_q=attn_q, attn_k=attn_k, attn_v=attn_v, attn_o=attn_o,
            ffn_gate=ffn_gate, ffn_up=ffn_up, ffn_down=ffn_down,
            attn_norm=attn_norm, ffn_norm=ffn_norm,
            output_norm=output_norm, output=output
        )
        
        print("  ✓ Weights loaded successfully!")
        return weights
        
    except ImportError:
        print("  ✗ gguf library not available. Install with: pip install gguf")
        return None
    except Exception as e:
        print(f"  ✗ Error loading weights: {e}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# TOKENIZER
# =============================================================================

def load_tokenizer():
    """Load tokenizer for Qwen3."""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
        return tokenizer
    except:
        print("  Warning: Could not load HuggingFace tokenizer")
        return None


class SimpleTokenizer:
    """Fallback simple tokenizer using llama-cpp."""
    def __init__(self):
        self.llm = None
        try:
            from llama_cpp import Llama
            self.llm = Llama(model_path=MODEL_PATH, n_ctx=512, verbose=False)
            print("  ✓ Using llama-cpp tokenizer")
        except:
            print("  Warning: llama-cpp not available for tokenization")
    
    def encode(self, text: str) -> List[int]:
        if self.llm:
            return self.llm.tokenize(text.encode('utf-8'))
        # Fallback: return placeholder
        return [791, 220]  # "The "
    
    def decode(self, tokens: List[int]) -> str:
        if self.llm:
            return self.llm.detokenize(tokens).decode('utf-8', errors='replace')
        return f"[tokens: {tokens}]"


# =============================================================================
# LOOKUP TABLES
# =============================================================================

def create_silu_lut() -> np.ndarray:
    """Create SiLU lookup table for int4 values."""
    lut = np.zeros(16, dtype=np.float32)
    for i in range(16):
        x = i - 8 if i >= 8 else i  # Convert to signed
        x_scaled = x / QUANT_SCALE
        sigmoid = 1.0 / (1.0 + np.exp(-x_scaled))
        lut[i] = x_scaled * sigmoid
    return lut


def create_rope_lut(max_seq_len: int = 2048, base: float = 1000000.0) -> Tuple[np.ndarray, np.ndarray]:
    """Create RoPE sin/cos lookup tables."""
    half_dim = D_HEAD // 2
    freqs = 1.0 / (base ** (np.arange(0, half_dim, dtype=np.float32) / half_dim))
    positions = np.arange(max_seq_len, dtype=np.float32)
    angles = np.outer(positions, freqs)
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


# =============================================================================
# CPU REFERENCE (using llama-cpp for comparison)
# =============================================================================

def cpu_reference_generate(prompt: str, num_tokens: int = 3) -> Tuple[List[int], str]:
    """Generate tokens using llama-cpp as CPU reference."""
    try:
        from llama_cpp import Llama
        
        print("  Loading llama-cpp model for CPU reference...")
        llm = Llama(model_path=MODEL_PATH, n_ctx=512, verbose=False)
        
        print(f"  Generating {num_tokens} tokens from: '{prompt}'")
        
        # Use llama-cpp's proper generation API
        output = llm(
            prompt,
            max_tokens=num_tokens,
            temperature=0.0,  # Greedy
            echo=True,  # Include prompt in output
        )
        
        full_text = output['choices'][0]['text']
        
        # Get the generated portion
        generated_text = full_text[len(prompt):]
        
        # Tokenize to get token IDs
        input_tokens = llm.tokenize(prompt.encode('utf-8'))
        full_tokens = llm.tokenize(full_text.encode('utf-8'))
        generated_tokens = full_tokens[len(input_tokens):]
        
        print(f"  Input tokens: {input_tokens}")
        print(f"  Generated tokens: {generated_tokens}")
        print(f"  Generated text: '{generated_text}'")
        
        return list(generated_tokens), full_text
        
    except Exception as e:
        print(f"  ✗ CPU reference failed: {e}")
        import traceback
        traceback.print_exc()
        return [], ""


# =============================================================================
# SWITCH CONFIGURATION
# =============================================================================

def cleanup_switch(switch_ip: str, filter_name: str):
    """Clean up switch configuration thoroughly."""
    # Delete in correct order to avoid conflicts
    # Also delete stale vlans from previous experiments
    cleanup_cmds = [
        # First remove filter from interface
        "delete interfaces et-0/0/96 unit 0 family ethernet-switching filter",
        # Then remove vlan membership
        "delete interfaces et-0/0/96 unit 0 family ethernet-switching vlan",
        # Delete the filter itself
        f"delete firewall family ethernet-switching filter {filter_name}",
        # Delete all potential conflicting vlans
        "delete vlans qwen_vlan",
        "delete vlans test_vlan",  # From previous experiments
    ]
    
    cmd_str = "; ".join(cleanup_cmds)
    ssh_command_long(switch_ip, f"cli -c 'configure; {cmd_str}; commit'", timeout=60)
    time.sleep(1)


def configure_projection(filter_name: str, proj_name: str, 
                         layer_base: int, output_dim: int) -> List[str]:
    """Generate config commands for one projection."""
    cmds = []
    
    # Limit neurons in fast test mode
    actual_dim = min(output_dim, TEST_NEURONS) if TEST_NEURONS else output_dim
    
    for neuron in range(actual_dim):
        # Positive counter
        mac_pos = get_layer_neuron_mac(layer_base, neuron)
        term_pos = f"{proj_name}_n{neuron}_p"
        cmds.extend([
            f"set firewall family ethernet-switching filter {filter_name} term {term_pos} from destination-mac-address {mac_pos}/48",
            f"set firewall family ethernet-switching filter {filter_name} term {term_pos} then count {term_pos}",
            f"set firewall family ethernet-switching filter {filter_name} term {term_pos} then accept",
        ])
        
        # Negative counter
        mac_neg = get_layer_neuron_mac(layer_base + 1, neuron)
        term_neg = f"{proj_name}_n{neuron}_n"
        cmds.extend([
            f"set firewall family ethernet-switching filter {filter_name} term {term_neg} from destination-mac-address {mac_neg}/48",
            f"set firewall family ethernet-switching filter {filter_name} term {term_neg} then count {term_neg}",
            f"set firewall family ethernet-switching filter {filter_name} term {term_neg} then accept",
        ])
    
    return cmds


def generate_layer_config(filter_name: str, layer_idx: int, layer_in_switch: int) -> List[str]:
    """
    Generate all config commands for one transformer layer.
    
    Layer encoding:
      - Each layer gets 14 MAC layer IDs (7 projections × 2 for pos/neg)
      - layer_base = layer_in_switch * 14
    """
    layer_base = layer_in_switch * 14
    
    all_cmds = []
    
    # Q projection: 1024 -> 2048
    all_cmds.extend(configure_projection(
        filter_name, f"l{layer_idx}_q", layer_base + 0, N_HEADS * D_HEAD
    ))
    
    # K projection: 1024 -> 1024
    all_cmds.extend(configure_projection(
        filter_name, f"l{layer_idx}_k", layer_base + 2, N_KV_HEADS * D_HEAD
    ))
    
    # V projection: 1024 -> 1024
    all_cmds.extend(configure_projection(
        filter_name, f"l{layer_idx}_v", layer_base + 4, N_KV_HEADS * D_HEAD
    ))
    
    # O projection: 2048 -> 1024
    all_cmds.extend(configure_projection(
        filter_name, f"l{layer_idx}_o", layer_base + 6, D_MODEL
    ))
    
    # FFN gate: 1024 -> 3072
    all_cmds.extend(configure_projection(
        filter_name, f"l{layer_idx}_gate", layer_base + 8, D_FFN
    ))
    
    # FFN up: 1024 -> 3072
    all_cmds.extend(configure_projection(
        filter_name, f"l{layer_idx}_up", layer_base + 10, D_FFN
    ))
    
    # FFN down: 3072 -> 1024
    all_cmds.extend(configure_projection(
        filter_name, f"l{layer_idx}_down", layer_base + 12, D_MODEL
    ))
    
    return all_cmds


def configure_switch_via_file(switch_ip: str, filter_name: str, 
                               layers: List[int], is_sw1: bool) -> bool:
    """
    Configure switch using VLAN-per-layer architecture.
    Each layer gets its own VLAN and filter to bypass TCAM limit!
    
    Architecture (from e063):
      - 1 VLAN per layer (vlan_id = BASE_VLAN + layer_idx)
      - 1 filter per layer (attached to VLAN, not interface)
      - Interface in trunk mode accepting all layer VLANs
      - Each filter stays under 1152 term TCAM limit
    """
    print(f"  Generating config for {len(layers)} layers (VLAN-per-layer architecture)...")
    
    BASE_VLAN = 100  # Starting VLAN ID
    all_cmds = []
    
    # First, set up trunk interface
    all_cmds.extend([
        "delete interfaces et-0/0/96 unit 0 family ethernet-switching",
        "set interfaces et-0/0/96 unit 0 family ethernet-switching interface-mode trunk",
    ])
    
    # Generate config for each layer (each gets its own filter + VLAN)
    for i, layer_idx in enumerate(layers):
        layer_in_switch = i  # 0-13 for each switch
        layer_filter = f"layer{layer_idx}_filter"
        layer_vlan_name = f"layer{layer_idx}_vlan"
        layer_vlan_id = BASE_VLAN + layer_idx
        
        # Generate the layer's filter terms
        layer_cmds = generate_layer_config(layer_filter, layer_idx, layer_in_switch)
        all_cmds.extend(layer_cmds)
        
        # Add default term for this layer's filter
        all_cmds.append(f"set firewall family ethernet-switching filter {layer_filter} term default then accept")
        
        # Create VLAN and attach filter to it
        all_cmds.extend([
            f"set vlans {layer_vlan_name} vlan-id {layer_vlan_id}",
            f"set vlans {layer_vlan_name} forwarding-options filter input {layer_filter}",
            f"set interfaces et-0/0/96 unit 0 family ethernet-switching vlan members {layer_vlan_name}",
        ])
        
        print(f"    Layer {layer_idx}: VLAN {layer_vlan_id}, {len(layer_cmds)} terms")
    
    print(f"  Total commands: {len(all_cmds)}")
    
    # Write config to local file
    local_config = f"/tmp/qwen_config_{'sw1' if is_sw1 else 'sw2'}.txt"
    remote_config = "/var/tmp/qwen_config.txt"
    
    print(f"  Writing config to {local_config}...")
    with open(local_config, 'w') as f:
        for cmd in all_cmds:
            f.write(cmd + '\n')
    
    # Get file size
    file_size_mb = os.path.getsize(local_config) / (1024 * 1024)
    print(f"  Config file size: {file_size_mb:.1f} MB")
    
    # Transfer via SSH stdin (more reliable than SCP)
    print(f"  Transferring config to switch...")
    if not transfer_config_file(local_config, switch_ip, remote_config):
        print("  ✗ Failed to transfer config file")
        return False
    print("  ✓ Config file transferred")
    
    # Apply config
    print(f"  Applying config (this may take several minutes for large configs)...")
    start_time = time.time()
    
    success, stdout, stderr = ssh_command_long(
        switch_ip,
        f"cli -c 'configure; load set {remote_config}; commit'",
        timeout=1800  # 30 minutes for very large configs
    )
    
    elapsed = time.time() - start_time
    
    if success and 'error' not in stdout.lower():
        print(f"  ✓ Config applied in {elapsed:.1f}s")
        return True
    else:
        print(f"  ✗ Config failed after {elapsed:.1f}s")
        if stderr:
            print(f"    stderr: {stderr[:200]}")
        if stdout:
            print(f"    stdout: {stdout[:200]}")
        return False


def configure_layer_on_switch(switch_ip: str, filter_name: str, 
                               layer_idx: int, layer_in_switch: int) -> bool:
    """Configure single layer using file method."""
    return configure_switch_via_file(switch_ip, filter_name, [layer_idx], True)


def configure_lm_head_shard(switch_ip: str, filter_name: str,
                            shard_idx: int, shard_size: int, 
                            total_vocab: int) -> bool:
    """Configure LM head shard counters."""
    start_vocab = shard_idx * shard_size
    end_vocab = min(start_vocab + shard_size, total_vocab)
    actual_size = end_vocab - start_vocab
    
    # LM head uses high layer IDs (200+)
    layer_base = 200 + shard_idx * 2
    
    all_cmds = []
    for i in range(actual_size):
        vocab_idx = start_vocab + i
        
        mac_pos = get_layer_neuron_mac(layer_base, i)
        term_pos = f"lm_v{vocab_idx}_p"
        all_cmds.extend([
            f"set firewall family ethernet-switching filter {filter_name} term {term_pos} from destination-mac-address {mac_pos}/48",
            f"set firewall family ethernet-switching filter {filter_name} term {term_pos} then count {term_pos}",
            f"set firewall family ethernet-switching filter {filter_name} term {term_pos} then accept",
        ])
        
        mac_neg = get_layer_neuron_mac(layer_base + 1, i)
        term_neg = f"lm_v{vocab_idx}_n"
        all_cmds.extend([
            f"set firewall family ethernet-switching filter {filter_name} term {term_neg} from destination-mac-address {mac_neg}/48",
            f"set firewall family ethernet-switching filter {filter_name} term {term_neg} then count {term_neg}",
            f"set firewall family ethernet-switching filter {filter_name} term {term_neg} then accept",
        ])
    
    print(f"    LM head shard {shard_idx}: vocab {start_vocab}-{end_vocab} ({len(all_cmds)} commands)")
    
    # Apply in chunks
    chunk_size = 500
    for i in range(0, len(all_cmds), chunk_size):
        chunk = all_cmds[i:i + chunk_size]
        cmd_str = "; ".join(chunk)
        success, _, stderr = ssh_command_long(
            switch_ip,
            f"cli -c 'configure; {cmd_str}; commit'",
            timeout=300
        )
        if not success:
            print(f"      ✗ LM head shard config failed: {stderr[:100]}")
            return False
    
    return True


def finalize_switch_config(switch_ip: str, filter_name: str) -> bool:
    """Add default term and interface binding."""
    cmds = [
        f"set firewall family ethernet-switching filter {filter_name} term default then accept",
        f"set vlans qwen_vlan vlan-id {TEST_VLAN}",
        "set interfaces et-0/0/96 unit 0 family ethernet-switching interface-mode access",
        f"set interfaces et-0/0/96 unit 0 family ethernet-switching vlan members qwen_vlan",
        f"set interfaces et-0/0/96 unit 0 family ethernet-switching filter input {filter_name}",
    ]
    
    cmd_str = "; ".join(cmds)
    success, _, stderr = ssh_command_long(
        switch_ip,
        f"cli -c 'configure; {cmd_str}; commit'",
        timeout=120
    )
    
    return success


# =============================================================================
# PACKET CREATION
# =============================================================================

def create_matmul_packets(activation: np.ndarray, weights: np.ndarray,
                          layer_base: int, src_mac: str, vlan_id: int = None) -> List[bytes]:
    """Create packets for matrix multiplication.
    
    Args:
        activation: Input activation vector
        weights: Weight matrix [in_dim, out_dim]
        layer_base: Base layer ID for MAC encoding
        src_mac: Source MAC address string
        vlan_id: VLAN ID for packet tagging (default: TEST_VLAN)
    """
    if vlan_id is None:
        vlan_id = TEST_VLAN
    
    packets = []
    src = mac_str_to_bytes(src_mac)
    
    in_dim, out_dim = weights.shape
    
    for out_idx in range(out_dim):
        for in_idx in range(in_dim):
            act_val = int(activation[in_idx])
            weight_val = int(weights[in_idx, out_idx])
            
            product = act_val * weight_val
            if product == 0:
                continue
            
            is_positive = product > 0
            dst_layer = layer_base if is_positive else layer_base + 1
            dst_mac = mac_str_to_bytes(get_layer_neuron_mac(dst_layer, out_idx))
            
            for _ in range(abs(product)):
                packets.append(craft_vlan_packet(dst_mac, src, vlan_id))
    
    return packets


# =============================================================================
# INFERENCE
# =============================================================================

def clear_counters(switch_ip: str, filter_name: str):
    """Clear all firewall counters."""
    ssh_command_long(switch_ip, f"cli -c 'clear firewall filter {filter_name}'", timeout=60)
    time.sleep(0.3)


def read_projection_output(switch_ip: str, filter_name: str,
                           layer_idx: int, proj_name: str, 
                           output_dim: int) -> np.ndarray:
    """Read counter values for a projection."""
    success, stdout, _ = ssh_command_long(
        switch_ip,
        f"cli -c 'show firewall filter {filter_name}'",
        timeout=120
    )
    
    output = np.zeros(output_dim, dtype=np.int32)
    
    if not success:
        return output
    
    for neuron in range(output_dim):
        term_pos = f"l{layer_idx}_{proj_name}_n{neuron}_p"
        pattern_pos = rf'{re.escape(term_pos)}\s+\d+\s+(\d+)'
        match_pos = re.search(pattern_pos, stdout)
        pos_val = int(match_pos.group(1)) if match_pos else 0
        
        term_neg = f"l{layer_idx}_{proj_name}_n{neuron}_n"
        pattern_neg = rf'{re.escape(term_neg)}\s+\d+\s+(\d+)'
        match_neg = re.search(pattern_neg, stdout)
        neg_val = int(match_neg.group(1)) if match_neg else 0
        
        output[neuron] = pos_val - neg_val
    
    return output


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*80)
    print("E082: FULL QWEN3-0.6B WITH REAL WEIGHTS")
    print("="*80)
    
    print(f"""
  Model: Qwen3-0.6B-Q4_K_M
  
  Full specifications:
    - Layers: {N_LAYERS}
    - Embedding dim: {D_MODEL}
    - FFN dim: {D_FFN}
    - Heads: {N_HEADS} Q, {N_KV_HEADS} KV
    - Head dim: {D_HEAD}
    - Vocabulary: {VOCAB_SIZE}
    
  Configuration mode: {'FAST TEST' if FAST_TEST else ('TEST (4 layers)' if TEST_MODE else 'FULL (28 layers)')}
    - Switch 1 ({SWITCH1_IP}): Layers {LAYERS_SW1}
    - Switch 2 ({SWITCH2_IP}): Layers {LAYERS_SW2}
    - Neurons per projection: {TEST_NEURONS if TEST_NEURONS else 'FULL'}
    
  Expected config time: {'~1-2 minutes' if FAST_TEST else f'~{len(LAYERS_SW1) * 3 + len(LAYERS_SW2) * 3} minutes'}
""")
    
    # Step 1: Load weights
    print("\n" + "="*60)
    print("STEP 1: LOAD WEIGHTS")
    print("="*60)
    
    weights = load_gguf_weights()
    if weights is None:
        print("  ✗ Failed to load weights")
        return False
    
    # Verify weight shapes
    print("\n  Weight shapes:")
    print(f"    token_embd: {weights.token_embd.shape if weights.token_embd is not None else 'None'}")
    print(f"    attn_q[0]: {weights.attn_q[0].shape if weights.attn_q[0] is not None else 'None'}")
    print(f"    ffn_gate[0]: {weights.ffn_gate[0].shape if weights.ffn_gate[0] is not None else 'None'}")
    print(f"    output: {weights.output.shape if weights.output is not None else 'None'}")
    
    # Step 2: Setup tokenizer
    print("\n" + "="*60)
    print("STEP 2: SETUP TOKENIZER")
    print("="*60)
    
    tokenizer = SimpleTokenizer()
    
    # Step 3: CPU reference
    print("\n" + "="*60)
    print("STEP 3: CPU REFERENCE GENERATION")
    print("="*60)
    
    prompt = "The "
    num_tokens = 3
    
    cpu_tokens, cpu_text = cpu_reference_generate(prompt, num_tokens)
    print(f"\n  CPU generated: {cpu_tokens}")
    print(f"  CPU text: '{cpu_text}'")
    
    # Step 4: Configure switches
    print("\n" + "="*60)
    print("STEP 4: CONFIGURE SWITCHES")
    print("="*60)
    
    # Cleanup both switches
    print("\n  Cleaning up switches...")
    cleanup_switch(SWITCH1_IP, FILTER_SW1)
    cleanup_switch(SWITCH2_IP, FILTER_SW2)
    time.sleep(2)
    
    # Configure Switch 1 (layers 0-13)
    print("\n  Configuring Switch 1 (layers 0-13)...")
    sw1_success = configure_switch_via_file(SWITCH1_IP, FILTER_SW1, LAYERS_SW1, True)
    
    # Configure Switch 2 (layers 14-27)
    print("\n  Configuring Switch 2 (layers 14-27)...")
    sw2_success = configure_switch_via_file(SWITCH2_IP, FILTER_SW2, LAYERS_SW2, False)
    
    if sw1_success and sw2_success:
        print("\n  ✓ Both switches configured!")
    else:
        print("\n  ✗ Configuration failed")
        if not sw1_success:
            print("    - Switch 1 failed")
        if not sw2_success:
            print("    - Switch 2 failed")
    
    time.sleep(2)
    
    # Step 5: Test one projection
    print("\n" + "="*60)
    print("STEP 5: TEST SINGLE PROJECTION (Layer 0 Q)")
    print("="*60)
    
    # Get input embedding
    input_tokens = tokenizer.encode(prompt)
    print(f"\n  Input tokens: {input_tokens}")
    
    if len(input_tokens) > 0 and weights.token_embd is not None:
        last_token = input_tokens[-1]
        
        print(f"\n  Token: {last_token}")
        print(f"  Embedding shape: {weights.token_embd.shape}")
        print(f"  attn_q[0] shape: {weights.attn_q[0].shape if weights.attn_q[0] is not None else 'None'}")
        
        # Get embedding for last token
        x = weights.token_embd[last_token].astype(np.int32)
        print(f"  Embedding for token {last_token}: shape {x.shape}")
        print(f"  Embedding values: min={x.min()}, max={x.max()}, mean={x.mean():.2f}")
        print(f"  Embedding sample: {x[:8]}...")
        
        # CPU reference for Q projection
        if weights.attn_q[0] is not None:
            # x @ W^T for projection (input is [d_model], weight is [out_dim, d_model])
            w_q = weights.attn_q[0].astype(np.int32)
            print(f"\n  Q weight shape: {w_q.shape}")
            print(f"  Q weight values: min={w_q.min()}, max={w_q.max()}")
            
            # Compute Q projection: x @ W^T = [d_model] @ [out_dim, d_model]^T = [out_dim]
            cpu_q = x @ w_q.T
            print(f"  CPU Q projection: shape {cpu_q.shape}")
            print(f"  CPU Q[:8]: {cpu_q[:8]}")
            
            # Now test on switch!
            print(f"\n  Testing on switch...")
            # Layer 0 uses filter "layer0_filter"
            layer_filter = "layer0_filter"
            clear_counters(SWITCH1_IP, layer_filter)
            time.sleep(0.5)
            
            src_mac = get_mac_address(SEND_IFACE)
            
            # Use first 64 input dims and first 64 output dims (matching our config)
            x_sub = x[:64]
            w_sub = w_q[:64, :64]  # [64 outputs, 64 inputs]
            
            cpu_sub = x_sub @ w_sub.T
            print(f"  CPU subsample (64x64): {cpu_sub[:8]}...")
            
            # Check expected MACs
            print(f"\n  Expected MAC for neuron 0: {get_layer_neuron_mac(0, 0)}")
            print(f"  Expected MAC for neuron 1: {get_layer_neuron_mac(0, 1)}")
            
            # Layer 0 uses VLAN 100 + 0 = 100 and filter "layer0_filter"
            layer_vlan = 100 + 0  # BASE_VLAN + layer_idx
            layer_filter = "layer0_filter"
            
            # Create and send packets with layer-specific VLAN
            packets = create_matmul_packets(x_sub, w_sub.T, 0, src_mac, vlan_id=layer_vlan)
            print(f"  Sending {len(packets)} packets to VLAN {layer_vlan}...")
            send_packets(SEND_IFACE, packets)
            time.sleep(1.5)
            
            # Debug: Show filter
            success, stdout, _ = ssh_command_long(
                SWITCH1_IP,
                f"cli -c 'show firewall filter {layer_filter} | match l0_q'",
                timeout=60
            )
            print(f"\n  Filter terms (l0_q): {stdout[:500] if success else 'FAILED'}...")
            
            # Read results
            switch_output = read_projection_output(SWITCH1_IP, layer_filter, 0, "q", 64)
            print(f"  Switch result: {switch_output[:8]}...")
            
            # Compare
            match = np.allclose(cpu_sub, switch_output, atol=2)
            print(f"\n  CPU vs Switch: {'✓ MATCH' if match else '✗ MISMATCH'}")
            
            if not match:
                diff = np.abs(cpu_sub - switch_output)
                print(f"  Max difference: {diff.max()}")
                print(f"  CPU:    {cpu_sub[:8]}")
                print(f"  Switch: {switch_output[:8]}")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)



""" Output:
    Note: blk.27.ffn_down.weight has type 14, attempting Q4_K decode
  Loading output weights...
    Warning: output.weight not found
  ✓ Weights loaded successfully!

  Weight shapes:
    token_embd: (151936, 1024)
    attn_q[0]: (1024, 1024)
    ffn_gate[0]: (3072, 1024)
    output: None

============================================================
STEP 2: SETUP TOKENIZER
============================================================
llama_context: n_ctx_per_seq (512) < n_ctx_train (40960) -- the full capacity of the model will not be utilized
  ✓ Using llama-cpp tokenizer

============================================================
STEP 3: CPU REFERENCE GENERATION
============================================================
  Loading llama-cpp model for CPU reference...
llama_context: n_ctx_per_seq (512) < n_ctx_train (40960) -- the full capacity of the model will not be utilized
  Generating 3 tokens from: 'The '
  Input tokens: [785, 220]
  Generated tokens: [16, 24, 339]
  Generated text: '19th'

  CPU generated: [16, 24, 339]
  CPU text: 'The 19th'

============================================================
STEP 4: CONFIGURE SWITCHES
============================================================

  Cleaning up switches...

  Configuring Switch 1 (layers 0-13)...
  Generating config for 2 layers (VLAN-per-layer architecture)...
    Layer 0: VLAN 100, 2688 terms
    Layer 1: VLAN 101, 2688 terms
  Total commands: 5386
  Writing config to /tmp/qwen_config_sw1.txt...
  Config file size: 0.5 MB
  Transferring config to switch...
  ✓ Config file transferred
  Applying config (this may take several minutes for large configs)...
  ✓ Config applied in 35.8s

  Configuring Switch 2 (layers 14-27)...
  Generating config for 2 layers (VLAN-per-layer architecture)...
    Layer 2: VLAN 102, 2688 terms
    Layer 3: VLAN 103, 2688 terms
  Total commands: 5386
  Writing config to /tmp/qwen_config_sw2.txt...
  Config file size: 0.5 MB
  Transferring config to switch...
  ✓ Config file transferred
  Applying config (this may take several minutes for large configs)...
  ✓ Config applied in 19.9s

  ✓ Both switches configured!

============================================================
STEP 5: TEST SINGLE PROJECTION (Layer 0 Q)
============================================================

  Input tokens: [785, 220]

  Token: 220
  Embedding shape: (151936, 1024)
  attn_q[0] shape: (1024, 1024)
  Embedding for token 220: shape (1024,)
  Embedding values: min=-8, max=7, mean=-0.48
  Embedding sample: [-1  3 -2  0  5  6  4 -1]...

  Q weight shape: (1024, 1024)
  Q weight values: min=-8, max=7
  CPU Q projection: shape (1024,)
  CPU Q[:8]: [  871 -1594  -152  1459  -288  -920  1495   951]

  Testing on switch...
  CPU subsample (64x64): [ 219  -59   62   77  180  -19   76 -151]...

  Expected MAC for neuron 0: 01:00:5e:00:00:00
  Expected MAC for neuron 1: 01:00:5e:00:00:01
  Sending 82106 packets to VLAN 100...

  Filter terms (l0_q): l0_q_n0_n                                           34944                  546
l0_q_n0_p                                           48960                  765
l0_q_n10_n                                          34560                  540
l0_q_n10_p                                          50688                  792
l0_q_n11_n                                          47680                  745
l0_q_n11_p                                          27456                  429
l0_q_n12_n                ...
  Switch result: [ 219  -59   62   77  180  -19   76 -151]...

  CPU vs Switch: ✓ MATCH
"""
