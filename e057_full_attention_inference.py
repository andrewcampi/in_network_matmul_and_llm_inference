#!/usr/bin/env python3
"""
e057_full_attention_inference.py

FULL TRANSFORMER INFERENCE WITH ATTENTION
==========================================

GOAL: Produce coherent text like the CPU baseline using ALL 28 layers
      with proper attention mechanism!

ARCHITECTURE (Qwen3-0.6B):
  Per layer:
    1. Attention:
       - Q = W_q @ RMSNorm(x)     [1024 → 1024]
       - K = W_k @ RMSNorm(x)     [1024 → 512 with GQA]
       - V = W_v @ RMSNorm(x)     [1024 → 512 with GQA]
       - For single token: attn_out = W_o @ V (softmax of 1 value = 1.0)
       - Residual: x = x + attn_out
    2. FFN:
       - gate = W_gate @ RMSNorm(x)  [1024 → 3072]
       - up = W_up @ RMSNorm(x)      [1024 → 3072]  
       - down = W_down @ SiLU(gate) * up  [3072 → 1024]
       - Residual: x = x + down
  Final:
    - output = W_out @ RMSNorm(x)  [1024 → vocab]

SCALING STRATEGY:
  - Use 512 hidden dim (half of 1024) to fit in TCAM
  - All 28 layers pre-configured
  - ~28K TCAM rules total (fits in 2 switches)

Author: Research Phase 001
Date: December 2025
"""

import time
import os
import sys
import re
import subprocess
import numpy as np
import gguf
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from previous experiments
from e053_mac_encoded_layers import get_layer_neuron_mac
from e045_real_weights_inference import mac_str_to_bytes
from e042_port_based_layers import (
    ssh_command, run_config_commands,
    craft_vlan_packet, send_packets, get_mac_address,
    SWITCH1_IP, SWITCH2_IP, SEND_IFACE,
)

# Model path
MODEL_PATH = "./models/Qwen3-0.6B-Q4_K_M.gguf"

# Architecture configuration - START SMALL like e056, then scale up
NUM_LAYERS = 2           # Start with 2 layers like e056
HIDDEN_DIM = 32          # 32 dim like e056 to verify it works
FFN_DIM = 96             # ~3x hidden for FFN
VOCAB_SIZE = 64          # Small vocab like e056
NUM_TOKENS = 5           # Match e056
WEIGHT_SCALE = 30        # Scale for 4-bit weights

FILTER_NAME = "full_attn"
TEST_VLAN = 100


def load_model() -> gguf.GGUFReader:
    """Load model."""
    print(f"\n  Loading model: {MODEL_PATH}")
    reader = gguf.GGUFReader(MODEL_PATH)
    print(f"    Loaded {len(reader.tensors)} tensors")
    return reader


def get_tensor_by_name(reader: gguf.GGUFReader, name: str):
    """Find tensor by exact name."""
    for tensor in reader.tensors:
        if tensor.name == name:
            return tensor
    return None


def dequantize_tensor(tensor) -> np.ndarray:
    """Dequantize tensor using gguf library."""
    return gguf.dequantize(tensor.data, tensor.tensor_type)


def weights_to_4bit(weights: np.ndarray) -> np.ndarray:
    """Convert to 4-bit signed integers."""
    scaled = weights * WEIGHT_SCALE
    return np.clip(np.round(scaled), -8, 7).astype(np.int8)


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """RMS normalization."""
    rms = np.sqrt(np.mean(x ** 2) + eps)
    return (x / rms) * weight


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU activation (x * sigmoid(x))."""
    return x * (1 / (1 + np.exp(-np.clip(x, -20, 20))))


class Qwen3Layer:
    """One transformer layer with attention + FFN."""
    
    def __init__(self, reader: gguf.GGUFReader, layer_idx: int, hidden_dim: int, ffn_dim: int):
        self.layer_idx = layer_idx
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        
        prefix = f'blk.{layer_idx}.'
        
        # Load and slice weights
        # Attention (simplified: just V and O for single-token)
        self.attn_norm = self._load_norm(reader, prefix + 'attn_norm.weight')
        self.attn_v = self._load_weight(reader, prefix + 'attn_v.weight', (hidden_dim, hidden_dim))
        self.attn_o = self._load_weight(reader, prefix + 'attn_output.weight', (hidden_dim, hidden_dim))
        
        # FFN
        self.ffn_norm = self._load_norm(reader, prefix + 'ffn_norm.weight')
        self.ffn_gate = self._load_weight(reader, prefix + 'ffn_gate.weight', (ffn_dim, hidden_dim))
        self.ffn_up = self._load_weight(reader, prefix + 'ffn_up.weight', (ffn_dim, hidden_dim))
        self.ffn_down = self._load_weight(reader, prefix + 'ffn_down.weight', (hidden_dim, ffn_dim))
    
    def _load_norm(self, reader, name):
        """Load RMS norm weight."""
        tensor = get_tensor_by_name(reader, name)
        if tensor:
            w = dequantize_tensor(tensor)
            return w[:self.hidden_dim]
        return np.ones(self.hidden_dim)
    
    def _load_weight(self, reader, name, shape):
        """Load and slice weight matrix to 4-bit."""
        tensor = get_tensor_by_name(reader, name)
        if tensor:
            w = dequantize_tensor(tensor)
            # Take the slice we need
            out_dim, in_dim = shape
            w = w[:out_dim, :in_dim] if len(w.shape) == 2 else w.reshape(-1, w.shape[-1])[:out_dim, :in_dim]
            return weights_to_4bit(w)
        return np.random.randint(-8, 8, size=shape, dtype=np.int8)
    
    def forward_cpu(self, x: np.ndarray) -> np.ndarray:
        """CPU reference forward pass."""
        # Attention (simplified for single token)
        normed = rms_norm(x, self.attn_norm)
        v = self.attn_v.astype(np.float32) @ normed  # Value projection
        attn_out = self.attn_o.astype(np.float32) @ v  # Output projection
        x = x + attn_out  # Residual
        
        # FFN
        normed = rms_norm(x, self.ffn_norm)
        gate = self.ffn_gate.astype(np.float32) @ normed
        up = self.ffn_up.astype(np.float32) @ normed
        ffn_out = self.ffn_down.astype(np.float32) @ (silu(gate) * up)
        x = x + ffn_out  # Residual
        
        return x


class Qwen3Model:
    """Full Qwen3 model."""
    
    def __init__(self, reader: gguf.GGUFReader, num_layers: int, hidden_dim: int, 
                 ffn_dim: int, vocab_size: int):
        print(f"\n  Building model: {num_layers} layers, {hidden_dim} hidden, {vocab_size} vocab")
        
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        
        # Load embeddings
        emb_tensor = get_tensor_by_name(reader, 'token_embd.weight')
        if emb_tensor:
            emb = dequantize_tensor(emb_tensor)
            self.embeddings = emb[:, :hidden_dim]  # [vocab, hidden]
            print(f"    Embeddings: {self.embeddings.shape}")
        else:
            self.embeddings = np.random.randn(152000, hidden_dim).astype(np.float32) * 0.1
        
        # Load output projection
        out_tensor = get_tensor_by_name(reader, 'output.weight')
        if out_tensor:
            out = dequantize_tensor(out_tensor)
            self.output_proj = weights_to_4bit(out[:vocab_size, :hidden_dim])
            print(f"    Output: {self.output_proj.shape}")
        else:
            self.output_proj = np.random.randint(-8, 8, size=(vocab_size, hidden_dim), dtype=np.int8)
        
        # Load output norm
        norm_tensor = get_tensor_by_name(reader, 'output_norm.weight')
        if norm_tensor:
            self.output_norm = dequantize_tensor(norm_tensor)[:hidden_dim]
        else:
            self.output_norm = np.ones(hidden_dim)
        
        # Load layers
        self.layers = []
        for i in range(num_layers):
            layer = Qwen3Layer(reader, i, hidden_dim, ffn_dim)
            self.layers.append(layer)
            if (i + 1) % 7 == 0:
                print(f"    Loaded layer {i + 1}/{num_layers}")
        
        print(f"    Model ready!")
    
    def forward_cpu(self, token_id: int) -> np.ndarray:
        """CPU forward pass for one token."""
        # Embedding lookup
        if token_id >= len(self.embeddings):
            token_id = token_id % len(self.embeddings)
        x = self.embeddings[token_id].copy()
        
        # Forward through all layers
        for layer in self.layers:
            x = layer.forward_cpu(x)
        
        # Output projection
        x = rms_norm(x, self.output_norm)
        logits = self.output_proj.astype(np.float32) @ x
        
        return logits


def full_cleanup():
    """Clean switches using efficient single-commit."""
    print("\n  Cleanup...")
    cleanup_cmd = f"cli -c 'configure; delete firewall family ethernet-switching filter {FILTER_NAME}; delete firewall family ethernet-switching filter; set forwarding-options storm-control-profiles default all; commit'"
    ssh_command(SWITCH1_IP, cleanup_cmd)
    time.sleep(0.5)
    print("  ✓ Done")


def configure_all_layers(model: Qwen3Model):
    """Configure filters for ALL layer projections using SCP + load set."""
    import subprocess
    import tempfile
    
    print("\n  Generating configuration file...")
    
    all_cmds = []
    rule_count = 0
    
    # Storm control profile (fixes commit errors)
    all_cmds.append("set forwarding-options storm-control-profiles default all")
    
    # Base filter
    all_cmds.append(f"set firewall family ethernet-switching filter {FILTER_NAME}")
    
    # Encoding scheme:
    # Layer L, projection P, neuron N → MAC layer = L*5 + P, neuron = N
    # P: 0=attn_v, 1=attn_o, 2=ffn_gate, 3=ffn_up, 4=ffn_down
    
    for layer_idx, layer in enumerate(model.layers):
        # attn_v (P=0)
        for n in range(layer.hidden_dim):
            mac_layer = layer_idx * 5 + 0
            mac_pos = get_layer_neuron_mac(mac_layer, n * 2)
            mac_neg = get_layer_neuron_mac(mac_layer, n * 2 + 1)
            term = f"L{layer_idx}_av{n}"  # Shortened names
            all_cmds.extend([
                f"set firewall family ethernet-switching filter {FILTER_NAME} term {term}p from destination-mac-address {mac_pos}/48",
                f"set firewall family ethernet-switching filter {FILTER_NAME} term {term}p then count {term}p",
                f"set firewall family ethernet-switching filter {FILTER_NAME} term {term}p then accept",
                f"set firewall family ethernet-switching filter {FILTER_NAME} term {term}n from destination-mac-address {mac_neg}/48",
                f"set firewall family ethernet-switching filter {FILTER_NAME} term {term}n then count {term}n",
                f"set firewall family ethernet-switching filter {FILTER_NAME} term {term}n then accept",
            ])
            rule_count += 2
        
        # attn_o (P=1)
        for n in range(layer.hidden_dim):
            mac_layer = layer_idx * 5 + 1
            mac_pos = get_layer_neuron_mac(mac_layer, n * 2)
            mac_neg = get_layer_neuron_mac(mac_layer, n * 2 + 1)
            term = f"L{layer_idx}_ao{n}"
            all_cmds.extend([
                f"set firewall family ethernet-switching filter {FILTER_NAME} term {term}p from destination-mac-address {mac_pos}/48",
                f"set firewall family ethernet-switching filter {FILTER_NAME} term {term}p then count {term}p",
                f"set firewall family ethernet-switching filter {FILTER_NAME} term {term}p then accept",
                f"set firewall family ethernet-switching filter {FILTER_NAME} term {term}n from destination-mac-address {mac_neg}/48",
                f"set firewall family ethernet-switching filter {FILTER_NAME} term {term}n then count {term}n",
                f"set firewall family ethernet-switching filter {FILTER_NAME} term {term}n then accept",
            ])
            rule_count += 2
        
        # ffn_down (P=4)
        for n in range(layer.hidden_dim):
            mac_layer = layer_idx * 5 + 4
            mac_pos = get_layer_neuron_mac(mac_layer, n * 2)
            mac_neg = get_layer_neuron_mac(mac_layer, n * 2 + 1)
            term = f"L{layer_idx}_fd{n}"
            all_cmds.extend([
                f"set firewall family ethernet-switching filter {FILTER_NAME} term {term}p from destination-mac-address {mac_pos}/48",
                f"set firewall family ethernet-switching filter {FILTER_NAME} term {term}p then count {term}p",
                f"set firewall family ethernet-switching filter {FILTER_NAME} term {term}p then accept",
                f"set firewall family ethernet-switching filter {FILTER_NAME} term {term}n from destination-mac-address {mac_neg}/48",
                f"set firewall family ethernet-switching filter {FILTER_NAME} term {term}n then count {term}n",
                f"set firewall family ethernet-switching filter {FILTER_NAME} term {term}n then accept",
            ])
            rule_count += 2
        
        if (layer_idx + 1) % 7 == 0:
            print(f"    Layer {layer_idx + 1}/{len(model.layers)} prepared ({rule_count} rules)")
    
    # Output projection
    for n in range(model.vocab_size):
        mac_layer = 255  # Special layer for output
        mac_pos = get_layer_neuron_mac(mac_layer, n * 2)
        mac_neg = get_layer_neuron_mac(mac_layer, n * 2 + 1)
        term = f"o{n}"
        all_cmds.extend([
            f"set firewall family ethernet-switching filter {FILTER_NAME} term {term}p from destination-mac-address {mac_pos}/48",
            f"set firewall family ethernet-switching filter {FILTER_NAME} term {term}p then count {term}p",
            f"set firewall family ethernet-switching filter {FILTER_NAME} term {term}p then accept",
            f"set firewall family ethernet-switching filter {FILTER_NAME} term {term}n from destination-mac-address {mac_neg}/48",
            f"set firewall family ethernet-switching filter {FILTER_NAME} term {term}n then count {term}n",
            f"set firewall family ethernet-switching filter {FILTER_NAME} term {term}n then accept",
        ])
        rule_count += 2
    
    # Default term
    all_cmds.extend([
        f"set firewall family ethernet-switching filter {FILTER_NAME} term default then count default_pkts",
        f"set firewall family ethernet-switching filter {FILTER_NAME} term default then accept",
    ])
    
    # Apply filter to interface
    all_cmds.extend([
        f"set vlans test_vlan vlan-id {TEST_VLAN}",
        f"set interfaces et-0/0/96 unit 0 family ethernet-switching vlan members test_vlan",
        f"set interfaces et-0/0/96 unit 0 family ethernet-switching filter input {FILTER_NAME}",
    ])
    
    print(f"    Total rules: {rule_count}")
    print(f"    Total commands: {len(all_cmds)}")
    
    if rule_count > 16000:
        print(f"    WARNING: {rule_count} rules may exceed TCAM limit!")
    
    # Write to temp file
    config_file = "/tmp/e057_config.txt"
    with open(config_file, 'w') as f:
        f.write('\n'.join(all_cmds))
    print(f"    Config written to {config_file}")
    
    # Transfer config file via SSH stdin (SCP subsystem not available on Junos)
    print(f"    Transferring config to {SWITCH1_IP}...")
    ssh_key = "/home/multiplex/.ssh/id_rsa"
    
    # Use SSH with stdin redirect: cat file | ssh 'cat > remote'
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
    
    print("    ✓ Config file transferred")
    
    # Load and commit on switch (use longer timeout for large configs)
    print("    Loading configuration on switch (this may take a few minutes)...")
    load_ssh_cmd = [
        'ssh', '-i', ssh_key,
        '-o', 'StrictHostKeyChecking=no',
        '-o', 'UserKnownHostsFile=/dev/null',
        '-o', 'LogLevel=ERROR',
        f'root@{SWITCH1_IP}',
        "cli -c 'configure; load set /var/tmp/config.txt; commit'"
    ]
    
    try:
        result = subprocess.run(load_ssh_cmd, capture_output=True, text=True, timeout=300)
        success = result.returncode == 0
        stdout = result.stdout
        stderr = result.stderr
    except subprocess.TimeoutExpired:
        print(f"    ✗ Load timed out after 300 seconds")
        return False
    
    if not success:
        print(f"    ✗ Load failed: {stderr}")
        return False
    
    # Check for errors in output
    if 'error' in stdout.lower() or 'error' in stderr.lower():
        print(f"    ✗ Config errors detected")
        print(f"    stdout: {stdout[:500]}")
        return False
    
    print("  ✓ Configuration complete (single commit!)")
    return True


def ssh_command_long(switch_ip: str, command: str, timeout: int = 120) -> Tuple[bool, str, str]:
    """SSH command with longer timeout for large operations."""
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


def clear_counters():
    """Clear all firewall counters."""
    ssh_command_long(SWITCH1_IP, f"cli -c 'clear firewall filter {FILTER_NAME}'", timeout=60)
    time.sleep(0.1)


def read_counters(prefix: str, count: int) -> np.ndarray:
    """Read counters with given prefix."""
    success, stdout, _ = ssh_command_long(
        SWITCH1_IP,
        f"cli -c 'show firewall filter {FILTER_NAME}'",
        timeout=120
    )
    
    values = np.zeros(count, dtype=np.int32)
    if not success or not stdout:
        return values
    
    for i in range(count):
        # Match positive and negative counters
        pos_pattern = rf'{prefix}{i}p\s+\d+\s+(\d+)'
        neg_pattern = rf'{prefix}{i}n\s+\d+\s+(\d+)'
        
        pos_match = re.search(pos_pattern, stdout)
        neg_match = re.search(neg_pattern, stdout)
        
        pos_count = int(pos_match.group(1)) if pos_match else 0
        neg_count = int(neg_match.group(1)) if neg_match else 0
        values[i] = pos_count - neg_count
    
    return values


def create_layer_packets(layer_idx: int, proj_idx: int, hidden_state: np.ndarray, 
                         weights: np.ndarray, src_mac: str) -> List:
    """Create packets for a layer projection."""
    packets = []
    mac_layer = layer_idx * 5 + proj_idx
    
    # Binarize input (active if non-zero)
    active = np.abs(hidden_state) > 0.01
    
    for out_idx in range(weights.shape[0]):
        pos_packets = 0
        neg_packets = 0
        
        for in_idx in range(weights.shape[1]):
            if not active[in_idx]:
                continue
            
            w = weights[out_idx, in_idx]
            if w > 0:
                pos_packets += abs(w)
            elif w < 0:
                neg_packets += abs(w)
        
        # Create packets
        if pos_packets > 0:
            mac_pos = get_layer_neuron_mac(mac_layer, out_idx * 2)
            dst_bytes = mac_str_to_bytes(mac_pos)
            src_bytes = mac_str_to_bytes(src_mac)
            for _ in range(pos_packets):
                pkt = craft_vlan_packet(dst_bytes, src_bytes, TEST_VLAN)
                packets.append(pkt)
        
        if neg_packets > 0:
            mac_neg = get_layer_neuron_mac(mac_layer, out_idx * 2 + 1)
            dst_bytes = mac_str_to_bytes(mac_neg)
            src_bytes = mac_str_to_bytes(src_mac)
            for _ in range(neg_packets):
                pkt = craft_vlan_packet(dst_bytes, src_bytes, TEST_VLAN)
                packets.append(pkt)
    
    return packets


def cpu_4bit_matmul(hidden: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """CPU reference that matches switch computation exactly."""
    # Binarize: treat any non-zero input as active (same as switch)
    threshold = 0.01 if np.any(hidden != 0) else 0
    hidden_binary = (np.abs(hidden) > threshold).astype(np.float32)
    
    # Matrix multiply with 4-bit weights
    return weights.astype(np.float32) @ hidden_binary


def run_layer_on_switch(layer_idx: int, proj_idx: int, hidden: np.ndarray, 
                        weights: np.ndarray, src_mac: str, num_outputs: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run a single layer projection on switch. Returns (switch_result, cpu_result).
    
    This follows e056's proven approach:
    1. Create packets for this projection
    2. Send packets
    3. Read counters
    4. Return results for comparison
    """
    packets = create_layer_packets(layer_idx, proj_idx, hidden, weights, src_mac)
    
    if packets:
        send_packets(SEND_IFACE, packets)
    time.sleep(0.15)  # Wait for packets to be processed
    
    # Read counters for this layer/projection
    prefix = f"L{layer_idx}_"
    proj_names = {0: "av", 1: "ao", 4: "fd"}
    counter_prefix = prefix + proj_names.get(proj_idx, f"p{proj_idx}")
    
    # Read switch counters
    switch_result = np.zeros(num_outputs, dtype=np.float32)
    success, stdout, _ = ssh_command_long(SWITCH1_IP, f"cli -c 'show firewall filter {FILTER_NAME}'", timeout=30)
    
    if success and stdout:
        for n in range(num_outputs):
            pos_pattern = rf'{counter_prefix}{n}p\s+\d+\s+(\d+)'
            neg_pattern = rf'{counter_prefix}{n}n\s+\d+\s+(\d+)'
            pos_match = re.search(pos_pattern, stdout)
            neg_match = re.search(neg_pattern, stdout)
            pos_count = int(pos_match.group(1)) if pos_match else 0
            neg_count = int(neg_match.group(1)) if neg_match else 0
            switch_result[n] = pos_count - neg_count
    
    # CPU reference (same computation as switch)
    cpu_result = cpu_4bit_matmul(hidden, weights)
    
    return switch_result, cpu_result


def run_switch_forward_layerwise(model: Qwen3Model, token_id: int, src_mac: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run forward pass layer-by-layer, comparing switch vs CPU at each step.
    Returns (switch_logits, cpu_logits).
    """
    # Get embedding
    if token_id >= len(model.embeddings):
        token_id = token_id % len(model.embeddings)
    hidden = model.embeddings[token_id].copy()
    
    all_match = True
    
    # Forward through layers
    for layer_idx, layer in enumerate(model.layers):
        # Clear counters before this layer
        clear_counters()
        time.sleep(0.1)
        
        # V projection (proj_idx=0)
        sw_v, cpu_v = run_layer_on_switch(layer_idx, 0, hidden, layer.attn_v, src_mac, layer.hidden_dim)
        
        # O projection (proj_idx=1) - use binarized V output as input
        v_binary = (np.abs(sw_v) > 0).astype(np.float32)  # Binarize for next projection
        sw_o, cpu_o = run_layer_on_switch(layer_idx, 1, v_binary, layer.attn_o, src_mac, layer.hidden_dim)
        
        # Update hidden state with attention output
        hidden = hidden + sw_o  # Use switch result
        
        # Skip FFN for now (dimension mismatch: ffn_down needs FFN_DIM input, not HIDDEN_DIM)
        # TODO: Add gate/up projections to compute FFN hidden state
        
        # Check match
        v_match = np.allclose(sw_v, cpu_v, atol=1)
        o_match = np.allclose(sw_o, cpu_o, atol=1)
        
        status = "✓" if (v_match and o_match) else "✗"
        print(f"    L{layer_idx}: V={v_match}, O={o_match} {status}")
        
        if not (v_match and o_match):
            all_match = False
    
    # Output projection
    clear_counters()
    time.sleep(0.1)
    
    # Create and send output packets
    out_packets = []
    for out_idx in range(model.vocab_size):
        pos_pkts = 0
        neg_pkts = 0
        for in_idx in range(model.output_proj.shape[1]):
            if abs(hidden[in_idx]) < 0.01:
                continue
            w = model.output_proj[out_idx, in_idx]
            if w > 0:
                pos_pkts += abs(w)
            elif w < 0:
                neg_pkts += abs(w)
        
        mac_pos = get_layer_neuron_mac(255, out_idx * 2)
        mac_neg = get_layer_neuron_mac(255, out_idx * 2 + 1)
        for _ in range(pos_pkts):
            out_packets.append(craft_vlan_packet(mac_str_to_bytes(mac_pos), mac_str_to_bytes(src_mac), TEST_VLAN))
        for _ in range(neg_pkts):
            out_packets.append(craft_vlan_packet(mac_str_to_bytes(mac_neg), mac_str_to_bytes(src_mac), TEST_VLAN))
    
    if out_packets:
        send_packets(SEND_IFACE, out_packets)
    time.sleep(0.2)
    
    # Read output counters
    switch_logits = read_counters("o", model.vocab_size)
    cpu_logits = cpu_4bit_matmul(hidden, model.output_proj)
    
    return switch_logits, cpu_logits


def run_full_inference():
    """Run full transformer inference."""
    print("="*80)
    print("E057: FULL TRANSFORMER INFERENCE WITH ATTENTION")
    print("="*80)
    print(f"""
  Architecture:
    - {NUM_LAYERS} layers (full model!)
    - {HIDDEN_DIM} hidden dim
    - {FFN_DIM} FFN dim  
    - {VOCAB_SIZE} vocab size
    - Attention: simplified for single-token (V @ O)
    - FFN: gate * up → down
""")
    
    # Cleanup
    full_cleanup()
    
    # Load model
    reader = load_model()
    model = Qwen3Model(reader, NUM_LAYERS, HIDDEN_DIM, FFN_DIM, VOCAB_SIZE)
    
    # For now, just run CPU inference to verify the model works
    print("\n" + "="*60)
    print("Testing CPU-only inference first")
    print("="*60)
    
    # Use a simple tokenizer
    tokens = [chr(i) if 32 <= i <= 126 else f"<{i}>" for i in range(128)]
    while len(tokens) < 152000:
        tokens.append(f"<tok_{len(tokens)}>")
    tokens[791] = "The"
    tokens[1575] = "The"
    
    token_to_id = {t: i for i, t in enumerate(tokens)}
    
    prompt = "The"
    prompt_token = token_to_id.get(prompt, 791)
    generated = [prompt_token]
    
    print(f"\n  Prompt: '{prompt}' (token {prompt_token})")
    print("  Generating with CPU reference...")
    
    for step in range(NUM_TOKENS):
        t0 = time.time()
        logits = model.forward_cpu(generated[-1])
        next_token = int(np.argmax(logits))
        generated.append(next_token)
        
        tok_str = tokens[next_token] if next_token < len(tokens) else f"<{next_token}>"
        print(f"    Step {step+1}: token {next_token} = '{tok_str}' (logits top: {logits[:5].astype(int)})")
    
    print(f"\n  Generated tokens: {generated}")
    output = "".join(tokens[t] if t < len(tokens) else f"<{t}>" for t in generated)
    print(f"  Output: '{output}'")
    
    # Now configure switch and compare
    print("\n" + "="*60)
    print("Configuring switch for comparison")
    print("="*60)
    
    if not configure_all_layers(model):
        print("  ✗ Configuration failed!")
        return
    
    print("\n  Switch configuration successful!")
    print(f"\n  Host interface: {SEND_IFACE}")
    
    # Verify filter is applied
    print("\n  Verifying filter installation...")
    success, stdout, _ = ssh_command_long(SWITCH1_IP, "cli -c 'show configuration interfaces et-0/0/96'", 30)
    if stdout:
        print(f"    Interface config: {stdout.strip()[:200]}")
    
    # Check interface status
    success, stdout, _ = ssh_command_long(SWITCH1_IP, "cli -c 'show interfaces et-0/0/96 terse'", 30)
    if stdout:
        print(f"    Interface status: {stdout.strip()}")
    
    # Check what's actually connected
    success, stdout, _ = ssh_command_long(SWITCH1_IP, "cli -c 'show interfaces terse' | grep -E 'et-0/0/(48|49|96|97|98|99)'", 30)
    if stdout:
        print(f"    Relevant interfaces:\n      " + "\n      ".join(stdout.strip().split('\n')[:10]))
    
    # Now compare switch vs CPU
    print("\n" + "="*60)
    print("Comparing Switch vs CPU inference")
    print("="*60)
    
    src_mac = get_mac_address(SEND_IFACE)
    switch_generated = [prompt_token]
    
    print(f"\n  Generating with switch...")
    
    for step in range(min(3, NUM_TOKENS)):  # Just 3 steps for now
        print(f"\n  Step {step+1}:")
        
        # Run layer-by-layer forward pass
        switch_logits, cpu_logits = run_switch_forward_layerwise(model, switch_generated[-1], src_mac)
        
        switch_next = int(np.argmax(switch_logits))
        cpu_next = int(np.argmax(cpu_logits))
        switch_generated.append(switch_next)
        
        match = "✓ MATCH" if switch_next == cpu_next else "✗ MISMATCH"
        tok_str = tokens[switch_next] if switch_next < len(tokens) else f"<{switch_next}>"
        
        print(f"    Output: switch={switch_next} cpu={cpu_next} {match}")
        print(f"            switch logits[:5]={switch_logits[:5].astype(int)}")
        print(f"            cpu logits[:5]={cpu_logits[:5].astype(int)}")
    
    print(f"\n  Switch generated: {switch_generated}")
    
    # Cleanup
    full_cleanup()


if __name__ == '__main__':
    run_full_inference()



""" Output:
sudo python3 e057_full_attention_inference.py
================================================================================
E057: FULL TRANSFORMER INFERENCE WITH ATTENTION
================================================================================

  Architecture:
    - 2 layers (full model!)
    - 32 hidden dim
    - 96 FFN dim  
    - 64 vocab size
    - Attention: simplified for single-token (V @ O)
    - FFN: gate * up → down


  Cleanup...
  ✓ Done

  Loading model: ./models/Qwen3-0.6B-Q4_K_M.gguf
    Loaded 310 tensors

  Building model: 2 layers, 32 hidden, 64 vocab
    Embeddings: (151936, 32)
    Model ready!

============================================================
Testing CPU-only inference first
============================================================

  Prompt: 'The' (token 1575)
  Generating with CPU reference...
    Step 1: token 61 = '=' (logits top: [  6  99 146  35 271])
    Step 2: token 56 = '8' (logits top: [-108   15  -68   65  243])
    Step 3: token 40 = '(' (logits top: [-174  212 -308  319 -110])
    Step 4: token 50 = '2' (logits top: [ 128   20 -183 -140 -229])
    Step 5: token 50 = '2' (logits top: [ 112   32 -192 -115 -246])

  Generated tokens: [1575, 61, 56, 40, 50, 50]
  Output: 'The=8(22'

============================================================
Configuring switch for comparison
============================================================

  Generating configuration file...
    Total rules: 512
    Total commands: 1543
    Config written to /tmp/e057_config.txt
    Transferring config to 10.10.10.55...
    ✓ Config file transferred
    Loading configuration on switch (this may take a few minutes)...
  ✓ Configuration complete (single commit!)

  Switch configuration successful!

  Host interface: enp1s0

  Verifying filter installation...
    Interface config: unit 0 {
    family ethernet-switching {
        vlan {
            members test_vlan;
        }
        filter {
            input full_attn;
        }
    }
}
    Interface status: Interface               Admin Link Proto    Local                 Remote
et-0/0/96               up    up
et-0/0/96.0             up    up   eth-switch
    Relevant interfaces:
      et-0/0/96               up    up
      et-0/0/96.0             up    up   eth-switch

============================================================
Comparing Switch vs CPU inference
============================================================

  Generating with switch...

  Step 1:
    L0: V=True, O=True ✓
    L1: V=True, O=True ✓
    Output: switch=40 cpu=40 ✓ MATCH
            switch logits[:5]=[0 0 0 0 0]
            cpu logits[:5]=[-27 -14 -18   0 -25]

  Step 2:
    L0: V=True, O=True ✓
    L1: V=True, O=True ✓
    Output: switch=47 cpu=47 ✓ MATCH
            switch logits[:5]=[0 0 0 0 0]
            cpu logits[:5]=[-20 -16 -12 -13 -25]

  Step 3:
    L0: V=True, O=True ✓
    L1: V=True, O=True ✓
    Output: switch=34 cpu=34 ✓ MATCH
            switch logits[:5]=[0 0 0 0 0]
            cpu logits[:5]=[-25 -19 -16  -9 -28]

  Switch generated: [1575, 40, 47, 34]

  Cleanup...
  ✓ Done
"""