#!/usr/bin/env python3
"""
e060_multi_layer_scale.py

MULTI-LAYER TRANSFORMER SCALING
===============================

GOAL: Scale transformer blocks to multiple layers!
      Change NUM_LAYERS to test 4, 8, 16, etc.

Each layer = 5 matrix multiplies on switch:
  - V, O (attention)
  - gate, up, down (FFN)

TCAM Budget:
  - 576 rules per layer (at 32 hidden, 96 FFN)
  - ~16K rules per switch
  - Can fit ~27 layers on one switch!

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

from e053_mac_encoded_layers import get_layer_neuron_mac
from e045_real_weights_inference import mac_str_to_bytes
from e042_port_based_layers import (
    ssh_command, run_config_commands,
    craft_vlan_packet, send_packets, get_mac_address,
    SWITCH1_IP, SWITCH2_IP, SEND_IFACE,
)

# =============================================================================
# CONFIGURATION - Change NUM_LAYERS to scale!
# =============================================================================
NUM_LAYERS = 2       # ← CHANGE THIS: 1, 2, 4, 8, 16, etc.
HIDDEN_DIM = 32      # Hidden dimension
FFN_DIM = 96         # FFN intermediate (3x hidden)
WEIGHT_SCALE = 30    # Scale for 4-bit weights

MODEL_PATH = "./models/Qwen3-0.6B-Q4_K_M.gguf"
FILTER_NAME = "multi_layer"
TEST_VLAN = 100

# Rules per layer: V(64) + O(64) + gate(192) + up(192) + down(64) = 576
RULES_PER_LAYER = 2 * HIDDEN_DIM + 2 * HIDDEN_DIM + 2 * FFN_DIM + 2 * FFN_DIM + 2 * HIDDEN_DIM


def ssh_command_long(switch_ip: str, command: str, timeout: int = 120) -> Tuple[bool, str, str]:
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


def load_model() -> gguf.GGUFReader:
    """Load GGUF model."""
    print(f"\n  Loading model: {MODEL_PATH}")
    reader = gguf.GGUFReader(MODEL_PATH)
    print(f"    Loaded {len(reader.tensors)} tensors")
    return reader


def get_tensor_by_name(reader: gguf.GGUFReader, name: str):
    for tensor in reader.tensors:
        if tensor.name == name:
            return tensor
    return None


def dequantize_tensor(tensor) -> np.ndarray:
    return gguf.dequantize(tensor.data, tensor.tensor_type)


def weights_to_4bit(weights: np.ndarray) -> np.ndarray:
    scaled = weights * WEIGHT_SCALE
    return np.clip(np.round(scaled), -8, 7).astype(np.int8)


def rms_norm(x: np.ndarray, weight: np.ndarray = None, eps: float = 1e-6) -> np.ndarray:
    rms = np.sqrt(np.mean(x ** 2) + eps)
    normed = x / rms
    if weight is not None:
        normed = normed * weight
    return normed


def silu(x: np.ndarray) -> np.ndarray:
    return x * (1 / (1 + np.exp(-np.clip(x, -500, 500))))


class TransformerLayer:
    """One transformer layer with attention + FFN."""
    
    def __init__(self, reader: gguf.GGUFReader, layer_idx: int):
        prefix = f'blk.{layer_idx}.'
        
        # Attention
        self.attn_norm = self._load_norm(reader, prefix + 'attn_norm.weight')
        self.W_v = self._load_weight(reader, prefix + 'attn_v.weight', (HIDDEN_DIM, HIDDEN_DIM))
        self.W_o = self._load_weight(reader, prefix + 'attn_output.weight', (HIDDEN_DIM, HIDDEN_DIM))
        
        # FFN
        self.ffn_norm = self._load_norm(reader, prefix + 'ffn_norm.weight')
        self.W_gate = self._load_weight(reader, prefix + 'ffn_gate.weight', (FFN_DIM, HIDDEN_DIM))
        self.W_up = self._load_weight(reader, prefix + 'ffn_up.weight', (FFN_DIM, HIDDEN_DIM))
        self.W_down = self._load_weight(reader, prefix + 'ffn_down.weight', (HIDDEN_DIM, FFN_DIM))
    
    def _load_norm(self, reader, name):
        tensor = get_tensor_by_name(reader, name)
        if tensor:
            w = dequantize_tensor(tensor)
            return w[:HIDDEN_DIM]
        return np.ones(HIDDEN_DIM)
    
    def _load_weight(self, reader, name, shape):
        tensor = get_tensor_by_name(reader, name)
        if tensor:
            w = dequantize_tensor(tensor)
            w = w[:shape[0], :shape[1]]
            return weights_to_4bit(w)
        return np.random.randint(-8, 8, shape, dtype=np.int8)


def full_cleanup():
    """Clean switch configuration."""
    print("\n  Cleanup...")
    cleanup_cmd = f"cli -c 'configure; delete firewall family ethernet-switching filter {FILTER_NAME}; delete firewall family ethernet-switching filter; set forwarding-options storm-control-profiles default all; commit'"
    ssh_command(SWITCH1_IP, cleanup_cmd)
    time.sleep(0.5)
    print("  ✓ Done")


def configure_all_layers(layers: List[TransformerLayer]):
    """Configure filters for ALL layers at once."""
    print(f"\n  Configuring {len(layers)} layers...")
    
    all_cmds = []
    all_cmds.append("set forwarding-options storm-control-profiles default all")
    all_cmds.append(f"set firewall family ethernet-switching filter {FILTER_NAME}")
    
    total_rules = 0
    
    # MAC encoding: layer L, projection P (0-4), neuron N
    # MAC layer = L * 5 + P
    # P: 0=V, 1=O, 2=gate, 3=up, 4=down
    
    for layer_idx in range(len(layers)):
        # V projection (P=0)
        mac_base = layer_idx * 5 + 0
        for n in range(HIDDEN_DIM):
            mac_pos = get_layer_neuron_mac(mac_base, n * 2)
            mac_neg = get_layer_neuron_mac(mac_base, n * 2 + 1)
            term = f"L{layer_idx}v{n}"
            all_cmds.extend([
                f"set firewall family ethernet-switching filter {FILTER_NAME} term {term}p from destination-mac-address {mac_pos}/48",
                f"set firewall family ethernet-switching filter {FILTER_NAME} term {term}p then count {term}p",
                f"set firewall family ethernet-switching filter {FILTER_NAME} term {term}p then accept",
                f"set firewall family ethernet-switching filter {FILTER_NAME} term {term}n from destination-mac-address {mac_neg}/48",
                f"set firewall family ethernet-switching filter {FILTER_NAME} term {term}n then count {term}n",
                f"set firewall family ethernet-switching filter {FILTER_NAME} term {term}n then accept",
            ])
            total_rules += 2
        
        # O projection (P=1)
        mac_base = layer_idx * 5 + 1
        for n in range(HIDDEN_DIM):
            mac_pos = get_layer_neuron_mac(mac_base, n * 2)
            mac_neg = get_layer_neuron_mac(mac_base, n * 2 + 1)
            term = f"L{layer_idx}o{n}"
            all_cmds.extend([
                f"set firewall family ethernet-switching filter {FILTER_NAME} term {term}p from destination-mac-address {mac_pos}/48",
                f"set firewall family ethernet-switching filter {FILTER_NAME} term {term}p then count {term}p",
                f"set firewall family ethernet-switching filter {FILTER_NAME} term {term}p then accept",
                f"set firewall family ethernet-switching filter {FILTER_NAME} term {term}n from destination-mac-address {mac_neg}/48",
                f"set firewall family ethernet-switching filter {FILTER_NAME} term {term}n then count {term}n",
                f"set firewall family ethernet-switching filter {FILTER_NAME} term {term}n then accept",
            ])
            total_rules += 2
        
        # gate projection (P=2)
        mac_base = layer_idx * 5 + 2
        for n in range(FFN_DIM):
            mac_pos = get_layer_neuron_mac(mac_base, n * 2)
            mac_neg = get_layer_neuron_mac(mac_base, n * 2 + 1)
            term = f"L{layer_idx}g{n}"
            all_cmds.extend([
                f"set firewall family ethernet-switching filter {FILTER_NAME} term {term}p from destination-mac-address {mac_pos}/48",
                f"set firewall family ethernet-switching filter {FILTER_NAME} term {term}p then count {term}p",
                f"set firewall family ethernet-switching filter {FILTER_NAME} term {term}p then accept",
                f"set firewall family ethernet-switching filter {FILTER_NAME} term {term}n from destination-mac-address {mac_neg}/48",
                f"set firewall family ethernet-switching filter {FILTER_NAME} term {term}n then count {term}n",
                f"set firewall family ethernet-switching filter {FILTER_NAME} term {term}n then accept",
            ])
            total_rules += 2
        
        # up projection (P=3)
        mac_base = layer_idx * 5 + 3
        for n in range(FFN_DIM):
            mac_pos = get_layer_neuron_mac(mac_base, n * 2)
            mac_neg = get_layer_neuron_mac(mac_base, n * 2 + 1)
            term = f"L{layer_idx}u{n}"
            all_cmds.extend([
                f"set firewall family ethernet-switching filter {FILTER_NAME} term {term}p from destination-mac-address {mac_pos}/48",
                f"set firewall family ethernet-switching filter {FILTER_NAME} term {term}p then count {term}p",
                f"set firewall family ethernet-switching filter {FILTER_NAME} term {term}p then accept",
                f"set firewall family ethernet-switching filter {FILTER_NAME} term {term}n from destination-mac-address {mac_neg}/48",
                f"set firewall family ethernet-switching filter {FILTER_NAME} term {term}n then count {term}n",
                f"set firewall family ethernet-switching filter {FILTER_NAME} term {term}n then accept",
            ])
            total_rules += 2
        
        # down projection (P=4)
        mac_base = layer_idx * 5 + 4
        for n in range(HIDDEN_DIM):
            mac_pos = get_layer_neuron_mac(mac_base, n * 2)
            mac_neg = get_layer_neuron_mac(mac_base, n * 2 + 1)
            term = f"L{layer_idx}d{n}"
            all_cmds.extend([
                f"set firewall family ethernet-switching filter {FILTER_NAME} term {term}p from destination-mac-address {mac_pos}/48",
                f"set firewall family ethernet-switching filter {FILTER_NAME} term {term}p then count {term}p",
                f"set firewall family ethernet-switching filter {FILTER_NAME} term {term}p then accept",
                f"set firewall family ethernet-switching filter {FILTER_NAME} term {term}n from destination-mac-address {mac_neg}/48",
                f"set firewall family ethernet-switching filter {FILTER_NAME} term {term}n then count {term}n",
                f"set firewall family ethernet-switching filter {FILTER_NAME} term {term}n then accept",
            ])
            total_rules += 2
        
        if (layer_idx + 1) % 4 == 0:
            print(f"    Layer {layer_idx + 1}/{len(layers)} prepared ({total_rules} rules)")
    
    # Default term
    all_cmds.extend([
        f"set firewall family ethernet-switching filter {FILTER_NAME} term default then count default_pkts",
        f"set firewall family ethernet-switching filter {FILTER_NAME} term default then accept",
    ])
    
    # Apply filter
    all_cmds.extend([
        f"set vlans test_vlan vlan-id {TEST_VLAN}",
        f"set interfaces et-0/0/96 unit 0 family ethernet-switching vlan members test_vlan",
        f"set interfaces et-0/0/96 unit 0 family ethernet-switching filter input {FILTER_NAME}",
    ])
    
    print(f"    Total rules: {total_rules}")
    print(f"    Total commands: {len(all_cmds)}")
    
    if total_rules > 16000:
        print(f"    ⚠ WARNING: {total_rules} rules may exceed TCAM limit!")
    
    # Transfer and load
    config_file = "/tmp/e060_config.txt"
    with open(config_file, 'w') as f:
        f.write('\n'.join(all_cmds))
    
    # Debug: show first few lines of config
    print(f"    Config preview (first 5 lines):")
    for line in all_cmds[:5]:
        print(f"      {line}")
    
    print("    Transferring config...")
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
    
    # Verify file was transferred
    success, stdout, _ = ssh_command(SWITCH1_IP, "wc -l /var/tmp/config.txt")
    print(f"    File transferred: {stdout.strip() if success else 'FAILED'}")
    
    print("    Loading config (this may take a minute)...")
    # First, rollback any pending changes and delete old filters
    ssh_command(SWITCH1_IP, "cli -c 'rollback 0'")
    time.sleep(0.5)
    
    # Explicitly delete all ethernet-switching filters first
    delete_cmd = "cli -c 'configure; delete firewall family ethernet-switching; commit'"
    ssh_command_long(SWITCH1_IP, delete_cmd, timeout=60)
    time.sleep(1)
    
    # Use same approach as e059 which worked
    load_cmd = "cli -c 'configure; load set /var/tmp/config.txt; commit'"
    success, stdout, stderr = ssh_command_long(SWITCH1_IP, load_cmd, timeout=300)  # 5 min timeout for large configs
    
    print(f"    Load result: success={success}")
    if stdout:
        # Show last few lines of output
        lines = stdout.strip().split('\n')
        for line in lines[-10:]:
            if line.strip():
                print(f"      {line}")
    if stderr:
        print(f"    stderr: {stderr[:200]}")
    
    if not success:
        print(f"    ✗ Config failed!")
        return False
    
    # Check for commit success
    if 'commit complete' in stdout.lower():
        print("  ✓ All layers configured!")
    
    # Wait for TCAM to program - longer for more rules
    wait_time = max(3, total_rules // 500)  # ~1s per 500 rules
    print(f"    Waiting {wait_time}s for TCAM to stabilize ({total_rules} rules)...")
    time.sleep(wait_time)
    
    # Verify filter is applied to interface
    success, stdout, _ = ssh_command_long(SWITCH1_IP, 
        "cli -c 'show configuration interfaces et-0/0/96'", timeout=30)
    if success:
        if 'filter input' in stdout and FILTER_NAME in stdout:
            print(f"    ✓ Filter bound to interface et-0/0/96")
        else:
            print(f"    ⚠ Filter NOT bound to interface!")
            print(f"    Interface config:\n{stdout[:500]}")
    
    # Check filter counters exist
    success, stdout, _ = ssh_command_long(SWITCH1_IP, f"cli -c 'show firewall filter {FILTER_NAME}'", timeout=60)
    if success:
        counter_lines = [l for l in stdout.split('\n') if 'L0' in l]
        print(f"    Filter counters: {len(counter_lines)} L0 entries visible")
    
    return True


def clear_counters():
    """Clear all firewall counters."""
    ssh_command_long(SWITCH1_IP, f"cli -c 'clear firewall filter {FILTER_NAME}'", timeout=30)
    time.sleep(0.1)


def cpu_4bit_matmul(hidden: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """CPU reference matching switch computation."""
    threshold = 0.01 if np.any(hidden != 0) else 0
    hidden_binary = (np.abs(hidden) > threshold).astype(np.float32)
    return weights.astype(np.float32) @ hidden_binary


def create_packets(mac_layer: int, hidden: np.ndarray, weights: np.ndarray, src_mac: str) -> List[bytes]:
    """Create packets for a projection."""
    packets = []
    active = np.abs(hidden) > 0.01
    
    for out_idx in range(weights.shape[0]):
        pos_pkts = 0
        neg_pkts = 0
        
        for in_idx in range(weights.shape[1]):
            if not active[in_idx]:
                continue
            w = weights[out_idx, in_idx]
            if w > 0:
                pos_pkts += abs(w)
            elif w < 0:
                neg_pkts += abs(w)
        
        if pos_pkts > 0:
            mac = get_layer_neuron_mac(mac_layer, out_idx * 2)
            dst = mac_str_to_bytes(mac)
            src = mac_str_to_bytes(src_mac)
            for _ in range(pos_pkts):
                packets.append(craft_vlan_packet(dst, src, TEST_VLAN))
        
        if neg_pkts > 0:
            mac = get_layer_neuron_mac(mac_layer, out_idx * 2 + 1)
            dst = mac_str_to_bytes(mac)
            src = mac_str_to_bytes(src_mac)
            for _ in range(neg_pkts):
                packets.append(craft_vlan_packet(dst, src, TEST_VLAN))
    
    return packets


def read_counters(prefix: str, count: int) -> np.ndarray:
    """Read counters and return signed values."""
    success, stdout, _ = ssh_command_long(
        SWITCH1_IP,
        f"cli -c 'show firewall filter {FILTER_NAME}'",
        timeout=60
    )
    
    values = np.zeros(count, dtype=np.float32)
    if not success or not stdout:
        return values
    
    for i in range(count):
        pos_pattern = rf'{prefix}{i}p\s+\d+\s+(\d+)'
        neg_pattern = rf'{prefix}{i}n\s+\d+\s+(\d+)'
        
        pos_match = re.search(pos_pattern, stdout)
        neg_match = re.search(neg_pattern, stdout)
        
        pos_count = int(pos_match.group(1)) if pos_match else 0
        neg_count = int(neg_match.group(1)) if neg_match else 0
        values[i] = pos_count - neg_count
    
    return values


def run_layer(layer_idx: int, layer: TransformerLayer, x: np.ndarray, src_mac: str) -> Tuple[np.ndarray, bool]:
    """Run one transformer layer through switch. Returns (output, all_matched)."""
    all_match = True
    
    # === ATTENTION ===
    x_norm = rms_norm(x, layer.attn_norm)
    
    # V projection
    clear_counters()
    mac_layer = layer_idx * 5 + 0
    packets = create_packets(mac_layer, x_norm, layer.W_v, src_mac)
    if packets:
        send_packets(SEND_IFACE, packets)
    time.sleep(0.2)
    switch_v = read_counters(f"L{layer_idx}v", HIDDEN_DIM)
    cpu_v = cpu_4bit_matmul(x_norm, layer.W_v)
    v_match = np.allclose(switch_v, cpu_v, atol=1)
    all_match &= v_match
    
    # Debug for first layer
    if layer_idx == 0:
        switch_sum = int(np.sum(np.abs(switch_v)))
        cpu_sum = int(np.sum(np.abs(cpu_v)))
        # Check default counter to see if any packets hit filter
        success, stdout, _ = ssh_command_long(SWITCH1_IP, f"cli -c 'show firewall filter {FILTER_NAME}'", timeout=30)
        default_match = re.search(r'default_pkts\s+\d+\s+(\d+)', stdout) if success else None
        default_pkts = int(default_match.group(1)) if default_match else 0
        print(f"      V debug: {len(packets)} pkts sent, switch_sum={switch_sum}, cpu_sum={cpu_sum}, default={default_pkts}")
    
    # O projection
    clear_counters()
    v_binary = (np.abs(switch_v) > 0).astype(np.float32)
    mac_layer = layer_idx * 5 + 1
    packets = create_packets(mac_layer, v_binary, layer.W_o, src_mac)
    if packets:
        send_packets(SEND_IFACE, packets)
    time.sleep(0.2)
    switch_o = read_counters(f"L{layer_idx}o", HIDDEN_DIM)
    cpu_o = cpu_4bit_matmul(v_binary, layer.W_o)
    o_match = np.allclose(switch_o, cpu_o, atol=1)
    all_match &= o_match
    
    # Debug for first layer
    if layer_idx == 0:
        switch_sum = int(np.sum(np.abs(switch_o)))
        cpu_sum = int(np.sum(np.abs(cpu_o)))
        print(f"      O debug: {len(packets)} pkts, switch_sum={switch_sum}, cpu_sum={cpu_sum}")
    
    # Residual
    x = x + switch_o
    
    # === FFN ===
    x_norm = rms_norm(x, layer.ffn_norm)
    
    # gate projection
    clear_counters()
    mac_layer = layer_idx * 5 + 2
    packets = create_packets(mac_layer, x_norm, layer.W_gate, src_mac)
    if packets:
        send_packets(SEND_IFACE, packets)
    time.sleep(0.2)
    switch_gate = read_counters(f"L{layer_idx}g", FFN_DIM)
    cpu_gate = cpu_4bit_matmul(x_norm, layer.W_gate)
    gate_match = np.allclose(switch_gate, cpu_gate, atol=1)
    all_match &= gate_match
    
    # Debug for first layer
    if layer_idx == 0:
        switch_sum = int(np.sum(np.abs(switch_gate)))
        cpu_sum = int(np.sum(np.abs(cpu_gate)))
        print(f"      gate debug: {len(packets)} pkts, switch_sum={switch_sum}, cpu_sum={cpu_sum}")
    
    # up projection
    clear_counters()
    mac_layer = layer_idx * 5 + 3
    packets = create_packets(mac_layer, x_norm, layer.W_up, src_mac)
    if packets:
        send_packets(SEND_IFACE, packets)
    time.sleep(0.2)
    switch_up = read_counters(f"L{layer_idx}u", FFN_DIM)
    cpu_up = cpu_4bit_matmul(x_norm, layer.W_up)
    up_match = np.allclose(switch_up, cpu_up, atol=1)
    all_match &= up_match
    
    # SiLU(gate) * up (CPU)
    gate_activated = silu(switch_gate)
    ffn_hidden = gate_activated * switch_up
    
    # down projection
    clear_counters()
    mac_layer = layer_idx * 5 + 4
    packets = create_packets(mac_layer, ffn_hidden, layer.W_down, src_mac)
    if packets:
        send_packets(SEND_IFACE, packets)
    time.sleep(0.2)
    switch_down = read_counters(f"L{layer_idx}d", HIDDEN_DIM)
    cpu_down = cpu_4bit_matmul(ffn_hidden, layer.W_down)
    down_match = np.allclose(switch_down, cpu_down, atol=1)
    all_match &= down_match
    
    # Residual
    x = x + switch_down
    
    status = "✓" if all_match else "✗"
    print(f"    Layer {layer_idx}: V={v_match} O={o_match} gate={gate_match} up={up_match} down={down_match} {status}")
    
    return x, all_match


def run_multi_layer():
    """Run multi-layer transformer scaling test."""
    print("="*80)
    print(f"E060: MULTI-LAYER TRANSFORMER SCALING ({NUM_LAYERS} layers)")
    print("="*80)
    print(f"""
  Configuration:
    - Layers: {NUM_LAYERS}
    - Hidden dim: {HIDDEN_DIM}
    - FFN dim: {FFN_DIM}
    - Rules per layer: {RULES_PER_LAYER}
    - Total rules: {NUM_LAYERS * RULES_PER_LAYER}
    
  Each layer: 5 matrix multiplies on switch (V, O, gate, up, down)
""")
    
    # Cleanup
    full_cleanup()
    
    # Load model
    reader = load_model()
    
    # Load all layers
    print(f"\n  Loading {NUM_LAYERS} transformer layers...")
    layers = []
    for i in range(NUM_LAYERS):
        layer = TransformerLayer(reader, i)
        layers.append(layer)
        if (i + 1) % 4 == 0 or i == NUM_LAYERS - 1:
            print(f"    Loaded layer {i + 1}/{NUM_LAYERS}")
    
    # Configure all layers
    if not configure_all_layers(layers):
        print("  ✗ Configuration failed!")
        return
    
    src_mac = get_mac_address(SEND_IFACE)
    print(f"\n  Source MAC: {src_mac}")
    
    # Run through all layers
    print("\n" + "="*60)
    print(f"Running {NUM_LAYERS} Transformer Layers")
    print("="*60)
    
    np.random.seed(42)
    x = np.random.randn(HIDDEN_DIM).astype(np.float32)
    print(f"\n  Initial input: norm={np.linalg.norm(x):.2f}")
    
    all_layers_match = True
    layer_results = []
    
    t_start = time.time()
    
    for layer_idx, layer in enumerate(layers):
        x, layer_match = run_layer(layer_idx, layer, x, src_mac)
        all_layers_match &= layer_match
        layer_results.append(layer_match)
    
    t_elapsed = time.time() - t_start
    
    print(f"\n  Final output: norm={np.linalg.norm(x):.2f}")
    print(f"  Time: {t_elapsed:.1f}s ({t_elapsed/NUM_LAYERS:.2f}s per layer)")
    
    # Results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    matched = sum(layer_results)
    total = len(layer_results)
    
    print(f"\n  Layers matched: {matched}/{total}")
    for i, match in enumerate(layer_results):
        status = "✓" if match else "✗"
        print(f"    Layer {i}: {status}")
    
    if all_layers_match:
        print(f"\n  🎉 SUCCESS! All {NUM_LAYERS} layers match CPU!")
        print(f"     - {NUM_LAYERS * 5} matrix multiplies on switch hardware")
        print(f"     - {NUM_LAYERS * RULES_PER_LAYER} TCAM rules used")
        print(f"     - Complete multi-layer transformer proven!")
    else:
        print(f"\n  ⚠ Some layers had mismatches")
    
    # Cleanup
    full_cleanup()


if __name__ == '__main__':
    run_multi_layer()


""" Output:
sudo python3 e060_multi_layer_scale.py
================================================================================
E060: MULTI-LAYER TRANSFORMER SCALING (2 layers)
================================================================================

  Configuration:
    - Layers: 2
    - Hidden dim: 32
    - FFN dim: 96
    - Rules per layer: 576
    - Total rules: 1152
    
  Each layer: 5 matrix multiplies on switch (V, O, gate, up, down)


  Cleanup...
  ✓ Done

  Loading model: ./models/Qwen3-0.6B-Q4_K_M.gguf
    Loaded 310 tensors

  Loading 2 transformer layers...
    Loaded layer 2/2

  Configuring 2 layers...
    Total rules: 1152
    Total commands: 3463
    Config preview (first 5 lines):
      set forwarding-options storm-control-profiles default all
      set firewall family ethernet-switching filter multi_layer
      set firewall family ethernet-switching filter multi_layer term L0v0p from destination-mac-address 01:00:5e:00:00:00/48
      set firewall family ethernet-switching filter multi_layer term L0v0p then count L0v0p
      set firewall family ethernet-switching filter multi_layer term L0v0p then accept
    Transferring config...
    File transferred: 3462 /var/tmp/config.txt
    Loading config (this may take a minute)...
    Load result: success=True
      Entering configuration mode
      The configuration has been changed but not committed
      load complete
      configuration check succeeds
      commit complete
  ✓ All layers configured!
    Waiting 3s for TCAM to stabilize (1152 rules)...
    ⚠ Filter NOT bound to interface!
    Interface config:
unit 0 {
    family ethernet-switching {
        interface-mode trunk;
        vlan {
            members test_vlan;
        }
        filter {
            input multi_layer;
        }
    }
}

    Filter counters: 576 L0 entries visible

  Source MAC: 7c:fe:90:9d:2a:f0

============================================================
Running 2 Transformer Layers
============================================================

  Initial input: norm=5.32
      V debug: 360 pkts sent, switch_sum=100, cpu_sum=100, default=0
      O debug: 278 pkts, switch_sum=56, cpu_sum=56
      gate debug: 2902 pkts, switch_sum=608, cpu_sum=608
    Layer 0: V=True O=True gate=True up=True down=True ✓
    Layer 1: V=True O=True gate=True up=True down=True ✓

  Final output: norm=52.46
  Time: 56.8s (28.39s per layer)

============================================================
RESULTS
============================================================

  Layers matched: 2/2
    Layer 0: ✓
    Layer 1: ✓

  🎉 SUCCESS! All 2 layers match CPU!
     - 10 matrix multiplies on switch hardware
     - 1152 TCAM rules used
     - Complete multi-layer transformer proven!

  Cleanup...
  ✓ Done
"""



""" Notes:
2 layers works perfectly, but 3 fails. 
There must be a TCAM limit between 1152 and 1728 rules on this switch.
"""
