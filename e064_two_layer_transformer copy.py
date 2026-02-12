#!/usr/bin/env python3
"""
e064_two_layer_transformer.py

Demonstrate two transformer layers (attention + FFN) running through
VLAN-sharded filters. Each filter holds only 16 neurons (two shards)
so the per-filter term count stays well below the ~1,500 limit.
We reuse the MAC encoding, packet crafting, and GGUF parsing logic
from earlier experiments.

Steps:
  1. Configure VLAN-sharded filters for layers 0-1 with positive/negative MAC terms.
  2. Load the first two layers from the Qwen3-0.6B GGUF.
  3. For each projection (V, O, gate, up, down) and layer, send packets.
  4. After each projection, read positive/negative counters and compare to CPU counts.
  5. The test proves the full transformer block works while keeping each filter within budget.
"""

import re
import subprocess
import time
from typing import Dict, List, Tuple

import gguf
import numpy as np

from e042_port_based_layers import (
    craft_vlan_packet,
    cleanup_switch,
    get_mac_address,
    run_config_commands,
    send_packets,
    ssh_command,
    SWITCH1_IP,
    SEND_IFACE,
)
from e053_mac_encoded_layers import get_layer_neuron_mac
from e045_real_weights_inference import mac_str_to_bytes
from e063_layer_to_vlan_sharding import counter_name


NUM_LAYERS = 8
HIDDEN_DIM = 32
FFN_DIM = 96
SHARDS_PER_LAYER = 2
NEURONS_PER_LAYER = max(HIDDEN_DIM, FFN_DIM)
NEURONS_PER_SHARD = NEURONS_PER_LAYER // SHARDS_PER_LAYER
BASE_VLAN = 400
WEIGHT_SCALE = 30
MODEL_PATH = "./models/Qwen3-0.6B-Q4_K_M.gguf"

INTERFACE = "et-0/0/96"
DEFAULT_FILTER = "transformer_layer_filter"


class TransformerLayer:
    """Load weights for a transformer layer (same as e060)."""

    def __init__(self, reader: gguf.GGUFReader, layer_idx: int):
        prefix = f"blk.{layer_idx}."
        self.attn_norm = self._load_norm(reader, prefix + "attn_norm.weight")
        self.W_v = self._load_weight(reader, prefix + "attn_v.weight", (HIDDEN_DIM, HIDDEN_DIM))
        self.W_o = self._load_weight(reader, prefix + "attn_output.weight", (HIDDEN_DIM, HIDDEN_DIM))

        self.ffn_norm = self._load_norm(reader, prefix + "ffn_norm.weight")
        self.W_gate = self._load_weight(reader, prefix + "ffn_gate.weight", (FFN_DIM, HIDDEN_DIM))
        self.W_up = self._load_weight(reader, prefix + "ffn_up.weight", (FFN_DIM, HIDDEN_DIM))
        self.W_down = self._load_weight(reader, prefix + "ffn_down.weight", (HIDDEN_DIM, FFN_DIM))

    def _load_norm(self, reader, name: str) -> np.ndarray:
        tensor = self._find_tensor(reader, name)
        if tensor is not None:
            return self._dequantize(tensor)[:HIDDEN_DIM]
        return np.ones(HIDDEN_DIM, dtype=np.float32)

    def _load_weight(self, reader, name: str, shape: Tuple[int, int]) -> np.ndarray:
        tensor = self._find_tensor(reader, name)
        if tensor is not None:
            weights = self._dequantize(tensor)
            return weights[: shape[0], : shape[1]]
        return np.random.randint(-8, 8, shape, dtype=np.int8)

    @staticmethod
    def _find_tensor(reader: gguf.GGUFReader, name: str):
        for tensor in reader.tensors:
            if tensor.name == name:
                return tensor
        return None

    @staticmethod
    @staticmethod
    def _dequantize(tensor):
        return gguf.dequantize(tensor.data, tensor.tensor_type)


def weights_to_4bit(weights: np.ndarray) -> np.ndarray:
    scaled = weights * WEIGHT_SCALE
    return np.clip(np.round(scaled), -8, 7).astype(np.int8)


def rms_norm(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(x**2) + 1e-6)
    return (x / rms) * weight


def silu(x: np.ndarray) -> np.ndarray:
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -50, 50))))


def cpu_4bit_matmul(hidden: np.ndarray, weights: np.ndarray) -> np.ndarray:
    threshold = 0.01 if np.any(hidden != 0) else 0
    hidden_binary = (np.abs(hidden) > threshold).astype(np.float32)
    return weights.astype(np.float32) @ hidden_binary


def configure_transformer_filters():
    """Build commands for VLAN-sharded transformer filters."""
    commands: List[str] = []
    base_vlan = "layer0_shard0_vlan"
    commands.append(f"delete interfaces {INTERFACE} unit 0 family ethernet-switching")
    commands.append(f"set vlans {base_vlan} vlan-id {BASE_VLAN}")
    commands.append(f"set interfaces {INTERFACE} unit 0 family ethernet-switching interface-mode trunk")
    commands.append(f"set interfaces {INTERFACE} unit 0 family ethernet-switching vlan members {base_vlan}")
    commands.append("set interfaces et-0/0/100 unit 0 family ethernet-switching interface-mode trunk")
    commands.append(f"set interfaces et-0/0/100 unit 0 family ethernet-switching vlan members {base_vlan}")

    for layer in range(NUM_LAYERS):
        for shard in range(SHARDS_PER_LAYER):
            vlan_name = f"layer{layer}_shard{shard}_vlan"
            vlan_id = BASE_VLAN + layer * SHARDS_PER_LAYER + shard
            filter_name = f"{DEFAULT_FILTER}_l{layer}_s{shard}"

            commands.append(f"delete firewall family ethernet-switching filter {filter_name}")
            commands.append(f"set vlans {vlan_name} vlan-id {vlan_id}")
            commands.append(f"set interfaces {INTERFACE} unit 0 family ethernet-switching vlan members {vlan_name}")

            neurons = range(shard * NEURONS_PER_SHARD, (shard + 1) * NEURONS_PER_SHARD)
            for neuron in neurons:
                mac_base = layer * 5
                mac_pos = get_layer_neuron_mac(mac_base, neuron * 2)
                mac_neg = get_layer_neuron_mac(mac_base, neuron * 2 + 1)
                term_p = f"l{layer}s{shard}n{neuron}p"
                term_n = f"l{layer}s{shard}n{neuron}n"
                commands.extend([
                    f"set firewall family ethernet-switching filter {filter_name} term {term_p} from destination-mac-address {mac_pos}",
                    f"set firewall family ethernet-switching filter {filter_name} term {term_p} then count {term_p}",
                    f"set firewall family ethernet-switching filter {filter_name} term {term_p} then accept",
                    f"set firewall family ethernet-switching filter {filter_name} term {term_n} from destination-mac-address {mac_neg}",
                    f"set firewall family ethernet-switching filter {filter_name} term {term_n} then count {term_n}",
                    f"set firewall family ethernet-switching filter {filter_name} term {term_n} then accept",
                ])

            commands.extend([
                f"set firewall family ethernet-switching filter {filter_name} term default then accept",
                f"set vlans {vlan_name} forwarding-options filter input {filter_name}",
            ])

    config_file = "/tmp/e064_transformer_config.txt"
    with open(config_file, "w") as f:
        f.write("\n".join(commands))

    ssh_key = "/home/multiplex/.ssh/id_rsa"
    transfer = [
        'ssh', '-i', ssh_key,
        '-o', 'StrictHostKeyChecking=no',
        '-o', 'UserKnownHostsFile=/dev/null',
        '-o', 'LogLevel=ERROR',
        f"root@{SWITCH1_IP}",
        'cat > /var/tmp/e064_transformer_config.txt'
    ]
    with open(config_file, 'rb') as f:
        subprocess.run(transfer, stdin=f, check=True)

    load_cmd = "cli -c 'configure; load set /var/tmp/e064_transformer_config.txt; commit'"
    ssh_command(SWITCH1_IP, load_cmd)


def load_model() -> gguf.GGUFReader:
    print(f"\nLoading model: {MODEL_PATH}")
    reader = gguf.GGUFReader(MODEL_PATH)
    print(f"  Loaded {len(reader.tensors)} tensors")
    return reader


def send_projection(layer_idx: int, weights: np.ndarray, hidden: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Send packets corresponding to one projection and return pos/neg counts."""
    weights_q = weights_to_4bit(weights)
    active = np.abs(hidden) > 0.01
    num_out = weights.shape[0]
    pos_counts = np.zeros(num_out, dtype=int)
    neg_counts = np.zeros(num_out, dtype=int)

    packets = []
    src = mac_str_to_bytes(get_mac_address(SEND_IFACE))

    for out_idx in range(num_out):
        shard = out_idx // NEURONS_PER_SHARD
        vlan = BASE_VLAN + layer_idx * SHARDS_PER_LAYER + shard
        mac_layer = layer_idx * 5
        mac_pos = mac_str_to_bytes(get_layer_neuron_mac(mac_layer, out_idx * 2))
        mac_neg = mac_str_to_bytes(get_layer_neuron_mac(mac_layer, out_idx * 2 + 1))

        for in_idx, active_flag in enumerate(active):
            if not active_flag:
                continue
            w = int(weights_q[out_idx, in_idx])
            if w > 0:
                pos_counts[out_idx] += abs(w)
            elif w < 0:
                neg_counts[out_idx] += abs(w)

        for _ in range(pos_counts[out_idx]):
            packets.append(craft_vlan_packet(mac_pos, src, vlan))
        for _ in range(neg_counts[out_idx]):
            packets.append(craft_vlan_packet(mac_neg, src, vlan))

    if packets:
        send_packets(SEND_IFACE, packets)
    return pos_counts, neg_counts


def clear_layer_counters(layer_idx: int):
    for shard in range(SHARDS_PER_LAYER):
        filter_name = f"{DEFAULT_FILTER}_l{layer_idx}_s{shard}"
        ssh_command(SWITCH1_IP, f"cli -c 'clear firewall filter {filter_name}'")


def read_layer_counts(layer_idx: int) -> Dict[int, Tuple[int, int]]:
    """Return {neuron: (pos_count, neg_count)} for a layer."""
    data: Dict[int, Tuple[int, int]] = {}
    for shard in range(SHARDS_PER_LAYER):
        filter_name = f"{DEFAULT_FILTER}_l{layer_idx}_s{shard}"
        success, stdout, _ = ssh_command(SWITCH1_IP, f"cli -c 'show firewall filter {filter_name}'")
        if not success:
            continue

        neurons = range(shard * NEURONS_PER_SHARD, (shard + 1) * NEURONS_PER_SHARD)
        for neuron in neurons:
            term_p = f"l{layer_idx}s{shard}n{neuron}p"
            term_n = f"l{layer_idx}s{shard}n{neuron}n"
            pos = 0
            neg = 0
            for line in stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith(term_p):
                    parts = line.split()
                    if parts and parts[-1].isdigit():
                        pos = int(parts[-1])
                elif line.startswith(term_n):
                    parts = line.split()
                    if parts and parts[-1].isdigit():
                        neg = int(parts[-1])
            data[neuron] = (pos, neg)
    return data


def verify_projection(layer_idx: int, projection: str, weights: np.ndarray, hidden: np.ndarray):
    pos_counts, neg_counts = send_projection(layer_idx, weights, hidden)
    time.sleep(0.5)
    actual = read_layer_counts(layer_idx)
    print(f"\nLayer {layer_idx} projection {projection} verification:")
    all_good = True
    for neuron in range(weights.shape[0]):
        expected_pos = pos_counts[neuron]
        expected_neg = neg_counts[neuron]
        actual_pos, actual_neg = actual.get(neuron, (0, 0))
        ok = expected_pos == actual_pos and expected_neg == actual_neg
        status = "✓" if ok else "✗"
        print(f"  Neuron {neuron}: pos {actual_pos}/{expected_pos} neg {actual_neg}/{expected_neg} {status}")
        if not ok:
            all_good = False
    clear_layer_counters(layer_idx)
    return all_good


def summarize_results(results: List[Tuple[int, str, bool]], success_layers: int, total_layers: int):
    print("\nSUMMARY")
    print("=" * 80)
    failures = [f"Layer {layer} projection {name}" for layer, name, ok in results if not ok]
    if failures:
        print("  STATUS: FAIL")
        for fail in failures:
            print(f"    {fail}")
    else:
        print("  STATUS: PASS – all projections matched the CPU reference.")
    print(f"  Layers verified: {success_layers}/{total_layers}")


def run():
    print("=" * 80)
    print("E064: Two-Layer Transformer (VLAN Sharded Filters)")
    print("=" * 80)

    print("\nSTEP 1: Cleanup")
    cleanup_switch(SWITCH1_IP)
    time.sleep(1)

    print("\nSTEP 2: Configure VLAN filters")
    configure_transformer_filters()
    time.sleep(1)

    reader = load_model()
    layers = [TransformerLayer(reader, idx) for idx in range(NUM_LAYERS)]

    # Random input vector
    x = np.random.uniform(-1, 1, HIDDEN_DIM).astype(np.float32)

    results: List[Tuple[int, str, bool]] = []
    layer_results = {idx: True for idx in range(NUM_LAYERS)}
    for idx, layer in enumerate(layers):
        print(f"\n=== Layer {idx} ===")
        attn_in = rms_norm(x, layer.attn_norm)
        v = cpu_4bit_matmul(attn_in, layer.W_v)
        ok = verify_projection(idx, "V", layer.W_v, attn_in)
        results.append((idx, "V", ok))
        if not ok:
            layer_results[idx] = False
        o = cpu_4bit_matmul(v, layer.W_o)
        ok = verify_projection(idx, "O", layer.W_o, v)
        results.append((idx, "O", ok))
        if not ok:
            layer_results[idx] = False
        x = x + o

        ffn_in = rms_norm(x, layer.ffn_norm)
        gate = cpu_4bit_matmul(ffn_in, layer.W_gate)
        ok = verify_projection(idx, "Gate", layer.W_gate, ffn_in)
        results.append((idx, "Gate", ok))
        if not ok:
            layer_results[idx] = False
        up = cpu_4bit_matmul(ffn_in, layer.W_up)
        ok = verify_projection(idx, "Up", layer.W_up, ffn_in)
        results.append((idx, "Up", ok))
        if not ok:
            layer_results[idx] = False
        down = cpu_4bit_matmul(gate * silu(up), layer.W_down)
        ok = verify_projection(idx, "Down", layer.W_down, gate * silu(up))
        results.append((idx, "Down", ok))
        if not ok:
            layer_results[idx] = False
        x = x + down

    success_layers = sum(1 for v in layer_results.values() if v)
    print(f"\nCompleted! {success_layers}/{NUM_LAYERS} layers verified across VLAN-sharded filters.")
    summarize_results(results, success_layers, NUM_LAYERS)


if __name__ == "__main__":
    run()



""" Output:
sudo python3 e064_two_layer_transformer.py 
================================================================================
E064: Two-Layer Transformer (VLAN Sharded Filters)
================================================================================

STEP 1: Cleanup

  Cleaning up 10.10.10.55...
    Found 4 VLANs: ['layer0_shard0_vlan', 'layer1_shard1_vlan', 'layer1_shard0_vlan', 'layer0_shard1_vlan']
    Deleting 4 VLANs...
    ✓ Cleanup complete

STEP 2: Configure VLAN filters

Loading model: ./models/Qwen3-0.6B-Q4_K_M.gguf
  Loaded 310 tensors

=== Layer 0 ===

Layer 0 projection V verification:
  Neuron 0: pos 4/4 neg 12/12 ✓
  Neuron 1: pos 2/2 neg 3/3 ✓
  Neuron 2: pos 4/4 neg 3/3 ✓
  Neuron 3: pos 7/7 neg 4/4 ✓
  Neuron 4: pos 4/4 neg 8/8 ✓
  Neuron 5: pos 4/4 neg 7/7 ✓
  Neuron 6: pos 2/2 neg 6/6 ✓
  Neuron 7: pos 4/4 neg 6/6 ✓
  Neuron 8: pos 5/5 neg 6/6 ✓
  Neuron 9: pos 12/12 neg 5/5 ✓
  Neuron 10: pos 2/2 neg 9/9 ✓
  Neuron 11: pos 2/2 neg 6/6 ✓
  Neuron 12: pos 10/10 neg 16/16 ✓
  Neuron 13: pos 6/6 neg 5/5 ✓
  Neuron 14: pos 4/4 neg 3/3 ✓
  Neuron 15: pos 1/1 neg 6/6 ✓
  Neuron 16: pos 7/7 neg 7/7 ✓
  Neuron 17: pos 6/6 neg 9/9 ✓
  Neuron 18: pos 7/7 neg 7/7 ✓
  Neuron 19: pos 6/6 neg 6/6 ✓
  Neuron 20: pos 8/8 neg 7/7 ✓
  Neuron 21: pos 5/5 neg 8/8 ✓
  Neuron 22: pos 9/9 neg 1/1 ✓
  Neuron 23: pos 4/4 neg 11/11 ✓
  Neuron 24: pos 6/6 neg 3/3 ✓
  Neuron 25: pos 2/2 neg 6/6 ✓
  Neuron 26: pos 7/7 neg 5/5 ✓
  Neuron 27: pos 5/5 neg 3/3 ✓
  Neuron 28: pos 7/7 neg 8/8 ✓
  Neuron 29: pos 1/1 neg 3/3 ✓
  Neuron 30: pos 3/3 neg 6/6 ✓
  Neuron 31: pos 3/3 neg 6/6 ✓

Layer 0 projection O verification:
  Neuron 0: pos 4/4 neg 3/3 ✓
  Neuron 1: pos 2/2 neg 0/0 ✓
  Neuron 2: pos 6/6 neg 5/5 ✓
  Neuron 3: pos 6/6 neg 4/4 ✓
  Neuron 4: pos 2/2 neg 2/2 ✓
  Neuron 5: pos 4/4 neg 5/5 ✓
  Neuron 6: pos 5/5 neg 6/6 ✓
  Neuron 7: pos 3/3 neg 3/3 ✓
  Neuron 8: pos 11/11 neg 4/4 ✓
  Neuron 9: pos 9/9 neg 2/2 ✓
  Neuron 10: pos 3/3 neg 2/2 ✓
  Neuron 11: pos 5/5 neg 5/5 ✓
  Neuron 12: pos 5/5 neg 5/5 ✓
  Neuron 13: pos 4/4 neg 5/5 ✓
  Neuron 14: pos 5/5 neg 3/3 ✓
  Neuron 15: pos 4/4 neg 5/5 ✓
  Neuron 16: pos 5/5 neg 5/5 ✓
  Neuron 17: pos 3/3 neg 9/9 ✓
  Neuron 18: pos 4/4 neg 6/6 ✓
  Neuron 19: pos 5/5 neg 3/3 ✓
  Neuron 20: pos 10/10 neg 7/7 ✓
  Neuron 21: pos 3/3 neg 6/6 ✓
  Neuron 22: pos 7/7 neg 3/3 ✓
  Neuron 23: pos 3/3 neg 6/6 ✓
  Neuron 24: pos 7/7 neg 5/5 ✓
  Neuron 25: pos 1/1 neg 1/1 ✓
  Neuron 26: pos 4/4 neg 4/4 ✓
  Neuron 27: pos 5/5 neg 3/3 ✓
  Neuron 28: pos 7/7 neg 3/3 ✓
  Neuron 29: pos 3/3 neg 5/5 ✓
  Neuron 30: pos 5/5 neg 2/2 ✓
  Neuron 31: pos 6/6 neg 5/5 ✓

Layer 0 projection Gate verification:
  Neuron 0: pos 19/19 neg 26/26 ✓
  Neuron 1: pos 11/11 neg 10/10 ✓
  Neuron 2: pos 23/23 neg 17/17 ✓
  Neuron 3: pos 12/12 neg 12/12 ✓
  Neuron 4: pos 17/17 neg 13/13 ✓
  Neuron 5: pos 20/20 neg 9/9 ✓
  Neuron 6: pos 22/22 neg 25/25 ✓
  Neuron 7: pos 7/7 neg 15/15 ✓
  Neuron 8: pos 15/15 neg 16/16 ✓
  Neuron 9: pos 44/44 neg 25/25 ✓
  Neuron 10: pos 12/12 neg 11/11 ✓
  Neuron 11: pos 13/13 neg 15/15 ✓
  Neuron 12: pos 17/17 neg 12/12 ✓
  Neuron 13: pos 18/18 neg 21/21 ✓
  Neuron 14: pos 15/15 neg 7/7 ✓
  Neuron 15: pos 23/23 neg 9/9 ✓
  Neuron 16: pos 8/8 neg 12/12 ✓
  Neuron 17: pos 13/13 neg 14/14 ✓
  Neuron 18: pos 23/23 neg 17/17 ✓
  Neuron 19: pos 12/12 neg 3/3 ✓
  Neuron 20: pos 14/14 neg 13/13 ✓
  Neuron 21: pos 14/14 neg 3/3 ✓
  Neuron 22: pos 23/23 neg 11/11 ✓
  Neuron 23: pos 4/4 neg 4/4 ✓
  Neuron 24: pos 4/4 neg 3/3 ✓
  Neuron 25: pos 19/19 neg 9/9 ✓
  Neuron 26: pos 9/9 neg 9/9 ✓
  Neuron 27: pos 19/19 neg 10/10 ✓
  Neuron 28: pos 16/16 neg 12/12 ✓
  Neuron 29: pos 42/42 neg 27/27 ✓
  Neuron 30: pos 17/17 neg 11/11 ✓
  Neuron 31: pos 18/18 neg 22/22 ✓
  Neuron 32: pos 2/2 neg 9/9 ✓
  Neuron 33: pos 8/8 neg 9/9 ✓
  Neuron 34: pos 9/9 neg 14/14 ✓
  Neuron 35: pos 24/24 neg 25/25 ✓
  Neuron 36: pos 17/17 neg 8/8 ✓
  Neuron 37: pos 14/14 neg 14/14 ✓
  Neuron 38: pos 31/31 neg 49/49 ✓
  Neuron 39: pos 6/6 neg 7/7 ✓
  Neuron 40: pos 6/6 neg 9/9 ✓
  Neuron 41: pos 15/15 neg 14/14 ✓
  Neuron 42: pos 28/28 neg 12/12 ✓
  Neuron 43: pos 10/10 neg 14/14 ✓
  Neuron 44: pos 7/7 neg 7/7 ✓
  Neuron 45: pos 16/16 neg 11/11 ✓
  Neuron 46: pos 16/16 neg 15/15 ✓
  Neuron 47: pos 9/9 neg 5/5 ✓
  Neuron 48: pos 7/7 neg 4/4 ✓
  Neuron 49: pos 7/7 neg 13/13 ✓
  Neuron 50: pos 12/12 neg 25/25 ✓
  Neuron 51: pos 11/11 neg 16/16 ✓
  Neuron 52: pos 16/16 neg 21/21 ✓
  Neuron 53: pos 19/19 neg 6/6 ✓
  Neuron 54: pos 8/8 neg 7/7 ✓
  Neuron 55: pos 9/9 neg 17/17 ✓
  Neuron 56: pos 9/9 neg 8/8 ✓
  Neuron 57: pos 7/7 neg 7/7 ✓
  Neuron 58: pos 4/4 neg 12/12 ✓
  Neuron 59: pos 9/9 neg 15/15 ✓
  Neuron 60: pos 26/26 neg 9/9 ✓
  Neuron 61: pos 13/13 neg 4/4 ✓
  Neuron 62: pos 31/31 neg 15/15 ✓
  Neuron 63: pos 20/20 neg 29/29 ✓
  Neuron 64: pos 31/31 neg 17/17 ✓
  Neuron 65: pos 29/29 neg 18/18 ✓
  Neuron 66: pos 11/11 neg 19/19 ✓
  Neuron 67: pos 12/12 neg 17/17 ✓
  Neuron 68: pos 9/9 neg 15/15 ✓
  Neuron 69: pos 24/24 neg 21/21 ✓
  Neuron 70: pos 25/25 neg 20/20 ✓
  Neuron 71: pos 21/21 neg 31/31 ✓
  Neuron 72: pos 18/18 neg 19/19 ✓
  Neuron 73: pos 6/6 neg 7/7 ✓
  Neuron 74: pos 10/10 neg 8/8 ✓
  Neuron 75: pos 4/4 neg 17/17 ✓
  Neuron 76: pos 15/15 neg 7/7 ✓
  Neuron 77: pos 18/18 neg 8/8 ✓
  Neuron 78: pos 8/8 neg 7/7 ✓
  Neuron 79: pos 8/8 neg 6/6 ✓
  Neuron 80: pos 18/18 neg 7/7 ✓
  Neuron 81: pos 29/29 neg 25/25 ✓
  Neuron 82: pos 9/9 neg 5/5 ✓
  Neuron 83: pos 39/39 neg 21/21 ✓
  Neuron 84: pos 31/31 neg 20/20 ✓
  Neuron 85: pos 9/9 neg 15/15 ✓
  Neuron 86: pos 13/13 neg 18/18 ✓
  Neuron 87: pos 21/21 neg 12/12 ✓
  Neuron 88: pos 9/9 neg 15/15 ✓
  Neuron 89: pos 11/11 neg 19/19 ✓
  Neuron 90: pos 32/32 neg 31/31 ✓
  Neuron 91: pos 24/24 neg 14/14 ✓
  Neuron 92: pos 32/32 neg 11/11 ✓
  Neuron 93: pos 24/24 neg 18/18 ✓
  Neuron 94: pos 9/9 neg 13/13 ✓
  Neuron 95: pos 21/21 neg 19/19 ✓

Layer 0 projection Up verification:
  Neuron 0: pos 14/14 neg 9/9 ✓
  Neuron 1: pos 11/11 neg 12/12 ✓
  Neuron 2: pos 6/6 neg 7/7 ✓
  Neuron 3: pos 7/7 neg 10/10 ✓
  Neuron 4: pos 17/17 neg 11/11 ✓
  Neuron 5: pos 15/15 neg 7/7 ✓
  Neuron 6: pos 0/0 neg 5/5 ✓
  Neuron 7: pos 12/12 neg 13/13 ✓
  Neuron 8: pos 5/5 neg 15/15 ✓
  Neuron 9: pos 5/5 neg 15/15 ✓
  Neuron 10: pos 10/10 neg 10/10 ✓
  Neuron 11: pos 8/8 neg 9/9 ✓
  Neuron 12: pos 12/12 neg 10/10 ✓
  Neuron 13: pos 10/10 neg 7/7 ✓
  Neuron 14: pos 10/10 neg 3/3 ✓
  Neuron 15: pos 8/8 neg 8/8 ✓
  Neuron 16: pos 6/6 neg 8/8 ✓
  Neuron 17: pos 4/4 neg 8/8 ✓
  Neuron 18: pos 27/27 neg 27/27 ✓
  Neuron 19: pos 2/2 neg 5/5 ✓
  Neuron 20: pos 10/10 neg 1/1 ✓
  Neuron 21: pos 8/8 neg 5/5 ✓
  Neuron 22: pos 2/2 neg 11/11 ✓
  Neuron 23: pos 5/5 neg 6/6 ✓
  Neuron 24: pos 6/6 neg 10/10 ✓
  Neuron 25: pos 11/11 neg 10/10 ✓
  Neuron 26: pos 15/15 neg 2/2 ✓
  Neuron 27: pos 6/6 neg 7/7 ✓
  Neuron 28: pos 5/5 neg 14/14 ✓
  Neuron 29: pos 9/9 neg 10/10 ✓
  Neuron 30: pos 7/7 neg 9/9 ✓
  Neuron 31: pos 5/5 neg 5/5 ✓
  Neuron 32: pos 5/5 neg 5/5 ✓
  Neuron 33: pos 15/15 neg 3/3 ✓
  Neuron 34: pos 13/13 neg 12/12 ✓
  Neuron 35: pos 6/6 neg 9/9 ✓
  Neuron 36: pos 9/9 neg 6/6 ✓
  Neuron 37: pos 10/10 neg 13/13 ✓
  Neuron 38: pos 13/13 neg 6/6 ✓
  Neuron 39: pos 5/5 neg 10/10 ✓
  Neuron 40: pos 12/12 neg 5/5 ✓
  Neuron 41: pos 7/7 neg 5/5 ✓
  Neuron 42: pos 5/5 neg 8/8 ✓
  Neuron 43: pos 6/6 neg 16/16 ✓
  Neuron 44: pos 11/11 neg 7/7 ✓
  Neuron 45: pos 5/5 neg 12/12 ✓
  Neuron 46: pos 4/4 neg 7/7 ✓
  Neuron 47: pos 8/8 neg 11/11 ✓
  Neuron 48: pos 4/4 neg 7/7 ✓
  Neuron 49: pos 8/8 neg 7/7 ✓
  Neuron 50: pos 8/8 neg 4/4 ✓
  Neuron 51: pos 11/11 neg 7/7 ✓
  Neuron 52: pos 7/7 neg 8/8 ✓
  Neuron 53: pos 7/7 neg 4/4 ✓
  Neuron 54: pos 8/8 neg 7/7 ✓
  Neuron 55: pos 5/5 neg 2/2 ✓
  Neuron 56: pos 6/6 neg 9/9 ✓
  Neuron 57: pos 1/1 neg 5/5 ✓
  Neuron 58: pos 6/6 neg 3/3 ✓
  Neuron 59: pos 4/4 neg 9/9 ✓
  Neuron 60: pos 2/2 neg 7/7 ✓
  Neuron 61: pos 9/9 neg 15/15 ✓
  Neuron 62: pos 4/4 neg 4/4 ✓
  Neuron 63: pos 14/14 neg 5/5 ✓
  Neuron 64: pos 5/5 neg 8/8 ✓
  Neuron 65: pos 9/9 neg 8/8 ✓
  Neuron 66: pos 4/4 neg 7/7 ✓
  Neuron 67: pos 9/9 neg 8/8 ✓
  Neuron 68: pos 13/13 neg 6/6 ✓
  Neuron 69: pos 8/8 neg 4/4 ✓
  Neuron 70: pos 6/6 neg 4/4 ✓
  Neuron 71: pos 5/5 neg 4/4 ✓
  Neuron 72: pos 8/8 neg 5/5 ✓
  Neuron 73: pos 11/11 neg 6/6 ✓
  Neuron 74: pos 9/9 neg 6/6 ✓
  Neuron 75: pos 7/7 neg 8/8 ✓
  Neuron 76: pos 8/8 neg 8/8 ✓
  Neuron 77: pos 10/10 neg 4/4 ✓
  Neuron 78: pos 11/11 neg 10/10 ✓
  Neuron 79: pos 5/5 neg 12/12 ✓
  Neuron 80: pos 11/11 neg 5/5 ✓
  Neuron 81: pos 12/12 neg 2/2 ✓
  Neuron 82: pos 10/10 neg 10/10 ✓
  Neuron 83: pos 8/8 neg 21/21 ✓
  Neuron 84: pos 9/9 neg 7/7 ✓
  Neuron 85: pos 3/3 neg 6/6 ✓
  Neuron 86: pos 8/8 neg 9/9 ✓
  Neuron 87: pos 7/7 neg 7/7 ✓
  Neuron 88: pos 16/16 neg 17/17 ✓
  Neuron 89: pos 4/4 neg 7/7 ✓
  Neuron 90: pos 8/8 neg 9/9 ✓
  Neuron 91: pos 12/12 neg 12/12 ✓
  Neuron 92: pos 8/8 neg 5/5 ✓
  Neuron 93: pos 13/13 neg 11/11 ✓
  Neuron 94: pos 4/4 neg 6/6 ✓
  Neuron 95: pos 5/5 neg 4/4 ✓

Layer 0 projection Down verification:
  Neuron 0: pos 17/17 neg 8/8 ✓
  Neuron 1: pos 3/3 neg 3/3 ✓
  Neuron 2: pos 7/7 neg 14/14 ✓
  Neuron 3: pos 27/27 neg 20/20 ✓
  Neuron 4: pos 4/4 neg 4/4 ✓
  Neuron 5: pos 13/13 neg 12/12 ✓
  Neuron 6: pos 12/12 neg 13/13 ✓
  Neuron 7: pos 27/27 neg 10/10 ✓
  Neuron 8: pos 13/13 neg 12/12 ✓
  Neuron 9: pos 13/13 neg 15/15 ✓
  Neuron 10: pos 10/10 neg 5/5 ✓
  Neuron 11: pos 11/11 neg 12/12 ✓
  Neuron 12: pos 12/12 neg 11/11 ✓
  Neuron 13: pos 9/9 neg 6/6 ✓
  Neuron 14: pos 10/10 neg 9/9 ✓
  Neuron 15: pos 13/13 neg 6/6 ✓
  Neuron 16: pos 14/14 neg 18/18 ✓
  Neuron 17: pos 13/13 neg 7/7 ✓
  Neuron 18: pos 16/16 neg 6/6 ✓
  Neuron 19: pos 10/10 neg 7/7 ✓
  Neuron 20: pos 11/11 neg 10/10 ✓
  Neuron 21: pos 16/16 neg 8/8 ✓
  Neuron 22: pos 8/8 neg 3/3 ✓
  Neuron 23: pos 23/23 neg 14/14 ✓
  Neuron 24: pos 11/11 neg 11/11 ✓
  Neuron 25: pos 4/4 neg 5/5 ✓
  Neuron 26: pos 13/13 neg 8/8 ✓
  Neuron 27: pos 13/13 neg 5/5 ✓
  Neuron 28: pos 2/2 neg 4/4 ✓
  Neuron 29: pos 12/12 neg 10/10 ✓
  Neuron 30: pos 8/8 neg 8/8 ✓
  Neuron 31: pos 14/14 neg 16/16 ✓

=== Layer 1 ===

Layer 1 projection V verification:
  Neuron 0: pos 3/3 neg 4/4 ✓
  Neuron 1: pos 10/10 neg 1/1 ✓
  Neuron 2: pos 10/10 neg 9/9 ✓
  Neuron 3: pos 7/7 neg 3/3 ✓
  Neuron 4: pos 6/6 neg 5/5 ✓
  Neuron 5: pos 1/1 neg 3/3 ✓
  Neuron 6: pos 1/1 neg 3/3 ✓
  Neuron 7: pos 4/4 neg 4/4 ✓
  Neuron 8: pos 3/3 neg 8/8 ✓
  Neuron 9: pos 9/9 neg 2/2 ✓
  Neuron 10: pos 7/7 neg 3/3 ✓
  Neuron 11: pos 2/2 neg 0/0 ✓
  Neuron 12: pos 1/1 neg 4/4 ✓
  Neuron 13: pos 6/6 neg 2/2 ✓
  Neuron 14: pos 6/6 neg 2/2 ✓
  Neuron 15: pos 5/5 neg 4/4 ✓
  Neuron 16: pos 2/2 neg 4/4 ✓
  Neuron 17: pos 7/7 neg 4/4 ✓
  Neuron 18: pos 4/4 neg 3/3 ✓
  Neuron 19: pos 6/6 neg 4/4 ✓
  Neuron 20: pos 9/9 neg 4/4 ✓
  Neuron 21: pos 9/9 neg 2/2 ✓
  Neuron 22: pos 4/4 neg 4/4 ✓
  Neuron 23: pos 1/1 neg 6/6 ✓
  Neuron 24: pos 5/5 neg 4/4 ✓
  Neuron 25: pos 5/5 neg 5/5 ✓
  Neuron 26: pos 3/3 neg 1/1 ✓
  Neuron 27: pos 2/2 neg 5/5 ✓
  Neuron 28: pos 6/6 neg 5/5 ✓
  Neuron 29: pos 3/3 neg 3/3 ✓
  Neuron 30: pos 4/4 neg 6/6 ✓
  Neuron 31: pos 6/6 neg 2/2 ✓

Layer 1 projection O verification:
  Neuron 0: pos 10/10 neg 10/10 ✓
  Neuron 1: pos 3/3 neg 3/3 ✓
  Neuron 2: pos 9/9 neg 12/12 ✓
  Neuron 3: pos 16/16 neg 10/10 ✓
  Neuron 4: pos 10/10 neg 8/8 ✓
  Neuron 5: pos 11/11 neg 16/16 ✓
  Neuron 6: pos 8/8 neg 9/9 ✓
  Neuron 7: pos 12/12 neg 12/12 ✓
  Neuron 8: pos 15/15 neg 6/6 ✓
  Neuron 9: pos 12/12 neg 6/6 ✓
  Neuron 10: pos 5/5 neg 5/5 ✓
  Neuron 11: pos 15/15 neg 13/13 ✓
  Neuron 12: pos 7/7 neg 15/15 ✓
  Neuron 13: pos 8/8 neg 6/6 ✓
  Neuron 14: pos 10/10 neg 10/10 ✓
  Neuron 15: pos 8/8 neg 9/9 ✓
  Neuron 16: pos 8/8 neg 8/8 ✓
  Neuron 17: pos 6/6 neg 21/21 ✓
  Neuron 18: pos 7/7 neg 10/10 ✓
  Neuron 19: pos 5/5 neg 13/13 ✓
  Neuron 20: pos 11/11 neg 6/6 ✓
  Neuron 21: pos 10/10 neg 9/9 ✓
  Neuron 22: pos 8/8 neg 6/6 ✓
  Neuron 23: pos 13/13 neg 10/10 ✓
  Neuron 24: pos 13/13 neg 14/14 ✓
  Neuron 25: pos 6/6 neg 4/4 ✓
  Neuron 26: pos 6/6 neg 9/9 ✓
  Neuron 27: pos 3/3 neg 7/7 ✓
  Neuron 28: pos 5/5 neg 6/6 ✓
  Neuron 29: pos 14/14 neg 6/6 ✓
  Neuron 30: pos 5/5 neg 5/5 ✓
  Neuron 31: pos 9/9 neg 9/9 ✓

Layer 1 projection Gate verification:
  Neuron 0: pos 13/13 neg 13/13 ✓
  Neuron 1: pos 8/8 neg 10/10 ✓
  Neuron 2: pos 12/12 neg 23/23 ✓
  Neuron 3: pos 14/14 neg 13/13 ✓
  Neuron 4: pos 14/14 neg 10/10 ✓
  Neuron 5: pos 17/17 neg 6/6 ✓
  Neuron 6: pos 18/18 neg 8/8 ✓
  Neuron 7: pos 12/12 neg 16/16 ✓
  Neuron 8: pos 19/19 neg 19/19 ✓
  Neuron 9: pos 16/16 neg 8/8 ✓
  Neuron 10: pos 8/8 neg 6/6 ✓
  Neuron 11: pos 14/14 neg 8/8 ✓
  Neuron 12: pos 24/24 neg 18/18 ✓
  Neuron 13: pos 11/11 neg 10/10 ✓
  Neuron 14: pos 16/16 neg 8/8 ✓
  Neuron 15: pos 10/10 neg 7/7 ✓
  Neuron 16: pos 8/8 neg 4/4 ✓
  Neuron 17: pos 12/12 neg 12/12 ✓
  Neuron 18: pos 21/21 neg 11/11 ✓
  Neuron 19: pos 7/7 neg 12/12 ✓
  Neuron 20: pos 7/7 neg 3/3 ✓
  Neuron 21: pos 17/17 neg 6/6 ✓
  Neuron 22: pos 16/16 neg 12/12 ✓
  Neuron 23: pos 19/19 neg 7/7 ✓
  Neuron 24: pos 16/16 neg 6/6 ✓
  Neuron 25: pos 17/17 neg 41/41 ✓
  Neuron 26: pos 24/24 neg 11/11 ✓
  Neuron 27: pos 8/8 neg 4/4 ✓
  Neuron 28: pos 13/13 neg 5/5 ✓
  Neuron 29: pos 14/14 neg 10/10 ✓
  Neuron 30: pos 18/18 neg 4/4 ✓
  Neuron 31: pos 3/3 neg 6/6 ✓
  Neuron 32: pos 12/12 neg 4/4 ✓
  Neuron 33: pos 14/14 neg 36/36 ✓
  Neuron 34: pos 13/13 neg 14/14 ✓
  Neuron 35: pos 20/20 neg 6/6 ✓
  Neuron 36: pos 11/11 neg 9/9 ✓
  Neuron 37: pos 21/21 neg 18/18 ✓
  Neuron 38: pos 22/22 neg 11/11 ✓
  Neuron 39: pos 15/15 neg 17/17 ✓
  Neuron 40: pos 6/6 neg 7/7 ✓
  Neuron 41: pos 24/24 neg 17/17 ✓
  Neuron 42: pos 8/8 neg 4/4 ✓
  Neuron 43: pos 10/10 neg 12/12 ✓
  Neuron 44: pos 14/14 neg 12/12 ✓
  Neuron 45: pos 9/9 neg 5/5 ✓
  Neuron 46: pos 5/5 neg 16/16 ✓
  Neuron 47: pos 7/7 neg 5/5 ✓
  Neuron 48: pos 19/19 neg 13/13 ✓
  Neuron 49: pos 29/29 neg 15/15 ✓
  Neuron 50: pos 12/12 neg 8/8 ✓
  Neuron 51: pos 31/31 neg 14/14 ✓
  Neuron 52: pos 8/8 neg 12/12 ✓
  Neuron 53: pos 11/11 neg 9/9 ✓
  Neuron 54: pos 17/17 neg 13/13 ✓
  Neuron 55: pos 3/3 neg 6/6 ✓
  Neuron 56: pos 12/12 neg 7/7 ✓
  Neuron 57: pos 15/15 neg 9/9 ✓
  Neuron 58: pos 10/10 neg 2/2 ✓
  Neuron 59: pos 3/3 neg 9/9 ✓
  Neuron 60: pos 1/1 neg 7/7 ✓
  Neuron 61: pos 16/16 neg 25/25 ✓
  Neuron 62: pos 19/19 neg 10/10 ✓
  Neuron 63: pos 26/26 neg 10/10 ✓
  Neuron 64: pos 25/25 neg 6/6 ✓
  Neuron 65: pos 20/20 neg 17/17 ✓
  Neuron 66: pos 10/10 neg 4/4 ✓
  Neuron 67: pos 11/11 neg 12/12 ✓
  Neuron 68: pos 8/8 neg 11/11 ✓
  Neuron 69: pos 22/22 neg 9/9 ✓
  Neuron 70: pos 28/28 neg 6/6 ✓
  Neuron 71: pos 18/18 neg 24/24 ✓
  Neuron 72: pos 8/8 neg 11/11 ✓
  Neuron 73: pos 10/10 neg 11/11 ✓
  Neuron 74: pos 21/21 neg 8/8 ✓
  Neuron 75: pos 15/15 neg 3/3 ✓
  Neuron 76: pos 28/28 neg 13/13 ✓
  Neuron 77: pos 7/7 neg 11/11 ✓
  Neuron 78: pos 33/33 neg 8/8 ✓
  Neuron 79: pos 7/7 neg 5/5 ✓
  Neuron 80: pos 18/18 neg 13/13 ✓
  Neuron 81: pos 4/4 neg 8/8 ✓
  Neuron 82: pos 12/12 neg 9/9 ✓
  Neuron 83: pos 25/25 neg 8/8 ✓
  Neuron 84: pos 8/8 neg 10/10 ✓
  Neuron 85: pos 25/25 neg 19/19 ✓
  Neuron 86: pos 6/6 neg 4/4 ✓
  Neuron 87: pos 22/22 neg 17/17 ✓
  Neuron 88: pos 3/3 neg 11/11 ✓
  Neuron 89: pos 15/15 neg 10/10 ✓
  Neuron 90: pos 10/10 neg 9/9 ✓
  Neuron 91: pos 12/12 neg 9/9 ✓
  Neuron 92: pos 23/23 neg 11/11 ✓
  Neuron 93: pos 10/10 neg 7/7 ✓
  Neuron 94: pos 18/18 neg 14/14 ✓
  Neuron 95: pos 33/33 neg 15/15 ✓

Layer 1 projection Up verification:
  Neuron 0: pos 7/7 neg 6/6 ✓
  Neuron 1: pos 11/11 neg 6/6 ✓
  Neuron 2: pos 5/5 neg 15/15 ✓
  Neuron 3: pos 7/7 neg 0/0 ✓
  Neuron 4: pos 6/6 neg 5/5 ✓
  Neuron 5: pos 5/5 neg 4/4 ✓
  Neuron 6: pos 3/3 neg 3/3 ✓
  Neuron 7: pos 9/9 neg 4/4 ✓
  Neuron 8: pos 11/11 neg 8/8 ✓
  Neuron 9: pos 6/6 neg 5/5 ✓
  Neuron 10: pos 2/2 neg 1/1 ✓
  Neuron 11: pos 4/4 neg 4/4 ✓
  Neuron 12: pos 10/10 neg 7/7 ✓
  Neuron 13: pos 7/7 neg 2/2 ✓
  Neuron 14: pos 3/3 neg 3/3 ✓
  Neuron 15: pos 3/3 neg 3/3 ✓
  Neuron 16: pos 3/3 neg 3/3 ✓
  Neuron 17: pos 1/1 neg 4/4 ✓
  Neuron 18: pos 5/5 neg 7/7 ✓
  Neuron 19: pos 4/4 neg 3/3 ✓
  Neuron 20: pos 2/2 neg 8/8 ✓
  Neuron 21: pos 4/4 neg 2/2 ✓
  Neuron 22: pos 5/5 neg 7/7 ✓
  Neuron 23: pos 4/4 neg 2/2 ✓
  Neuron 24: pos 5/5 neg 7/7 ✓
  Neuron 25: pos 16/16 neg 8/8 ✓
  Neuron 26: pos 8/8 neg 5/5 ✓
  Neuron 27: pos 3/3 neg 2/2 ✓
  Neuron 28: pos 2/2 neg 0/0 ✓
  Neuron 29: pos 4/4 neg 3/3 ✓
  Neuron 30: pos 5/5 neg 4/4 ✓
  Neuron 31: pos 3/3 neg 6/6 ✓
  Neuron 32: pos 3/3 neg 3/3 ✓
  Neuron 33: pos 5/5 neg 8/8 ✓
  Neuron 34: pos 8/8 neg 3/3 ✓
  Neuron 35: pos 3/3 neg 5/5 ✓
  Neuron 36: pos 3/3 neg 5/5 ✓
  Neuron 37: pos 10/10 neg 9/9 ✓
  Neuron 38: pos 11/11 neg 4/4 ✓
  Neuron 39: pos 8/8 neg 4/4 ✓
  Neuron 40: pos 5/5 neg 2/2 ✓
  Neuron 41: pos 3/3 neg 8/8 ✓
  Neuron 42: pos 1/1 neg 3/3 ✓
  Neuron 43: pos 5/5 neg 10/10 ✓
  Neuron 44: pos 7/7 neg 5/5 ✓
  Neuron 45: pos 2/2 neg 2/2 ✓
  Neuron 46: pos 6/6 neg 7/7 ✓
  Neuron 47: pos 5/5 neg 3/3 ✓
  Neuron 48: pos 9/9 neg 3/3 ✓
  Neuron 49: pos 9/9 neg 6/6 ✓
  Neuron 50: pos 3/3 neg 1/1 ✓
  Neuron 51: pos 6/6 neg 4/4 ✓
  Neuron 52: pos 7/7 neg 6/6 ✓
  Neuron 53: pos 1/1 neg 5/5 ✓
  Neuron 54: pos 2/2 neg 3/3 ✓
  Neuron 55: pos 2/2 neg 9/9 ✓
  Neuron 56: pos 1/1 neg 8/8 ✓
  Neuron 57: pos 2/2 neg 8/8 ✓
  Neuron 58: pos 1/1 neg 8/8 ✓
  Neuron 59: pos 7/7 neg 2/2 ✓
  Neuron 60: pos 6/6 neg 9/9 ✓
  Neuron 61: pos 6/6 neg 2/2 ✓
  Neuron 62: pos 5/5 neg 7/7 ✓
  Neuron 63: pos 1/1 neg 16/16 ✓
  Neuron 64: pos 6/6 neg 2/2 ✓
  Neuron 65: pos 8/8 neg 7/7 ✓
  Neuron 66: pos 6/6 neg 3/3 ✓
  Neuron 67: pos 9/9 neg 2/2 ✓
  Neuron 68: pos 7/7 neg 5/5 ✓
  Neuron 69: pos 8/8 neg 2/2 ✓
  Neuron 70: pos 9/9 neg 4/4 ✓
  Neuron 71: pos 6/6 neg 10/10 ✓
  Neuron 72: pos 3/3 neg 2/2 ✓
  Neuron 73: pos 3/3 neg 1/1 ✓
  Neuron 74: pos 5/5 neg 9/9 ✓
  Neuron 75: pos 9/9 neg 3/3 ✓
  Neuron 76: pos 5/5 neg 6/6 ✓
  Neuron 77: pos 6/6 neg 4/4 ✓
  Neuron 78: pos 6/6 neg 3/3 ✓
  Neuron 79: pos 4/4 neg 3/3 ✓
  Neuron 80: pos 5/5 neg 2/2 ✓
  Neuron 81: pos 5/5 neg 0/0 ✓
  Neuron 82: pos 9/9 neg 1/1 ✓
  Neuron 83: pos 5/5 neg 4/4 ✓
  Neuron 84: pos 7/7 neg 8/8 ✓
  Neuron 85: pos 5/5 neg 3/3 ✓
  Neuron 86: pos 4/4 neg 4/4 ✓
  Neuron 87: pos 17/17 neg 11/11 ✓
  Neuron 88: pos 10/10 neg 7/7 ✓
  Neuron 89: pos 7/7 neg 6/6 ✓
  Neuron 90: pos 6/6 neg 5/5 ✓
  Neuron 91: pos 8/8 neg 0/0 ✓
  Neuron 92: pos 3/3 neg 9/9 ✓
  Neuron 93: pos 3/3 neg 3/3 ✓
  Neuron 94: pos 4/4 neg 4/4 ✓
  Neuron 95: pos 3/3 neg 6/6 ✓

Layer 1 projection Down verification:
  Neuron 0: pos 6/6 neg 3/3 ✓
  Neuron 1: pos 3/3 neg 5/5 ✓
  Neuron 2: pos 3/3 neg 9/9 ✓
  Neuron 3: pos 7/7 neg 5/5 ✓
  Neuron 4: pos 2/2 neg 2/2 ✓
  Neuron 5: pos 6/6 neg 2/2 ✓
  Neuron 6: pos 13/13 neg 9/9 ✓
  Neuron 7: pos 6/6 neg 7/7 ✓
  Neuron 8: pos 1/1 neg 5/5 ✓
  Neuron 9: pos 9/9 neg 12/12 ✓
  Neuron 10: pos 2/2 neg 1/1 ✓
  Neuron 11: pos 2/2 neg 4/4 ✓
  Neuron 12: pos 5/5 neg 9/9 ✓
  Neuron 13: pos 6/6 neg 3/3 ✓
  Neuron 14: pos 3/3 neg 10/10 ✓
  Neuron 15: pos 1/1 neg 2/2 ✓
  Neuron 16: pos 8/8 neg 4/4 ✓
  Neuron 17: pos 4/4 neg 6/6 ✓
  Neuron 18: pos 5/5 neg 4/4 ✓
  Neuron 19: pos 3/3 neg 6/6 ✓
  Neuron 20: pos 3/3 neg 4/4 ✓
  Neuron 21: pos 5/5 neg 3/3 ✓
  Neuron 22: pos 0/0 neg 2/2 ✓
  Neuron 23: pos 10/10 neg 9/9 ✓
  Neuron 24: pos 5/5 neg 5/5 ✓
  Neuron 25: pos 1/1 neg 3/3 ✓
  Neuron 26: pos 4/4 neg 5/5 ✓
  Neuron 27: pos 0/0 neg 1/1 ✓
  Neuron 28: pos 2/2 neg 1/1 ✓
  Neuron 29: pos 10/10 neg 7/7 ✓
  Neuron 30: pos 7/7 neg 4/4 ✓
  Neuron 31: pos 5/5 neg 4/4 ✓

Completed! Two layers verified across VLAN-sharded filters.
"""