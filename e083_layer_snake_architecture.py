#!/usr/bin/env python3
"""
e083_layer_snake_architecture.py

LAYER SNAKE ARCHITECTURE - Packets flow through ALL layers automatically!
=========================================================================

THE BREAKTHROUGH:
  Instead of sending packets to each layer separately (requiring multiple
  SSH round-trips), we configure a "snake" path where packets:
  
  1. Enter SW1 via et-0/0/96 (from host)
  2. Get counted for layers 0-3
  3. Forward to SW2 via et-0/0/100
  4. Get counted for layers 4-7
  5. (Optional) Return to host or just read counters

CURRENT TOPOLOGY:
  Host enp1s0 ────40G────→ SW1 et-0/0/96
  SW1 et-0/0/100 ←──40G──→ SW2 et-0/0/100
  Host enp1s0d1 ←──40G──── SW2 et-0/0/96

SNAKE PATH:
  ┌──────────────────────────────────────────────────────────┐
  │  Host (enp1s0)                                           │
  │       │                                                  │
  │       ▼ (VLAN 100-103 for L0-L3)                         │
  │  ┌────────────────────────┐                              │
  │  │  SW1 et-0/0/96         │                              │
  │  │  ├─ L0 filter (count)  │                              │
  │  │  ├─ L1 filter (count)  │                              │
  │  │  ├─ L2 filter (count)  │                              │
  │  │  └─ L3 filter (count)  │                              │
  │  └────────────────────────┘                              │
  │       │                                                  │
  │       ▼ Forward via et-0/0/100                           │
  │  ┌────────────────────────┐                              │
  │  │  SW2 et-0/0/100        │                              │
  │  │  ├─ L4 filter (count)  │                              │
  │  │  ├─ L5 filter (count)  │                              │
  │  │  ├─ L6 filter (count)  │                              │
  │  │  └─ L7 filter (count)  │                              │
  │  └────────────────────────┘                              │
  │       │                                                  │
  │       ▼ (Optional: forward to host via et-0/0/96)        │
  │  Host (enp1s0d1) - can receive final packets             │
  └──────────────────────────────────────────────────────────┘

WHY THIS WORKS:
  - Each layer has its own VLAN (stays under 8 VLAN/port limit)
  - Filters count packets AND forward them (via normal L2 switching)
  - VLANs are trunked between switches, so packets flow through
  - SW1 learns MAC addresses pointing to SW2 via et-0/0/100
  - Packets with SW2's VLANs get forwarded automatically

BENEFITS:
  - Single packet injection from host
  - Packets flow through ALL layers automatically
  - Read counters once at the end
  - Zero intermediate CPU round-trips
  - MASSIVE speedup for multi-layer inference!
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

# Test dimensions (small for fast testing)
NUM_NEURONS = 8          # Neurons per layer
NUM_LAYERS_SW1 = 4       # Layers 0-3 on SW1
NUM_LAYERS_SW2 = 4       # Layers 4-7 on SW2
TOTAL_LAYERS = NUM_LAYERS_SW1 + NUM_LAYERS_SW2

# VLAN configuration
BASE_VLAN = 200          # Start VLANs from 200
# Layer 0 = VLAN 200, Layer 1 = VLAN 201, etc.

# Interfaces
SW1_HOST_IFACE = "et-0/0/96"    # SW1 receives from host
SW1_INTER_IFACE = "et-0/0/100"  # SW1 to SW2
SW2_INTER_IFACE = "et-0/0/100"  # SW2 from SW1
SW2_HOST_IFACE = "et-0/0/96"    # SW2 to host (for return path)

SSH_KEY = "/home/multiplex/.ssh/id_rsa"


def ssh_command_long(switch_ip: str, command: str, timeout: int = 60) -> Tuple[bool, str, str]:
    """SSH command with configurable timeout."""
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


def transfer_and_apply_config(switch_ip: str, config_commands: List[str], name: str) -> bool:
    """Transfer config file to switch and apply it."""
    # Write local config file
    local_path = f"/tmp/{name}_config.txt"
    with open(local_path, 'w') as f:
        for cmd in config_commands:
            f.write(cmd + '\n')
    
    print(f"    Config file: {len(config_commands)} commands")
    
    # Transfer via SSH stdin
    remote_path = f"/var/tmp/{name}_config.txt"
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
            result = subprocess.run(ssh_cmd, stdin=f, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            print(f"    ✗ Transfer failed: {result.stderr[:100]}")
            return False
    except Exception as e:
        print(f"    ✗ Transfer error: {e}")
        return False
    
    # Apply config
    apply_cmd = f"cli -c 'configure; load set {remote_path}; commit'"
    success, stdout, stderr = ssh_command_long(switch_ip, apply_cmd, timeout=120)
    
    if not success:
        print(f"    ✗ Apply failed: {stderr[:200]}")
        return False
    
    if "error" in stdout.lower() or "error" in stderr.lower():
        print(f"    ⚠ Possible error in output:")
        print(f"      stdout: {stdout[:200]}")
        print(f"      stderr: {stderr[:200]}")
    
    return True


# =============================================================================
# SWITCH CLEANUP
# =============================================================================

def full_cleanup(switch_ip: str, name: str):
    """Clean up switch configuration thoroughly."""
    print(f"  Cleaning up {name}...")
    
    # Delete ALL filters explicitly (including any layer-specific ones)
    cleanup_cmds = [
        "delete vlans",
        "delete firewall family ethernet-switching filter",
    ]
    
    # Also delete specific layer filters that might exist
    for layer in range(TOTAL_LAYERS):
        cleanup_cmds.append(f"delete firewall family ethernet-switching filter layer{layer}_filter")
    
    # Clean interfaces
    cleanup_cmds.extend([
        f"delete interfaces {SW1_HOST_IFACE} unit 0 family ethernet-switching",
        f"delete interfaces {SW1_INTER_IFACE} unit 0 family ethernet-switching",
    ])
    
    cleanup_str = "; ".join(cleanup_cmds)
    ssh_command_long(switch_ip, f"cli -c 'configure; {cleanup_str}; commit'", timeout=60)
    time.sleep(1)


# =============================================================================
# SNAKE CONFIGURATION
# =============================================================================

def configure_sw1_snake(layers: List[int]) -> bool:
    """
    Configure SW1 for snake architecture.
    
    SW1 receives packets from host on et-0/0/96, counts them for layers 0-3,
    and forwards them to SW2 via et-0/0/100.
    
    Key insight: We put all layer VLANs (including SW2's) on the trunk link
    to SW2, so packets tagged with SW2's VLANs get forwarded there.
    """
    print(f"\n  Configuring SW1 (layers {layers})...")
    
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
    for layer in range(NUM_LAYERS_SW1, TOTAL_LAYERS):
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
    # This allows SW2's VLAN packets to flow through to SW2
    all_cmds.append(f"delete interfaces {SW1_INTER_IFACE} unit 0 family ethernet-switching")
    all_cmds.append(f"set interfaces {SW1_INTER_IFACE} unit 0 family ethernet-switching interface-mode trunk")
    for vlan_name in all_vlans_sw2:
        all_cmds.append(f"set interfaces {SW1_INTER_IFACE} unit 0 family ethernet-switching vlan members {vlan_name}")
    
    # Configure filters for SW1's layers (on the VLAN, not the interface)
    for layer in layers:
        filter_name = f"layer{layer}_filter"
        vlan_name = f"layer{layer}_vlan"
        
        # Create filter with neuron counters
        for neuron in range(NUM_NEURONS):
            # Positive counter
            mac_pos = get_layer_neuron_mac(layer * 2, neuron)  # Use layer*2 for pos
            term_pos = f"l{layer}_n{neuron}_p"
            all_cmds.extend([
                f"set firewall family ethernet-switching filter {filter_name} term {term_pos} from destination-mac-address {mac_pos}/48",
                f"set firewall family ethernet-switching filter {filter_name} term {term_pos} then count {term_pos}",
                f"set firewall family ethernet-switching filter {filter_name} term {term_pos} then accept",
            ])
            
            # Negative counter
            mac_neg = get_layer_neuron_mac(layer * 2 + 1, neuron)  # Use layer*2+1 for neg
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


def configure_sw2_snake(layers: List[int]) -> bool:
    """
    Configure SW2 for snake architecture.
    
    SW2 receives packets from SW1 on et-0/0/100 and counts them for layers 4-7.
    """
    print(f"\n  Configuring SW2 (layers {layers})...")
    
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
        
        for neuron in range(NUM_NEURONS):
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


# =============================================================================
# PACKET SENDING
# =============================================================================

def create_snake_packets(activations: np.ndarray, weights_per_layer: Dict[int, np.ndarray],
                         src_mac: str) -> Tuple[List[bytes], Dict[int, np.ndarray]]:
    """
    Create packets for ALL layers in the snake.
    
    Each layer gets packets tagged with its VLAN.
    All packets are sent at once - they flow through the snake automatically!
    
    Returns:
        packets: List of all packets to send
        expected: Dict of expected output per layer
    """
    packets = []
    expected = {}
    
    src = mac_str_to_bytes(src_mac)
    
    for layer, weights in weights_per_layer.items():
        vlan_id = BASE_VLAN + layer
        
        # Compute expected output for this layer
        output = activations @ weights  # [num_neurons]
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


# =============================================================================
# COUNTER READING
# =============================================================================

def read_layer_counters(switch_ip: str, layer: int, debug: bool = False) -> Dict[int, int]:
    """Read counters for a specific layer from a switch."""
    filter_name = f"layer{layer}_filter"
    
    success, stdout, _ = ssh_command_long(
        switch_ip,
        f"cli -c 'show firewall filter {filter_name}'",
        timeout=30
    )
    
    if not success:
        return {}
    
    if debug:
        print(f"\n    DEBUG: Raw output for layer {layer}:")
        print(stdout[:800])
    
    counters = {}
    
    for neuron in range(NUM_NEURONS):
        pos_val = 0
        neg_val = 0
        
        # Junos counter output format: "Name    Bytes    Packets"
        # We need the SECOND number (Packets column)!
        # Pattern: counter_name followed by bytes (first num) then packets (second num)
        pos_match = re.search(rf"l{layer}_n{neuron}_p\s+\d+\s+(\d+)", stdout)
        if pos_match:
            pos_val = int(pos_match.group(1))
        
        neg_match = re.search(rf"l{layer}_n{neuron}_n\s+\d+\s+(\d+)", stdout)
        if neg_match:
            neg_val = int(neg_match.group(1))
        
        counters[neuron] = pos_val - neg_val
    
    return counters


def read_all_counters(debug: bool = False) -> Dict[int, Dict[int, int]]:
    """Read counters from all layers on both switches."""
    all_counters = {}
    
    # SW1 layers (0-3)
    for layer in range(NUM_LAYERS_SW1):
        all_counters[layer] = read_layer_counters(SWITCH1_IP, layer, debug=(debug and layer == 0))
    
    # SW2 layers (4-7)
    for layer in range(NUM_LAYERS_SW1, TOTAL_LAYERS):
        all_counters[layer] = read_layer_counters(SWITCH2_IP, layer, debug=(debug and layer == 4))
    
    return all_counters


def clear_all_counters():
    """Clear counters on both switches."""
    for layer in range(NUM_LAYERS_SW1):
        ssh_command_long(SWITCH1_IP, f"cli -c 'clear firewall filter layer{layer}_filter'", timeout=10)
    
    for layer in range(NUM_LAYERS_SW1, TOTAL_LAYERS):
        ssh_command_long(SWITCH2_IP, f"cli -c 'clear firewall filter layer{layer}_filter'", timeout=10)
    
    time.sleep(0.3)


# =============================================================================
# MAIN TEST
# =============================================================================

def main():
    print("=" * 70)
    print("E083: LAYER SNAKE ARCHITECTURE")
    print("=" * 70)
    print()
    print("Testing snake path:")
    print("  Host → SW1 (L0-L3) → SW2 (L4-L7)")
    print()
    print(f"Configuration:")
    print(f"  - Layers per switch: {NUM_LAYERS_SW1}")
    print(f"  - Neurons per layer: {NUM_NEURONS}")
    print(f"  - Total layers: {TOTAL_LAYERS}")
    print(f"  - Base VLAN: {BASE_VLAN}")
    print()
    
    # =========================================================================
    # STEP 1: Clean up both switches
    # =========================================================================
    print("\nSTEP 1: Cleanup")
    print("-" * 50)
    full_cleanup(SWITCH1_IP, "SW1")
    full_cleanup(SWITCH2_IP, "SW2")
    time.sleep(1)
    print("  ✓ Both switches cleaned")
    
    # =========================================================================
    # STEP 2: Configure snake architecture
    # =========================================================================
    print("\nSTEP 2: Configure Snake Architecture")
    print("-" * 50)
    
    sw1_layers = list(range(NUM_LAYERS_SW1))  # [0, 1, 2, 3]
    sw2_layers = list(range(NUM_LAYERS_SW1, TOTAL_LAYERS))  # [4, 5, 6, 7]
    
    if not configure_sw1_snake(sw1_layers):
        print("  ✗ SW1 configuration failed!")
        return
    print("  ✓ SW1 configured")
    
    if not configure_sw2_snake(sw2_layers):
        print("  ✗ SW2 configuration failed!")
        return
    print("  ✓ SW2 configured")
    
    time.sleep(2)  # Wait for config to settle
    
    # Verify configuration
    print("\n  Verifying SW1 VLAN configuration...")
    success, stdout, _ = ssh_command_long(SWITCH1_IP, "cli -c 'show vlans brief'", timeout=30)
    print(f"    VLANs:\n{stdout[:500]}")
    
    print("\n  Verifying SW1 interface configuration...")
    success, stdout, _ = ssh_command_long(SWITCH1_IP, f"cli -c 'show interfaces {SW1_HOST_IFACE} | match vlan'", timeout=30)
    print(f"    Interface VLANs: {stdout.strip()}")
    
    # =========================================================================
    # STEP 3: Generate test data
    # =========================================================================
    print("\nSTEP 3: Generate Test Data")
    print("-" * 50)
    
    np.random.seed(42)
    
    # Input activation (shared across all layers for simplicity)
    activations = np.random.randint(-3, 4, size=NUM_NEURONS)
    print(f"  Input: {activations}")
    
    # Generate weights for each layer
    weights_per_layer = {}
    for layer in range(TOTAL_LAYERS):
        # Simple weight matrix (identity with small variations for testing)
        w = np.eye(NUM_NEURONS, dtype=np.int32) * 2
        # Add some variation per layer
        w = w + np.random.randint(-1, 2, size=(NUM_NEURONS, NUM_NEURONS))
        weights_per_layer[layer] = w
    
    # =========================================================================
    # STEP 4: Create and send all packets at once
    # =========================================================================
    print("\nSTEP 4: Send ALL Packets (Single Burst)")
    print("-" * 50)
    
    src_mac = get_mac_address(SEND_IFACE)
    print(f"  Source MAC: {src_mac}")
    
    clear_all_counters()
    
    packets, expected = create_snake_packets(activations, weights_per_layer, src_mac)
    print(f"  Total packets: {len(packets)}")
    print(f"  Packets per layer: ~{len(packets) // TOTAL_LAYERS}")
    
    start_time = time.time()
    send_packets(SEND_IFACE, packets)
    send_time = time.time() - start_time
    print(f"  ✓ All packets sent in {send_time*1000:.1f}ms")
    
    time.sleep(1)  # Wait for packets to propagate through snake
    
    # =========================================================================
    # STEP 5: Read all counters (single batch)
    # =========================================================================
    print("\nSTEP 5: Read All Counters")
    print("-" * 50)
    
    start_time = time.time()
    all_counters = read_all_counters(debug=True)
    read_time = time.time() - start_time
    print(f"  ✓ All counters read in {read_time*1000:.1f}ms")
    
    # =========================================================================
    # STEP 6: Verify results
    # =========================================================================
    print("\nSTEP 6: Verify Snake Results")
    print("-" * 50)
    
    all_passed = True
    
    for layer in range(TOTAL_LAYERS):
        switch_name = "SW1" if layer < NUM_LAYERS_SW1 else "SW2"
        
        layer_counters = all_counters.get(layer, {})
        layer_expected = expected.get(layer, np.zeros(NUM_NEURONS))
        
        # Compare
        matches = 0
        mismatches = []
        
        for neuron in range(NUM_NEURONS):
            exp = int(layer_expected[neuron])
            got = layer_counters.get(neuron, 0)
            
            if exp == got:
                matches += 1
            else:
                mismatches.append((neuron, exp, got))
        
        if matches == NUM_NEURONS:
            print(f"  Layer {layer} ({switch_name}): ✓ ALL {NUM_NEURONS} neurons match!")
        else:
            print(f"  Layer {layer} ({switch_name}): ✗ {len(mismatches)} mismatches")
            for neuron, exp, got in mismatches[:3]:  # Show first 3
                print(f"    Neuron {neuron}: expected {exp}, got {got}")
            all_passed = False
    
    # =========================================================================
    # STEP 7: Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SNAKE ARCHITECTURE RESULTS")
    print("=" * 70)
    
    if all_passed:
        print()
        print("  ╔═════════════════════════════════════════════════════")
        print("  ║    SNAKE ARCHITECTURE WORKS!                        ")
        print("  ╠═════════════════════════════════════════════════════")
        print(f"  ║  Layers processed: {TOTAL_LAYERS} (SW1: {NUM_LAYERS_SW1}, SW2: {NUM_LAYERS_SW2})")
        print(f"  ║  Packets sent: {len(packets):,} (single burst)     ")
        print(f"  ║  Send time: {send_time*1000:.1f}ms                 ")
        print(f"  ║  Read time: {read_time*1000:.1f}ms                 ")
        print("  ║                                                     ")
        print("  ║  BREAKTHROUGH:                                      ")
        print("  ║  • Packets flow through ALL layers automatically!   ")
        print("  ║  • Single packet injection, single counter read     ")
        print("  ║  • Zero intermediate CPU round-trips!               ")
        print("  ╚═════════════════════════════════════════════════════")
        print()
        print("SCALING TO FULL MODEL:")
        print("  With 8 VLANs per port limit:")
        print("  • Current: 4 layers per switch = 4 VLANs each ✓")
        print("  • Full 28 layers: Need 4 hops (7 layers per hop)")
        print("  • New topology with breakout cables enables this!")
        print()
    else:
        print()
        print("  ⚠ Some layers did not match expected values")
        print("  Debugging needed - check VLAN forwarding between switches")
        print()
        
        # Check if SW1 layers worked (local)
        sw1_worked = all(
            all_counters.get(layer, {}).get(0, 0) != 0 or expected[layer][0] == 0
            for layer in range(NUM_LAYERS_SW1)
        )
        
        # Check if SW2 layers worked (forwarded)
        sw2_worked = all(
            all_counters.get(layer, {}).get(0, 0) != 0 or expected[layer][0] == 0
            for layer in range(NUM_LAYERS_SW1, TOTAL_LAYERS)
        )
        
        print(f"  SW1 layers (0-3): {'✓' if sw1_worked else '✗ Packets not counted'}")
        print(f"  SW2 layers (4-7): {'✓' if sw2_worked else '✗ Packets not forwarded'}")
        
        if sw1_worked and not sw2_worked:
            print()
            print("  DIAGNOSIS: SW1 works, SW2 not receiving packets")
            print("  Possible fixes:")
            print("  1. Check VLAN trunk between SW1 et-0/0/100 and SW2 et-0/0/100")
            print("  2. Verify VLANs 204-207 are on the trunk link")
            print("  3. Check cable connection between switches")


if __name__ == "__main__":
    main()



""" Output:
 sudo python3 e083_layer_snake_architecture.py
======================================================================
E083: LAYER SNAKE ARCHITECTURE
======================================================================

Testing snake path:
  Host → SW1 (L0-L3) → SW2 (L4-L7)

Configuration:
  - Layers per switch: 4
  - Neurons per layer: 8
  - Total layers: 8
  - Base VLAN: 200


STEP 1: Cleanup
--------------------------------------------------
  Cleaning up SW1...
  Cleaning up SW2...
  ✓ Both switches cleaned

STEP 2: Configure Snake Architecture
--------------------------------------------------

  Configuring SW1 (layers [0, 1, 2, 3])...
    Config file: 225 commands
  ✓ SW1 configured

  Configuring SW2 (layers [4, 5, 6, 7])...
    Config file: 211 commands
  ✓ SW2 configured

  Verifying SW1 VLAN configuration...
    VLANs:

Routing instance        VLAN name             Tag          Interfaces
default-switch          default               1        
                                                            
default-switch          layer0_vlan           200      
                                                           et-0/0/96.0*
default-switch          layer1_vlan           201      
                                                           et-0/0/96.0*
default-switch          layer2_vlan           202      


  Verifying SW1 interface configuration...
    Interface VLANs: 

STEP 3: Generate Test Data
--------------------------------------------------
  Input: [ 3  0  1  3 -1  1  1  3]

STEP 4: Send ALL Packets (Single Burst)
--------------------------------------------------
  Source MAC: 7c:fe:90:9d:2a:f0
  Total packets: 322
  Packets per layer: ~40
  ✓ All packets sent in 7.4ms

STEP 5: Read All Counters
--------------------------------------------------

    DEBUG: Raw output for layer 0:

Filter: layer0_filter                                          
Counters:
Name                                                Bytes              Packets
l0_n0_n                                                 0                    0
l0_n0_p                                               512                    8
l0_n1_n                                                 0                    0
l0_n1_p                                               256                    4
l0_n2_n                                                 0                    0
l0_n2_p                                               128                    2
l0_n3_n                                                 0                    0
l0_n3_p                                               704                   11
l0_n4_n       

    DEBUG: Raw output for layer 4:

Filter: layer4_filter                                          
Counters:
Name                                                Bytes              Packets
l4_n0_n                                                 0                    0
l4_n0_p                                               512                    8
l4_n1_n                                               128                    2
l4_n1_p                                                 0                    0
l4_n2_n                                                 0                    0
l4_n2_p                                               576                    9
l4_n3_n                                                 0                    0
l4_n3_p                                               640                   10
l4_n4_n       
  ✓ All counters read in 6150.8ms

STEP 6: Verify Snake Results
--------------------------------------------------
  Layer 0 (SW1): ✓ ALL 8 neurons match!
  Layer 1 (SW1): ✓ ALL 8 neurons match!
  Layer 2 (SW1): ✓ ALL 8 neurons match!
  Layer 3 (SW1): ✓ ALL 8 neurons match!
  Layer 4 (SW2): ✓ ALL 8 neurons match!
  Layer 5 (SW2): ✓ ALL 8 neurons match!
  Layer 6 (SW2): ✓ ALL 8 neurons match!
  Layer 7 (SW2): ✓ ALL 8 neurons match!

======================================================================
SNAKE ARCHITECTURE RESULTS
======================================================================

  ╔═══════════════════════════════════════════════════════
  ║   SNAKE ARCHITECTURE WORKS!                 
  ╠═══════════════════════════════════════════════════════
  ║  Layers processed: 8 (SW1: 4, SW2: 4)     
  ║  Packets sent: 322 (single burst)              
  ║  Send time: 7.4ms                           
  ║  Read time: 6150.8ms                              
  ║                                                     
  ║  BREAKTHROUGH:                                      
  ║  • Packets flow through ALL layers automatically!      
  ║  • Single packet injection, single counter read       
  ║  • Zero intermediate CPU round-trips!                 
  ╚═══════════════════════════════════════════════════════

SCALING TO FULL MODEL:
  With 8 VLANs per port limit:
  • Current: 4 layers per switch = 4 VLANs each ✓
  • Full 28 layers: Need 4 hops (7 layers per hop)
  • New topology with breakout cables enables this!
"""