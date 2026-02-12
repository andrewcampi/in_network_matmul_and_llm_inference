#!/usr/bin/env python3
"""
e053_mac_encoded_layers.py

MAC-ENCODED LAYER IDENTIFICATION
================================

BREAKTHROUGH: Encode layer ID directly in destination MAC address!

Since VLAN matching doesn't work for layer isolation, we encode BOTH
the layer ID AND neuron ID in the destination MAC address.

MAC FORMAT:
  01:00:5e:LL:NN:NN
  
  Where:
    01:00:5e = Multicast prefix (we control this)
    LL = Layer ID (0-255, supports 256 layers)
    NN:NN = Neuron ID (0-65535, supports 65K neurons per layer)

EXAMPLE:
  Layer 0, Neuron 5    → 01:00:5e:00:00:05
  Layer 1, Neuron 5    → 01:00:5e:01:00:05
  Layer 47, Neuron 2047 → 01:00:5e:2F:07:FF

ADVANTAGE:
  - All layers pre-configured at startup (no reconfiguration!)
  - MAC matching works perfectly (proven in e044-e051)
  - Simple: packet's destination MAC tells switch exactly which (layer, neuron) counter

Author: Research Phase 001
Date: December 2025
"""

import time
import os
import sys
import re
import numpy as np
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import proven infrastructure
from e042_port_based_layers import (
    ssh_command, run_config_commands,
    craft_vlan_packet, send_packets, get_mac_address,
    SWITCH1_IP, SWITCH2_IP, SEND_IFACE,
)

from e045_real_weights_inference import mac_str_to_bytes

# Configuration
NUM_LAYERS = 3
NUM_NEURONS = 8
FILTER_NAME = "layer_mac_filter"
TEST_VLAN = 100  # Single VLAN for all packets (just for transport)


def get_layer_neuron_mac(layer: int, neuron: int) -> str:
    """
    Generate MAC address encoding both layer and neuron.
    
    Format: 01:00:5e:LL:NN:NN
      LL = layer (0-255)
      NN:NN = neuron (0-65535)
    """
    if layer > 255:
        raise ValueError(f"Layer {layer} exceeds 255")
    if neuron > 65535:
        raise ValueError(f"Neuron {neuron} exceeds 65535")
    
    neuron_hi = (neuron >> 8) & 0xFF
    neuron_lo = neuron & 0xFF
    
    return f"01:00:5e:{layer:02x}:{neuron_hi:02x}:{neuron_lo:02x}"


def full_cleanup():
    """Clean switch thoroughly."""
    print("\n  Full cleanup...")
    cleanup_cmds = [
        "delete firewall family ethernet-switching filter",
        "delete interfaces et-0/0/96 unit 0 family ethernet-switching",
        "delete vlans",
    ]
    for cmd in cleanup_cmds:
        run_config_commands(SWITCH1_IP, [cmd], debug=False)
    time.sleep(0.5)
    print("  ✓ Cleanup complete")


def configure_multi_layer_mac_filter(num_layers: int, 
                                      num_neurons: int,
                                      debug: bool = False) -> Tuple[bool, float]:
    """
    Configure filter with terms for ALL (layer, neuron) combinations.
    
    Each term matches a unique MAC that encodes both layer and neuron.
    All layers pre-configured at once!
    """
    total_terms = num_layers * num_neurons
    print(f"\n  Configuring {num_layers} layers × {num_neurons} neurons = {total_terms} terms")
    print(f"  Each (layer, neuron) gets unique MAC address")
    
    start_time = time.time()
    
    # Step 1: Create VLAN and configure port
    setup_cmds = [
        f"set vlans test_vlan vlan-id {TEST_VLAN}",
        "set interfaces et-0/0/96 unit 0 family ethernet-switching interface-mode trunk",
        "set interfaces et-0/0/96 unit 0 family ethernet-switching vlan members test_vlan",
    ]
    print("    Setting up VLAN and port...")
    if not run_config_commands(SWITCH1_IP, setup_cmds, debug=debug):
        return False, 0
    
    # Step 2: Create filter terms for each (layer, neuron)
    print("    Creating filter terms...")
    filter_cmds = []
    
    for layer in range(num_layers):
        for neuron in range(num_neurons):
            mac = get_layer_neuron_mac(layer, neuron)
            term_name = f"L{layer}_N{neuron}"
            counter_name = f"layer{layer}_neuron{neuron}_pkts"
            
            filter_cmds.extend([
                f"set firewall family ethernet-switching filter {FILTER_NAME} "
                f"term {term_name} from destination-mac-address {mac}",
                f"set firewall family ethernet-switching filter {FILTER_NAME} "
                f"term {term_name} then count {counter_name}",
                f"set firewall family ethernet-switching filter {FILTER_NAME} "
                f"term {term_name} then accept",
            ])
    
    # Default term
    filter_cmds.extend([
        f"set firewall family ethernet-switching filter {FILTER_NAME} "
        f"term default then count unmatched_pkts",
        f"set firewall family ethernet-switching filter {FILTER_NAME} "
        f"term default then accept",
    ])
    
    # Batch the commands
    batch_size = 50
    total_batches = (len(filter_cmds) + batch_size - 1) // batch_size
    
    for i in range(0, len(filter_cmds), batch_size):
        batch = filter_cmds[i:i+batch_size]
        batch_num = i // batch_size + 1
        print(f"      Batch {batch_num}/{total_batches}...")
        if not run_config_commands(SWITCH1_IP, batch, debug=debug):
            print(f"    ✗ Failed at batch {batch_num}")
            return False, 0
    
    # Step 3: Apply filter to port
    apply_cmds = [
        f"set interfaces et-0/0/96 unit 0 family ethernet-switching filter input {FILTER_NAME}",
    ]
    print("    Applying filter to port...")
    if not run_config_commands(SWITCH1_IP, apply_cmds, debug=debug):
        return False, 0
    
    config_time = time.time() - start_time
    print(f"  ✓ All {total_terms} terms configured in {config_time:.1f}s")
    
    return True, config_time


def send_layer_neuron_packet(layer: int, neuron: int, count: int = 1) -> int:
    """
    Send packets to a specific (layer, neuron) using MAC-encoded addressing.
    """
    dst_mac = mac_str_to_bytes(get_layer_neuron_mac(layer, neuron))
    src_mac = mac_str_to_bytes(get_mac_address(SEND_IFACE))
    
    packets = []
    for i in range(count):
        pkt = craft_vlan_packet(
            dst_mac=dst_mac,
            src_mac=src_mac,
            vlan_id=TEST_VLAN,
            payload=f"L{layer}N{neuron}P{i}".encode()
        )
        packets.append(pkt)
    
    return send_packets(SEND_IFACE, packets)


def clear_counters():
    """Clear filter counters."""
    ssh_command(SWITCH1_IP, f"cli -c 'clear firewall filter {FILTER_NAME}'")


def read_layer_neuron_counters(num_layers: int, 
                                num_neurons: int,
                                debug: bool = False) -> Dict[Tuple[int, int], int]:
    """
    Read counters for all (layer, neuron) combinations.
    
    Returns: {(layer, neuron): packet_count}
    """
    success, stdout, _ = ssh_command(
        SWITCH1_IP,
        f"cli -c 'show firewall filter {FILTER_NAME}'"
    )
    
    if debug:
        print(f"\n  [DEBUG] Raw counter output:\n{stdout[:1500]}")
    
    results = {}
    
    if not success:
        return results
    
    # Parse counters
    for layer in range(num_layers):
        for neuron in range(num_neurons):
            counter_name = f"layer{layer}_neuron{neuron}_pkts"
            # Format: counter_name    bytes    packets
            pattern = rf'{counter_name}\s+\d+\s+(\d+)'
            match = re.search(pattern, stdout)
            if match:
                results[(layer, neuron)] = int(match.group(1))
            else:
                results[(layer, neuron)] = 0
    
    # Also get unmatched
    pattern = r'unmatched_pkts\s+\d+\s+(\d+)'
    match = re.search(pattern, stdout)
    if match:
        results[(-1, -1)] = int(match.group(1))
    
    return results


def run_mac_layer_experiment():
    """
    Test MAC-encoded layer identification.
    """
    print("="*80)
    print("E053: MAC-ENCODED LAYER IDENTIFICATION")
    print("="*80)
    
    print(f"""
  Configuration:
    Layers: {NUM_LAYERS}
    Neurons per layer: {NUM_NEURONS}
    Total terms: {NUM_LAYERS * NUM_NEURONS}
    
  MAC Encoding:
    01:00:5e:LL:NN:NN
    LL = Layer ID (0-255)
    NN:NN = Neuron ID (0-65535)
    
  Example MACs:
    Layer 0, Neuron 0: {get_layer_neuron_mac(0, 0)}
    Layer 1, Neuron 0: {get_layer_neuron_mac(1, 0)}
    Layer 2, Neuron 7: {get_layer_neuron_mac(2, 7)}
    
  Why this works:
    - Each (layer, neuron) has UNIQUE MAC address
    - MAC matching works (proven in previous experiments)
    - ALL layers configured at startup - NO reconfiguration!
""")
    
    # Step 1: Cleanup
    print("\n" + "="*60)
    print("STEP 1: CLEANUP")
    print("="*60)
    full_cleanup()
    time.sleep(1)
    
    # Step 2: Configure multi-layer filter
    print("\n" + "="*60)
    print("STEP 2: CONFIGURE ALL LAYERS")
    print("="*60)
    
    success, config_time = configure_multi_layer_mac_filter(
        NUM_LAYERS, NUM_NEURONS, debug=False
    )
    
    if not success:
        print("  ✗ Configuration failed!")
        return
    
    time.sleep(1)
    
    # Step 3: Clear counters
    print("\n" + "="*60)
    print("STEP 3: CLEAR COUNTERS")
    print("="*60)
    clear_counters()
    time.sleep(0.5)
    print("  ✓ Counters cleared")
    
    # Step 4: Send test packets
    print("\n" + "="*60)
    print("STEP 4: SEND TEST PACKETS")
    print("="*60)
    
    # Test pattern: different counts for each (layer, neuron)
    test_cases = [
        # (layer, neuron, packet_count)
        (0, 0, 5),   # Layer 0, Neuron 0
        (1, 0, 3),   # Layer 1, Neuron 0 (SAME neuron, different layer!)
        (2, 0, 7),   # Layer 2, Neuron 0
        (0, 5, 4),   # Layer 0, Neuron 5
        (1, 5, 6),   # Layer 1, Neuron 5
        (2, 5, 2),   # Layer 2, Neuron 5
        (0, 7, 8),   # Layer 0, Neuron 7
        (1, 7, 1),   # Layer 1, Neuron 7
        (2, 7, 9),   # Layer 2, Neuron 7
    ]
    
    print("\n  Sending test packets:")
    for layer, neuron, count in test_cases:
        mac = get_layer_neuron_mac(layer, neuron)
        sent = send_layer_neuron_packet(layer, neuron, count)
        print(f"    Layer {layer}, Neuron {neuron} ({mac}): sent {sent} packets")
    
    time.sleep(1)
    
    # Step 5: Read counters
    print("\n" + "="*60)
    print("STEP 5: READ COUNTERS")
    print("="*60)
    
    counters = read_layer_neuron_counters(NUM_LAYERS, NUM_NEURONS, debug=True)
    
    # Show non-zero counters by layer
    print("\n  Counter values by layer:")
    for layer in range(NUM_LAYERS):
        print(f"\n    Layer {layer}:")
        for neuron in range(NUM_NEURONS):
            count = counters.get((layer, neuron), 0)
            if count > 0:
                print(f"      Neuron {neuron}: {count} packets")
    
    unmatched = counters.get((-1, -1), 0)
    if unmatched > 0:
        print(f"\n    Unmatched: {unmatched} packets")
    
    # Step 6: Verify
    print("\n" + "="*60)
    print("STEP 6: VERIFY RESULTS")
    print("="*60)
    
    all_correct = True
    print("\n  Verification:")
    
    for layer, neuron, expected in test_cases:
        actual = counters.get((layer, neuron), 0)
        status = "✓" if actual == expected else "✗"
        if actual != expected:
            all_correct = False
        print(f"    Layer {layer}, Neuron {neuron}: expected={expected}, actual={actual} {status}")
    
    # Check cross-layer isolation
    print("\n  Cross-layer isolation check:")
    isolation_ok = True
    
    # Verify that neurons with no packets have zero counts
    tested_pairs = set((l, n) for l, n, _ in test_cases)
    for layer in range(NUM_LAYERS):
        for neuron in range(NUM_NEURONS):
            if (layer, neuron) not in tested_pairs:
                count = counters.get((layer, neuron), 0)
                if count != 0:
                    print(f"    ✗ Layer {layer}, Neuron {neuron}: expected=0, got={count}")
                    isolation_ok = False
    
    if isolation_ok:
        print("    ✓ All untested (layer, neuron) pairs have zero counts!")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if all_correct and isolation_ok:
        print(f"""
  🎉🎉🎉 MAC-ENCODED LAYER IDENTIFICATION WORKS! 🎉🎉🎉
  
  What we proved:
    ✓ Same neuron ID maps to DIFFERENT counters based on layer
    ✓ Layer 0, 1, and 2 are completely isolated
    ✓ All {NUM_LAYERS * NUM_NEURONS} terms pre-configured in {config_time:.1f}s
    ✓ NO reconfiguration needed between layers!
    
  MAC encoding scheme:
    01:00:5e:LL:NN:NN (LL=layer, NN:NN=neuron)
    
  Capacity:
    - 256 layers supported (LL: 0-255)
    - 65,536 neurons per layer (NN:NN: 0-65535)
    - Total: 16 million unique (layer, neuron) combinations!
    
  This enables TRUE counter-free multi-layer inference:
    1. Configure all layers at startup (once!)
    2. For each layer, send packets with layer-encoded MACs
    3. Read counters once at the end
    
  The 25-30 second reconfiguration per layer is ELIMINATED!
""")
    else:
        print(f"\n  ⚠ Some tests failed - needs investigation")
    
    # Cleanup
    print("\n  Final cleanup...")
    full_cleanup()
    print("  ✓ Done")


if __name__ == '__main__':
    run_mac_layer_experiment()


""" Output:
sudo python3 e053_mac_encoded_layers.py 
================================================================================
E053: MAC-ENCODED LAYER IDENTIFICATION
================================================================================

  Configuration:
    Layers: 3
    Neurons per layer: 8
    Total terms: 24
    
  MAC Encoding:
    01:00:5e:LL:NN:NN
    LL = Layer ID (0-255)
    NN:NN = Neuron ID (0-65535)
    
  Example MACs:
    Layer 0, Neuron 0: 01:00:5e:00:00:00
    Layer 1, Neuron 0: 01:00:5e:01:00:00
    Layer 2, Neuron 7: 01:00:5e:02:00:07
    
  Why this works:
    - Each (layer, neuron) has UNIQUE MAC address
    - MAC matching works (proven in previous experiments)
    - ALL layers configured at startup - NO reconfiguration!


============================================================
STEP 1: CLEANUP
============================================================

  Full cleanup...
  ✓ Cleanup complete

============================================================
STEP 2: CONFIGURE ALL LAYERS
============================================================

  Configuring 3 layers × 8 neurons = 24 terms
  Each (layer, neuron) gets unique MAC address
    Setting up VLAN and port...
    Creating filter terms...
      Batch 1/2...
      Batch 2/2...
    Applying filter to port...
  ✓ All 24 terms configured in 19.9s

============================================================
STEP 3: CLEAR COUNTERS
============================================================
  ✓ Counters cleared

============================================================
STEP 4: SEND TEST PACKETS
============================================================

  Sending test packets:
    Layer 0, Neuron 0 (01:00:5e:00:00:00): sent 5 packets
    Layer 1, Neuron 0 (01:00:5e:01:00:00): sent 3 packets
    Layer 2, Neuron 0 (01:00:5e:02:00:00): sent 7 packets
    Layer 0, Neuron 5 (01:00:5e:00:00:05): sent 4 packets
    Layer 1, Neuron 5 (01:00:5e:01:00:05): sent 6 packets
    Layer 2, Neuron 5 (01:00:5e:02:00:05): sent 2 packets
    Layer 0, Neuron 7 (01:00:5e:00:00:07): sent 8 packets
    Layer 1, Neuron 7 (01:00:5e:01:00:07): sent 1 packets
    Layer 2, Neuron 7 (01:00:5e:02:00:07): sent 9 packets

============================================================
STEP 5: READ COUNTERS
============================================================

  [DEBUG] Raw counter output:

Filter: layer_mac_filter                                       
Counters:
Name                                                Bytes              Packets
layer0_neuron0_pkts                                   320                    5
layer0_neuron1_pkts                                     0                    0
layer0_neuron2_pkts                                     0                    0
layer0_neuron3_pkts                                     0                    0
layer0_neuron4_pkts                                     0                    0
layer0_neuron5_pkts                                   256                    4
layer0_neuron6_pkts                                     0                    0
layer0_neuron7_pkts                                   512                    8
layer1_neuron0_pkts                                   192                    3
layer1_neuron1_pkts                                     0                    0
layer1_neuron2_pkts                                     0                    0
layer1_neuron3_pkts                                     0                    0
layer1_neuron4_pkts                                     0                    0
layer1_neuron5_pkts                                   384                    6
layer1_neuron6_pkts                                     0                    0
layer1_neuron7_pkts                                    64                    1
layer2_neuron0_pkts                                   448                    7
lay

  Counter values by layer:

    Layer 0:
      Neuron 0: 5 packets
      Neuron 5: 4 packets
      Neuron 7: 8 packets

    Layer 1:
      Neuron 0: 3 packets
      Neuron 5: 6 packets
      Neuron 7: 1 packets

    Layer 2:
      Neuron 0: 7 packets
      Neuron 5: 2 packets
      Neuron 7: 9 packets

============================================================
STEP 6: VERIFY RESULTS
============================================================

  Verification:
    Layer 0, Neuron 0: expected=5, actual=5 ✓
    Layer 1, Neuron 0: expected=3, actual=3 ✓
    Layer 2, Neuron 0: expected=7, actual=7 ✓
    Layer 0, Neuron 5: expected=4, actual=4 ✓
    Layer 1, Neuron 5: expected=6, actual=6 ✓
    Layer 2, Neuron 5: expected=2, actual=2 ✓
    Layer 0, Neuron 7: expected=8, actual=8 ✓
    Layer 1, Neuron 7: expected=1, actual=1 ✓
    Layer 2, Neuron 7: expected=9, actual=9 ✓

  Cross-layer isolation check:
    ✓ All untested (layer, neuron) pairs have zero counts!

================================================================================
SUMMARY
================================================================================

  🎉🎉🎉 MAC-ENCODED LAYER IDENTIFICATION WORKS! 🎉🎉🎉
  
  What we proved:
    ✓ Same neuron ID maps to DIFFERENT counters based on layer
    ✓ Layer 0, 1, and 2 are completely isolated
    ✓ All 24 terms pre-configured in 19.9s
    ✓ NO reconfiguration needed between layers!
    
  MAC encoding scheme:
    01:00:5e:LL:NN:NN (LL=layer, NN:NN=neuron)
    
  Capacity:
    - 256 layers supported (LL: 0-255)
    - 65,536 neurons per layer (NN:NN: 0-65535)
    - Total: 16 million unique (layer, neuron) combinations!
    
  This enables TRUE counter-free multi-layer inference:
    1. Configure all layers at startup (once!)
    2. For each layer, send packets with layer-encoded MACs
    3. Read counters once at the end
    
  The 25-30 second reconfiguration per layer is ELIMINATED!


  Final cleanup...

  Full cleanup...
  ✓ Cleanup complete
  ✓ Done
"""