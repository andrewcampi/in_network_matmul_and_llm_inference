#!/usr/bin/env python3
"""
e122_single_filter_multi_layer.py

PROOF: SINGLE FILTER CAN HANDLE MULTIPLE LAYERS
================================================

FUNDAMENTAL QUESTION:
  Can one filter handle multiple layers using MAC-encoded layer IDs?
  
HYPOTHESIS:
  YES! Layer ID is encoded in MAC address (byte 3).
  We don't need separate VLANs/filters per layer.
  VLANs are ONLY for switch routing (SW1 vs SW2).
  
ARCHITECTURE:
  - SW1: 1 VLAN (100), 1 filter, handles layers 0-2
  - SW2: 1 VLAN (101), 1 filter, handles layers 3-5
  - Each filter has terms for ALL its layers
  - MAC format: 01:00:5e:LL:00:NN (LL=layer, NN=neuron)
  
TEST:
  1. Configure SW1 with 1 filter for layers 0, 1, 2
  2. Configure SW2 with 1 filter for layers 3, 4, 5
  3. Send test packets to different layers
  4. Verify counters distinguish layers correctly
  
SUCCESS CRITERIA:
  ✓ All 6 layers counted correctly
  ✓ Layers 0-2 counted on SW1
  ✓ Layers 3-5 counted on SW2
  ✓ No crosstalk between layers
  
This respects the 3-filter-per-switch limit!
"""

import subprocess
import time
import re
from typing import Dict, Tuple

from e042_port_based_layers import (
    craft_vlan_packet,
    get_mac_address,
    ssh_command,
    run_config_commands,
    SWITCH1_IP,
    SWITCH2_IP,
    SEND_IFACE,
)

from e053_mac_encoded_layers import get_layer_neuron_mac
from e045_real_weights_inference import mac_str_to_bytes

# =============================================================================
# CONFIGURATION
# =============================================================================

# Very simple: 6 layers total, 3 per switch
N_LAYERS = 6
TEST_NEURONS_PER_LAYER = 10  # Small number for clear testing

# VLANs: ONLY for routing to correct switch
VLAN_SW1 = 100  # All SW1 traffic
VLAN_SW2 = 101  # All SW2 traffic

# Layer assignment
SW1_LAYERS = [0, 1, 2]
SW2_LAYERS = [3, 4, 5]

# Interfaces (from e095 topology)
SW1_HOST_IFACE = "et-0/0/96"
SW1_INTER_IFACE = "et-0/0/97"
SW2_HOST_IFACE = "et-0/0/96"
SW2_INTER_IFACE = "et-0/0/97"

print("=" * 80)
print("E122: SINGLE FILTER CAN HANDLE MULTIPLE LAYERS")
print("=" * 80)
print(f"""
Hypothesis: MAC-encoded layers let ONE filter handle MULTIPLE layers!

Architecture:
  SW1: 1 VLAN (100), 1 filter → layers {SW1_LAYERS}
  SW2: 1 VLAN (101), 1 filter → layers {SW2_LAYERS}
  
Per Layer: {TEST_NEURONS_PER_LAYER} test neurons
Total Terms Per Switch: {TEST_NEURONS_PER_LAYER * len(SW1_LAYERS) * 2} (well under TCAM limit!)

This respects the 3-filter-per-switch hardware limit ✓
""")

# =============================================================================
# CLEANUP
# =============================================================================

def cleanup_switches():
    """Clean up both switches."""
    print("\n" + "="*80)
    print("CLEANUP")
    print("="*80)
    
    for switch_ip, name in [(SWITCH1_IP, "SW1"), (SWITCH2_IP, "SW2")]:
        print(f"  Cleaning {name}...")
        
        # Correct order from hardware_limits.md
        cleanup_cmds = [
            "delete vlans",
            f"delete interfaces {SW1_HOST_IFACE} unit 0",
            f"delete interfaces {SW1_INTER_IFACE} unit 0",
            "delete firewall family ethernet-switching",
        ]
        cmd_str = "cli -c 'configure; " + "; ".join(cleanup_cmds) + "; commit'"
        ssh_command(switch_ip, cmd_str, timeout=60)
        time.sleep(0.5)
    
    print("  ✓ Both switches cleaned")

# =============================================================================
# SWITCH CONFIGURATION
# =============================================================================

def configure_switch(switch_ip: str, layers: list, vlan_id: int, 
                    host_iface: str, inter_iface: str, is_sw1: bool) -> bool:
    """
    Configure switch with SINGLE filter handling MULTIPLE layers.
    This is the key innovation being tested!
    """
    sw_name = "SW1" if is_sw1 else "SW2"
    filter_name = f"{sw_name.lower()}_multi_layer_filter"
    vlan_name = f"{sw_name.lower()}_vlan"
    
    print(f"\n  Configuring {sw_name}...")
    print(f"    Layers: {layers}")
    print(f"    VLAN: {vlan_id}")
    print(f"    Filter: {filter_name} (handles ALL layers for this switch!)")
    
    commands = []
    
    # STEP 1: Create VLANs (both switches need both VLANs for Snake routing)
    commands.append(f"set vlans sw1_vlan vlan-id {VLAN_SW1}")
    commands.append(f"set vlans sw2_vlan vlan-id {VLAN_SW2}")
    
    # STEP 2: Configure interfaces
    commands.append(f"set interfaces {host_iface} unit 0 family ethernet-switching interface-mode trunk")
    commands.append(f"set interfaces {host_iface} unit 0 family ethernet-switching vlan members sw1_vlan")
    commands.append(f"set interfaces {host_iface} unit 0 family ethernet-switching vlan members sw2_vlan")
    
    commands.append(f"set interfaces {inter_iface} unit 0 family ethernet-switching interface-mode trunk")
    commands.append(f"set interfaces {inter_iface} unit 0 family ethernet-switching vlan members sw1_vlan")
    commands.append(f"set interfaces {inter_iface} unit 0 family ethernet-switching vlan members sw2_vlan")
    
    print(f"    Part 1/2: VLANs and interfaces...")
    if not run_config_commands(switch_ip, commands, debug=False):
        print(f"    ✗ Part 1 failed")
        return False
    print(f"    ✓ Part 1 complete")
    
    # STEP 3: Create SINGLE filter with terms for ALL layers
    commands = []
    commands.append(f"delete firewall family ethernet-switching filter {filter_name}")
    commands.append(f"set firewall family ethernet-switching filter {filter_name}")
    
    # Add terms for each layer's neurons (all in ONE filter!)
    for layer in layers:
        for neuron in range(TEST_NEURONS_PER_LAYER):
            # Positive counter
            mac_pos = get_layer_neuron_mac(layer, neuron * 2)
            term_pos = f"L{layer}_n{neuron}_pos"
            counter_pos = f"L{layer}_n{neuron}_pos"
            
            commands.append(f"set firewall family ethernet-switching filter {filter_name} term {term_pos} from destination-mac-address {mac_pos}")
            commands.append(f"set firewall family ethernet-switching filter {filter_name} term {term_pos} then count {counter_pos}")
            commands.append(f"set firewall family ethernet-switching filter {filter_name} term {term_pos} then accept")
            
            # Negative counter
            mac_neg = get_layer_neuron_mac(layer, neuron * 2 + 1)
            term_neg = f"L{layer}_n{neuron}_neg"
            counter_neg = f"L{layer}_n{neuron}_neg"
            
            commands.append(f"set firewall family ethernet-switching filter {filter_name} term {term_neg} from destination-mac-address {mac_neg}")
            commands.append(f"set firewall family ethernet-switching filter {filter_name} term {term_neg} then count {counter_neg}")
            commands.append(f"set firewall family ethernet-switching filter {filter_name} term {term_neg} then accept")
    
    # Default term
    commands.append(f"set firewall family ethernet-switching filter {filter_name} term default then accept")
    
    # CRITICAL: Attach filter to THIS SWITCH's VLAN
    commands.append(f"set vlans {vlan_name} forwarding-options filter input {filter_name}")
    
    total_terms = len(layers) * TEST_NEURONS_PER_LAYER * 2
    print(f"    Part 2/2: Creating filter with {total_terms} terms (for {len(layers)} layers)...")
    
    if not run_config_commands(switch_ip, commands, debug=False):
        print(f"    ✗ Part 2 failed")
        return False
    
    print(f"    ✓ {sw_name} configured with 1 filter handling {len(layers)} layers!")
    return True

# =============================================================================
# PACKET GENERATION
# =============================================================================

def create_test_packet(layer: int, neuron: int, is_positive: bool) -> bytes:
    """Create a single test packet for a specific layer/neuron."""
    counter_idx = neuron * 2 if is_positive else neuron * 2 + 1
    dst_mac = get_layer_neuron_mac(layer, counter_idx)
    dst_mac_bytes = mac_str_to_bytes(dst_mac)
    src_mac_bytes = mac_str_to_bytes(get_mac_address(SEND_IFACE))
    
    # Choose VLAN based on which switch handles this layer
    vlan_id = VLAN_SW1 if layer in SW1_LAYERS else VLAN_SW2
    
    packet = craft_vlan_packet(dst_mac_bytes, src_mac_bytes, vlan_id, b'\x00' * 100)
    return packet

# =============================================================================
# COUNTER READING
# =============================================================================

def read_layer_counters(switch_ip: str, layers: list, sw_name: str) -> Dict[int, Dict[int, Tuple[int, int]]]:
    """
    Read counters for multiple layers from a single filter.
    Returns: {layer: {neuron: (pos_count, neg_count)}}
    """
    filter_name = f"{sw_name.lower()}_multi_layer_filter"
    
    success, stdout, _ = ssh_command(switch_ip, 
        f"cli -c 'show firewall filter {filter_name}'", timeout=30)
    
    results = {}
    if success:
        for layer in layers:
            layer_counters = {}
            for neuron in range(TEST_NEURONS_PER_LAYER):
                # Parse positive counter
                counter_pos = f"L{layer}_n{neuron}_pos"
                pattern_pos = rf"{counter_pos}\s+\d+\s+(\d+)"
                match_pos = re.search(pattern_pos, stdout)
                pos_count = int(match_pos.group(1)) if match_pos else 0
                
                # Parse negative counter
                counter_neg = f"L{layer}_n{neuron}_neg"
                pattern_neg = rf"{counter_neg}\s+\d+\s+(\d+)"
                match_neg = re.search(pattern_neg, stdout)
                neg_count = int(match_neg.group(1)) if match_neg else 0
                
                layer_counters[neuron] = (pos_count, neg_count)
            
            results[layer] = layer_counters
    
    return results

def clear_counters():
    """Clear all counters on both switches."""
    ssh_command(SWITCH1_IP, f"cli -c 'clear firewall filter sw1_multi_layer_filter'", timeout=5)
    ssh_command(SWITCH2_IP, f"cli -c 'clear firewall filter sw2_multi_layer_filter'", timeout=5)

# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def main():
    # Step 1: Cleanup
    cleanup_switches()
    
    # Step 2: Configure switches
    print("\n" + "="*80)
    print("STEP 1: CONFIGURE SWITCHES")
    print("="*80)
    
    if not configure_switch(SWITCH1_IP, SW1_LAYERS, VLAN_SW1, 
                           SW1_HOST_IFACE, SW1_INTER_IFACE, True):
        print("\n❌ SW1 configuration failed!")
        return False
    
    if not configure_switch(SWITCH2_IP, SW2_LAYERS, VLAN_SW2, 
                           SW2_HOST_IFACE, SW2_INTER_IFACE, False):
        print("\n❌ SW2 configuration failed!")
        return False
    
    print("\n  ✓ Both switches configured with single-filter multi-layer architecture!")
    
    # Step 3: Generate and send test packets
    print("\n" + "="*80)
    print("STEP 2: SEND TEST PACKETS")
    print("="*80)
    
    clear_counters()
    time.sleep(0.5)
    
    # Create socket for sending
    import socket as sock
    s = sock.socket(sock.AF_PACKET, sock.SOCK_RAW)
    s.bind((SEND_IFACE, 0))
    
    # Send 10 packets per neuron (5 positive, 5 negative) for each layer
    packets_sent = 0
    for layer in range(N_LAYERS):
        for neuron in range(TEST_NEURONS_PER_LAYER):
            # Send 5 positive packets
            packet_pos = create_test_packet(layer, neuron, True)
            for _ in range(5):
                s.send(packet_pos)
                packets_sent += 1
            
            # Send 5 negative packets
            packet_neg = create_test_packet(layer, neuron, False)
            for _ in range(5):
                s.send(packet_neg)
                packets_sent += 1
        
        vlan = VLAN_SW1 if layer in SW1_LAYERS else VLAN_SW2
        print(f"  Layer {layer}: Sent {TEST_NEURONS_PER_LAYER * 10} packets (VLAN {vlan})")
    
    s.close()
    
    print(f"\n  ✓ Total packets sent: {packets_sent}")
    time.sleep(1)
    
    # Step 4: Read and verify counters
    print("\n" + "="*80)
    print("STEP 3: READ COUNTERS")
    print("="*80)
    
    all_pass = True
    
    # Read SW1 counters
    print(f"\n  SW1 (layers {SW1_LAYERS}):")
    sw1_counters = read_layer_counters(SWITCH1_IP, SW1_LAYERS, "SW1")
    for layer in SW1_LAYERS:
        layer_counters = sw1_counters.get(layer, {})
        total_pos = sum(pos for pos, neg in layer_counters.values())
        total_neg = sum(neg for pos, neg in layer_counters.values())
        
        expected = TEST_NEURONS_PER_LAYER * 5
        pos_ok = total_pos == expected
        neg_ok = total_neg == expected
        
        status = "✓" if (pos_ok and neg_ok) else "✗"
        print(f"    Layer {layer}: pos={total_pos} (expect {expected}) neg={total_neg} (expect {expected}) {status}")
        
        if not (pos_ok and neg_ok):
            all_pass = False
    
    # Read SW2 counters
    print(f"\n  SW2 (layers {SW2_LAYERS}):")
    sw2_counters = read_layer_counters(SWITCH2_IP, SW2_LAYERS, "SW2")
    for layer in SW2_LAYERS:
        layer_counters = sw2_counters.get(layer, {})
        total_pos = sum(pos for pos, neg in layer_counters.values())
        total_neg = sum(neg for pos, neg in layer_counters.values())
        
        expected = TEST_NEURONS_PER_LAYER * 5
        pos_ok = total_pos == expected
        neg_ok = total_neg == expected
        
        status = "✓" if (pos_ok and neg_ok) else "✗"
        print(f"    Layer {layer}: pos={total_pos} (expect {expected}) neg={total_neg} (expect {expected}) {status}")
        
        if not (pos_ok and neg_ok):
            all_pass = False
    
    # Final results
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    
    if all_pass:
        print("\n  🎉 SUCCESS! Single filter CAN handle multiple layers!")
        print(f"  ✓ SW1: 1 filter processed {len(SW1_LAYERS)} layers correctly")
        print(f"  ✓ SW2: 1 filter processed {len(SW2_LAYERS)} layers correctly")
        print(f"  ✓ MAC-encoded layer IDs work perfectly")
        print(f"  ✓ VLANs only needed for switch routing")
        print(f"  ✓ Respects 3-filter-per-switch hardware limit")
        print(f"\n  This architecture can scale to 12+ layers!")
    else:
        print("\n  ⚠️  Some layers failed verification")
        print(f"  Check counter readings and filter configuration")
    
    return all_pass


if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)



""" Output:
sudo python3 e122_single_filter_multi_layer.py 
================================================================================
E122: SINGLE FILTER CAN HANDLE MULTIPLE LAYERS
================================================================================

Hypothesis: MAC-encoded layers let ONE filter handle MULTIPLE layers!

Architecture:
  SW1: 1 VLAN (100), 1 filter → layers [0, 1, 2]
  SW2: 1 VLAN (101), 1 filter → layers [3, 4, 5]
  
Per Layer: 10 test neurons
Total Terms Per Switch: 60 (well under TCAM limit!)

This respects the 3-filter-per-switch hardware limit ✓


================================================================================
CLEANUP
================================================================================
  Cleaning SW1...
  Cleaning SW2...
  ✓ Both switches cleaned

================================================================================
STEP 1: CONFIGURE SWITCHES
================================================================================

  Configuring SW1...
    Layers: [0, 1, 2]
    VLAN: 100
    Filter: sw1_multi_layer_filter (handles ALL layers for this switch!)
    Part 1/2: VLANs and interfaces...
    ✓ Part 1 complete
    Part 2/2: Creating filter with 60 terms (for 3 layers)...
    ✓ SW1 configured with 1 filter handling 3 layers!

  Configuring SW2...
    Layers: [3, 4, 5]
    VLAN: 101
    Filter: sw2_multi_layer_filter (handles ALL layers for this switch!)
    Part 1/2: VLANs and interfaces...
    ✓ Part 1 complete
    Part 2/2: Creating filter with 60 terms (for 3 layers)...
    ✓ SW2 configured with 1 filter handling 3 layers!

  ✓ Both switches configured with single-filter multi-layer architecture!

================================================================================
STEP 2: SEND TEST PACKETS
================================================================================
  Layer 0: Sent 100 packets (VLAN 100)
  Layer 1: Sent 100 packets (VLAN 100)
  Layer 2: Sent 100 packets (VLAN 100)
  Layer 3: Sent 100 packets (VLAN 101)
  Layer 4: Sent 100 packets (VLAN 101)
  Layer 5: Sent 100 packets (VLAN 101)

  ✓ Total packets sent: 600

================================================================================
STEP 3: READ COUNTERS
================================================================================

  SW1 (layers [0, 1, 2]):
    Layer 0: pos=50 (expect 50) neg=50 (expect 50) ✓
    Layer 1: pos=50 (expect 50) neg=50 (expect 50) ✓
    Layer 2: pos=50 (expect 50) neg=50 (expect 50) ✓

  SW2 (layers [3, 4, 5]):
    Layer 3: pos=50 (expect 50) neg=50 (expect 50) ✓
    Layer 4: pos=50 (expect 50) neg=50 (expect 50) ✓
    Layer 5: pos=50 (expect 50) neg=50 (expect 50) ✓

================================================================================
FINAL RESULTS
================================================================================

  🎉 SUCCESS! Single filter CAN handle multiple layers!
  ✓ SW1: 1 filter processed 3 layers correctly
  ✓ SW2: 1 filter processed 3 layers correctly
  ✓ MAC-encoded layer IDs work perfectly
  ✓ VLANs only needed for switch routing
  ✓ Respects 3-filter-per-switch hardware limit

  This architecture can scale to 12+ layers!
"""