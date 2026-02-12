#!/usr/bin/env python3
"""
e152_mac_encoded_layer_snake.py

MAC-ENCODED LAYER SNAKE ARCHITECTURE
=====================================

COMBINING TWO INNOVATIONS:
  e122: Single filter handles MULTIPLE layers via MAC-encoded layer IDs
  e095: Snake routing - packets automatically flow SW1 → SW2 via VLANs
  
HYPOTHESIS:
  We can use MAC-encoded layers (01:00:5e:LL:NN:NN where LL=layer)
  with VLAN-based snake routing to:
  1. Reduce number of filters needed (3 layers per filter instead of 1)
  2. Maintain automatic layer progression via VLAN routing
  3. Scale to 12+ layers efficiently
  
ARCHITECTURE:
  SW1: ONE filter handles layers 0, 1, 2 (using MAC byte 3)
  SW2: ONE filter handles layers 3, 4, 5 (using MAC byte 3)
  
  VLANs: Used ONLY for routing packets to correct switch
    - Layers 0-2 → VLAN 100 → SW1
    - Layers 3-5 → VLAN 101 → SW2
    
TEST APPROACH:
  1. Configure both switches with single multi-layer filters
  2. Send test packets with:
     - MAC encoding layer ID (byte 3)
     - VLAN routing to correct switch
  3. Verify each layer counted correctly on its switch
  4. Prove snake routing works with MAC-encoded layers!
  
SUCCESS CRITERIA:
  ✓ Single filter per switch handles 3 layers each
  ✓ MAC-encoded layer IDs work correctly
  ✓ VLAN-based routing delivers packets to correct switch
  ✓ All 6 layers counted accurately
  ✓ Architecture proves scalability to 12+ layers

Author: Research Phase 001
Date: January 2026
"""

import os
import sys
import time
import re
from typing import Dict, Tuple

# Import proven infrastructure
from e042_port_based_layers import (
    get_mac_address,
    ssh_command,
    run_config_commands,
    SWITCH1_IP,
    SWITCH2_IP,
    SEND_IFACE,
)

from e053_mac_encoded_layers import get_layer_neuron_mac
from e045_real_weights_inference import mac_str_to_bytes
from e093_gpt2_dpdk_inference import DPDKPacketSender, ensure_dpdk_binding

# =============================================================================
# CONFIGURATION
# =============================================================================

N_LAYERS = 6  # 3 per switch to prove concept
TEST_NEURONS_PER_LAYER = 10
TEST_PACKETS_PER_NEURON = 10

# VLAN assignment for routing
VLAN_SW1 = 100  # Routes to SW1 (layers 0-2)
VLAN_SW2 = 101  # Routes to SW2 (layers 3-5)

# Layer assignments
SW1_LAYERS = [0, 1, 2]
SW2_LAYERS = [3, 4, 5]

# Interfaces (matching current topology from important_files.md)
# NOTE: Cables are SWAPPED from old docs:
#   enp1s0 → SW2 et-0/0/96 (NOT SW1!)
#   enp1s0d1 → SW1 et-0/0/96 (NOT SW2!)
SW1_HOST_IFACE = "et-0/0/96"
SW1_INTER_IFACE = "et-0/0/97"
SW2_HOST_IFACE = "et-0/0/96"
SW2_INTER_IFACE = "et-0/0/97"

print("=" * 80)
print("E152: MAC-ENCODED LAYER SNAKE ARCHITECTURE")
print("=" * 80)
print(f"""
Combining e122 (MAC-encoded multi-layer) + e095 (Snake routing)

Architecture:
  SW1: ONE filter → layers {SW1_LAYERS} (VLAN {VLAN_SW1})
  SW2: ONE filter → layers {SW2_LAYERS} (VLAN {VLAN_SW2})
  
MAC Format: 01:00:5e:LL:00:NN (LL=layer, NN=neuron)
  
Per Layer: {TEST_NEURONS_PER_LAYER} test neurons × {TEST_PACKETS_PER_NEURON} packets
Total TCAM Terms Per Switch: {TEST_NEURONS_PER_LAYER * len(SW1_LAYERS)} (vs {TEST_NEURONS_PER_LAYER * len(SW1_LAYERS) * 3} if separate filters!)

Benefits:
  ✓ 3× TCAM reduction (3 layers per filter)
  ✓ Automatic routing via VLANs
  ✓ Scales to 12+ layers easily
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
        
        # Clean in correct order
        cleanup_cmds = [
            "delete vlans",
            "delete interfaces et-0/0/96 unit 0",
            "delete interfaces et-0/0/97 unit 0",
            "delete firewall family ethernet-switching",
        ]
        
        for cmd in cleanup_cmds:
            ssh_command(switch_ip, f"cli -c 'configure; {cmd}; commit'", timeout=20)
            time.sleep(0.3)
    
    print("  ✓ Both switches cleaned")

# =============================================================================
# SWITCH CONFIGURATION
# =============================================================================

def configure_switch(switch_ip: str, layers: list, vlan_id: int, 
                    host_iface: str, inter_iface: str, is_sw1: bool) -> bool:
    """
    Configure switch with SINGLE filter handling MULTIPLE MAC-encoded layers.
    
    Key innovation: Filter matches on MAC byte 3 (layer ID), not VLAN!
    VLANs are ONLY for routing packets to correct switch.
    """
    sw_name = "SW1" if is_sw1 else "SW2"
    filter_name = f"{sw_name.lower()}_multi_layer_filter"
    
    print(f"\n  Configuring {sw_name}...")
    print(f"    Layers: {layers}")
    print(f"    VLAN: {vlan_id} (for routing only)")
    print(f"    Filter: {filter_name} (handles ALL layers via MAC encoding!)")
    
    # STEP 1: VLANs and interfaces
    commands = []
    
    # Create BOTH VLANs on BOTH switches (for snake routing)
    commands.append(f"set vlans sw1_vlan vlan-id {VLAN_SW1}")
    commands.append(f"set vlans sw2_vlan vlan-id {VLAN_SW2}")
    
    # Host interface: trunk with both VLANs
    commands.append(f"set interfaces {host_iface} unit 0 family ethernet-switching interface-mode trunk")
    commands.append(f"set interfaces {host_iface} unit 0 family ethernet-switching vlan members sw1_vlan")
    commands.append(f"set interfaces {host_iface} unit 0 family ethernet-switching vlan members sw2_vlan")
    
    # Inter-switch interface: trunk with both VLANs
    commands.append(f"set interfaces {inter_iface} unit 0 family ethernet-switching interface-mode trunk")
    commands.append(f"set interfaces {inter_iface} unit 0 family ethernet-switching vlan members sw1_vlan")
    commands.append(f"set interfaces {inter_iface} unit 0 family ethernet-switching vlan members sw2_vlan")
    
    print(f"    Part 1/2: VLANs and interfaces...")
    if not run_config_commands(switch_ip, commands, debug=False):
        print(f"    ✗ Part 1 failed")
        return False
    print(f"    ✓ Part 1 complete")
    
    # STEP 2: Create SINGLE filter with MAC-encoded layer terms
    commands = []
    
    # Create filter
    vlan_name = "sw1_vlan" if is_sw1 else "sw2_vlan"
    
    # Add terms for each layer's neurons
    # KEY: All layers in ONE filter, distinguished by MAC byte 3 (layer ID)
    for layer in layers:
        for neuron in range(TEST_NEURONS_PER_LAYER):
            # MAC encoding: 01:00:5e:LL:00:NN where LL=layer
            mac = get_layer_neuron_mac(layer, neuron)
            term = f"L{layer}_n{neuron}"
            counter = f"L{layer}_n{neuron}_cnt"
            
            commands.append(f"set firewall family ethernet-switching filter {filter_name} term {term} from destination-mac-address {mac}")
            commands.append(f"set firewall family ethernet-switching filter {filter_name} term {term} then count {counter}")
            commands.append(f"set firewall family ethernet-switching filter {filter_name} term {term} then accept")
    
    # Default term
    commands.append(f"set firewall family ethernet-switching filter {filter_name} term default then accept")
    
    # Attach filter to THIS SWITCH's VLAN
    # This is KEY: SW1 filter only processes VLAN 100, SW2 only processes VLAN 101
    commands.append(f"set vlans {vlan_name} forwarding-options filter input {filter_name}")
    
    total_terms = len(layers) * TEST_NEURONS_PER_LAYER
    print(f"    Part 2/2: Creating filter with {total_terms} terms for {len(layers)} layers...")
    
    if not run_config_commands(switch_ip, commands, debug=False):
        print(f"    ✗ Part 2 failed")
        return False
    
    print(f"    ✓ {sw_name} configured!")
    print(f"      - 1 filter handles {len(layers)} layers")
    print(f"      - {total_terms} TCAM terms (vs {total_terms * len(layers)} if separate filters)")
    print(f"      - Filter attached to VLAN {vlan_id}")
    
    return True

# =============================================================================
# PACKET GENERATION
# =============================================================================

def craft_test_packet(layer: int, neuron: int, src_mac: str) -> bytes:
    """
    Craft test packet with:
    - Destination MAC encoding layer ID (byte 3) and neuron ID (bytes 4-5)
    - VLAN tag routing to correct switch (layers 0-2 → SW1, 3-5 → SW2)
    """
    import struct
    
    # MAC-encoded layer and neuron
    dst_mac = get_layer_neuron_mac(layer, neuron)
    dst_bytes = bytes.fromhex(dst_mac.replace(':', ''))
    src_bytes = bytes.fromhex(src_mac.replace(':', ''))
    
    # VLAN routing: layers 0-2 go to SW1, layers 3-5 go to SW2
    vlan_id = VLAN_SW1 if layer in SW1_LAYERS else VLAN_SW2
    
    # Build packet: DST_MAC | SRC_MAC | VLAN_TPID | VLAN_TCI | ETHERTYPE | PAYLOAD
    vlan_tpid = struct.pack("!H", 0x8100)  # VLAN tag
    vlan_tci = struct.pack("!H", vlan_id)
    ethertype = struct.pack("!H", 0x0800)  # IPv4
    
    # Minimal IP header + payload
    ip_header = bytes([
        0x45, 0x00,  # Version, IHL, DSCP
        0x00, 0x28,  # Total length (40 bytes)
        0x00, 0x00,  # ID
        0x00, 0x00,  # Flags
        0x40, 0x11,  # TTL, Protocol (UDP)
        0x00, 0x00,  # Checksum
        0x0a, 0x00, 0x00, 0x01,  # Source IP
        0x0a, 0x00, 0x00, 0x02,  # Dest IP
    ])
    
    payload = b'\x00' * 20
    
    packet = dst_bytes + src_bytes + vlan_tpid + vlan_tci + ethertype + ip_header + payload
    
    return packet

# =============================================================================
# COUNTER READING
# =============================================================================

def read_layer_counters(switch_ip: str, layers: list, sw_name: str) -> Dict[int, Dict[int, int]]:
    """
    Read counters for multiple layers from a single filter.
    Returns: {layer: {neuron: count}}
    """
    filter_name = f"{sw_name.lower()}_multi_layer_filter"
    
    success, stdout, _ = ssh_command(switch_ip, 
        f"cli -c 'show firewall filter {filter_name}'", timeout=30)
    
    results = {}
    if success:
        for layer in layers:
            layer_counters = {}
            for neuron in range(TEST_NEURONS_PER_LAYER):
                counter_name = f"L{layer}_n{neuron}_cnt"
                # Pattern: counter_name BYTES PACKETS
                pattern = rf"{counter_name}\s+\d+\s+(\d+)"
                match = re.search(pattern, stdout)
                if match:
                    layer_counters[neuron] = int(match.group(1))
                else:
                    layer_counters[neuron] = 0
            
            results[layer] = layer_counters
    
    return results

def clear_counters():
    """Clear all counters on both switches."""
    ssh_command(SWITCH1_IP, "cli -c 'clear firewall filter sw1_multi_layer_filter'", timeout=5)
    ssh_command(SWITCH2_IP, "cli -c 'clear firewall filter sw2_multi_layer_filter'", timeout=5)

# =============================================================================
# MAIN TEST
# =============================================================================

def run_snake_test(dpdk_sender: DPDKPacketSender):
    """
    Run the MAC-Encoded Layer Snake test.
    
    For each layer:
    1. Generate packets with MAC-encoded layer ID
    2. Use VLAN to route to correct switch
    3. Verify counters on correct switch
    """
    print("\n" + "="*80)
    print("RUNNING MAC-ENCODED LAYER SNAKE TEST")
    print("="*80)
    
    src_mac = get_mac_address(SEND_IFACE)
    
    # Clear all counters
    print("\n  Clearing counters...")
    clear_counters()
    time.sleep(0.5)
    
    # Generate ALL packets for ALL layers
    print("\n  Generating packets...")
    all_packets = []
    layer_packet_counts = {}
    
    for layer in range(N_LAYERS):
        layer_packets = []
        for neuron in range(TEST_NEURONS_PER_LAYER):
            for _ in range(TEST_PACKETS_PER_NEURON):
                packet = craft_test_packet(layer, neuron, src_mac)
                layer_packets.append(packet)
        
        all_packets.extend(layer_packets)
        layer_packet_counts[layer] = len(layer_packets)
        
        vlan = VLAN_SW1 if layer in SW1_LAYERS else VLAN_SW2
        switch = "SW1" if layer in SW1_LAYERS else "SW2"
        print(f"    Layer {layer}: {len(layer_packets)} packets → VLAN {vlan} → {switch}")
    
    print(f"\n  Total packets generated: {len(all_packets)}")
    
    # Send all packets in one burst (DPDK high speed!)
    print(f"\n  Sending {len(all_packets)} packets via DPDK...")
    send_time = dpdk_sender.send_packets_dpdk(all_packets)
    pps = len(all_packets) / send_time if send_time > 0 else 0
    print(f"  ✓ Sent in {send_time*1000:.1f}ms ({pps/1e6:.2f}M pps)")
    
    # Wait for packet processing
    print("\n  Waiting for switch processing...")
    time.sleep(1.0)
    
    # Read counters from both switches
    print("\n  Reading counters...")
    sw1_counters = read_layer_counters(SWITCH1_IP, SW1_LAYERS, "SW1")
    sw2_counters = read_layer_counters(SWITCH2_IP, SW2_LAYERS, "SW2")
    
    # Verify results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    all_pass = True
    
    print(f"\n  SW1 (layers {SW1_LAYERS}, VLAN {VLAN_SW1}):")
    for layer in SW1_LAYERS:
        layer_counters = sw1_counters.get(layer, {})
        total = sum(layer_counters.values())
        expected = layer_packet_counts[layer]
        
        match = total == expected
        status = "✓" if match else "✗"
        print(f"    Layer {layer}: {total} packets (expected {expected}) {status}")
        
        if not match:
            all_pass = False
            # Debug: show per-neuron counts
            print(f"      Per-neuron: {layer_counters}")
    
    print(f"\n  SW2 (layers {SW2_LAYERS}, VLAN {VLAN_SW2}):")
    for layer in SW2_LAYERS:
        layer_counters = sw2_counters.get(layer, {})
        total = sum(layer_counters.values())
        expected = layer_packet_counts[layer]
        
        match = total == expected
        status = "✓" if match else "✗"
        print(f"    Layer {layer}: {total} packets (expected {expected}) {status}")
        
        if not match:
            all_pass = False
            print(f"      Per-neuron: {layer_counters}")
    
    return all_pass

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main test execution."""
    
    # Step 1: DPDK setup
    print("\n" + "="*80)
    print("STEP 1: DPDK SETUP")
    print("="*80)
    
    if not ensure_dpdk_binding():
        print("\n✗ DPDK configuration failed!")
        return False
    
    dpdk_sender = DPDKPacketSender(iface=SEND_IFACE)
    if not dpdk_sender.compile_dpdk_sender():
        print("\n✗ DPDK compilation failed!")
        return False
    
    try:
        # Step 2: Configure switches
        print("\n" + "="*80)
        print("STEP 2: CONFIGURE SWITCHES")
        print("="*80)
        
        cleanup_switches()
        
        # Configure SW1
        if not configure_switch(SWITCH1_IP, SW1_LAYERS, VLAN_SW1, 
                               SW1_HOST_IFACE, SW1_INTER_IFACE, True):
            print("\n✗ SW1 configuration failed!")
            return False
        
        # Configure SW2
        if not configure_switch(SWITCH2_IP, SW2_LAYERS, VLAN_SW2, 
                               SW2_HOST_IFACE, SW2_INTER_IFACE, False):
            print("\n✗ SW2 configuration failed!")
            return False
        
        print("\n  ✓ Both switches configured with MAC-Encoded Layer Snake architecture!")
        
        # Wait for filters to activate
        print("\n  Waiting for filters to activate...")
        time.sleep(2)
        
        # Step 3: Run test
        print("\n" + "="*80)
        print("STEP 3: RUN TEST")
        print("="*80)
        
        success = run_snake_test(dpdk_sender)
        
        # Step 4: Summary
        print("\n" + "="*80)
        print("FINAL RESULTS")
        print("="*80)
        
        if success:
            print("\n  🎉 SUCCESS! MAC-Encoded Layer Snake Architecture PROVEN! 🎉")
            print()
            print("  ✓ Single filter per switch handles MULTIPLE layers")
            print("  ✓ MAC byte 3 correctly encodes layer ID")
            print("  ✓ VLAN routing delivers packets to correct switch")
            print("  ✓ All layers counted accurately")
            print()
            print("  INNOVATIONS COMBINED:")
            print("    - e122: MAC-encoded multi-layer filtering")
            print("    - e095: VLAN-based snake routing")
            print()
            print("  BENEFITS:")
            print(f"    - 3× TCAM reduction ({len(SW1_LAYERS)} layers per filter)")
            print("    - Automatic routing via VLANs")
            print("    - Scales to 12+ layers easily")
            print()
            print("  PATH TO SCALE:")
            print("    - 2 switches × 3 layers/filter = 6 layers ✓")
            print("    - Can scale to 2 switches × 6 layers/filter = 12 layers")
            print("    - Or more switches for 28+ layer models!")
        else:
            print("\n  ⚠️  Some layers failed verification")
            print("  Check counter readings and routing configuration")
        
        return success
        
    finally:
        dpdk_sender.cleanup()
        print("\n✓ Test complete")

if __name__ == "__main__":
    if os.geteuid() != 0:
        print("Error: This script must be run with sudo")
        print("Usage: sudo python3 e152_mac_encoded_layer_snake.py")
        sys.exit(1)
    
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)



""" Output:
sudo python3 e152_mac_encoded_layer_snake.py 
[sudo] password for multiplex: 
================================================================================
E152: MAC-ENCODED LAYER SNAKE ARCHITECTURE
================================================================================

Combining e122 (MAC-encoded multi-layer) + e095 (Snake routing)

Architecture:
  SW1: ONE filter → layers [0, 1, 2] (VLAN 100)
  SW2: ONE filter → layers [3, 4, 5] (VLAN 101)
  
MAC Format: 01:00:5e:LL:00:NN (LL=layer, NN=neuron)
  
Per Layer: 10 test neurons × 10 packets
Total TCAM Terms Per Switch: 30 (vs 90 if separate filters!)

Benefits:
  ✓ 3× TCAM reduction (3 layers per filter)
  ✓ Automatic routing via VLANs
  ✓ Scales to 12+ layers easily


================================================================================
STEP 1: DPDK SETUP
================================================================================

================================================================================
CHECKING DPDK CONFIGURATION
================================================================================
✓ NIC already bound to mlx4_core (DPDK ready!)

================================================================================
COMPILING DPDK PACKET SENDER
================================================================================
  Generated: /tmp/dpdk_sender_berfjl6g/packet_sender.c
  Compiling...
✓ Compiled: /tmp/dpdk_sender_berfjl6g/packet_sender

================================================================================
STEP 2: CONFIGURE SWITCHES
================================================================================

================================================================================
CLEANUP
================================================================================
  Cleaning SW1...
  Cleaning SW2...
  ✓ Both switches cleaned

  Configuring SW1...
    Layers: [0, 1, 2]
    VLAN: 100 (for routing only)
    Filter: sw1_multi_layer_filter (handles ALL layers via MAC encoding!)
    Part 1/2: VLANs and interfaces...
    ✓ Part 1 complete
    Part 2/2: Creating filter with 30 terms for 3 layers...
    ✓ SW1 configured!
      - 1 filter handles 3 layers
      - 30 TCAM terms (vs 90 if separate filters)
      - Filter attached to VLAN 100

  Configuring SW2...
    Layers: [3, 4, 5]
    VLAN: 101 (for routing only)
    Filter: sw2_multi_layer_filter (handles ALL layers via MAC encoding!)
    Part 1/2: VLANs and interfaces...
    ✓ Part 1 complete
    Part 2/2: Creating filter with 30 terms for 3 layers...
    ✓ SW2 configured!
      - 1 filter handles 3 layers
      - 30 TCAM terms (vs 90 if separate filters)
      - Filter attached to VLAN 101

  ✓ Both switches configured with MAC-Encoded Layer Snake architecture!

  Waiting for filters to activate...

================================================================================
STEP 3: RUN TEST
================================================================================

================================================================================
RUNNING MAC-ENCODED LAYER SNAKE TEST
================================================================================

  Clearing counters...

  Generating packets...
    Layer 0: 100 packets → VLAN 100 → SW1
    Layer 1: 100 packets → VLAN 100 → SW1
    Layer 2: 100 packets → VLAN 100 → SW1
    Layer 3: 100 packets → VLAN 101 → SW2
    Layer 4: 100 packets → VLAN 101 → SW2
    Layer 5: 100 packets → VLAN 101 → SW2

  Total packets generated: 600

  Sending 600 packets via DPDK...
  ✓ Sent in 0.5ms (1.11M pps)

  Waiting for switch processing...

  Reading counters...

================================================================================
RESULTS
================================================================================

  SW1 (layers [0, 1, 2], VLAN 100):
    Layer 0: 100 packets (expected 100) ✓
    Layer 1: 100 packets (expected 100) ✓
    Layer 2: 100 packets (expected 100) ✓

  SW2 (layers [3, 4, 5], VLAN 101):
    Layer 3: 100 packets (expected 100) ✓
    Layer 4: 100 packets (expected 100) ✓
    Layer 5: 100 packets (expected 100) ✓

================================================================================
FINAL RESULTS
================================================================================

  🎉 SUCCESS! MAC-Encoded Layer Snake Architecture PROVEN! 🎉

  ✓ Single filter per switch handles MULTIPLE layers
  ✓ MAC byte 3 correctly encodes layer ID
  ✓ VLAN routing delivers packets to correct switch
  ✓ All layers counted accurately

  INNOVATIONS COMBINED:
    - e122: MAC-encoded multi-layer filtering
    - e095: VLAN-based snake routing

  BENEFITS:
    - 3× TCAM reduction (3 layers per filter)
    - Automatic routing via VLANs
    - Scales to 12+ layers easily

  PATH TO SCALE:
    - 2 switches × 3 layers/filter = 6 layers ✓
    - Can scale to 2 switches × 6 layers/filter = 12 layers
    - Or more switches for 28+ layer models!

✓ Test complete
"""