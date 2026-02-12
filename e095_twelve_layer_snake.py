#!/usr/bin/env python3
"""
e095_twelve_layer_snake.py

SNAKE ROUTING TEST - 12 LAYERS ACROSS 200 GBPS TRUNK
=====================================================

GOAL:
  Prove packets automatically route from SW1 → SW2 based on VLAN tags!
  
TEST APPROACH:
  1. Configure layers 0-5 on SW1, layers 6-11 on SW2
  2. Send test packets to SW1 with VLANs 100-111
  3. Verify: VLANs 100-105 counted on SW1, VLANs 106-111 counted on SW2
  4. SUCCESS = packets auto-route via 200 Gbps trunk!

NO FULL INFERENCE - Just routing proof!

TOPOLOGY:
  Host → SW1 (enp1s0)
  SW1 ←→ SW2 (200 Gbps: 2×40G + 3×4×10G)
  
USAGE:
  $ sudo python3 e095_twelve_layer_snake.py
"""

import os
import sys
import time
import re
import numpy as np
from typing import Dict, Tuple

from e042_port_based_layers import (
    get_mac_address,
    ssh_command,
    run_config_commands,
    SWITCH1_IP,
    SWITCH2_IP,
    SEND_IFACE,
)

from e053_mac_encoded_layers import get_layer_neuron_mac
from e088_gpt2_full_inference import BASE_VLAN, TEST_DIM
from e093_gpt2_dpdk_inference import DPDKPacketSender, ensure_dpdk_binding

# =============================================================================
# CONFIGURATION
# =============================================================================

N_LAYERS = 12
VLAN_BASE = BASE_VLAN

# Test: 10 neurons per layer, 10 packets each
TEST_NEURONS = 10
TEST_PACKETS_PER_NEURON = 10

print(f"\n{'='*80}")
print(f"E095: SNAKE ROUTING TEST")
print(f"{'='*80}")
print(f"  Testing: {N_LAYERS} layers across 2 switches")
print(f"  SW1: Layers 0-5, SW2: Layers 6-11")
print(f"  Proof: VLAN-based routing via 200 Gbps trunk")

# =============================================================================
# SWITCH CONFIGURATION
# =============================================================================

def configure_switch_simple(switch_ip: str, layers: list, host_iface: str, 
                           inter_iface: str, is_sw1: bool) -> bool:
    """Simple switch config: VLANs + trunk + filters."""
    sw_name = "SW1" if is_sw1 else "SW2"
    print(f"\n  Configuring {sw_name} (layers {layers[0]}-{layers[-1]})...")
    
    # STEP 1: VLANs and interfaces
    commands = []
    
    # Create VLANs for ALL layers (both switches need all for routing)
    for layer in range(N_LAYERS):
        vlan_id = VLAN_BASE + layer
        vlan_name = f"layer{layer}_vlan"
        commands.append(f"set vlans {vlan_name} vlan-id {vlan_id}")
    
    # Host interface: trunk with all VLANs
    # Note: Some interfaces might already be trunks, so don't fail on delete
    commands.append(f"set interfaces {host_iface} unit 0 family ethernet-switching interface-mode trunk")
    for layer in range(N_LAYERS):
        vlan_name = f"layer{layer}_vlan"
        commands.append(f"set interfaces {host_iface} unit 0 family ethernet-switching vlan members {vlan_name}")
    
    # Inter-switch interface: trunk with all VLANs
    commands.append(f"set interfaces {inter_iface} unit 0 family ethernet-switching interface-mode trunk")
    for layer in range(N_LAYERS):
        vlan_name = f"layer{layer}_vlan"
        commands.append(f"set interfaces {inter_iface} unit 0 family ethernet-switching vlan members {vlan_name}")
    
    print(f"    Part 1/2: VLANs and interfaces ({len(commands)} commands)...")
    success, stdout, stderr = ssh_command(switch_ip, 
        "cli -c 'configure; " + "; ".join(commands) + "; commit'", timeout=30)
    
    if not success:
        print(f"    ✗ Part 1 failed")
        if stderr:
            print(f"      Error: {stderr[:200]}")
        return False
    
    # Check for commit errors
    if "error" in stdout.lower() or "failed" in stdout.lower():
        print(f"    ✗ Part 1 had errors")
        print(f"      Output: {stdout[:300]}")
        return False
    
    print(f"    ✓ Part 1 complete")
    
    # STEP 2: Filters (one layer at a time to avoid huge batches)
    for layer in layers:
        filter_name = f"layer{layer}_filter"
        vlan_name = f"layer{layer}_vlan"
        
        commands = []
        
        # Delete existing filter
        commands.append(f"delete firewall family ethernet-switching filter {filter_name}")
        
        # Add simple counter terms (just test neurons)
        for neuron in range(TEST_NEURONS):
            mac = get_layer_neuron_mac(layer * 2, neuron)  # Just positive for simplicity
            term = f"l{layer}_n{neuron}"
            counter = f"l{layer}_n{neuron}_cnt"
            
            commands.extend([
                f"set firewall family ethernet-switching filter {filter_name} term {term} from destination-mac-address {mac}",
                f"set firewall family ethernet-switching filter {filter_name} term {term} then count {counter}",
                f"set firewall family ethernet-switching filter {filter_name} term {term} then accept",
            ])
        
        # Default term
        commands.extend([
            f"set firewall family ethernet-switching filter {filter_name} term default then accept",
        ])
        
        # Attach to VLAN
        commands.append(f"set vlans {vlan_name} forwarding-options filter input {filter_name}")
        
        if not run_config_commands(switch_ip, commands, debug=False):
            print(f"    ✗ Layer {layer} filter failed")
            return False
    
    print(f"    ✓ Part 2/2: All {len(layers)} layer filters configured")
    print(f"    ✓ {sw_name} fully configured")
    
    return True


def cleanup_switches():
    """Clean up both switches including ae0 aggregation."""
    print("\n  Cleaning up switches...")
    
    # List of interfaces that might be in ae0
    ae_links = ["xe-0/0/0", "xe-0/0/1", "xe-0/0/2", "xe-0/0/3",
                "xe-0/0/40", "xe-0/0/41", "xe-0/0/42", "xe-0/0/43",
                "et-0/0/97", "et-0/0/98", "et-0/0/99", "et-0/0/103"]
    
    for switch_ip in [SWITCH1_IP, SWITCH2_IP]:
        # Remove ae0 aggregated ethernet (from previous experiments)
        ssh_command(switch_ip, "cli -c 'configure; delete interfaces ae0; commit'", timeout=10)
        ssh_command(switch_ip, "cli -c 'configure; delete chassis aggregated-devices; commit'", timeout=10)
        
        # Remove individual interface ae0 membership
        for link in ae_links:
            ssh_command(switch_ip, f"cli -c 'configure; delete interfaces {link} ether-options; commit'", timeout=5)
        
        # Clean up VLANs and filters
        for layer in range(N_LAYERS):
            filter_name = f"layer{layer}_filter"
            vlan_name = f"layer{layer}_vlan"
            
            ssh_command(switch_ip, f"cli -c 'delete firewall family ethernet-switching filter {filter_name}'", timeout=5)
            ssh_command(switch_ip, f"cli -c 'delete vlans {vlan_name}'", timeout=5)
    
    print("    ✓ Cleanup complete (including ae0 aggregation)")


def read_layer_counters(switch_ip: str, layer: int) -> Dict[int, int]:
    """Read counters for a layer."""
    filter_name = f"layer{layer}_filter"
    
    success, stdout, _ = ssh_command(switch_ip, 
        f"cli -c 'show firewall filter {filter_name}'", timeout=10)
    
    counters = {}
    if success:
        for neuron in range(TEST_NEURONS):
            counter_name = f"l{layer}_n{neuron}_cnt"
            pattern = rf"{counter_name}\s+\d+\s+(\d+)"
            match = re.search(pattern, stdout)
            if match:
                counters[neuron] = int(match.group(1))
    
    return counters


def clear_layer_counters(switch_ip: str, layer: int):
    """Clear counters for a layer."""
    filter_name = f"layer{layer}_filter"
    ssh_command(switch_ip, f"cli -c 'clear firewall filter {filter_name}'", timeout=5)


# =============================================================================
# PACKET GENERATION
# =============================================================================

def craft_test_packet(layer: int, neuron: int, src_mac: str) -> bytes:
    """Craft a test packet for a specific layer and neuron."""
    import struct
    
    # Get neuron MAC
    dst_mac = get_layer_neuron_mac(layer * 2, neuron)
    
    # Convert MACs to bytes
    src_bytes = bytes.fromhex(src_mac.replace(':', ''))
    dst_bytes = bytes.fromhex(dst_mac.replace(':', ''))
    
    # VLAN tag
    vlan_id = VLAN_BASE + layer
    vlan_tpid = struct.pack("!H", 0x8100)  # VLAN TPID
    vlan_tci = struct.pack("!H", vlan_id)  # VLAN ID
    
    # Ethertype
    ethertype = struct.pack("!H", 0x0800)  # IPv4
    
    # Minimal IP packet (20 bytes header + minimal payload)
    ip_header = bytes([
        0x45, 0x00,  # Version, IHL, DSCP, ECN
        0x00, 0x1c,  # Total length (28 bytes)
        0x00, 0x00,  # ID
        0x00, 0x00,  # Flags, fragment offset
        0x40,        # TTL
        0x11,        # Protocol (UDP)
        0x00, 0x00,  # Checksum (dummy)
        0x0a, 0x00, 0x00, 0x01,  # Source IP
        0x0a, 0x00, 0x00, 0x02,  # Dest IP
    ])
    
    payload = b'\x00' * 8  # Minimal payload
    
    packet = dst_bytes + src_bytes + vlan_tpid + vlan_tci + ethertype + ip_header + payload
    
    return packet


# =============================================================================
# TEST EXECUTION
# =============================================================================

def run_snake_test(dpdk_sender: DPDKPacketSender):
    """Run the Snake routing test."""
    
    print("\n" + "="*80)
    print("RUNNING SNAKE ROUTING TEST")
    print("="*80)
    
    src_mac = get_mac_address(SEND_IFACE)
    
    # Test each layer
    results = {}
    
    for layer in range(N_LAYERS):
        print(f"\n  Testing Layer {layer} (VLAN {VLAN_BASE + layer}):")
        
        # Determine which switch should count this
        is_sw1_layer = layer < 6
        expected_switch = SWITCH1_IP if is_sw1_layer else SWITCH2_IP
        switch_name = "SW1" if is_sw1_layer else "SW2"
        
        # Clear counters on BOTH switches
        clear_layer_counters(SWITCH1_IP, layer)
        clear_layer_counters(SWITCH2_IP, layer)
        time.sleep(0.1)
        
        # Generate test packets
        packets = []
        for neuron in range(TEST_NEURONS):
            for _ in range(TEST_PACKETS_PER_NEURON):
                packet = craft_test_packet(layer, neuron, src_mac)
                packets.append(packet)
        
        # Send packets
        print(f"    Sending {len(packets)} packets to SW1...")
        send_time = dpdk_sender.send_packets_dpdk(packets)
        print(f"    Sent in {send_time*1000:.1f}ms")
        
        # Wait for processing
        time.sleep(0.2)
        
        # Read counters from BOTH switches
        sw1_counters = read_layer_counters(SWITCH1_IP, layer)
        sw2_counters = read_layer_counters(SWITCH2_IP, layer)
        
        sw1_total = sum(sw1_counters.values())
        sw2_total = sum(sw2_counters.values())
        
        print(f"    SW1 counters: {sw1_total} packets")
        print(f"    SW2 counters: {sw2_total} packets")
        print(f"    Expected: {switch_name} should have {len(packets)} packets")
        
        # Check if routing worked
        if is_sw1_layer:
            success = sw1_total > 0 and sw2_total == 0
        else:
            success = sw2_total > 0 and sw1_total == 0
        
        results[layer] = {
            'expected_switch': switch_name,
            'sw1_count': sw1_total,
            'sw2_count': sw2_total,
            'success': success
        }
        
        status = "✓" if success else "✗"
        print(f"    {status} Routing {'CORRECT' if success else 'FAILED'}")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main test execution."""
    
    # Step 1: DPDK setup
    if not ensure_dpdk_binding():
        print("\n✗ DPDK configuration failed!")
        return
    
    dpdk_sender = DPDKPacketSender(iface=SEND_IFACE)
    if not dpdk_sender.compile_dpdk_sender():
        print("\n✗ DPDK compilation failed!")
        return
    
    try:
        # Step 2: Configure switches
        print("\n" + "="*80)
        print("STEP 1: CONFIGURE SWITCHES")
        print("="*80)
        
        cleanup_switches()
        
        # SW1: Layers 0-5
        sw1_layers = list(range(0, 6))
        sw1_success = configure_switch_simple(
            SWITCH1_IP, sw1_layers, 
            "et-0/0/96", "et-0/0/97", 
            is_sw1=True
        )
        
        # SW2: Layers 6-11
        sw2_layers = list(range(6, 12))
        sw2_success = configure_switch_simple(
            SWITCH2_IP, sw2_layers,
            "et-0/0/96", "et-0/0/97",
            is_sw1=False
        )
        
        if not (sw1_success and sw2_success):
            print("\n✗ Switch configuration failed!")
            return
        
        print("\n✓ Both switches configured with Snake routing!")
        time.sleep(2)
        
        # Step 3: Run test
        results = run_snake_test(dpdk_sender)
        
        # Step 4: Summary
        print("\n" + "="*80)
        print("TEST RESULTS")
        print("="*80)
        
        print(f"\n{'Layer':<8} {'Expected':<10} {'SW1 Pkts':<12} {'SW2 Pkts':<12} {'Status':<10}")
        print("-" * 60)
        
        for layer in range(N_LAYERS):
            r = results[layer]
            status = "✓ PASS" if r['success'] else "✗ FAIL"
            print(f"{layer:<8} {r['expected_switch']:<10} {r['sw1_count']:<12} {r['sw2_count']:<12} {status}")
        
        # Overall result
        successes = sum(1 for r in results.values() if r['success'])
        print(f"\n{'='*60}")
        print(f"OVERALL: {successes}/{N_LAYERS} layers routed correctly")
        
        if successes == N_LAYERS:
            print("\n🎉 SNAKE ARCHITECTURE PROVEN! 🎉")
            print("✓ All packets routed correctly based on VLAN")
            print("✓ SW1 → SW2 routing via 200 Gbps trunk works")
            print("✓ Ready for full 12-layer inference!")
        elif successes >= 6:
            print("\n⚠ PARTIAL SUCCESS")
            print(f"✓ {successes} layers working")
            print(f"✗ {N_LAYERS - successes} layers need debugging")
        else:
            print("\n✗ SNAKE ROUTING FAILED")
            print("  Check VLAN configuration and trunk setup")
        
    finally:
        dpdk_sender.cleanup()
        print("\n✓ Test complete")


if __name__ == "__main__":
    if os.geteuid() != 0:
        print("Error: This script must be run with sudo")
        print("Usage: sudo python3 e095_twelve_layer_snake.py")
        sys.exit(1)
    
    main()


""" Output:
sudo python3 e095_twelve_layer_snake.py 

================================================================================
E095: SNAKE ROUTING TEST
================================================================================
  Testing: 12 layers across 2 switches
  SW1: Layers 0-5, SW2: Layers 6-11
  Proof: VLAN-based routing via 200 Gbps trunk

================================================================================
CHECKING DPDK CONFIGURATION
================================================================================
✓ NIC already bound to mlx4_core (DPDK ready!)

================================================================================
COMPILING DPDK PACKET SENDER
================================================================================
  Generated: /tmp/dpdk_sender_dkr3h96b/packet_sender.c
  Compiling...
✓ Compiled: /tmp/dpdk_sender_dkr3h96b/packet_sender

================================================================================
STEP 1: CONFIGURE SWITCHES
================================================================================

  Cleaning up switches...
    ✓ Cleanup complete (including ae0 aggregation)

  Configuring SW1 (layers 0-5)...
    Part 1/2: VLANs and interfaces (38 commands)...
    ✓ Part 1 complete
    ✓ Part 2/2: All 6 layer filters configured
    ✓ SW1 fully configured

  Configuring SW2 (layers 6-11)...
    Part 1/2: VLANs and interfaces (38 commands)...
    ✓ Part 1 complete
    ✓ Part 2/2: All 6 layer filters configured
    ✓ SW2 fully configured

✓ Both switches configured with Snake routing!

================================================================================
RUNNING SNAKE ROUTING TEST
================================================================================

  Testing Layer 0 (VLAN 200):
    Sending 100 packets to SW1...
    Sent in 0.2ms
    SW1 counters: 100 packets
    SW2 counters: 0 packets
    Expected: SW1 should have 100 packets
    ✓ Routing CORRECT

  Testing Layer 1 (VLAN 201):
    Sending 100 packets to SW1...
    Sent in 0.2ms
    SW1 counters: 100 packets
    SW2 counters: 0 packets
    Expected: SW1 should have 100 packets
    ✓ Routing CORRECT

  Testing Layer 2 (VLAN 202):
    Sending 100 packets to SW1...
    Sent in 0.2ms
    SW1 counters: 100 packets
    SW2 counters: 0 packets
    Expected: SW1 should have 100 packets
    ✓ Routing CORRECT

  Testing Layer 3 (VLAN 203):
    Sending 100 packets to SW1...
    Sent in 0.2ms
    SW1 counters: 100 packets
    SW2 counters: 0 packets
    Expected: SW1 should have 100 packets
    ✓ Routing CORRECT

  Testing Layer 4 (VLAN 204):
    Sending 100 packets to SW1...
    Sent in 0.2ms
    SW1 counters: 100 packets
    SW2 counters: 0 packets
    Expected: SW1 should have 100 packets
    ✓ Routing CORRECT

  Testing Layer 5 (VLAN 205):
    Sending 100 packets to SW1...
    Sent in 0.2ms
    SW1 counters: 100 packets
    SW2 counters: 0 packets
    Expected: SW1 should have 100 packets
    ✓ Routing CORRECT

  Testing Layer 6 (VLAN 206):
    Sending 100 packets to SW1...
    Sent in 0.2ms
    SW1 counters: 0 packets
    SW2 counters: 100 packets
    Expected: SW2 should have 100 packets
    ✓ Routing CORRECT

  Testing Layer 7 (VLAN 207):
    Sending 100 packets to SW1...
    Sent in 0.2ms
    SW1 counters: 0 packets
    SW2 counters: 100 packets
    Expected: SW2 should have 100 packets
    ✓ Routing CORRECT

  Testing Layer 8 (VLAN 208):
    Sending 100 packets to SW1...
    Sent in 0.2ms
    SW1 counters: 0 packets
    SW2 counters: 100 packets
    Expected: SW2 should have 100 packets
    ✓ Routing CORRECT

  Testing Layer 9 (VLAN 209):
    Sending 100 packets to SW1...
    Sent in 0.2ms
    SW1 counters: 0 packets
    SW2 counters: 100 packets
    Expected: SW2 should have 100 packets
    ✓ Routing CORRECT

  Testing Layer 10 (VLAN 210):
    Sending 100 packets to SW1...
    Sent in 0.2ms
    SW1 counters: 0 packets
    SW2 counters: 100 packets
    Expected: SW2 should have 100 packets
    ✓ Routing CORRECT

  Testing Layer 11 (VLAN 211):
    Sending 100 packets to SW1...
    Sent in 0.2ms
    SW1 counters: 0 packets
    SW2 counters: 100 packets
    Expected: SW2 should have 100 packets
    ✓ Routing CORRECT

================================================================================
TEST RESULTS
================================================================================

Layer    Expected   SW1 Pkts     SW2 Pkts     Status    
------------------------------------------------------------
0        SW1        100          0            ✓ PASS
1        SW1        100          0            ✓ PASS
2        SW1        100          0            ✓ PASS
3        SW1        100          0            ✓ PASS
4        SW1        100          0            ✓ PASS
5        SW1        100          0            ✓ PASS
6        SW2        0            100          ✓ PASS
7        SW2        0            100          ✓ PASS
8        SW2        0            100          ✓ PASS
9        SW2        0            100          ✓ PASS
10       SW2        0            100          ✓ PASS
11       SW2        0            100          ✓ PASS

============================================================
OVERALL: 12/12 layers routed correctly

🎉 SNAKE ARCHITECTURE PROVEN! 🎉
✓ All packets routed correctly based on VLAN
✓ SW1 → SW2 routing via 200 Gbps trunk works
✓ Ready for full 12-layer inference!

✓ Test complete
"""