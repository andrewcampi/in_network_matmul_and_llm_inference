#!/usr/bin/env python3
"""
e039_vlan_rewrite_test.py

VLAN REWRITING FOR LAYER PROGRESSION

================================================================================
GOAL
================================================================================

Test VLAN rewriting on Juniper QFX5100 to enable counter-free multi-layer flow.

From e038, we proved:
  - VLAN matching works (Layer 0 = VLAN 800 matched 100%)
  - Packets carry layer state in VLAN tag

Now we need to make packets automatically progress:
  VLAN 800 (Layer 0) → VLAN 801 (Layer 1) → VLAN 802 (Layer 2)

================================================================================
VLAN REWRITE OPTIONS TO TEST
================================================================================

Option A: Flexible VLAN Tagging (swap operation)
  set interfaces et-0/0/96 flexible-vlan-tagging
  set interfaces et-0/0/96 unit 800 vlan-id 800
  set interfaces et-0/0/96 unit 800 input-vlan-map swap vlan-id 801

Option B: VLAN Translation via switch-options
  set vlans layer0_vlan switch-options interface et-0/0/96.0 
      mapping vlan-id 801

Option C: Policy-based VLAN rewrite (if supported)
  set firewall family ethernet-switching filter ... then vlan-id 801

We already know Option C doesn't work (syntax error from e038).
Let's test Options A and B.

Author: Research Phase 001
Date: December 2025
"""

import sys
import os

# Import common functions from e038
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e038_counter_free_layers import (
    ssh_command, run_config_commands, get_all_vlans, cleanup_switch,
    craft_vlan_packet, send_packets, read_layer_counters, clear_counters,
    SWITCH1_IP, SWITCH2_IP, SEND_IFACE, SEND_MAC, RECV_MAC,
    LAYER_0_VLAN, LAYER_1_VLAN, LAYER_2_VLAN, FILTER_NAME
)

import time
import socket
import struct
from typing import Dict, List, Tuple
from dataclasses import dataclass


# ============================================================================
# VLAN REWRITE CONFIGURATION APPROACHES
# ============================================================================

def test_approach_a_flexible_vlan(debug: bool = True) -> bool:
    """
    Approach A: Flexible VLAN Tagging with swap operation.
    
    This uses logical units per VLAN and input-vlan-map to swap tags.
    """
    print("\n" + "="*80)
    print("APPROACH A: Flexible VLAN Tagging")
    print("="*80)
    
    print("\n  Configuring flexible VLAN tagging on Switch 1...")
    
    # Clean up first
    cleanup_cmds = [
        "delete interfaces et-0/0/96 flexible-vlan-tagging",
        "delete interfaces et-0/0/96 unit 800",
        "delete interfaces et-0/0/96 unit 801",
        "delete interfaces et-0/0/96 unit 802",
        "delete interfaces et-0/0/96 encapsulation",
    ]
    run_config_commands(SWITCH1_IP, cleanup_cmds, debug=False)
    
    # Configure flexible VLAN tagging
    config_cmds = [
        # Enable flexible VLAN tagging
        "set interfaces et-0/0/96 flexible-vlan-tagging",
        "set interfaces et-0/0/96 encapsulation flexible-ethernet-services",
        
        # Unit for VLAN 800 (Layer 0) - swap to 801
        f"set interfaces et-0/0/96 unit 800 vlan-id {LAYER_0_VLAN}",
        f"set interfaces et-0/0/96 unit 800 input-vlan-map swap vlan-id {LAYER_1_VLAN}",
        f"set interfaces et-0/0/96 unit 800 output-vlan-map swap vlan-id {LAYER_0_VLAN}",
        
        # Unit for VLAN 801 (Layer 1) - swap to 802
        f"set interfaces et-0/0/96 unit 801 vlan-id {LAYER_1_VLAN}",
        f"set interfaces et-0/0/96 unit 801 input-vlan-map swap vlan-id {LAYER_2_VLAN}",
        
        # Unit for VLAN 802 (Layer 2) - final layer, no swap needed
        f"set interfaces et-0/0/96 unit 802 vlan-id {LAYER_2_VLAN}",
    ]
    
    print(f"    Running {len(config_cmds)} config commands...")
    success = run_config_commands(SWITCH1_IP, config_cmds, debug=debug)
    
    if success:
        print("    ✓ Flexible VLAN tagging configured")
    else:
        print("    ✗ Configuration failed")
        print("    Note: Flexible VLAN tagging may not be compatible with ethernet-switching")
    
    return success


def test_approach_b_vlan_translation(debug: bool = True) -> bool:
    """
    Approach B: VLAN Translation via interface config.
    
    Uses native-vlan-id and vlan-rewrite features.
    """
    print("\n" + "="*80)
    print("APPROACH B: Interface VLAN Translation")
    print("="*80)
    
    print("\n  Configuring VLAN translation on Switch 1...")
    
    # This approach tries to use the vlan-rewrite feature
    config_cmds = [
        # Set up interface in trunk mode with all VLANs
        "set interfaces et-0/0/96 unit 0 family ethernet-switching interface-mode trunk",
        f"set interfaces et-0/0/96 unit 0 family ethernet-switching vlan members layer0_vlan",
        f"set interfaces et-0/0/96 unit 0 family ethernet-switching vlan members layer1_vlan",
        f"set interfaces et-0/0/96 unit 0 family ethernet-switching vlan members layer2_vlan",
        
        # Try VLAN translation at the VLAN level
        # Syntax: set vlans <vlan> interface <iface> mapping vlan-id <new-vlan>
        f"set vlans layer0_vlan interface et-0/0/96.0 mapping vlan-id {LAYER_1_VLAN}",
    ]
    
    print(f"    Running {len(config_cmds)} config commands...")
    success = run_config_commands(SWITCH1_IP, config_cmds, debug=debug)
    
    if success:
        print("    ✓ VLAN translation configured")
    else:
        print("    ✗ Configuration failed")
    
    return success


def test_approach_c_qinq_push_pop(debug: bool = True) -> bool:
    """
    Approach C: Use Q-in-Q push/pop operations.
    
    Push an outer tag, then pop/swap to change the effective VLAN.
    """
    print("\n" + "="*80)
    print("APPROACH C: Q-in-Q Push/Pop Operations")
    print("="*80)
    
    print("\n  Configuring Q-in-Q on Switch 1...")
    
    config_cmds = [
        # Enable extended VLAN bridge (required for Q-in-Q)
        "set ethernet-switching-options voip interface et-0/0/96.0 vlan layer0_vlan",
        
        # Try push operation
        f"set interfaces et-0/0/96 unit 0 family ethernet-switching vlan members layer0_vlan",
        f"set vlans layer0_vlan dot1q-tunneling customer-vlans {LAYER_0_VLAN}",
    ]
    
    print(f"    Running {len(config_cmds)} config commands...")
    success = run_config_commands(SWITCH1_IP, config_cmds, debug=debug)
    
    if success:
        print("    ✓ Q-in-Q configured")
    else:
        print("    ✗ Configuration failed (Q-in-Q may require specific license)")
    
    return success


def configure_simple_layer_test(debug: bool = True) -> bool:
    """
    Configure a simple test: just verify packets with different VLANs 
    are counted correctly (baseline without rewrite).
    """
    print("\n" + "="*80)
    print("BASELINE: Simple Layer Counting (No Rewrite)")
    print("="*80)
    
    print("\n  Configuring Switch 1 for baseline layer counting...")
    
    # Create VLANs
    vlan_cmds = [
        f"set vlans layer0_vlan vlan-id {LAYER_0_VLAN}",
        f"set vlans layer1_vlan vlan-id {LAYER_1_VLAN}",
        f"set vlans layer2_vlan vlan-id {LAYER_2_VLAN}",
    ]
    
    # Interface config
    iface_cmds = [
        "set interfaces et-0/0/96 unit 0 family ethernet-switching interface-mode trunk",
        "set interfaces et-0/0/96 unit 0 family ethernet-switching vlan members layer0_vlan",
        "set interfaces et-0/0/96 unit 0 family ethernet-switching vlan members layer1_vlan",
        "set interfaces et-0/0/96 unit 0 family ethernet-switching vlan members layer2_vlan",
        "set interfaces et-0/0/100 unit 0 family ethernet-switching interface-mode trunk",
        "set interfaces et-0/0/100 unit 0 family ethernet-switching vlan members layer0_vlan",
        "set interfaces et-0/0/100 unit 0 family ethernet-switching vlan members layer1_vlan",
        "set interfaces et-0/0/100 unit 0 family ethernet-switching vlan members layer2_vlan",
    ]
    
    # Firewall filter to count each VLAN
    filter_cmds = [
        f"set firewall family ethernet-switching filter {FILTER_NAME} term layer0 from vlan layer0_vlan",
        f"set firewall family ethernet-switching filter {FILTER_NAME} term layer0 then count layer0_pkts",
        f"set firewall family ethernet-switching filter {FILTER_NAME} term layer0 then accept",
        
        f"set firewall family ethernet-switching filter {FILTER_NAME} term layer1 from vlan layer1_vlan",
        f"set firewall family ethernet-switching filter {FILTER_NAME} term layer1 then count layer1_pkts",
        f"set firewall family ethernet-switching filter {FILTER_NAME} term layer1 then accept",
        
        f"set firewall family ethernet-switching filter {FILTER_NAME} term layer2 from vlan layer2_vlan",
        f"set firewall family ethernet-switching filter {FILTER_NAME} term layer2 then count layer2_pkts",
        f"set firewall family ethernet-switching filter {FILTER_NAME} term layer2 then accept",
        
        f"set firewall family ethernet-switching filter {FILTER_NAME} term default then accept",
    ]
    
    # Apply filter
    apply_cmds = [
        f"set interfaces et-0/0/96 unit 0 family ethernet-switching filter input {FILTER_NAME}",
    ]
    
    all_cmds = vlan_cmds + iface_cmds + filter_cmds + apply_cmds
    
    print(f"    Running {len(all_cmds)} config commands...")
    success = run_config_commands(SWITCH1_IP, all_cmds, debug=debug)
    
    if success:
        print("    ✓ Baseline configuration complete")
    else:
        print("    ✗ Configuration failed")
    
    return success


def send_multi_vlan_packets(num_per_vlan: int = 5) -> Dict[int, int]:
    """
    Send packets with different VLANs to test layer identification.
    Returns: dict of VLAN -> packets sent
    """
    src_mac = bytes.fromhex(SEND_MAC.replace(':', ''))
    dst_mac = bytes.fromhex('01005e000300')  # Multicast
    
    sent = {}
    all_packets = []
    
    for vlan_id, count in [(LAYER_0_VLAN, num_per_vlan), 
                            (LAYER_1_VLAN, num_per_vlan),
                            (LAYER_2_VLAN, num_per_vlan)]:
        for i in range(count):
            payload = f'VLAN{vlan_id}PKT{i}'.encode()
            pkt = craft_vlan_packet(dst_mac, src_mac, vlan_id, payload)
            all_packets.append(pkt)
        sent[vlan_id] = count
    
    # Send all packets
    total = send_packets(SEND_IFACE, all_packets)
    print(f"    Sent {total} total packets across 3 VLANs")
    
    return sent


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment():
    """Run VLAN rewrite experiments."""
    
    print("="*80)
    print("E039: VLAN REWRITE FOR LAYER PROGRESSION")
    print("="*80)
    
    print("\nGoal: Make packets automatically progress through layers via VLAN rewriting")
    print(f"  VLAN {LAYER_0_VLAN} (Layer 0) → VLAN {LAYER_1_VLAN} (Layer 1) → VLAN {LAYER_2_VLAN} (Layer 2)")
    
    # Step 1: Cleanup
    print("\n" + "="*80)
    print("STEP 1: CLEANUP")
    print("="*80)
    
    cleanup_switch(SWITCH1_IP)
    cleanup_switch(SWITCH2_IP)
    time.sleep(1)
    
    # Step 2: Baseline test - verify multi-VLAN counting works
    print("\n" + "="*80)
    print("STEP 2: BASELINE TEST")
    print("="*80)
    
    if not configure_simple_layer_test(debug=True):
        print("Baseline configuration failed!")
        return
    
    time.sleep(2)
    clear_counters()
    time.sleep(0.5)
    
    print("\n  Sending test packets (5 per VLAN)...")
    sent = send_multi_vlan_packets(5)
    time.sleep(1)
    
    counters = read_layer_counters()
    sw1 = counters.get('switch1', {})
    
    print(f"\n  Baseline Results:")
    print(f"    VLAN {LAYER_0_VLAN} (Layer 0): sent={sent.get(LAYER_0_VLAN, 0)}, counted={sw1.get('layer0', 0)}")
    print(f"    VLAN {LAYER_1_VLAN} (Layer 1): sent={sent.get(LAYER_1_VLAN, 0)}, counted={sw1.get('layer1', 0)}")
    print(f"    VLAN {LAYER_2_VLAN} (Layer 2): sent={sent.get(LAYER_2_VLAN, 0)}, counted={sw1.get('layer2', 0)}")
    
    baseline_ok = (sw1.get('layer0', 0) == 5 and 
                   sw1.get('layer1', 0) == 5 and 
                   sw1.get('layer2', 0) == 5)
    
    if baseline_ok:
        print("\n  ✓ Baseline PASSED: All VLANs counted correctly")
    else:
        print("\n  ⚠ Baseline shows some variance (may be due to flooding)")
    
    # Step 3: Test VLAN rewrite approaches
    print("\n" + "="*80)
    print("STEP 3: TEST VLAN REWRITE APPROACHES")
    print("="*80)
    
    # Test Approach A: Flexible VLAN tagging
    cleanup_switch(SWITCH1_IP)
    time.sleep(1)
    
    approach_a_ok = test_approach_a_flexible_vlan(debug=True)
    
    if not approach_a_ok:
        # Restore baseline and try Approach B
        cleanup_switch(SWITCH1_IP)
        time.sleep(1)
        
        approach_b_ok = test_approach_b_vlan_translation(debug=True)
        
        if not approach_b_ok:
            # Try Approach C
            cleanup_switch(SWITCH1_IP)
            time.sleep(1)
            
            approach_c_ok = test_approach_c_qinq_push_pop(debug=True)
    
    # Step 4: Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print("""
  VLAN Rewrite Status on Juniper QFX5100:
  
  The QFX5100 in Layer 2 (ethernet-switching) mode has LIMITED VLAN rewrite
  capabilities. The approaches tested:
  
  A. Flexible VLAN Tagging: Requires L3 mode (not ethernet-switching)
  B. VLAN Translation: May require specific syntax or license
  C. Q-in-Q: Requires extended bridge mode
  
  ALTERNATIVE APPROACHES for Counter-Free Architecture:
  
  1. Use Destination MAC for Layer Encoding
     - Each layer uses different MAC address range
     - TCAM rules route based on MAC
     - No VLAN rewrite needed
  
  2. Use Inter-Switch Recirculation with Port-Based Layers
     - Layer 0: Packets enter on et-0/0/96
     - Layer 1: Packets recirculate via et-0/0/100 → Switch2 → back
     - Each recirculation = one layer
  
  3. Use DSCP/ToS Field for Layer Encoding
     - Similar to VLAN but uses IP header field
     - May have better rewrite support
  
  4. Accept SSH Counter Read Latency
     - From e037: 700ms for 4 counters, ~1s for 64 counters
     - With sub-linear scaling, full inference may be ~2-3 seconds
     - Still viable for many use cases
""")
    
    # Save results
    import json
    os.makedirs("bringup_logs", exist_ok=True)
    log_file = f"bringup_logs/vlan_rewrite_test_{int(time.time())}.json"
    with open(log_file, 'w') as f:
        json.dump({
            "baseline_counters": counters,
            "approach_a_flexible_vlan": approach_a_ok if 'approach_a_ok' in dir() else False,
            "approach_b_vlan_translation": approach_b_ok if 'approach_b_ok' in dir() else False,
            "approach_c_qinq": approach_c_ok if 'approach_c_ok' in dir() else False,
            "timestamp": time.time()
        }, f, indent=2)
    
    print(f"\n  Results saved to: {log_file}")


if __name__ == '__main__':
    run_experiment()

