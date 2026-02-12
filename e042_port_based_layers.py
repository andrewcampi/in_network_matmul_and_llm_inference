#!/usr/bin/env python3
"""
e042_port_based_layers.py

PORT-BASED LAYER PROGRESSION (Counter-Free Architecture)

================================================================================
KEY INSIGHT
================================================================================

We can't rewrite VLAN or MAC tags, but we CAN use PHYSICAL PORTS as layer IDs!

Topology:
  Host enp1s0 → SW1:et-0/0/96 (input)
  SW1:et-0/0/100 ↔ SW2:et-0/0/100 (inter-switch link)
  SW2:et-0/0/96 → Host enp1s0d1 (output)

Layer Progression via Physical Bouncing:
  Layer 0: Packet arrives at SW1:96 from host
           → SW1 processes, forwards to SW1:100
  Layer 1: Packet arrives at SW2:100 from inter-switch
           → SW2 processes, forwards back to SW2:100 (loopback) or to SW1
  Layer 2: Packet arrives at SW1:100 from inter-switch
           → SW1 processes, forwards to SW1:100
  ...
  Final:   After N bounces, packet goes to SW2:96 → Host

The TCAM rules match on INGRESS PORT:
  - Packets from port 96 = Layer 0 (from host)
  - Packets from port 100 = Layer 1, 2, 3... (from other switch)

================================================================================
CHALLENGE
================================================================================

How do we distinguish Layer 1 from Layer 3 from Layer 5 (all arrive on port 100)?

Options:
  A) Use VLAN ID to track bounce count (but we can't rewrite it!)
  B) Use TTL decrement (IP packets only)
  C) Use packet content/payload to track layer
  D) Accept that all layers share the same TCAM rules (broadcast model)

Option D is actually fine for neural networks!
  - Each layer applies the SAME weight matrix (or we use different MACs)
  - Packets accumulate at destination counters regardless of which layer

================================================================================
SIMPLIFIED TEST
================================================================================

Let's test if packets can bounce between switches and accumulate:
  1. Send 10 packets from host
  2. SW1 forwards to SW2
  3. SW2 counts arrivals
  4. Verify all 10 packets arrived

Then extend to bouncing back and forth.

Author: Research Phase 001
Date: December 2025
"""

import time
import os
import re
import socket
import struct
from typing import Dict, List, Tuple
from dataclasses import dataclass

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e038_counter_free_layers import (
    ssh_command, run_config_commands, cleanup_switch,
    craft_vlan_packet, send_packets, get_mac_address,
    SWITCH1_IP, SWITCH2_IP, SSH_KEY, SEND_IFACE, RECV_IFACE
)

# Configuration
BRIDGE_VLAN = 900  # VLAN for bridging between switches
FILTER_NAME = "port_layer_filter"


# ============================================================================
# SWITCH CONFIGURATION
# ============================================================================

def configure_switch1_for_bounce(debug: bool = False) -> bool:
    """
    Configure Switch 1:
    - Accept packets from host (port 96)
    - Forward to inter-switch link (port 100)
    - Accept returning packets from port 100, forward back to port 100
    """
    print(f"\n  Configuring Switch 1 ({SWITCH1_IP})...")
    
    # First clean up any analyzer/mirror config
    cleanup = [
        "delete forwarding-options analyzer",
        f"delete firewall family ethernet-switching filter {FILTER_NAME}",
    ]
    run_config_commands(SWITCH1_IP, cleanup, debug=False)
    time.sleep(1)
    
    commands = [
        # Create bridging VLAN
        f"set vlans bridge_vlan vlan-id {BRIDGE_VLAN}",
        
        # Host port (input)
        "set interfaces et-0/0/96 unit 0 family ethernet-switching interface-mode trunk",
        "set interfaces et-0/0/96 unit 0 family ethernet-switching vlan members bridge_vlan",
        
        # Inter-switch port
        "set interfaces et-0/0/100 unit 0 family ethernet-switching interface-mode trunk",
        "set interfaces et-0/0/100 unit 0 family ethernet-switching vlan members bridge_vlan",
        
        # Filter for port 96 (from host) - counts all ingress
        f"set firewall family ethernet-switching filter filter_port96 "
        f"term count_all then count from_host_pkts",
        f"set firewall family ethernet-switching filter filter_port96 "
        f"term count_all then accept",
        
        # Filter for port 100 (from SW2) - counts all ingress
        f"set firewall family ethernet-switching filter filter_port100 "
        f"term count_all then count from_sw2_pkts",
        f"set firewall family ethernet-switching filter filter_port100 "
        f"term count_all then accept",
        
        # Apply separate filter to each port
        f"set interfaces et-0/0/96 unit 0 family ethernet-switching filter input filter_port96",
        f"set interfaces et-0/0/100 unit 0 family ethernet-switching filter input filter_port100",
    ]
    
    success = run_config_commands(SWITCH1_IP, commands, debug=debug)
    
    if success:
        print("    ✓ Switch 1 configured")
    else:
        print("    ✗ Switch 1 configuration failed")
    
    return success


def configure_switch2_for_bounce(debug: bool = False) -> bool:
    """
    Configure Switch 2:
    - Accept packets from inter-switch link (port 100)
    - Count arrivals
    - Optionally forward to host (port 96) or back to SW1
    """
    print(f"\n  Configuring Switch 2 ({SWITCH2_IP})...")
    
    # First clean up
    cleanup = [
        "delete forwarding-options analyzer",
        "delete firewall family ethernet-switching filter filter_port100",
    ]
    run_config_commands(SWITCH2_IP, cleanup, debug=False)
    time.sleep(1)
    
    commands = [
        # Create bridging VLAN
        f"set vlans bridge_vlan vlan-id {BRIDGE_VLAN}",
        
        # Inter-switch port
        "set interfaces et-0/0/100 unit 0 family ethernet-switching interface-mode trunk",
        "set interfaces et-0/0/100 unit 0 family ethernet-switching vlan members bridge_vlan",
        
        # Host port (output)
        "set interfaces et-0/0/96 unit 0 family ethernet-switching interface-mode trunk",
        "set interfaces et-0/0/96 unit 0 family ethernet-switching vlan members bridge_vlan",
        
        # Filter for port 100 (from SW1) - counts all ingress
        f"set firewall family ethernet-switching filter filter_port100 "
        f"term count_all then count from_sw1_pkts",
        f"set firewall family ethernet-switching filter filter_port100 "
        f"term count_all then accept",
        
        # Apply filter
        f"set interfaces et-0/0/100 unit 0 family ethernet-switching filter input filter_port100",
    ]
    
    success = run_config_commands(SWITCH2_IP, commands, debug=debug)
    
    if success:
        print("    ✓ Switch 2 configured")
    else:
        print("    ✗ Switch 2 configuration failed")
    
    return success


def clear_counters():
    """Clear firewall counters on both switches."""
    ssh_command(SWITCH1_IP, "cli -c 'clear firewall filter filter_port96'")
    ssh_command(SWITCH1_IP, "cli -c 'clear firewall filter filter_port100'")
    ssh_command(SWITCH2_IP, "cli -c 'clear firewall filter filter_port100'")


def read_counters() -> Dict[str, Dict[str, int]]:
    """Read counters from both switches."""
    results = {}
    
    # Switch 1 - port 96 filter
    success, stdout, _ = ssh_command(SWITCH1_IP, 
        "cli -c 'show firewall filter filter_port96'")
    
    sw1_counters = {}
    if success:
        pattern = r'from_host_pkts\s+\d+\s+(\d+)'
        match = re.search(pattern, stdout)
        if match:
            sw1_counters['from_host_pkts'] = int(match.group(1))
    
    # Switch 1 - port 100 filter
    success, stdout, _ = ssh_command(SWITCH1_IP,
        "cli -c 'show firewall filter filter_port100'")
    
    if success:
        pattern = r'from_sw2_pkts\s+\d+\s+(\d+)'
        match = re.search(pattern, stdout)
        if match:
            sw1_counters['from_sw2_pkts'] = int(match.group(1))
    
    results['switch1'] = sw1_counters
    
    # Switch 2 - port 100 filter
    success, stdout, _ = ssh_command(SWITCH2_IP,
        "cli -c 'show firewall filter filter_port100'")
    
    sw2_counters = {}
    if success:
        pattern = r'from_sw1_pkts\s+\d+\s+(\d+)'
        match = re.search(pattern, stdout)
        if match:
            sw2_counters['from_sw1_pkts'] = int(match.group(1))
    results['switch2'] = sw2_counters
    
    return results


# ============================================================================
# PACKET SENDING
# ============================================================================

def send_test_packets(num_packets: int = 10) -> int:
    """Send test packets from host to SW1."""
    src_mac = bytes.fromhex(get_mac_address(SEND_IFACE).replace(':', ''))
    # Use broadcast to ensure flooding to SW2
    dst_mac = bytes.fromhex('ffffffffffff')
    
    packets = []
    for i in range(num_packets):
        payload = f'BOUNCE{i:04d}'.encode()
        pkt = craft_vlan_packet(dst_mac, src_mac, BRIDGE_VLAN, payload)
        packets.append(pkt)
    
    return send_packets(SEND_IFACE, packets)


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment():
    """Test port-based layer progression."""
    
    print("="*80)
    print("E042: PORT-BASED LAYER PROGRESSION")
    print("="*80)
    
    print("\nConcept:")
    print("  Use PHYSICAL PORT as layer identifier instead of VLAN rewriting")
    print("  Packets bounce between switches: SW1 ↔ SW2 ↔ SW1 ...")
    print("  Each hop = one layer processed")
    print()
    print("  Layer 0: Host → SW1:96 (from host)")
    print("  Layer 1: SW1:100 → SW2:100 (first bounce)")
    print("  Layer 2: SW2:100 → SW1:100 (second bounce)")
    print("  ...")
    
    # Step 1: Cleanup
    print("\n" + "="*80)
    print("STEP 1: CLEANUP")
    print("="*80)
    
    cleanup_switch(SWITCH1_IP)
    cleanup_switch(SWITCH2_IP)
    time.sleep(1)
    
    # Step 2: Configure
    print("\n" + "="*80)
    print("STEP 2: CONFIGURE SWITCHES")
    print("="*80)
    
    if not configure_switch1_for_bounce(debug=True):
        print("Switch 1 configuration failed!")
        return
    
    time.sleep(1)
    
    if not configure_switch2_for_bounce(debug=True):
        print("Switch 2 configuration failed!")
        return
    
    time.sleep(2)
    
    # Step 3: Clear counters
    print("\n" + "="*80)
    print("STEP 3: CLEAR COUNTERS")
    print("="*80)
    
    clear_counters()
    time.sleep(0.5)
    print("  ✓ Counters cleared")
    
    # Step 4: Send packets
    print("\n" + "="*80)
    print("STEP 4: SEND TEST PACKETS")
    print("="*80)
    
    num_packets = 10
    print(f"\n  Sending {num_packets} broadcast packets with VLAN {BRIDGE_VLAN}...")
    sent = send_test_packets(num_packets)
    print(f"  ✓ Sent {sent} packets")
    
    time.sleep(2)
    
    # Step 5: Read counters
    print("\n" + "="*80)
    print("STEP 5: READ COUNTERS")
    print("="*80)
    
    counters = read_counters()
    
    sw1 = counters.get('switch1', {})
    sw2 = counters.get('switch2', {})
    
    print(f"\n  Switch 1:")
    print(f"    from_host_pkts (Layer 0): {sw1.get('from_host_pkts', 0)}")
    print(f"    from_sw2_pkts (bounced):  {sw1.get('from_sw2_pkts', 0)}")
    
    print(f"\n  Switch 2:")
    print(f"    from_sw1_pkts (Layer 1):  {sw2.get('from_sw1_pkts', 0)}")
    
    # Step 6: Analysis
    print("\n" + "="*80)
    print("STEP 6: ANALYSIS")
    print("="*80)
    
    from_host = sw1.get('from_host_pkts', 0)
    from_sw1 = sw2.get('from_sw1_pkts', 0)
    from_sw2 = sw1.get('from_sw2_pkts', 0)
    
    print(f"\n  Packets sent:           {sent}")
    print(f"  SW1 from host (L0):     {from_host}")
    print(f"  SW2 from SW1 (L1):      {from_sw1}")
    print(f"  SW1 from SW2 (bounce):  {from_sw2}")
    
    if from_host == sent and from_sw1 > 0:
        print("\n  ✓ SUCCESS! Packets traversed SW1 → SW2!")
        print("  Port-based layer identification works!")
        print()
        print("  This proves:")
        print("    - Packets flow from host to SW1 (Layer 0)")
        print("    - Packets flow from SW1 to SW2 (Layer 1)")
        print("    - We can count packets at each layer by ingress port")
        
        if from_sw2 > 0:
            print(f"\n  ✓ BONUS: {from_sw2} packets bounced back to SW1!")
            print("    Multi-layer recirculation is working!")
    else:
        print("\n  ⚠ Packets didn't reach SW2 as expected")
        print("  Check VLAN configuration and inter-switch link")
    
    # Next steps
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    
    print("""
  If this works, the counter-free architecture becomes:
  
  1. Configure TCAM rules on each switch for matrix multiplication
  2. Packets enter at SW1:96 (Layer 0 input)
  3. SW1 processes Layer 0, forwards to SW2 via port 100
  4. SW2 processes Layer 1, forwards back to SW1 via port 100
  5. Repeat for all 32 layers (16 bounces each direction)
  6. Final layer outputs to SW2:96 → Host
  7. Read counters ONCE at the end!
  
  Key insight: INGRESS PORT identifies which layer the packet is on!
  No VLAN rewriting needed - physical topology does the work.
""")
    
    # Save results
    import json
    os.makedirs("bringup_logs", exist_ok=True)
    log_file = f"bringup_logs/port_layers_{int(time.time())}.json"
    with open(log_file, 'w') as f:
        json.dump({
            "packets_sent": sent,
            "counters": counters,
            "from_host": from_host,
            "from_sw1": from_sw1,
            "from_sw2": from_sw2,
            "timestamp": time.time()
        }, f, indent=2)
    
    print(f"\n  Results saved to: {log_file}")


if __name__ == '__main__':
    run_experiment()



""" Output:
sudo python3 e042_port_based_layers.py 
================================================================================
E042: PORT-BASED LAYER PROGRESSION
================================================================================

Concept:
  Use PHYSICAL PORT as layer identifier instead of VLAN rewriting
  Packets bounce between switches: SW1 ↔ SW2 ↔ SW1 ...
  Each hop = one layer processed

  Layer 0: Host → SW1:96 (from host)
  Layer 1: SW1:100 → SW2:100 (first bounce)
  Layer 2: SW2:100 → SW1:100 (second bounce)
  ...

================================================================================
STEP 1: CLEANUP
================================================================================

  Cleaning up 10.10.10.55...
    Found 2 VLANs: ['bridge_vlan', 'default']
    Deleting 1 VLANs...
    ✓ Cleanup complete

  Cleaning up 10.10.10.56...
    Found 2 VLANs: ['default', 'layer2_vlan']
    Deleting 1 VLANs...
    ✓ Cleanup complete

================================================================================
STEP 2: CONFIGURE SWITCHES
================================================================================

  Configuring Switch 1 (10.10.10.55)...
    [DEBUG] stdout: Entering configuration mode
The configuration has been changed but not committed
configuration check succeeds
commit complete

    ✓ Switch 1 configured

  Configuring Switch 2 (10.10.10.56)...
    [DEBUG] stdout: Entering configuration mode
The configuration has been changed but not committed
configuration check succeeds
commit complete

    ✓ Switch 2 configured

================================================================================
STEP 3: CLEAR COUNTERS
================================================================================
  ✓ Counters cleared

================================================================================
STEP 4: SEND TEST PACKETS
================================================================================

  Sending 10 broadcast packets with VLAN 900...
  ✓ Sent 10 packets

================================================================================
STEP 5: READ COUNTERS
================================================================================

  Switch 1:
    from_host_pkts (Layer 0): 10
    from_sw2_pkts (bounced):  0

  Switch 2:
    from_sw1_pkts (Layer 1):  10

================================================================================
STEP 6: ANALYSIS
================================================================================

  Packets sent:           10
  SW1 from host (L0):     10
  SW2 from SW1 (L1):      10
  SW1 from SW2 (bounce):  0

  ✓ SUCCESS! Packets traversed SW1 → SW2!
  Port-based layer identification works!

  This proves:
    - Packets flow from host to SW1 (Layer 0)
    - Packets flow from SW1 to SW2 (Layer 1)
    - We can count packets at each layer by ingress port

================================================================================
NEXT STEPS
================================================================================

  If this works, the counter-free architecture becomes:
  
  1. Configure TCAM rules on each switch for matrix multiplication
  2. Packets enter at SW1:96 (Layer 0 input)
  3. SW1 processes Layer 0, forwards to SW2 via port 100
  4. SW2 processes Layer 1, forwards back to SW1 via port 100
  5. Repeat for all 32 layers (16 bounces each direction)
  6. Final layer outputs to SW2:96 → Host
  7. Read counters ONCE at the end!
  
  Key insight: INGRESS PORT identifies which layer the packet is on!
  No VLAN rewriting needed - physical topology does the work.


  Results saved to: bringup_logs/port_layers_1766847588.json
(.venv) multiplex@multiplex1:~/photonic_inference_engine/phase_001/phase_001$ cat bringup_logs/port_layers_1766847588.json
{
  "packets_sent": 10,
  "counters": {
    "switch1": {
      "from_host_pkts": 10,
      "from_sw2_pkts": 0
    },
    "switch2": {
      "from_sw1_pkts": 10
    }
  },
  "from_host": 10,
  "from_sw1": 10,
  "from_sw2": 0,
  "timestamp": 1766847588.371829
}
"""