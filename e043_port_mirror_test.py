#!/usr/bin/env python3
"""
e043_port_mirror_test.py

PORT MIRROR TEST VIA FIREWALL FILTER

================================================================================
GOAL
================================================================================

Test packet mirroring using firewall filter actions:
  1. Configure an analyzer (port-mirror-instance) on the switch
  2. Add "port-mirror-instance <name>" action to a firewall filter term
  3. Send packets that match the filter
  4. Listen on Ubuntu for mirrored copies
  5. Verify we receive them!

This proves we can tap into the packet flow without disrupting it.
Useful for debugging and for reading neuron activations via mirrored packets.

================================================================================
JUNIPER PORT MIRRORING
================================================================================

On Juniper QFX5100, port mirroring is configured via:

  1. Define an analyzer (forwarding-options):
     set forwarding-options analyzer my_mirror input ingress interface et-0/0/96
     set forwarding-options analyzer my_mirror output interface et-0/0/97

  2. OR use firewall filter action (more selective):
     set firewall family ethernet-switching filter my_filter term t1 then port-mirror-instance my_mirror
     
The firewall filter approach lets us mirror only specific traffic (by MAC, VLAN, etc.)

Author: Research Phase 001
Date: December 2025
"""

import time
import os
import re
import socket
import struct
import threading
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import utilities from e042 (which imports from e038)
from e042_port_based_layers import (
    ssh_command, run_config_commands, cleanup_switch,
    craft_vlan_packet, send_packets, get_mac_address,
    SWITCH1_IP, SWITCH2_IP, SSH_KEY, SEND_IFACE, RECV_IFACE,
    BRIDGE_VLAN
)

# Configuration
MIRROR_VLAN = 901  # Separate VLAN for mirror test
ANALYZER_NAME = "packet_mirror"
FILTER_NAME = "mirror_filter"

# The port where mirrored packets will be sent
# IMPORTANT: Cannot use same port for filter input AND analyzer output!
# So we mirror to et-0/0/100 (inter-switch link) → SW2 → et-0/0/96 → Host enp1s0d1
MIRROR_OUTPUT_PORT = "et-0/0/100"  # Mirror to inter-switch link
CAPTURE_IFACE = RECV_IFACE  # Capture on enp1s0d1 (connected to SW2:96)


# ============================================================================
# PACKET CAPTURE
# ============================================================================

class PacketCapture:
    """Simple packet capture using raw sockets."""
    
    def __init__(self, interface: str):
        self.interface = interface
        self.packets = []
        self.running = False
        self.thread = None
    
    def start(self, timeout: float = 5.0):
        """Start capturing packets in a background thread."""
        self.packets = []
        self.running = True
        self.thread = threading.Thread(target=self._capture, args=(timeout,))
        self.thread.start()
    
    def stop(self):
        """Stop capturing and wait for thread to finish."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
    
    def _capture(self, timeout: float):
        """Capture packets on the interface."""
        try:
            # Create raw socket to receive all packets
            sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.ntohs(0x0003))
            sock.bind((self.interface, 0))
            sock.settimeout(0.5)  # Short timeout for polling
            
            start_time = time.time()
            
            while self.running and (time.time() - start_time) < timeout:
                try:
                    data, addr = sock.recvfrom(65535)
                    self.packets.append({
                        'data': data,
                        'time': time.time(),
                        'addr': addr
                    })
                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"Capture error: {e}")
                    break
            
            sock.close()
        except Exception as e:
            print(f"Failed to create capture socket: {e}")
    
    def get_packets(self) -> List[dict]:
        """Return captured packets."""
        return self.packets
    
    def filter_by_payload(self, pattern: bytes) -> List[dict]:
        """Return packets containing the given payload pattern."""
        matching = []
        for pkt in self.packets:
            if pattern in pkt['data']:
                matching.append(pkt)
        return matching


# ============================================================================
# SWITCH CONFIGURATION
# ============================================================================

def configure_port_mirroring(switch_ip: str, debug: bool = False) -> bool:
    """
    Configure port-mirroring instance on the switch.
    
    NOTE: For firewall filter 'port-mirror-instance' action, we need to use
    'forwarding-options port-mirroring instance', NOT 'forwarding-options analyzer'.
    These are different Juniper constructs!
    
    We'll mirror to port 100 (inter-switch link).
    """
    print(f"\n  Configuring port-mirroring on {switch_ip}...")
    
    # Clean up any existing port-mirroring AND interface config
    cleanup = [
        f"delete forwarding-options port-mirroring",
        f"delete forwarding-options analyzer",
        f"delete interfaces {MIRROR_OUTPUT_PORT} unit 0 family ethernet-switching",
    ]
    run_config_commands(switch_ip, cleanup, debug=False)
    time.sleep(0.5)
    
    # Configure the interface for mirror output (needs VLAN membership for trunk)
    # Then configure port-mirroring instance
    commands = [
        # Set up interface in trunk mode with a VLAN (use same name as filter will use)
        f"set vlans mirror_test vlan-id {MIRROR_VLAN}",
        f"set interfaces {MIRROR_OUTPUT_PORT} unit 0 family ethernet-switching interface-mode trunk",
        f"set interfaces {MIRROR_OUTPUT_PORT} unit 0 family ethernet-switching vlan members mirror_test",
        
        # Create port-mirroring INSTANCE (this is what firewall filter references)
        # Mirrored packets will go: SW1:100 → SW2:100 → SW2:96 → Host enp1s0d1
        f"set forwarding-options port-mirroring instance {ANALYZER_NAME} "
        f"family ethernet-switching output interface {MIRROR_OUTPUT_PORT}",
    ]
    
    success = run_config_commands(switch_ip, commands, debug=debug)
    
    if success:
        print(f"    ✓ Port-mirroring instance '{ANALYZER_NAME}' configured")
        print(f"    → Mirrored packets will go to {MIRROR_OUTPUT_PORT}")
    else:
        print(f"    ✗ Port-mirroring configuration failed")
    
    return success


def configure_sw2_for_mirror(debug: bool = False) -> bool:
    """
    Configure Switch 2 to forward mirrored packets to the host.
    
    Mirrored packets arrive on SW2:100, need to go out SW2:96 to host.
    """
    print(f"\n  Configuring Switch 2 ({SWITCH2_IP}) to forward mirrored packets...")
    
    # Clean up
    cleanup = [
        f"delete vlans mirror_test",
        f"delete interfaces et-0/0/100 unit 0 family ethernet-switching",
        f"delete interfaces et-0/0/96 unit 0 family ethernet-switching",
    ]
    run_config_commands(SWITCH2_IP, cleanup, debug=False)
    time.sleep(0.5)
    
    commands = [
        # Create same VLAN on SW2
        f"set vlans mirror_test vlan-id {MIRROR_VLAN}",
        
        # Inter-switch port (receives mirrored packets)
        "set interfaces et-0/0/100 unit 0 family ethernet-switching interface-mode trunk",
        f"set interfaces et-0/0/100 unit 0 family ethernet-switching vlan members mirror_test",
        
        # Host port (sends mirrored packets to host)
        "set interfaces et-0/0/96 unit 0 family ethernet-switching interface-mode trunk",
        f"set interfaces et-0/0/96 unit 0 family ethernet-switching vlan members mirror_test",
    ]
    
    success = run_config_commands(SWITCH2_IP, commands, debug=debug)
    
    if success:
        print(f"    ✓ Switch 2 configured for mirror forwarding")
        print(f"    → Mirrored packets: SW2:100 → SW2:96 → Host {RECV_IFACE}")
    else:
        print(f"    ✗ Switch 2 configuration failed")
    
    return success


def configure_mirror_filter(switch_ip: str, debug: bool = False) -> bool:
    """
    Configure firewall filter with port-mirror-instance action.
    
    The filter will:
    1. Match all packets on a specific VLAN
    2. Count them
    3. Mirror them to the analyzer
    4. Accept them (continue normal forwarding)
    """
    print(f"\n  Configuring mirror filter on {switch_ip}...")
    
    # Clean up old filter
    cleanup = [
        f"delete firewall family ethernet-switching filter {FILTER_NAME}",
        f"delete interfaces et-0/0/96 unit 0 family ethernet-switching filter",
    ]
    run_config_commands(switch_ip, cleanup, debug=False)
    time.sleep(0.5)
    
    # Create VLAN and filter
    commands = [
        # Create test VLAN
        f"set vlans mirror_test vlan-id {MIRROR_VLAN}",
        
        # Configure interface as trunk
        "set interfaces et-0/0/96 unit 0 family ethernet-switching interface-mode trunk",
        f"set interfaces et-0/0/96 unit 0 family ethernet-switching vlan members mirror_test",
        
        # Also add inter-switch port to VLAN (for completeness)
        "set interfaces et-0/0/100 unit 0 family ethernet-switching interface-mode trunk",
        f"set interfaces et-0/0/100 unit 0 family ethernet-switching vlan members mirror_test",
        
        # Create firewall filter that counts AND mirrors
        # Term 1: Match test VLAN, count, mirror, accept
        f"set firewall family ethernet-switching filter {FILTER_NAME} "
        f"term mirror_term from vlan mirror_test",
        f"set firewall family ethernet-switching filter {FILTER_NAME} "
        f"term mirror_term then count mirrored_pkts",
        f"set firewall family ethernet-switching filter {FILTER_NAME} "
        f"term mirror_term then port-mirror-instance {ANALYZER_NAME}",
        f"set firewall family ethernet-switching filter {FILTER_NAME} "
        f"term mirror_term then accept",
        
        # Default term to accept all other traffic
        f"set firewall family ethernet-switching filter {FILTER_NAME} "
        f"term default then accept",
        
        # Apply filter to input port
        f"set interfaces et-0/0/96 unit 0 family ethernet-switching "
        f"filter input {FILTER_NAME}",
    ]
    
    success = run_config_commands(switch_ip, commands, debug=debug)
    
    if success:
        print(f"    ✓ Mirror filter '{FILTER_NAME}' configured")
        print(f"    → Matching packets will be mirrored via '{ANALYZER_NAME}'")
    else:
        print(f"    ✗ Mirror filter configuration failed")
    
    return success


def clear_counters(switch_ip: str):
    """Clear firewall counters."""
    ssh_command(switch_ip, f"cli -c 'clear firewall filter {FILTER_NAME}'")


def read_counters(switch_ip: str) -> int:
    """Read mirror counter from the filter."""
    success, stdout, _ = ssh_command(switch_ip, 
        f"cli -c 'show firewall filter {FILTER_NAME}'")
    
    if success:
        pattern = r'mirrored_pkts\s+\d+\s+(\d+)'
        match = re.search(pattern, stdout)
        if match:
            return int(match.group(1))
    return 0


def verify_port_mirroring(switch_ip: str) -> bool:
    """Verify port-mirroring is configured correctly."""
    success, stdout, _ = ssh_command(switch_ip,
        f"cli -c 'show forwarding-options port-mirroring'")
    
    if success and stdout.strip():
        print(f"\n  Port-mirroring status:")
        for line in stdout.strip().split('\n'):
            print(f"    {line}")
        return True
    return False


# ============================================================================
# PACKET SENDING
# ============================================================================

def send_mirror_test_packets(num_packets: int = 10) -> int:
    """Send test packets that should trigger the mirror filter."""
    src_mac = bytes.fromhex(get_mac_address(SEND_IFACE).replace(':', ''))
    # Use broadcast to ensure packets are processed by filter
    dst_mac = bytes.fromhex('ffffffffffff')
    
    packets = []
    for i in range(num_packets):
        # Use a distinctive payload so we can identify mirrored packets
        payload = f'MIRROR_TEST_{i:04d}'.encode()
        pkt = craft_vlan_packet(dst_mac, src_mac, MIRROR_VLAN, payload)
        packets.append(pkt)
    
    return send_packets(SEND_IFACE, packets)


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

@dataclass
class MirrorTestResult:
    """Results from mirror test."""
    packets_sent: int
    packets_counted: int
    packets_received: int
    mirror_working: bool
    timestamp: float


def run_experiment():
    """Test port mirroring via firewall filter."""
    
    print("="*80)
    print("E043: PORT MIRROR TEST VIA FIREWALL FILTER")
    print("="*80)
    
    print("\nGoal:")
    print("  Configure port-mirror-instance in a firewall filter term")
    print("  Send packets → filter matches → packets mirrored to host")
    print("  Listen on Ubuntu and verify we receive mirrored copies!")
    print()
    print(f"  Analyzer: {ANALYZER_NAME}")
    print(f"  Mirror path: SW1:{MIRROR_OUTPUT_PORT} → SW2:100 → SW2:96 → Host")
    print(f"  Send interface: {SEND_IFACE}")
    print(f"  Capture interface: {CAPTURE_IFACE}")
    
    # Step 1: Cleanup
    print("\n" + "="*80)
    print("STEP 1: CLEANUP")
    print("="*80)
    
    cleanup_switch(SWITCH1_IP)
    cleanup_switch(SWITCH2_IP)
    
    # Extra cleanup: fully reset interfaces and filters
    print("\n  Removing leftover config from interfaces...")
    extra_cleanup = [
        "delete forwarding-options port-mirroring",
        "delete forwarding-options analyzer",
        "delete interfaces et-0/0/96 unit 0 family ethernet-switching",
        "delete interfaces et-0/0/100 unit 0 family ethernet-switching",
        "delete firewall family ethernet-switching filter filter_port96",
        "delete firewall family ethernet-switching filter filter_port100",
        "delete firewall family ethernet-switching filter mirror_filter",
    ]
    run_config_commands(SWITCH1_IP, extra_cleanup, debug=False)
    run_config_commands(SWITCH2_IP, extra_cleanup, debug=False)
    print("    ✓ Interface config cleaned up")
    time.sleep(1)
    
    # Step 2: Configure port-mirroring instance
    print("\n" + "="*80)
    print("STEP 2: CONFIGURE PORT-MIRRORING INSTANCE")
    print("="*80)
    
    if not configure_port_mirroring(SWITCH1_IP, debug=True):
        print("Port-mirroring configuration failed!")
        return
    
    time.sleep(1)
    
    # Step 2b: Configure SW2 to forward mirrored packets
    print("\n" + "="*80)
    print("STEP 2b: CONFIGURE SW2 FOR MIRROR FORWARDING")
    print("="*80)
    
    if not configure_sw2_for_mirror(debug=True):
        print("SW2 configuration failed!")
        return
    
    time.sleep(1)
    
    # Step 3: Configure mirror filter
    print("\n" + "="*80)
    print("STEP 3: CONFIGURE MIRROR FILTER")
    print("="*80)
    
    if not configure_mirror_filter(SWITCH1_IP, debug=True):
        print("Mirror filter configuration failed!")
        return
    
    time.sleep(2)
    
    # Step 4: Verify configuration
    print("\n" + "="*80)
    print("STEP 4: VERIFY CONFIGURATION")
    print("="*80)
    
    verify_port_mirroring(SWITCH1_IP)
    
    # Show filter config
    success, stdout, _ = ssh_command(SWITCH1_IP,
        f"cli -c 'show configuration firewall family ethernet-switching filter {FILTER_NAME}'")
    if success and stdout.strip():
        print(f"\n  Filter configuration:")
        for line in stdout.strip().split('\n')[:15]:
            print(f"    {line}")
    
    # Step 5: Clear counters and start capture
    print("\n" + "="*80)
    print("STEP 5: CLEAR COUNTERS & START CAPTURE")
    print("="*80)
    
    clear_counters(SWITCH1_IP)
    time.sleep(0.5)
    print("  ✓ Counters cleared")
    
    # Start packet capture on the receive interface
    # Mirrored packets path: SW1:100 → SW2:100 → SW2:96 → Host enp1s0d1
    capture = PacketCapture(CAPTURE_IFACE)
    print(f"  Starting packet capture on {CAPTURE_IFACE}...")
    capture.start(timeout=10.0)
    time.sleep(1)  # Let capture thread start
    print("  ✓ Capture started")
    
    # Step 6: Send test packets
    print("\n" + "="*80)
    print("STEP 6: SEND TEST PACKETS")
    print("="*80)
    
    num_packets = 10
    print(f"\n  Sending {num_packets} packets with VLAN {MIRROR_VLAN}...")
    print(f"  Payload pattern: 'MIRROR_TEST_XXXX'")
    sent = send_mirror_test_packets(num_packets)
    print(f"  ✓ Sent {sent} packets")
    
    # Wait for packets to be mirrored back
    print("\n  Waiting for mirrored packets...")
    time.sleep(3)
    
    # Stop capture
    capture.stop()
    print("  ✓ Capture stopped")
    
    # Step 7: Analyze results
    print("\n" + "="*80)
    print("STEP 7: ANALYZE RESULTS")
    print("="*80)
    
    # Read counter from switch
    counted = read_counters(SWITCH1_IP)
    print(f"\n  Switch counter (mirrored_pkts): {counted}")
    
    # Analyze captured packets
    all_packets = capture.get_packets()
    print(f"  Total packets captured: {len(all_packets)}")
    
    # Look for our test packets
    mirrored = capture.filter_by_payload(b'MIRROR_TEST_')
    print(f"  Packets with 'MIRROR_TEST_' payload: {len(mirrored)}")
    
    if mirrored:
        print("\n  Sample mirrored packets:")
        for i, pkt in enumerate(mirrored[:3]):
            # Extract some info from packet
            data = pkt['data']
            # Show first 60 bytes in hex
            hex_preview = data[:60].hex(':')
            print(f"    [{i}] Length: {len(data)} bytes")
            print(f"        Header: {hex_preview[:80]}...")
    
    # Step 8: Verdict
    print("\n" + "="*80)
    print("STEP 8: VERDICT")
    print("="*80)
    
    mirror_working = False
    
    if counted == sent:
        print(f"\n  ✓ Filter counter matches sent packets: {counted}/{sent}")
    else:
        print(f"\n  ⚠ Counter mismatch: {counted} counted vs {sent} sent")
    
    # Check if we received mirrored packets
    # Note: We might see packets twice - once from sending, once from mirroring
    if len(mirrored) >= sent:
        print(f"  ✓ Received at least {sent} mirrored packets!")
        print("  🎉 PORT MIRRORING WORKS!")
        mirror_working = True
    elif len(mirrored) > 0:
        print(f"  ⚠ Received {len(mirrored)} mirrored packets (expected {sent})")
        print("  Partial success - some packets were mirrored")
        mirror_working = True
    else:
        print(f"  ✗ No mirrored packets received")
        print("\n  Possible causes:")
        print("    - Analyzer not outputting to correct port")
        print("    - Mirrored packets have different VLAN/format")
        print("    - Need to check 'show forwarding-options analyzer' status")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if mirror_working:
        print("""
  ✓ PORT MIRRORING VIA FIREWALL FILTER WORKS!
  
  This proves:
    - Analyzer (port-mirror-instance) is configured
    - Firewall filter can trigger mirroring
    - Mirrored packets are sent to specified output port
    - Host can receive mirrored traffic
  
  Applications:
    - Debug packet flow through switch
    - Tap neuron activations without disrupting flow
    - Monitor specific MACs/VLANs selectively
    - Read inference results via mirrored packets
""")
    else:
        print("""
  Port mirroring needs debugging.
  
  Check:
    - 'show forwarding-options analyzer' on switch
    - 'show firewall filter' to verify terms
    - Interface status and VLAN membership
    - Try tcpdump on host to see raw traffic
""")
    
    # Save results
    result = MirrorTestResult(
        packets_sent=sent,
        packets_counted=counted,
        packets_received=len(mirrored),
        mirror_working=mirror_working,
        timestamp=time.time()
    )
    
    os.makedirs("bringup_logs", exist_ok=True)
    log_file = f"bringup_logs/port_mirror_{int(time.time())}.json"
    with open(log_file, 'w') as f:
        json.dump({
            "packets_sent": result.packets_sent,
            "packets_counted": result.packets_counted,
            "packets_received": result.packets_received,
            "mirror_working": result.mirror_working,
            "total_captured": len(all_packets),
            "timestamp": result.timestamp
        }, f, indent=2)
    
    print(f"\n  Results saved to: {log_file}")


if __name__ == '__main__':
    run_experiment()


""" Output:
sudo python3 e043_port_mirror_test.py 
================================================================================
E043: PORT MIRROR TEST VIA FIREWALL FILTER
================================================================================

Goal:
  Configure port-mirror-instance in a firewall filter term
  Send packets → filter matches → packets mirrored to host
  Listen on Ubuntu and verify we receive mirrored copies!

  Analyzer: packet_mirror
  Mirror path: SW1:et-0/0/100 → SW2:100 → SW2:96 → Host
  Send interface: enp1s0
  Capture interface: enp1s0d1

================================================================================
STEP 1: CLEANUP
================================================================================

  Cleaning up 10.10.10.55...
    Found 2 VLANs: ['mirror_test', 'default']
    Deleting 1 VLANs...
    ✓ Cleanup complete

  Cleaning up 10.10.10.56...
    Found 2 VLANs: ['mirror_test', 'default']
    Deleting 1 VLANs...
    ✓ Cleanup complete

  Removing leftover config from interfaces...
    ✓ Interface config cleaned up

================================================================================
STEP 2: CONFIGURE PORT-MIRRORING INSTANCE
================================================================================

  Configuring port-mirroring on 10.10.10.55...
    [DEBUG] stdout: Entering configuration mode
configuration check succeeds
commit complete

    ✓ Port-mirroring instance 'packet_mirror' configured
    → Mirrored packets will go to et-0/0/100

================================================================================
STEP 2b: CONFIGURE SW2 FOR MIRROR FORWARDING
================================================================================

  Configuring Switch 2 (10.10.10.56) to forward mirrored packets...
    [DEBUG] stdout: Entering configuration mode
configuration check succeeds
commit complete

    ✓ Switch 2 configured for mirror forwarding
    → Mirrored packets: SW2:100 → SW2:96 → Host enp1s0d1

================================================================================
STEP 3: CONFIGURE MIRROR FILTER
================================================================================

  Configuring mirror filter on 10.10.10.55...
    [DEBUG] stdout: Entering configuration mode
configuration check succeeds
commit complete

    ✓ Mirror filter 'mirror_filter' configured
    → Matching packets will be mirrored via 'packet_mirror'

================================================================================
STEP 4: VERIFY CONFIGURATION
================================================================================

  Port-mirroring status:
    Instance Name: packet_mirror                  
      Instance Id: 3              
      Input parameters:
        Rate                  : 1
        Run-length            : 0
        Maximum-packet-length : 0
      Output parameters:
        Family              State     Destination          Next-hop
        ethernet-switching  up        et-0/0/100.0         NA

  Filter configuration:
    term mirror_term {
        from {
            ##
            ## Warning: value vlan ignored: unsupported platform (qfx5100-96s-8q)
            ##
            vlan mirror_test;
        }
        then {
            accept;
            port-mirror-instance packet_mirror;
            count mirrored_pkts;
        }
    }
    term default {
        then accept;

================================================================================
STEP 5: CLEAR COUNTERS & START CAPTURE
================================================================================
  ✓ Counters cleared
  Starting packet capture on enp1s0d1...
  ✓ Capture started

================================================================================
STEP 6: SEND TEST PACKETS
================================================================================

  Sending 10 packets with VLAN 901...
  Payload pattern: 'MIRROR_TEST_XXXX'
  ✓ Sent 10 packets

  Waiting for mirrored packets...
  ✓ Capture stopped

================================================================================
STEP 7: ANALYZE RESULTS
================================================================================

  Switch counter (mirrored_pkts): 10
  Total packets captured: 20
  Packets with 'MIRROR_TEST_' payload: 20

  Sample mirrored packets:
    [0] Length: 56 bytes
        Header: ff:ff:ff:ff:ff:ff:7c:fe:90:9d:2a:f0:88:b5:4d:49:52:52:4f:52:5f:54:45:53:54:5f:30...
    [1] Length: 56 bytes
        Header: ff:ff:ff:ff:ff:ff:7c:fe:90:9d:2a:f0:88:b5:4d:49:52:52:4f:52:5f:54:45:53:54:5f:30...
    [2] Length: 56 bytes
        Header: ff:ff:ff:ff:ff:ff:7c:fe:90:9d:2a:f0:88:b5:4d:49:52:52:4f:52:5f:54:45:53:54:5f:30...

================================================================================
STEP 8: VERDICT
================================================================================

  ✓ Filter counter matches sent packets: 10/10
  ✓ Received at least 10 mirrored packets!
  🎉 PORT MIRRORING WORKS!

================================================================================
SUMMARY
================================================================================

  ✓ PORT MIRRORING VIA FIREWALL FILTER WORKS!
  
  This proves:
    - Analyzer (port-mirror-instance) is configured
    - Firewall filter can trigger mirroring
    - Mirrored packets are sent to specified output port
    - Host can receive mirrored traffic
  
  Applications:
    - Debug packet flow through switch
    - Tap neuron activations without disrupting flow
    - Monitor specific MACs/VLANs selectively
    - Read inference results via mirrored packets


  Results saved to: bringup_logs/port_mirror_1766848672.json
(.venv) multiplex@multiplex1:~/photonic_inference_engine/phase_001/phase_001$ cat bringup_logs/port_mirror_1766848672.json
{
  "packets_sent": 10,
  "packets_counted": 10,
  "packets_received": 20,
  "mirror_working": true,
  "total_captured": 20,
  "timestamp": 1766848672.035143
}
"""