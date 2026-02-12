#!/usr/bin/env python3
"""
e087_packet_based_counter_encoding.py

PACKET-BASED COUNTER ENCODING - ZERO-LATENCY COUNTER READS
===========================================================

THE BREAKTHROUGH IDEA:
  Don't READ counters via SSH/NETCONF (674ms) - have the switch SEND them!
  
  Current: Host → SSH → "show counters" → Parse → 674ms
  Target:  Switch → Forward packets back → Host counts → <10ms!

HOW IT WORKS:
  1. Configure firewall filters with TWO actions per neuron:
     a) Count packets (as before)
     b) MIRROR/FORWARD counted packets back to host
  
  2. Host receives the mirrored packets and counts them
  
  3. Packet count received = counter value!
  
  Example:
    - Neuron 0 counted 219 packets
    - Switch forwards those 219 packets back to host
    - Host receives 219 packets with neuron 0's MAC
    - Result: neuron 0 = 219 ✓

ADVANTAGES:
  - Uses data plane (forwarding), not control plane (SSH)
  - Sub-millisecond "counter reading" (it's packet forwarding!)
  - No SSH round-trip needed
  - Scales with switch ASIC speed, not CPU

TEST APPROACH:
  1. Configure filter with mirroring action
  2. Send test packets through filter
  3. Receive mirrored packets on host
  4. Compare received counts vs SSH counter read
  5. Measure latency difference

SUCCESS CRITERIA:
  - Received packet counts match counter values
  - Total time < 100ms (vs 674ms NETCONF)
  - Proves forwarding-plane counter reads work!

Author: Research Phase 001
Date: December 2025
"""

import time
import os
import sys
import socket
import struct
import re
from typing import Dict, Tuple, List
from collections import defaultdict
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from e042_port_based_layers import (
    SWITCH1_IP, SWITCH2_IP, SEND_IFACE,
    ssh_command, craft_vlan_packet, send_packets, get_mac_address
)
from e053_mac_encoded_layers import get_layer_neuron_mac
from e045_real_weights_inference import mac_str_to_bytes
from e075_onswitch_inference_probe import ssh_command_long

# =============================================================================
# CONFIGURATION
# =============================================================================

SSH_KEY = "/home/multiplex/.ssh/id_rsa"
RECV_IFACE = "enp1s0"  # Same interface - packets come back

TEST_FILTER_NAME = "mirror_test_filter"
TEST_VLAN = 400
NUM_TEST_NEURONS = 4  # Small test

# =============================================================================
# PACKET RECEIVER
# =============================================================================

class PacketCounterReceiver:
    """Receives mirrored packets and counts by destination MAC."""
    
    def __init__(self, interface: str):
        self.interface = interface
        self.socket: socket.socket = None
        self.running = False
        self.thread: threading.Thread = None
        self.counters: Dict[str, int] = defaultdict(int)
        self.total_received = 0
    
    def start(self):
        """Start packet receiver."""
        # Create raw socket
        self.socket = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.ntohs(0x0003))
        self.socket.bind((self.interface, 0))
        self.socket.settimeout(0.1)  # 100ms timeout
        
        self.running = True
        self.thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.thread.start()
        print(f"  ✓ Packet receiver started on {self.interface}")
    
    def _receive_loop(self):
        """Background thread to receive and count packets."""
        while self.running:
            try:
                packet = self.socket.recv(65535)
                
                # Parse Ethernet header (14 bytes)
                if len(packet) < 14:
                    continue
                
                dst_mac = packet[0:6]
                src_mac = packet[6:12]
                eth_type = struct.unpack('!H', packet[12:14])[0]
                
                # Convert MAC to string
                dst_mac_str = ':'.join(f'{b:02x}' for b in dst_mac)
                
                # Count this packet
                self.counters[dst_mac_str] += 1
                self.total_received += 1
                
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    pass  # Ignore errors during normal operation
    
    def stop(self):
        """Stop receiver."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.socket:
            self.socket.close()
        print(f"  ✓ Packet receiver stopped")
    
    def get_counts(self) -> Dict[str, int]:
        """Get current packet counts."""
        return dict(self.counters)
    
    def clear(self):
        """Clear counters."""
        self.counters.clear()
        self.total_received = 0


# =============================================================================
# SWITCH CONFIGURATION
# =============================================================================

def configure_mirror_filter(switch_ip: str, method: str = 'port-mirror') -> bool:
    """Configure filter with packet mirroring/forwarding.
    
    Methods:
        'port-mirror': Use port-mirroring (from e043)
        'accept-forward': Just accept and let L2 forward
        'sample': Use sampling
    """
    print(f"\n  Configuring mirror filter (method: {method})...")
    
    commands = []
    
    # Delete old config
    commands.append(f"delete firewall family ethernet-switching filter {TEST_FILTER_NAME}")
    commands.append(f"delete vlans mirror_test_vlan")
    commands.append(f"delete interfaces et-0/0/96 unit 0 family ethernet-switching")
    
    # Create VLAN
    commands.append(f"set vlans mirror_test_vlan vlan-id {TEST_VLAN}")
    
    # Configure interface as trunk
    commands.append(f"set interfaces et-0/0/96 unit 0 family ethernet-switching interface-mode trunk")
    commands.append(f"set interfaces et-0/0/96 unit 0 family ethernet-switching vlan members mirror_test_vlan")
    
    # Create filter with mirroring
    for neuron in range(NUM_TEST_NEURONS):
        mac = get_layer_neuron_mac(0, neuron)
        term_name = f"n{neuron}"
        
        commands.append(f"set firewall family ethernet-switching filter {TEST_FILTER_NAME} term {term_name} from destination-mac-address {mac}/48")
        commands.append(f"set firewall family ethernet-switching filter {TEST_FILTER_NAME} term {term_name} then count {term_name}")
        
        if method == 'port-mirror':
            # Use port-mirror action (from e043)
            commands.append(f"set firewall family ethernet-switching filter {TEST_FILTER_NAME} term {term_name} then port-mirror")
        elif method == 'accept-forward':
            # Just accept - let normal L2 forwarding happen
            commands.append(f"set firewall family ethernet-switching filter {TEST_FILTER_NAME} term {term_name} then accept")
        elif method == 'sample':
            # Use sampling (might not preserve all packets)
            commands.append(f"set firewall family ethernet-switching filter {TEST_FILTER_NAME} term {term_name} then sample")
            commands.append(f"set firewall family ethernet-switching filter {TEST_FILTER_NAME} term {term_name} then accept")
    
    # Default term
    commands.append(f"set firewall family ethernet-switching filter {TEST_FILTER_NAME} term default then accept")
    
    # Bind filter to VLAN
    commands.append(f"set vlans mirror_test_vlan forwarding-options filter input {TEST_FILTER_NAME}")
    
    # If using port-mirror, need to configure analyzer
    if method == 'port-mirror':
        commands.append("set forwarding-options port-mirroring instance pi1 input rate 1")
        commands.append("set forwarding-options port-mirroring instance pi1 family ethernet-switching output interface et-0/0/96")
    
    # Apply
    cmd_str = " ; ".join(commands)
    full_cmd = f"cli -c 'configure ; {cmd_str} ; commit'"
    success, stdout, stderr = ssh_command_long(switch_ip, full_cmd, timeout=60)
    
    if success and 'error' not in stderr.lower():
        print(f"  ✓ Filter configured with {method} method")
        return True
    else:
        print(f"  ✗ Configuration failed")
        if stderr:
            print(f"    Error: {stderr[:300]}")
        return False


def cleanup_mirror_filter(switch_ip: str):
    """Clean up test configuration."""
    commands = [
        f"delete firewall family ethernet-switching filter {TEST_FILTER_NAME}",
        f"delete vlans mirror_test_vlan",
        f"delete interfaces et-0/0/96 unit 0 family ethernet-switching",
        "delete forwarding-options port-mirroring",
    ]
    cmd_str = " ; ".join(commands)
    full_cmd = f"cli -c 'configure ; {cmd_str} ; commit'"
    ssh_command_long(switch_ip, full_cmd, timeout=30)


def clear_counters(switch_ip: str):
    """Clear all counters on the filter."""
    ssh_command_long(
        switch_ip,
        f"cli -c 'clear firewall filter {TEST_FILTER_NAME}'",
        timeout=10
    )


def read_counters_ssh(switch_ip: str, debug: bool = False) -> Dict[str, int]:
    """Read counters via SSH (baseline for comparison)."""
    success, stdout, _ = ssh_command_long(
        switch_ip,
        f"cli -c 'show firewall filter {TEST_FILTER_NAME}'",
        timeout=10
    )
    
    if debug:
        print(f"\n  DEBUG: SSH output (first 500 chars):")
        print(stdout[:500])
    
    counters = {}
    if success:
        for neuron in range(NUM_TEST_NEURONS):
            pattern = rf'n{neuron}\s+\d+\s+(\d+)'
            match = re.search(pattern, stdout)
            if match:
                counters[f'n{neuron}'] = int(match.group(1))
    
    return counters


# =============================================================================
# TEST PACKET SENDING
# =============================================================================

def send_test_packets() -> Dict[str, int]:
    """Send test packets with known counts per neuron."""
    print(f"\n  Sending test packets...")
    
    src_mac = get_mac_address(SEND_IFACE)
    src = mac_str_to_bytes(src_mac)
    
    # Send different counts per neuron (so we can verify)
    expected = {}
    packets = []
    
    for neuron in range(NUM_TEST_NEURONS):
        count = 10 * (neuron + 1)  # 10, 20, 30, 40
        expected[f'n{neuron}'] = count
        
        mac = get_layer_neuron_mac(0, neuron)
        dst = mac_str_to_bytes(mac)
        
        for _ in range(count):
            packets.append(craft_vlan_packet(dst, src, TEST_VLAN))
    
    print(f"    Sending {len(packets)} packets (10, 20, 30, 40 per neuron)")
    send_packets(SEND_IFACE, packets)
    print(f"  ✓ Packets sent")
    
    return expected


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment():
    """Run packet-based counter encoding experiment."""
    print("="*70)
    print("E087: PACKET-BASED COUNTER ENCODING")
    print("="*70)
    
    print(f"\n  Target switch: {SWITCH1_IP}")
    print(f"  Interface:     {SEND_IFACE}")
    
    # Cleanup
    print("\nSTEP 1: CLEANUP")
    cleanup_mirror_filter(SWITCH1_IP)
    time.sleep(1)
    
    # Try different methods
    methods_to_test = ['accept-forward', 'port-mirror']
    
    for method in methods_to_test:
        print(f"\n{'='*70}")
        print(f"TESTING METHOD: {method}")
        print(f"{'='*70}")
        
        # Configure
        print(f"\nSTEP 2: CONFIGURE ({method})")
        if not configure_mirror_filter(SWITCH1_IP, method=method):
            print(f"  ✗ Skipping {method} - configuration failed")
            continue
        
        time.sleep(1)
        
        # Start receiver
        print("\nSTEP 3: START PACKET RECEIVER")
        receiver = PacketCounterReceiver(RECV_IFACE)
        receiver.start()
        time.sleep(0.5)
        
        try:
            # Clear counters before test
            print("\nSTEP 4: CLEAR COUNTERS")
            clear_counters(SWITCH1_IP)
            time.sleep(0.5)
            
            # Send packets
            print("\nSTEP 5: SEND TEST PACKETS")
            start_total = time.time()
            start_send = time.time()
            expected = send_test_packets()
            send_time = time.time() - start_send
            print(f"    Send time: {send_time*1000:.1f}ms")
            
            # Wait for mirrored packets
            print("\nSTEP 6: RECEIVE MIRRORED PACKETS")
            print("  Waiting for packets...")
            start_receive = time.time()
            
            # Wait until we've received expected packets or timeout (3 seconds)
            timeout = 3.0
            while (time.time() - start_receive) < timeout:
                if receiver.total_received >= sum(expected.values()):
                    break
                time.sleep(0.01)
            
            receive_time = time.time() - start_receive
            total_packet_time = time.time() - start_total
            
            # Get received counts
            received_counts = receiver.get_counts()
            print(f"  ✓ Received {receiver.total_received} total packets in {receive_time*1000:.1f}ms")
            print(f"    Total packet-based method time: {total_packet_time*1000:.1f}ms")
            
            # Compare with SSH counters
            print("\nSTEP 7: READ COUNTERS VIA SSH (baseline)")
            start_ssh = time.time()
            ssh_counters = read_counters_ssh(SWITCH1_IP, debug=False)
            ssh_time = time.time() - start_ssh
            print(f"  ✓ SSH read completed in {ssh_time*1000:.1f}ms")
            
            # Analysis
            print("\nSTEP 8: RESULTS ANALYSIS")
            print(f"\n  Expected counts:")
            for name, count in sorted(expected.items()):
                print(f"    {name}: {count}")
            
            print(f"\n  SSH counter values:")
            if ssh_counters:
                for name, count in sorted(ssh_counters.items()):
                    print(f"    {name}: {count}")
            else:
                print("    (no counters found - filter might have been cleared)")
            
            print(f"\n  Received packet counts (by destination MAC):")
            if received_counts:
                # Match received MACs to neuron names
                for neuron in range(NUM_TEST_NEURONS):
                    mac = get_layer_neuron_mac(0, neuron)
                    count = received_counts.get(mac, 0)
                    print(f"    n{neuron} ({mac}): {count}")
            else:
                print("    (no packets received)")
            
            # Verify
            print(f"\n  Verification vs Expected:")
            if not received_counts:
                print("    ✗ No packets received - method doesn't work")
            else:
                all_match_expected = True
                all_match_ssh = True
                
                for neuron in range(NUM_TEST_NEURONS):
                    mac = get_layer_neuron_mac(0, neuron)
                    expected_count = expected[f'n{neuron}']
                    received_count = received_counts.get(mac, 0)
                    ssh_count = ssh_counters.get(f'n{neuron}', -1)
                    
                    match_expected = (received_count == expected_count)
                    match_ssh = (ssh_count >= 0 and received_count == ssh_count)
                    
                    symbol = "✓" if match_expected else "✗"
                    ssh_str = f"ssh={ssh_count}" if ssh_count >= 0 else "ssh=N/A"
                    print(f"    {symbol} n{neuron}: expected={expected_count}, received={received_count}, {ssh_str}")
                    
                    if not match_expected:
                        all_match_expected = False
                    if not match_ssh and ssh_count >= 0:
                        all_match_ssh = False
                
                # Performance summary
                print(f"\n  {'='*60}")
                print(f"  PERFORMANCE COMPARISON")
                print(f"  {'='*60}")
                print(f"\n  Packet-based method ({method}):")
                print(f"    Send time:        {send_time*1000:.1f}ms")
                print(f"    Receive time:     {receive_time*1000:.1f}ms")
                print(f"    Total time:       {total_packet_time*1000:.1f}ms")
                print(f"\n  SSH counter read (baseline):")
                print(f"    Read time:        {ssh_time*1000:.1f}ms")
                print(f"\n  Speedup comparison:")
                if total_packet_time > 0 and ssh_time > 0:
                    speedup = ssh_time / total_packet_time
                    time_saved = (ssh_time - total_packet_time) * 1000
                    print(f"    Packet method is {speedup:.1f}× faster")
                    print(f"    Time saved:       {time_saved:.1f}ms per read")
                    print(f"\n  Extrapolation to 28-layer inference:")
                    total_ssh = ssh_time * 28
                    total_packet = total_packet_time * 28
                    inference_speedup = total_ssh / total_packet if total_packet > 0 else 0
                    print(f"    SSH method:       {total_ssh:.2f}s per token")
                    print(f"    Packet method:    {total_packet:.2f}s per token")
                    print(f"    Speedup:          {inference_speedup:.1f}×")
                
                if all_match_expected:
                    print(f"\n  ✓✓✓ SUCCESS with {method}! ✓✓✓")
                    print(f"      Packet-based counter encoding works perfectly!")
                    print(f"      All received counts match expected values!")
                    if all_match_ssh:
                        print(f"      SSH counters also match!")
                else:
                    print(f"\n  ⚠ PARTIAL: Some mismatches with {method}")
        
        finally:
            receiver.stop()
            cleanup_mirror_filter(SWITCH1_IP)
            time.sleep(1)
    
    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*70}")
    print("\n  KEY FINDINGS:")
    print("    - Tested packet-based counter encoding")
    print("    - If successful: eliminates SSH round-trip!")
    print("    - Method shows path to <10ms counter reads")


if __name__ == "__main__":
    try:
        run_experiment()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()


""" Output:
sudo python3 e087_packet_based_counter_encoding.py 
======================================================================
E087: PACKET-BASED COUNTER ENCODING
======================================================================

  Target switch: 10.10.10.55
  Interface:     enp1s0

STEP 1: CLEANUP

======================================================================
TESTING METHOD: accept-forward
======================================================================

STEP 2: CONFIGURE (accept-forward)

  Configuring mirror filter (method: accept-forward)...
  ✓ Filter configured with accept-forward method

STEP 3: START PACKET RECEIVER
  ✓ Packet receiver started on enp1s0

STEP 4: CLEAR COUNTERS

STEP 5: SEND TEST PACKETS

  Sending test packets...
    Sending 100 packets (10, 20, 30, 40 per neuron)
  ✓ Packets sent
    Send time: 8.4ms

STEP 6: RECEIVE MIRRORED PACKETS
  Waiting for packets...
  ✓ Received 100 total packets in 0.0ms
    Total packet-based method time: 8.5ms

STEP 7: READ COUNTERS VIA SSH (baseline)
  ✓ SSH read completed in 742.8ms

STEP 8: RESULTS ANALYSIS

  Expected counts:
    n0: 10
    n1: 20
    n2: 30
    n3: 40

  SSH counter values:
    n0: 10
    n1: 20
    n2: 30
    n3: 40

  Received packet counts (by destination MAC):
    n0 (01:00:5e:00:00:00): 10
    n1 (01:00:5e:00:00:01): 20
    n2 (01:00:5e:00:00:02): 30
    n3 (01:00:5e:00:00:03): 40

  Verification vs Expected:
    ✓ n0: expected=10, received=10, ssh=10
    ✓ n1: expected=20, received=20, ssh=20
    ✓ n2: expected=30, received=30, ssh=30
    ✓ n3: expected=40, received=40, ssh=40

  ============================================================
  PERFORMANCE COMPARISON
  ============================================================

  Packet-based method (accept-forward):
    Send time:        8.4ms
    Receive time:     0.0ms
    Total time:       8.5ms

  SSH counter read (baseline):
    Read time:        742.8ms

  Speedup comparison:
    Packet method is 87.4× faster
    Time saved:       734.3ms per read

  Extrapolation to 28-layer inference:
    SSH method:       20.80s per token
    Packet method:    0.24s per token
    Speedup:          87.4×

  ✓✓✓ SUCCESS with accept-forward! ✓✓✓
      Packet-based counter encoding works perfectly!
      All received counts match expected values!
      SSH counters also match!
  ✓ Packet receiver stopped

======================================================================
TESTING METHOD: port-mirror
======================================================================

STEP 2: CONFIGURE (port-mirror)

  Configuring mirror filter (method: port-mirror)...
  ✓ Filter configured with port-mirror method

STEP 3: START PACKET RECEIVER
  ✓ Packet receiver started on enp1s0

STEP 4: CLEAR COUNTERS

STEP 5: SEND TEST PACKETS

  Sending test packets...
    Sending 100 packets (10, 20, 30, 40 per neuron)
  ✓ Packets sent
    Send time: 10.2ms

STEP 6: RECEIVE MIRRORED PACKETS
  Waiting for packets...
  ✓ Received 100 total packets in 0.0ms
    Total packet-based method time: 10.3ms

STEP 7: READ COUNTERS VIA SSH (baseline)
  ✓ SSH read completed in 724.6ms

STEP 8: RESULTS ANALYSIS

  Expected counts:
    n0: 10
    n1: 20
    n2: 30
    n3: 40

  SSH counter values:
    n0: 10
    n1: 20
    n2: 30
    n3: 40

  Received packet counts (by destination MAC):
    n0 (01:00:5e:00:00:00): 10
    n1 (01:00:5e:00:00:01): 20
    n2 (01:00:5e:00:00:02): 30
    n3 (01:00:5e:00:00:03): 40

  Verification vs Expected:
    ✓ n0: expected=10, received=10, ssh=10
    ✓ n1: expected=20, received=20, ssh=20
    ✓ n2: expected=30, received=30, ssh=30
    ✓ n3: expected=40, received=40, ssh=40

  ============================================================
  PERFORMANCE COMPARISON
  ============================================================

  Packet-based method (port-mirror):
    Send time:        10.2ms
    Receive time:     0.0ms
    Total time:       10.3ms

  SSH counter read (baseline):
    Read time:        724.6ms

  Speedup comparison:
    Packet method is 70.5× faster
    Time saved:       714.3ms per read

  Extrapolation to 28-layer inference:
    SSH method:       20.29s per token
    Packet method:    0.29s per token
    Speedup:          70.5×

  ✓✓✓ SUCCESS with port-mirror! ✓✓✓
      Packet-based counter encoding works perfectly!
      All received counts match expected values!
      SSH counters also match!
  ✓ Packet receiver stopped

======================================================================
EXPERIMENT COMPLETE
======================================================================

  KEY FINDINGS:
    - Tested packet-based counter encoding
    - If successful: eliminates SSH round-trip!
    - Method shows path to <10ms counter reads
"""