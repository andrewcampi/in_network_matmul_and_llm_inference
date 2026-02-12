#!/usr/bin/env python3
"""
e017_read_physical_counters.py

Read PHYSICAL interface packet counters (not logical interface counters).
The QFX5100 counts packets at the physical layer for VLAN-tagged traffic.
"""

import subprocess
import re
import time
from typing import Dict, List, Tuple

class PhysicalCounterReader:
    """Read physical interface packet counters from Juniper switches."""
    
    def __init__(self, switch_ips: List[str], ssh_key_path: str = '/home/multiplex/.ssh/id_rsa'):
        self.switch_ips = switch_ips
        self.ssh_key_path = ssh_key_path
    
    def read_physical_counters(self, port: str) -> Dict[str, Dict[str, int]]:
        """
        Read physical interface traffic statistics.
        
        Returns dict of {switch_ip: {'input_packets': N, 'output_packets': M}}
        """
        results = {}
        
        for switch_ip in self.switch_ips:
            cmd = [
                'ssh',
                '-i', self.ssh_key_path,
                '-o', 'StrictHostKeyChecking=no',
                '-o', 'UserKnownHostsFile=/dev/null',
                '-o', 'LogLevel=ERROR',
                f'root@{switch_ip}',
                f"cli -c 'show interfaces {port} extensive'"
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                output = result.stdout
                
                # Parse physical interface Traffic statistics section
                # We want the FIRST occurrence (physical interface), not logical interface counters
                input_packets = 0
                output_packets = 0
                
                # Find the first "Traffic statistics:" section (physical interface)
                in_traffic_stats = False
                found_input = False
                found_output = False
                
                for line in output.split('\n'):
                    if 'Traffic statistics:' in line and not in_traffic_stats:
                        in_traffic_stats = True
                        continue
                    
                    if in_traffic_stats:
                        # Look for Input packets (with multiple spaces after "Input")
                        if not found_input and 'Input  packets:' in line:
                            match = re.search(r'Input\s+packets:\s+(\d+)', line)
                            if match:
                                input_packets = int(match.group(1))
                                found_input = True
                        # Look for Output packets
                        elif not found_output and 'Output packets:' in line:
                            match = re.search(r'Output\s+packets:\s+(\d+)', line)
                            if match:
                                output_packets = int(match.group(1))
                                found_output = True
                        
                        # Stop after we found both
                        if found_input and found_output:
                            break
                
                results[switch_ip] = {
                    'input_packets': input_packets,
                    'output_packets': output_packets
                }
                
            except Exception as e:
                print(f"Error reading {switch_ip} port {port}: {e}")
                results[switch_ip] = {
                    'input_packets': 0,
                    'output_packets': 0
                }
        
        return results
    
    def test_with_packets(self, port: str = 'et-0/0/96', num_packets: int = 10, 
                         vlan_id: int = 2, interface: str = 'enp1s0'):
        """
        Run a complete test: baseline, send packets, read delta.
        """
        print("="*100)
        print("PHYSICAL INTERFACE COUNTER TEST")
        print("="*100)
        print(f"\nPort: {port}")
        print(f"Switches: {', '.join(self.switch_ips)}")
        print(f"Test: Send {num_packets} packets with VLAN {vlan_id}")
        print()
        
        # Step 1: Baseline
        print("Step 1: Reading baseline counters...")
        baseline = self.read_physical_counters(port)
        
        for switch_ip, counters in baseline.items():
            print(f"  {switch_ip}: Input={counters['input_packets']}, Output={counters['output_packets']}")
        print()
        
        # Step 2: Send packets
        print(f"Step 2: Sending {num_packets} packets...")
        self._send_packets(num_packets, vlan_id, interface)
        
        print("Waiting 2 seconds for propagation...")
        time.sleep(2)
        print()
        
        # Step 3: Read post-test counters
        print("Step 3: Reading post-test counters...")
        post_test = self.read_physical_counters(port)
        
        for switch_ip, counters in post_test.items():
            print(f"  {switch_ip}: Input={counters['input_packets']}, Output={counters['output_packets']}")
        print()
        
        # Step 4: Calculate deltas
        print("Step 4: Delta analysis")
        print("="*100)
        
        total_input_delta = 0
        total_output_delta = 0
        
        for switch_ip in self.switch_ips:
            input_delta = post_test[switch_ip]['input_packets'] - baseline[switch_ip]['input_packets']
            output_delta = post_test[switch_ip]['output_packets'] - baseline[switch_ip]['output_packets']
            
            total_input_delta += input_delta
            total_output_delta += output_delta
            
            print(f"\n{switch_ip}:")
            print(f"  Input delta:  {input_delta} packets")
            print(f"  Output delta: {output_delta} packets")
        
        print(f"\n{'='*100}")
        print(f"TOTALS:")
        print(f"  Packets sent:         {num_packets}")
        print(f"  Total input delta:    {total_input_delta}")
        print(f"  Total output delta:   {total_output_delta}")
        print(f"  Input detection rate: {100.0 * total_input_delta / num_packets:.1f}%")
        
        if total_input_delta == num_packets:
            print(f"\n✅ SUCCESS: 100% packet delivery!")
        elif total_input_delta > 0:
            print(f"\n⚠ PARTIAL: {total_input_delta}/{num_packets} packets detected")
        else:
            print(f"\n❌ FAILURE: No packets detected")
        
        print(f"{'='*100}\n")
        
        return total_input_delta == num_packets
    
    def _send_packets(self, num_packets: int, vlan_id: int, interface: str):
        """Send test packets using raw sockets."""
        import socket
        from e001_packet_craft_and_parse import craft_ethernet_frame, BROADCAST_MAC
        
        # Get source MAC and convert to bytes
        with open(f'/sys/class/net/{interface}/address') as f:
            src_mac_str = f.read().strip()
        
        # Convert MAC string "aa:bb:cc:dd:ee:ff" to bytes
        src_mac = bytes.fromhex(src_mac_str.replace(':', ''))
        
        # Craft packet
        packet = craft_ethernet_frame(
            dst_mac=BROADCAST_MAC,
            src_mac=src_mac,
            vlan_id=vlan_id,
            payload=b'PHOTONIC_TEST'
        )
        
        # Send packets
        sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW)
        sock.bind((interface, 0))
        
        for i in range(num_packets):
            sock.send(packet)
        
        sock.close()
        print(f"✓ Sent {num_packets} packets")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Read physical interface packet counters')
    parser.add_argument('--switches', nargs='+', default=['10.10.10.55', '10.10.10.56'],
                       help='Switch IP addresses')
    parser.add_argument('--port', default='et-0/0/96', help='Port to monitor')
    parser.add_argument('--packets', type=int, default=10, help='Number of test packets')
    parser.add_argument('--vlan', type=int, default=2, help='VLAN ID')
    parser.add_argument('--interface', default='enp1s0', help='Host network interface')
    
    args = parser.parse_args()
    
    reader = PhysicalCounterReader(args.switches)
    success = reader.test_with_packets(
        port=args.port,
        num_packets=args.packets,
        vlan_id=args.vlan,
        interface=args.interface
    )
    
    exit(0 if success else 1)



""" Output:
sudo python3 e017_read_physical_counters.py --packets 20 --vlan 2
====================================================================================================
PHYSICAL INTERFACE COUNTER TEST
====================================================================================================

Port: et-0/0/96
Switches: 10.10.10.55, 10.10.10.56
Test: Send 20 packets with VLAN 2

Step 1: Reading baseline counters...
  10.10.10.55: Input=160, Output=285
  10.10.10.56: Input=16, Output=404

Step 2: Sending 20 packets...
✓ Sent 20 packets
Waiting 2 seconds for propagation...

Step 3: Reading post-test counters...
  10.10.10.55: Input=180, Output=286
  10.10.10.56: Input=16, Output=404

Step 4: Delta analysis
====================================================================================================

10.10.10.55:
  Input delta:  20 packets
  Output delta: 1 packets

10.10.10.56:
  Input delta:  0 packets
  Output delta: 0 packets

====================================================================================================
TOTALS:
  Packets sent:         20
  Total input delta:    20
  Total output delta:   1
  Input detection rate: 100.0%

✅ SUCCESS: 100% packet delivery!
====================================================================================================

"""