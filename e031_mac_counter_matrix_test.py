#!/usr/bin/env python3
"""
e031_mac_counter_matrix_test.py

MAC-BASED COUNTER MATRIX MULTIPLICATION TEST

================================================================================
BREAKTHROUGH DISCOVERY
================================================================================

During e030 debugging, we discovered that:
1. Physical 10GbE ports don't exist without SFP+ transceivers installed
2. BUT - firewall filter counters CAN match on destination MAC address
3. Each destination MAC can have its own independent counter
4. This gives us HUNDREDS of "virtual neurons" without physical ports!

Proof from interactive testing:
  - Configured filter with terms matching MAC 01:00:5e:00:00:01 and :02
  - Sent 5 packets to MAC1, 3 packets to MAC2
  - Result: mac1_pkts=5, mac2_pkts=3 (PERFECT!)

================================================================================
ARCHITECTURE: MAC-BASED VIRTUAL NEURONS
================================================================================

Traditional approach (needs physical ports):
  Input → Physical Port → Port Counter
  
New approach (MAC-based counters):
  Input → Destination MAC → Firewall Filter Counter
  
Each OUTPUT neuron is represented by a unique destination MAC address.
Firewall filter terms match destination-mac and increment per-neuron counters.
The switch fabric handles packet forwarding, counters handle accumulation.

================================================================================
MATRIX MULTIPLICATION TEST
================================================================================

We implement y = W × x where:
  - x = input activation vector (4 neurons)
  - W = 4×4 binary weight matrix
  - y = output activation vector (4 neurons)

Weight Matrix W:
         In0  In1  In2  In3
  Out0:   1    0    1    0
  Out1:   0    1    1    1
  Out2:   1    1    0    0
  Out3:   0    0    1    1

Input activations x:
  x[0] = 3
  x[1] = 2
  x[2] = 4
  x[3] = 1

Expected output y = W × x:
  y[0] = 1×3 + 0×2 + 1×4 + 0×1 = 7
  y[1] = 0×3 + 1×2 + 1×4 + 1×1 = 7
  y[2] = 1×3 + 1×2 + 0×4 + 0×1 = 5
  y[3] = 0×3 + 0×2 + 1×4 + 1×1 = 5

Packet Generation:
  For each input i with activation x[i]:
    For each output j where W[j,i] = 1:
      Send x[i] packets with destination MAC = Output j's MAC

Counter Reading:
  Each output neuron's counter = accumulated sum = y[j]

Author: Research Phase 001
Date: December 2025
"""

import subprocess
import socket
import time
import json
import os
import re
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass

from e001_packet_craft_and_parse import craft_ethernet_frame


@dataclass 
class MatrixResult:
    """Results from MAC-based matrix multiplication test."""
    weight_matrix: np.ndarray
    input_activations: np.ndarray
    expected_output: np.ndarray
    actual_output: np.ndarray
    match: bool
    timestamp: float


class MACCounterMatrixTest:
    """
    Prove matrix multiplication using MAC-based firewall filter counters.
    
    Key insight: Each output neuron gets a unique destination MAC.
    Firewall filter counters accumulate packets per destination MAC.
    This eliminates the need for physical output ports!
    """
    
    def __init__(self,
                 switch_ip: str = '10.10.10.55',
                 ssh_key_path: str = '/home/multiplex/.ssh/id_rsa',
                 interface: str = 'enp1s0'):
        self.switch_ip = switch_ip
        self.ssh_key_path = ssh_key_path
        self.interface = interface
        
        self.input_port = 'et-0/0/96'
        self.vlan_id = 500
        self.vlan_name = 'matmul_test'
        
        # Output neuron MAC addresses (using multicast range)
        self.output_macs = {
            0: '01:00:5e:00:01:00',  # Output neuron 0
            1: '01:00:5e:00:01:01',  # Output neuron 1
            2: '01:00:5e:00:01:02',  # Output neuron 2
            3: '01:00:5e:00:01:03',  # Output neuron 3
        }
        
        # Get host MAC
        self.host_mac = self._get_mac_address(interface)
        print(f"Host MAC: {self.host_mac}")
        print(f"Switch: {switch_ip}")
        print(f"Interface: {interface}")
    
    def _get_mac_address(self, interface: str) -> str:
        try:
            with open(f'/sys/class/net/{interface}/address', 'r') as f:
                return f.read().strip()
        except Exception:
            return '00:00:00:00:00:00'
    
    def _ssh_command(self, command: str, timeout: int = 30) -> Tuple[bool, str, str]:
        """Execute SSH command on switch."""
        cmd = [
            'ssh',
            '-i', self.ssh_key_path,
            '-o', 'StrictHostKeyChecking=no',
            '-o', 'UserKnownHostsFile=/dev/null',
            '-o', 'LogLevel=ERROR',
            f'root@{self.switch_ip}',
            command
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            return (result.returncode == 0, result.stdout, result.stderr)
        except Exception as e:
            return (False, '', str(e))
    
    def _run_config_commands(self, commands: List[str]) -> bool:
        """Run configuration commands."""
        full_command = " ; ".join(commands)
        cli_command = f"configure ; {full_command} ; commit"
        
        cmd = [
            'ssh',
            '-i', self.ssh_key_path,
            '-o', 'StrictHostKeyChecking=no',
            '-o', 'UserKnownHostsFile=/dev/null',
            '-o', 'LogLevel=ERROR',
            f'root@{self.switch_ip}',
            f"cli -c '{cli_command}'"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
            output = result.stdout + result.stderr
            return 'commit complete' in output.lower()
        except Exception as e:
            print(f"  Config error: {e}")
            return False
    
    def setup_mac_counters(self, num_outputs: int = 4) -> bool:
        """
        Configure firewall filter with per-output-neuron counters.
        
        Each output neuron gets:
        - A unique destination MAC address
        - A firewall filter term that matches that MAC
        - A counter that accumulates matching packets
        """
        print(f"\n{'='*80}")
        print("STEP 1: CONFIGURE MAC-BASED COUNTERS")
        print(f"{'='*80}\n")
        
        print("Setting up firewall filter with per-output-neuron counters...")
        print()
        print("Output Neuron → MAC Address Mapping:")
        for i in range(num_outputs):
            print(f"  Output {i} → {self.output_macs[i]}")
        print()
        
        # Clean up previous config
        clean_commands = [
            "delete firewall family ethernet-switching filter matmul_counter",
            f"delete vlans {self.vlan_name}",
            f"delete interfaces {self.input_port} unit 0 family ethernet-switching",
        ]
        self._run_config_commands(clean_commands)
        
        # Build filter with one term per output neuron
        filter_commands = []
        for i in range(num_outputs):
            mac = self.output_macs[i]
            filter_commands.extend([
                f"set firewall family ethernet-switching filter matmul_counter "
                f"term out{i} from destination-mac-address {mac}",
                f"set firewall family ethernet-switching filter matmul_counter "
                f"term out{i} then count out{i}_pkts",
                f"set firewall family ethernet-switching filter matmul_counter "
                f"term out{i} then accept",
            ])
        
        # Default term to accept unmatched packets
        filter_commands.append(
            "set firewall family ethernet-switching filter matmul_counter "
            "term default then accept"
        )
        
        # Setup VLAN and apply filter
        vlan_commands = [
            f"set vlans {self.vlan_name} vlan-id {self.vlan_id}",
            f"set interfaces {self.input_port} unit 0 family ethernet-switching "
            f"interface-mode trunk",
            f"set interfaces {self.input_port} unit 0 family ethernet-switching "
            f"vlan members {self.vlan_name}",
            f"set interfaces {self.input_port} unit 0 family ethernet-switching "
            f"filter input matmul_counter",
        ]
        
        all_commands = filter_commands + vlan_commands
        success = self._run_config_commands(all_commands)
        
        if success:
            print("  ✓ MAC-based counters configured")
        else:
            print("  ⚠ Configuration may have issues")
        
        return success
    
    def clear_counters(self) -> bool:
        """Clear all firewall counters to zero."""
        success, stdout, stderr = self._ssh_command(
            "cli -c 'clear firewall filter matmul_counter'"
        )
        return success
    
    def read_counters(self, num_outputs: int = 4) -> Dict[int, int]:
        """Read all output neuron counters."""
        counters = {}
        
        success, stdout, stderr = self._ssh_command(
            "cli -c 'show firewall filter matmul_counter'"
        )
        
        if success:
            for i in range(num_outputs):
                pattern = rf'out{i}_pkts\s+\d+\s+(\d+)'
                match = re.search(pattern, stdout)
                if match:
                    counters[i] = int(match.group(1))
                else:
                    counters[i] = 0
        
        return counters
    
    def send_matrix_packets(self, 
                           weight_matrix: np.ndarray,
                           input_activations: np.ndarray) -> int:
        """
        Send packets to implement matrix multiplication.
        
        For each input neuron i with activation x[i]:
          For each output neuron j where W[j,i] = 1:
            Send x[i] packets with destination MAC = output j's MAC
        
        The firewall counters will accumulate y[j] = sum_i(W[j,i] × x[i])
        """
        print(f"\n{'='*80}")
        print("STEP 2: SEND MATRIX MULTIPLICATION PACKETS")
        print(f"{'='*80}\n")
        
        src_mac = bytes.fromhex(self.host_mac.replace(':', ''))
        total_sent = 0
        
        sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW)
        sock.bind((self.interface, 0))
        
        num_inputs = len(input_activations)
        num_outputs = weight_matrix.shape[0]
        
        print("Packet generation plan:")
        for i in range(num_inputs):
            activation = int(input_activations[i])
            connected_outputs = [j for j in range(num_outputs) if weight_matrix[j, i] == 1]
            if connected_outputs:
                print(f"  Input {i} (x={activation}) → Outputs {connected_outputs}")
        print()
        
        # Send packets for each input neuron
        for input_idx in range(num_inputs):
            activation = int(input_activations[input_idx])
            
            # Find which outputs this input connects to (W[j,i] = 1)
            for output_idx in range(num_outputs):
                if weight_matrix[output_idx, input_idx] == 1:
                    # Send 'activation' packets to this output's MAC
                    dst_mac = bytes.fromhex(
                        self.output_macs[output_idx].replace(':', '')
                    )
                    
                    packet = craft_ethernet_frame(
                        dst_mac=dst_mac,
                        src_mac=src_mac,
                        vlan_id=self.vlan_id,
                        payload=f'IN{input_idx}_OUT{output_idx}'.encode()
                    )
                    
                    for _ in range(activation):
                        sock.send(packet)
                        total_sent += 1
        
        sock.close()
        
        print(f"✓ Total packets sent: {total_sent}")
        return total_sent
    
    def run_discovery_demo(self):
        """
        Demonstrate the MAC-based counter discovery.
        This is the breakthrough that enables the full matrix multiply.
        """
        print("\n" + "="*80)
        print("DISCOVERY DEMO: MAC-BASED FIREWALL COUNTERS")
        print("="*80)
        print()
        print("Key insight: QFX5100 firewall filters can match on destination-mac-address")
        print("Each destination MAC can have its own independent counter!")
        print()
        print("This means:")
        print("  ❌ We DON'T need 96 SFP+ transceivers ($2000-5000)")
        print("  ✓ We CAN use destination MAC as 'virtual output neuron'")
        print("  ✓ Each output neuron = unique MAC address")
        print("  ✓ Firewall counter = accumulated activation")
        print()
        
        # Quick demo - send to two different MACs
        print("Quick verification test:")
        print("  Sending 5 packets to MAC A, 3 packets to MAC B...")
        
        # Setup minimal counter config
        self._run_config_commands([
            "delete firewall family ethernet-switching filter demo_counter",
            "set firewall family ethernet-switching filter demo_counter "
            "term a from destination-mac-address 01:00:5e:ff:00:01",
            "set firewall family ethernet-switching filter demo_counter "
            "term a then count mac_a_pkts",
            "set firewall family ethernet-switching filter demo_counter "
            "term a then accept",
            "set firewall family ethernet-switching filter demo_counter "
            "term b from destination-mac-address 01:00:5e:ff:00:02",
            "set firewall family ethernet-switching filter demo_counter "
            "term b then count mac_b_pkts",
            "set firewall family ethernet-switching filter demo_counter "
            "term b then accept",
            "set firewall family ethernet-switching filter demo_counter "
            "term default then accept",
            f"set interfaces {self.input_port} unit 0 family ethernet-switching "
            "filter input demo_counter",
        ])
        
        # Clear counters
        self._ssh_command("cli -c 'clear firewall filter demo_counter'")
        time.sleep(0.5)
        
        # Send test packets
        src_mac = bytes.fromhex(self.host_mac.replace(':', ''))
        sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW)
        sock.bind((self.interface, 0))
        
        # 5 packets to MAC A
        dst_a = bytes.fromhex('01005eff0001')
        for _ in range(5):
            pkt = craft_ethernet_frame(dst_a, src_mac, vlan_id=self.vlan_id, 
                                       payload=b'DEMO_A')
            sock.send(pkt)
        
        # 3 packets to MAC B
        dst_b = bytes.fromhex('01005eff0002')
        for _ in range(3):
            pkt = craft_ethernet_frame(dst_b, src_mac, vlan_id=self.vlan_id,
                                       payload=b'DEMO_B')
            sock.send(pkt)
        
        sock.close()
        time.sleep(1)
        
        # Read counters
        success, stdout, stderr = self._ssh_command(
            "cli -c 'show firewall filter demo_counter'"
        )
        
        mac_a_count = 0
        mac_b_count = 0
        if success:
            match_a = re.search(r'mac_a_pkts\s+\d+\s+(\d+)', stdout)
            match_b = re.search(r'mac_b_pkts\s+\d+\s+(\d+)', stdout)
            if match_a:
                mac_a_count = int(match_a.group(1))
            if match_b:
                mac_b_count = int(match_b.group(1))
        
        print()
        print(f"  Result: MAC A counter = {mac_a_count} (expected 5)")
        print(f"  Result: MAC B counter = {mac_b_count} (expected 3)")
        
        if mac_a_count == 5 and mac_b_count == 3:
            print()
            print("  🎉 DISCOVERY CONFIRMED!")
            print("  Each destination MAC has independent counting!")
            return True
        else:
            print()
            print("  ⚠ Counts don't match - check configuration")
            return False
    
    def run_matrix_test(self) -> MatrixResult:
        """
        Run complete matrix multiplication test.
        """
        print("\n" + "="*80)
        print("MATRIX MULTIPLICATION TEST: y = W × x")
        print("="*80)
        
        # Define the weight matrix (4×4)
        W = np.array([
            [1, 0, 1, 0],  # Output 0: connected to inputs 0, 2
            [0, 1, 1, 1],  # Output 1: connected to inputs 1, 2, 3
            [1, 1, 0, 0],  # Output 2: connected to inputs 0, 1
            [0, 0, 1, 1],  # Output 3: connected to inputs 2, 3
        ], dtype=np.int32)
        
        # Define input activations
        x = np.array([3, 2, 4, 1], dtype=np.int32)
        
        # Calculate expected output
        y_expected = W @ x
        
        print(f"\nWeight Matrix W ({W.shape[0]}×{W.shape[1]}):")
        print("         In0  In1  In2  In3")
        for i, row in enumerate(W):
            print(f"  Out{i}:   {row[0]}    {row[1]}    {row[2]}    {row[3]}")
        
        print(f"\nInput Activations x:")
        for i, val in enumerate(x):
            print(f"  x[{i}] = {val}")
        
        print(f"\nExpected Output y = W × x:")
        for i, val in enumerate(y_expected):
            print(f"  y[{i}] = {val}")
        
        # Step 1: Setup counters
        self.setup_mac_counters(num_outputs=4)
        
        # Clear counters
        print("\nClearing counters...")
        self.clear_counters()
        time.sleep(0.5)
        
        # Verify counters are zero
        baseline = self.read_counters(4)
        print(f"Baseline counters: {baseline}")
        
        # Step 2: Send packets
        total_packets = self.send_matrix_packets(W, x)
        
        print("\nWaiting for packet propagation...")
        time.sleep(2)
        
        # Step 3: Read final counters
        print(f"\n{'='*80}")
        print("STEP 3: READ OUTPUT COUNTERS")
        print(f"{'='*80}\n")
        
        final_counters = self.read_counters(4)
        
        # Calculate actual output
        y_actual = np.array([final_counters.get(i, 0) for i in range(4)])
        
        print("Counter Results:")
        all_match = True
        for i in range(4):
            expected = y_expected[i]
            actual = y_actual[i]
            match = "✓" if actual == expected else "✗"
            if actual != expected:
                all_match = False
            print(f"  Output {i}: counter={actual}, expected={expected} {match}")
        
        # Summary
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}\n")
        
        if all_match:
            print("🎉 MATRIX MULTIPLICATION SUCCESSFUL!")
            print()
            print("This proves the complete architecture:")
            print("  ✓ Input activations encoded as packet counts")
            print("  ✓ Weight matrix encoded via destination MAC selection")
            print("  ✓ Switch fabric routes packets")
            print("  ✓ Firewall counters accumulate = matrix multiply!")
            print()
            print("y = W × x computed entirely in switch hardware!")
            print()
            print("Key metrics:")
            print(f"  - Input neurons: {len(x)}")
            print(f"  - Output neurons: {len(y_expected)}")
            print(f"  - Total packets: {total_packets}")
            print(f"  - Weight matrix density: {np.sum(W)}/{W.size} = {np.sum(W)/W.size:.0%}")
        else:
            print("⚠ Matrix multiplication had errors")
            print(f"  Expected: {y_expected}")
            print(f"  Actual:   {y_actual}")
        
        # Save results
        result = MatrixResult(
            weight_matrix=W,
            input_activations=x,
            expected_output=y_expected,
            actual_output=y_actual,
            match=all_match,
            timestamp=time.time()
        )
        
        os.makedirs('bringup_logs', exist_ok=True)
        log_file = f"bringup_logs/mac_counter_matmul_{int(time.time())}.json"
        with open(log_file, 'w') as f:
            json.dump({
                'test_type': 'mac_counter_matrix_multiply',
                'weight_matrix': W.tolist(),
                'input_activations': x.tolist(),
                'expected_output': y_expected.tolist(),
                'actual_output': y_actual.tolist(),
                'match': all_match,
                'total_packets': total_packets,
                'output_macs': self.output_macs,
                'timestamp': result.timestamp
            }, f, indent=2)
        
        print(f"\nResults saved to: {log_file}")
        
        return result


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description="MAC-based counter matrix multiplication test"
    )
    parser.add_argument(
        '--switch',
        default='10.10.10.55',
        help='Switch IP address'
    )
    parser.add_argument(
        '--interface',
        default='enp1s0',
        help='Host network interface'
    )
    parser.add_argument(
        '--discovery-only',
        action='store_true',
        help='Only run the discovery demo, not the full matrix test'
    )
    
    args = parser.parse_args()
    
    test = MACCounterMatrixTest(
        switch_ip=args.switch,
        interface=args.interface
    )
    
    # Part 1: Demonstrate the discovery
    discovery_ok = test.run_discovery_demo()
    
    if not discovery_ok:
        print("\nDiscovery failed - aborting matrix test")
        import sys
        sys.exit(1)
    
    if args.discovery_only:
        import sys
        sys.exit(0)
    
    # Part 2: Full matrix multiplication test
    result = test.run_matrix_test()
    
    import sys
    sys.exit(0 if result.match else 1)



""" Output:
sudo python3 e031_mac_counter_matrix_test.py
Host MAC: 7c:fe:90:9d:2a:f0
Switch: 10.10.10.55
Interface: enp1s0

================================================================================
DISCOVERY DEMO: MAC-BASED FIREWALL COUNTERS
================================================================================

Key insight: QFX5100 firewall filters can match on destination-mac-address
Each destination MAC can have its own independent counter!

This means:
  ❌ We DON'T need 96 SFP+ transceivers ($2000-5000)
  ✓ We CAN use destination MAC as 'virtual output neuron'
  ✓ Each output neuron = unique MAC address
  ✓ Firewall counter = accumulated activation

Quick verification test:
  Sending 5 packets to MAC A, 3 packets to MAC B...
 
  Result: MAC A counter = 5 (expected 5)
  Result: MAC B counter = 3 (expected 3)

  🎉 DISCOVERY CONFIRMED!
  Each destination MAC has independent counting!

================================================================================
MATRIX MULTIPLICATION TEST: y = W × x
================================================================================

Weight Matrix W (4×4):
         In0  In1  In2  In3
  Out0:   1    0    1    0
  Out1:   0    1    1    1
  Out2:   1    1    0    0
  Out3:   0    0    1    1

Input Activations x:
  x[0] = 3
  x[1] = 2
  x[2] = 4
  x[3] = 1

Expected Output y = W × x:
  y[0] = 7
  y[1] = 7
  y[2] = 5
  y[3] = 5

================================================================================
STEP 1: CONFIGURE MAC-BASED COUNTERS
================================================================================

Setting up firewall filter with per-output-neuron counters...

Output Neuron → MAC Address Mapping:
  Output 0 → 01:00:5e:00:01:00
  Output 1 → 01:00:5e:00:01:01
  Output 2 → 01:00:5e:00:01:02
  Output 3 → 01:00:5e:00:01:03

  ✓ MAC-based counters configured

Clearing counters...
Baseline counters: {0: 0, 1: 0, 2: 0, 3: 0}

================================================================================
STEP 2: SEND MATRIX MULTIPLICATION PACKETS
================================================================================

Packet generation plan:
  Input 0 (x=3) → Outputs [0, 2]
  Input 1 (x=2) → Outputs [1, 2]
  Input 2 (x=4) → Outputs [0, 1, 3]
  Input 3 (x=1) → Outputs [1, 3]

✓ Total packets sent: 24

Waiting for packet propagation...

================================================================================
STEP 3: READ OUTPUT COUNTERS
================================================================================

Counter Results:
  Output 0: counter=7, expected=7 ✓
  Output 1: counter=7, expected=7 ✓
  Output 2: counter=5, expected=5 ✓
  Output 3: counter=5, expected=5 ✓

================================================================================
SUMMARY
================================================================================

🎉 MATRIX MULTIPLICATION SUCCESSFUL!

This proves the complete architecture:
  ✓ Input activations encoded as packet counts
  ✓ Weight matrix encoded via destination MAC selection
  ✓ Switch fabric routes packets
  ✓ Firewall counters accumulate = matrix multiply!

y = W × x computed entirely in switch hardware!

Key metrics:
  - Input neurons: 4
  - Output neurons: 4
  - Total packets: 24
  - Weight matrix density: 9/16 = 56%

Results saved to: bringup_logs/mac_counter_matmul_1766799922.json
(.venv) multiplex@multiplex1:~/photonic_inference_engine/phase_001/phase_001$ cat bringup_logs/mac_counter_matmul_1766799922.json
{
  "test_type": "mac_counter_matrix_multiply",
  "weight_matrix": [
    [
      1,
      0,
      1,
      0
    ],
    [
      0,
      1,
      1,
      1
    ],
    [
      1,
      1,
      0,
      0
    ],
    [
      0,
      0,
      1,
      1
    ]
  ],
  "input_activations": [
    3,
    2,
    4,
    1
  ],
  "expected_output": [
    7,
    7,
    5,
    5
  ],
  "actual_output": [
    7,
    7,
    5,
    5
  ],
  "match": true,
  "total_packets": 24,
  "output_macs": {
    "0": "01:00:5e:00:01:00",
    "1": "01:00:5e:00:01:01",
    "2": "01:00:5e:00:01:02",
    "3": "01:00:5e:00:01:03"
  },
  "timestamp": 1766799922.0337877
"""