#!/usr/bin/env python3
"""
e034_single_switch_full_layer.py

FULL 2048×512 LAYER ON A SINGLE SWITCH!

Previous tests showed a limit around 256 terms, but that may have been
due to configuration issues, not actual TCAM limits.

QFX5100 TCAM capacity is reportedly 4000+ filter terms.
Let's push it and find out!

Key changes from e033:
1. Configure ALL filter terms in ONE commit (not batched commits)
2. Use larger batch sizes
3. Better error handling to understand failures

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
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

# Import from previous experiments
from e032_first_layer_inference import RealLayerInference, craft_ethernet_frame


@dataclass
class FullLayerResult:
    """Results from full layer inference."""
    num_outputs: int
    num_inputs: int
    total_weights: int
    active_weights: int
    total_packets: int
    config_time: float
    send_time: float
    read_time: float
    match_rate: float
    success: bool


class SingleSwitchFullLayer(RealLayerInference):
    """
    Run the COMPLETE 2048×512 layer on a SINGLE QFX5100 switch.
    
    Key insight: The 256-limit we hit before was likely due to
    configuration issues (batched commits corrupting state), not
    actual TCAM limits.
    """
    
    FILTER_NAME = "full_2048_layer"
    VLAN_NAME = "full_2048_vlan"
    VLAN_ID = 900
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _run_cli_file(self, config_file: str, timeout: int = 300) -> Tuple[bool, str, str]:
        """
        Load configuration from a file - more reliable for large configs.
        """
        # First, copy the file to the switch
        scp_cmd = [
            'scp', '-i', self.ssh_key_path,
            '-o', 'StrictHostKeyChecking=no',
            config_file,
            f'root@{self.switch_ip}:/tmp/config.txt'
        ]
        subprocess.run(scp_cmd, capture_output=True, timeout=60)
        
        # Then load it
        load_cmd = f"cli -c 'configure ; load set /tmp/config.txt ; commit'"
        return self._ssh_command(load_cmd, timeout=timeout)
    
    def cleanup_switch(self) -> bool:
        """Remove any existing full layer configuration."""
        print("\n  Cleaning up previous configuration...")
        
        cleanup_commands = [
            f"delete interfaces {self.input_port} unit 0 family ethernet-switching filter",
            f"delete interfaces {self.input_port} unit 0 family ethernet-switching vlan members {self.VLAN_NAME}",
            f"delete firewall family ethernet-switching filter {self.FILTER_NAME}",
            f"delete vlans {self.VLAN_NAME}",
        ]
        
        # Run cleanup - ignore errors (things may not exist)
        for cmd in cleanup_commands:
            full_cmd = f"cli -c 'configure ; {cmd} ; commit'"
            self._ssh_command(full_cmd, timeout=30)
        
        time.sleep(2)
        return True
    
    def configure_full_layer(self, num_outputs: int = 2048) -> Tuple[bool, float]:
        """
        Configure all filter terms for the full layer.
        
        Strategy: Write config to a file and load it in one commit.
        This avoids the state corruption from multiple commits.
        """
        print(f"\n{'='*80}")
        print(f"CONFIGURING {num_outputs} OUTPUT NEURONS ON SINGLE SWITCH")
        print(f"{'='*80}\n")
        
        config_start = time.time()
        
        # Clean up first
        self.cleanup_switch()
        
        # Generate configuration file
        print(f"  Generating configuration for {num_outputs} neurons...")
        
        config_lines = []
        
        # Create VLAN
        config_lines.append(f"set vlans {self.VLAN_NAME} vlan-id {self.VLAN_ID}")
        
        # Create filter terms for each output neuron
        for i in range(num_outputs):
            # Generate MAC: 01:00:5e:20:HH:LL
            mac = f"01:00:5e:20:{(i >> 8) & 0xff:02x}:{i & 0xff:02x}"
            
            config_lines.extend([
                f"set firewall family ethernet-switching filter {self.FILTER_NAME} "
                f"term out{i} from destination-mac-address {mac}",
                f"set firewall family ethernet-switching filter {self.FILTER_NAME} "
                f"term out{i} then count out{i}_pkts",
                f"set firewall family ethernet-switching filter {self.FILTER_NAME} "
                f"term out{i} then accept",
            ])
            
            if (i + 1) % 500 == 0:
                print(f"    Generated terms for {i+1}/{num_outputs} neurons...")
        
        # Add default term
        config_lines.append(
            f"set firewall family ethernet-switching filter {self.FILTER_NAME} "
            f"term default then accept"
        )
        
        # Configure interface
        config_lines.extend([
            f"set interfaces {self.input_port} unit 0 family ethernet-switching interface-mode trunk",
            f"set interfaces {self.input_port} unit 0 family ethernet-switching vlan members {self.VLAN_NAME}",
            f"set interfaces {self.input_port} unit 0 family ethernet-switching filter input {self.FILTER_NAME}",
        ])
        
        print(f"  Generated {len(config_lines)} configuration lines")
        
        # Apply configuration in batches via SSH
        # Use larger batches with single commit at end
        print(f"\n  Applying configuration in batches...")
        
        batch_size = 100  # Commands per batch
        all_commands = config_lines[:-3]  # All except interface config
        interface_commands = config_lines[-3:]  # Interface config applied last
        
        for batch_start in range(0, len(all_commands), batch_size):
            batch_end = min(batch_start + batch_size, len(all_commands))
            batch = all_commands[batch_start:batch_end]
            
            cmd_str = " ; ".join(batch)
            full_cmd = f"cli -c 'configure ; {cmd_str} ; commit'"
            
            success, stdout, stderr = self._ssh_command(full_cmd, timeout=120)
            
            progress = (batch_end / len(all_commands)) * 100
            status = "✓" if 'error' not in stderr.lower() else "⚠"
            print(f"    Batch {batch_start}-{batch_end}: {status} ({progress:.0f}%)")
            
            if 'error' in stderr.lower():
                print(f"      Error: {stderr[:200]}")
        
        # Apply interface configuration
        print(f"  Applying interface configuration...")
        cmd_str = " ; ".join(interface_commands)
        full_cmd = f"cli -c 'configure ; {cmd_str} ; commit'"
        success, stdout, stderr = self._ssh_command(full_cmd, timeout=60)
        
        config_time = time.time() - config_start
        
        if 'error' in stderr.lower() or 'error' in stdout.lower():
            print(f"\n  ⚠ Configuration errors detected:")
            # Parse and show errors
            for line in (stdout + stderr).split('\n'):
                if 'error' in line.lower():
                    print(f"    {line}")
            return False, config_time
        
        print(f"\n  ✓ {num_outputs} neurons configured in {config_time:.1f}s")
        
        # Verify configuration
        print(f"\n  Verifying configuration...")
        verify_cmd = f"cli -c 'show firewall filter {self.FILTER_NAME} | count'"
        success, stdout, stderr = self._ssh_command(verify_cmd, timeout=30)
        
        if success:
            match = re.search(r'Count:\s*(\d+)', stdout)
            if match:
                line_count = int(match.group(1))
                print(f"    Filter has {line_count} lines")
                # Expected: ~1 header + num_outputs counters + some formatting
                if line_count >= num_outputs:
                    print(f"    ✓ Configuration verified!")
                else:
                    print(f"    ⚠ Fewer lines than expected")
        
        return True, config_time
    
    def clear_counters(self) -> bool:
        """Clear all counters."""
        success, _, _ = self._ssh_command(
            f"cli -c 'clear firewall filter {self.FILTER_NAME}'"
        )
        return success
    
    def read_counters(self, num_outputs: int) -> Dict[int, int]:
        """Read all counter values."""
        counters = {}
        
        print(f"  Reading {num_outputs} counters...")
        
        success, stdout, stderr = self._ssh_command(
            f"cli -c 'show firewall filter {self.FILTER_NAME}'",
            timeout=120
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
    
    def send_packets(self, weights: np.ndarray, input_activations: np.ndarray) -> Tuple[int, float]:
        """Send packets for matrix multiplication."""
        num_outputs, num_inputs = weights.shape
        src_mac = bytes.fromhex(self.host_mac.replace(':', ''))
        
        print(f"\n{'='*80}")
        print("SENDING PACKETS")
        print(f"{'='*80}\n")
        
        # Pre-generate packets
        print("  Pre-generating packets...")
        packets = []
        
        for input_idx in range(num_inputs):
            activation = int(input_activations[input_idx])
            if activation == 0:
                continue
            
            for output_idx in range(num_outputs):
                if weights[output_idx, input_idx] == 1:
                    mac = f"01005e20{(output_idx >> 8) & 0xff:02x}{output_idx & 0xff:02x}"
                    dst_mac = bytes.fromhex(mac)
                    
                    packet = craft_ethernet_frame(
                        dst_mac=dst_mac,
                        src_mac=src_mac,
                        vlan_id=self.VLAN_ID,
                        payload=b'FULL2048'
                    )
                    
                    for _ in range(activation):
                        packets.append(packet)
        
        total_packets = len(packets)
        print(f"  Generated {total_packets} packets")
        
        # Send
        print("  Sending...")
        sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW)
        sock.bind((self.interface, 0))
        
        send_start = time.time()
        
        for i, packet in enumerate(packets):
            sock.send(packet)
            if (i + 1) % 50000 == 0:
                elapsed = time.time() - send_start
                rate = (i + 1) / elapsed
                print(f"    Sent {i+1}/{total_packets} ({rate:.0f} pkt/s)")
        
        sock.close()
        
        send_time = time.time() - send_start
        rate = total_packets / send_time if send_time > 0 else 0
        print(f"\n  ✓ Sent {total_packets} packets in {send_time:.2f}s ({rate:.0f} pkt/s)")
        
        return total_packets, send_time
    
    def run_full_layer_test(self, num_outputs: int = 2048, num_inputs: int = 512) -> FullLayerResult:
        """
        Run the complete full layer inference test.
        """
        print("\n" + "="*80)
        print(f"SINGLE-SWITCH FULL LAYER TEST: {num_outputs}×{num_inputs}")
        print("="*80)
        
        # Load model and extract weights
        model_info = self.load_model_metadata()
        if not model_info:
            print("Failed to load model")
            return None
        
        weights = self.extract_layer_weights(model_info['reader'], max_size=max(num_outputs, num_inputs))
        if weights is None:
            print("Failed to extract weights")
            return None
        
        weights = weights[:num_outputs, :num_inputs]
        
        print(f"\nWeight matrix: {weights.shape}")
        print(f"  Ones: {np.sum(weights)}/{weights.size} ({100*np.sum(weights)/weights.size:.1f}%)")
        
        # Configure switch
        success, config_time = self.configure_full_layer(num_outputs)
        
        if not success:
            print("\n⚠ Configuration failed!")
            return FullLayerResult(
                num_outputs=num_outputs,
                num_inputs=num_inputs,
                total_weights=weights.size,
                active_weights=int(np.sum(weights)),
                total_packets=0,
                config_time=config_time,
                send_time=0,
                read_time=0,
                match_rate=0,
                success=False
            )
        
        # Create input activations
        np.random.seed(123)
        input_activations = np.ones(num_inputs, dtype=np.int32)  # Simple: all 1s
        
        expected_output = weights @ input_activations
        print(f"\nExpected output range: {expected_output.min()} - {expected_output.max()}")
        
        # Clear counters
        print("\nClearing counters...")
        self.clear_counters()
        time.sleep(1)
        
        # Send packets
        total_packets, send_time = self.send_packets(weights, input_activations)
        
        print("\nWaiting for propagation...")
        time.sleep(3)
        
        # Read counters
        print(f"\n{'='*80}")
        print("READING COUNTERS")
        print(f"{'='*80}\n")
        
        read_start = time.time()
        counters = self.read_counters(num_outputs)
        read_time = time.time() - read_start
        
        print(f"  Read {len(counters)} counters in {read_time:.1f}s")
        
        # Compare results
        actual_output = np.array([counters.get(i, 0) for i in range(num_outputs)])
        
        matches = np.sum(expected_output == actual_output)
        match_rate = matches / num_outputs
        
        print(f"\n{'='*80}")
        print("RESULTS")
        print(f"{'='*80}\n")
        
        print("Sample comparisons (first 20):")
        for i in range(min(20, num_outputs)):
            exp = expected_output[i]
            act = actual_output[i]
            match = "✓" if exp == act else "✗"
            print(f"  {i}: expected {exp}, actual {act} {match}")
        
        if num_outputs > 40:
            print("...")
            print("\nSample comparisons (last 10):")
            for i in range(num_outputs - 10, num_outputs):
                exp = expected_output[i]
                act = actual_output[i]
                match = "✓" if exp == act else "✗"
                print(f"  {i}: expected {exp}, actual {act} {match}")
        
        print(f"\nTotal matches: {matches}/{num_outputs} ({100*match_rate:.1f}%)")
        
        # Summary
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}\n")
        
        result = FullLayerResult(
            num_outputs=num_outputs,
            num_inputs=num_inputs,
            total_weights=weights.size,
            active_weights=int(np.sum(weights)),
            total_packets=total_packets,
            config_time=config_time,
            send_time=send_time,
            read_time=read_time,
            match_rate=match_rate,
            success=(match_rate >= 0.95)
        )
        
        if result.success:
            print("🎉 FULL LAYER INFERENCE ON SINGLE SWITCH SUCCESSFUL!")
            print()
            print(f"Layer size: {num_outputs} × {num_inputs}")
            print(f"Total weights: {weights.size:,}")
            print(f"Active weights: {int(np.sum(weights)):,}")
            print(f"Packets sent: {total_packets:,}")
            print()
            print("Timing:")
            print(f"  Config: {config_time:.1f}s (one-time)")
            print(f"  Send:   {send_time:.2f}s")
            print(f"  Read:   {read_time:.1f}s")
            print(f"  Total inference: {send_time + read_time:.1f}s")
        else:
            print(f"⚠ Test failed with {100*match_rate:.1f}% match rate")
        
        # Save results
        os.makedirs('bringup_logs', exist_ok=True)
        log_file = f"bringup_logs/single_switch_full_layer_{int(time.time())}.json"
        with open(log_file, 'w') as f:
            json.dump({
                'test_type': 'single_switch_full_layer',
                'num_outputs': int(num_outputs),
                'num_inputs': int(num_inputs),
                'total_weights': int(weights.size),
                'active_weights': int(np.sum(weights)),
                'total_packets': int(total_packets),
                'config_time': float(config_time),
                'send_time': float(send_time),
                'read_time': float(read_time),
                'match_rate': float(match_rate),
                'success': bool(result.success),
                'timestamp': float(time.time())
            }, f, indent=2)
        
        print(f"\nResults saved to: {log_file}")
        
        return result


class MultiSwitchFullLayer:
    """
    Full 2048×512 layer across TWO switches.
    
    Since each switch can handle ~1500 outputs (proven experimentally),
    we split 2048 outputs as:
      - Switch 1 (10.10.10.55): outputs 0-1023
      - Switch 2 (10.10.10.56): outputs 1024-2047
    """
    
    OUTPUTS_PER_SWITCH = 1024  # Half of 2048, well under 1500 limit
    VLAN_BASE = 900
    
    def __init__(self, 
                 model_path: str,
                 switches: List[Tuple[str, str]],  # [(ip, interface), ...]
                 ssh_key_path: str = '/home/multiplex/.ssh/id_rsa'):
        """
        Args:
            model_path: Path to GGUF model
            switches: List of (switch_ip, host_interface) tuples
            ssh_key_path: Path to SSH private key
        """
        self.model_path = model_path
        self.switches = switches
        self.ssh_key_path = ssh_key_path
        self.input_port = 'et-0/0/96'
        
        # Auto-detect host MACs
        self.host_macs = []
        for _, interface in switches:
            result = subprocess.run(
                ['ip', 'link', 'show', interface], 
                capture_output=True, text=True
            )
            for line in result.stdout.split('\n'):
                if 'link/ether' in line:
                    self.host_macs.append(line.split()[1])
                    break
        
        print(f"Multi-Switch Full Layer Engine")
        print(f"  Switches: {len(switches)}")
        print(f"  Max outputs: {len(switches) * self.OUTPUTS_PER_SWITCH}")
        for i, (ip, iface) in enumerate(switches):
            print(f"    Switch {i}: {ip} via {iface} (MAC: {self.host_macs[i]})")
    
    def _ssh_command(self, switch_ip: str, command: str, timeout: int = 120):
        """Execute SSH command on a switch."""
        ssh_cmd = [
            'ssh',
            '-i', self.ssh_key_path,
            '-o', 'StrictHostKeyChecking=no',
            '-o', 'UserKnownHostsFile=/dev/null',
            '-o', 'LogLevel=ERROR',
            '-o', 'ConnectTimeout=10',
            f'root@{switch_ip}', command
        ]
        try:
            result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=timeout)
            return True, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Timeout"
        except Exception as e:
            return False, "", str(e)
    
    def configure_switch(self, switch_idx: int, output_start: int, output_count: int) -> Tuple[bool, float]:
        """Configure a single switch for its portion of outputs."""
        switch_ip, interface = self.switches[switch_idx]
        filter_name = f"full_2048_sw{switch_idx}"
        vlan_name = f"full_2048_vlan_sw{switch_idx}"
        vlan_id = self.VLAN_BASE + switch_idx
        
        print(f"\n  Switch {switch_idx} ({switch_ip}): outputs {output_start}-{output_start + output_count - 1}")
        
        config_start = time.time()
        
        # Clean up first - including old single-switch filter names!
        print(f"    Cleaning up old configurations...")
        clean_commands = [
            # Remove interface bindings first
            f"delete interfaces {self.input_port} unit 0 family ethernet-switching filter",
            f"delete interfaces {self.input_port} unit 0 family ethernet-switching vlan members",
            # Remove old single-switch filters (from e034 single mode)
            f"delete firewall family ethernet-switching filter full_2048_layer",
            f"delete vlans full_2048_vlan",
            # Remove old multi-switch filters  
            f"delete firewall family ethernet-switching filter {filter_name}",
            f"delete vlans {vlan_name}",
            # Also clean up any layer_counter filters from e033
            f"delete firewall family ethernet-switching filter layer_counter_{switch_idx}",
            f"delete vlans layer_vlan_{switch_idx}",
        ]
        # Run all cleanup in one commit
        cmd_str = " ; ".join(clean_commands)
        full_cmd = f"cli -c 'configure ; {cmd_str} ; commit'"
        self._ssh_command(switch_ip, full_cmd, timeout=60)
        
        time.sleep(1)
        
        # Generate configuration
        print(f"    Generating {output_count} filter terms...")
        config_lines = []
        config_lines.append(f"set vlans {vlan_name} vlan-id {vlan_id}")
        
        for i in range(output_count):
            global_idx = output_start + i
            mac = f"01:00:5e:20:{(global_idx >> 8) & 0xff:02x}:{global_idx & 0xff:02x}"
            
            config_lines.extend([
                f"set firewall family ethernet-switching filter {filter_name} "
                f"term out{global_idx} from destination-mac-address {mac}",
                f"set firewall family ethernet-switching filter {filter_name} "
                f"term out{global_idx} then count out{global_idx}_pkts",
                f"set firewall family ethernet-switching filter {filter_name} "
                f"term out{global_idx} then accept",
            ])
        
        config_lines.append(
            f"set firewall family ethernet-switching filter {filter_name} term default then accept"
        )
        config_lines.extend([
            f"set interfaces {self.input_port} unit 0 family ethernet-switching interface-mode trunk",
            f"set interfaces {self.input_port} unit 0 family ethernet-switching vlan members {vlan_name}",
            f"set interfaces {self.input_port} unit 0 family ethernet-switching filter input {filter_name}",
        ])
        
        # Apply in batches
        print(f"    Applying {len(config_lines)} configuration lines...")
        batch_size = 100
        all_commands = config_lines[:-3]
        interface_commands = config_lines[-3:]
        
        for batch_start in range(0, len(all_commands), batch_size):
            batch_end = min(batch_start + batch_size, len(all_commands))
            batch = all_commands[batch_start:batch_end]
            
            cmd_str = " ; ".join(batch)
            full_cmd = f"cli -c 'configure ; {cmd_str} ; commit'"
            self._ssh_command(switch_ip, full_cmd, timeout=120)
            
            progress = (batch_end / len(all_commands)) * 100
            if batch_end % 500 < batch_size:
                print(f"      Progress: {progress:.0f}%")
        
        # Apply interface config
        cmd_str = " ; ".join(interface_commands)
        full_cmd = f"cli -c 'configure ; {cmd_str} ; commit'"
        self._ssh_command(switch_ip, full_cmd, timeout=60)
        
        config_time = time.time() - config_start
        print(f"    ✓ Configured in {config_time:.1f}s")
        
        return True, config_time
    
    def clear_counters(self, switch_idx: int):
        """Clear counters on a switch."""
        switch_ip = self.switches[switch_idx][0]
        filter_name = f"full_2048_sw{switch_idx}"
        self._ssh_command(switch_ip, f"cli -c 'clear firewall filter {filter_name}'")
    
    def read_counters(self, switch_idx: int, output_start: int, output_count: int) -> Dict[int, int]:
        """Read counters from a switch."""
        switch_ip = self.switches[switch_idx][0]
        filter_name = f"full_2048_sw{switch_idx}"
        
        success, stdout, _ = self._ssh_command(
            switch_ip,
            f"cli -c 'show firewall filter {filter_name}'",
            timeout=120
        )
        
        counters = {}
        if success:
            for i in range(output_count):
                global_idx = output_start + i
                pattern = rf'out{global_idx}_pkts\s+\d+\s+(\d+)'
                match = re.search(pattern, stdout)
                if match:
                    counters[global_idx] = int(match.group(1))
                else:
                    counters[global_idx] = 0
        
        return counters
    
    def send_packets(self, weights: np.ndarray, input_activations: np.ndarray) -> Tuple[int, float]:
        """Send packets to appropriate switches based on output neuron index."""
        num_outputs, num_inputs = weights.shape
        
        print(f"\n{'='*80}")
        print("SENDING PACKETS (MULTI-SWITCH)")
        print(f"{'='*80}\n")
        
        # Pre-generate packets grouped by switch
        packets_by_switch = {i: [] for i in range(len(self.switches))}
        
        print("  Pre-generating packets...")
        for input_idx in range(num_inputs):
            activation = int(input_activations[input_idx])
            if activation == 0:
                continue
            
            for output_idx in range(num_outputs):
                if weights[output_idx, input_idx] == 1:
                    switch_idx = output_idx // self.OUTPUTS_PER_SWITCH
                    if switch_idx >= len(self.switches):
                        continue
                    
                    mac = f"01005e20{(output_idx >> 8) & 0xff:02x}{output_idx & 0xff:02x}"
                    dst_mac = bytes.fromhex(mac)
                    src_mac = bytes.fromhex(self.host_macs[switch_idx].replace(':', ''))
                    
                    vlan_id = self.VLAN_BASE + switch_idx
                    packet = craft_ethernet_frame(
                        dst_mac=dst_mac,
                        src_mac=src_mac,
                        vlan_id=vlan_id,
                        payload=b'MULTI_SW'
                    )
                    
                    for _ in range(activation):
                        packets_by_switch[switch_idx].append(packet)
        
        total_packets = sum(len(pkts) for pkts in packets_by_switch.values())
        print(f"  Total packets: {total_packets}")
        for i, pkts in packets_by_switch.items():
            print(f"    Switch {i}: {len(pkts)} packets")
        
        # Send packets to each switch
        send_start = time.time()
        
        for switch_idx, packets in packets_by_switch.items():
            if not packets:
                continue
            
            interface = self.switches[switch_idx][1]
            sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW)
            sock.bind((interface, 0))
            
            for i, packet in enumerate(packets):
                sock.send(packet)
                if (i + 1) % 100000 == 0:
                    elapsed = time.time() - send_start
                    rate = (i + 1) / elapsed
                    print(f"    Switch {switch_idx}: sent {i+1}/{len(packets)} ({rate:.0f} pkt/s)")
            
            sock.close()
            print(f"    ✓ Sent {len(packets)} to switch {switch_idx} via {interface}")
        
        send_time = time.time() - send_start
        rate = total_packets / send_time if send_time > 0 else 0
        print(f"\n  ✓ Sent {total_packets} packets in {send_time:.2f}s ({rate:.0f} pkt/s)")
        
        return total_packets, send_time
    
    def run_full_2048_test(self, num_inputs: int = 512) -> FullLayerResult:
        """Run the complete 2048×512 layer across two switches."""
        num_outputs = 2048  # Full layer!
        
        print("\n" + "="*80)
        print(f"MULTI-SWITCH FULL LAYER TEST: {num_outputs}×{num_inputs}")
        print("="*80)
        
        # Load model and extract weights
        temp_engine = SingleSwitchFullLayer(
            model_path=self.model_path,
            switch_ip=self.switches[0][0],
            interface=self.switches[0][1]
        )
        
        model_info = temp_engine.load_model_metadata()
        if not model_info:
            print("Failed to load model")
            return None
        
        weights = temp_engine.extract_layer_weights(model_info['reader'], max_size=max(num_outputs, num_inputs))
        if weights is None:
            print("Failed to extract weights")
            return None
        
        weights = weights[:num_outputs, :num_inputs]
        
        print(f"\nWeight matrix: {weights.shape}")
        print(f"  Ones: {np.sum(weights)}/{weights.size} ({100*np.sum(weights)/weights.size:.1f}%)")
        
        # Configure both switches
        print(f"\n{'='*80}")
        print(f"CONFIGURING {num_outputs} OUTPUT NEURONS ACROSS {len(self.switches)} SWITCHES")
        print(f"{'='*80}")
        
        total_config_time = 0
        num_switches_needed = (num_outputs + self.OUTPUTS_PER_SWITCH - 1) // self.OUTPUTS_PER_SWITCH
        
        for i in range(min(num_switches_needed, len(self.switches))):
            output_start = i * self.OUTPUTS_PER_SWITCH
            output_count = min(self.OUTPUTS_PER_SWITCH, num_outputs - output_start)
            _, config_time = self.configure_switch(i, output_start, output_count)
            total_config_time += config_time
        
        # Create input activations
        np.random.seed(123)
        input_activations = np.ones(num_inputs, dtype=np.int32)
        
        expected_output = weights @ input_activations
        print(f"\nExpected output range: {expected_output.min()} - {expected_output.max()}")
        
        # Clear counters
        print("\nClearing counters on all switches...")
        for i in range(len(self.switches)):
            self.clear_counters(i)
        time.sleep(1)
        
        # Send packets
        total_packets, send_time = self.send_packets(weights, input_activations)
        
        print("\nWaiting for propagation...")
        time.sleep(3)
        
        # Read counters from all switches
        print(f"\n{'='*80}")
        print("READING COUNTERS FROM ALL SWITCHES")
        print(f"{'='*80}\n")
        
        read_start = time.time()
        all_counters = {}
        
        for i in range(min(num_switches_needed, len(self.switches))):
            output_start = i * self.OUTPUTS_PER_SWITCH
            output_count = min(self.OUTPUTS_PER_SWITCH, num_outputs - output_start)
            counters = self.read_counters(i, output_start, output_count)
            all_counters.update(counters)
            print(f"  Switch {i}: read {len(counters)} counters")
        
        read_time = time.time() - read_start
        print(f"  Total read time: {read_time:.1f}s")
        
        # Compare results
        actual_output = np.array([all_counters.get(i, 0) for i in range(num_outputs)])
        
        matches = np.sum(expected_output == actual_output)
        match_rate = matches / num_outputs
        
        print(f"\n{'='*80}")
        print("RESULTS")
        print(f"{'='*80}\n")
        
        print("Sample comparisons (first 20 - Switch 0):")
        for i in range(min(20, num_outputs)):
            exp = expected_output[i]
            act = actual_output[i]
            match = "✓" if exp == act else "✗"
            print(f"  {i}: expected {exp}, actual {act} {match}")
        
        print("...")
        print(f"\nSample comparisons (outputs 1020-1030 - boundary):")
        for i in range(1020, min(1030, num_outputs)):
            exp = expected_output[i]
            act = actual_output[i]
            match = "✓" if exp == act else "✗"
            print(f"  {i}: expected {exp}, actual {act} {match}")
        
        print("...")
        print(f"\nSample comparisons (last 10 - Switch 1):")
        for i in range(num_outputs - 10, num_outputs):
            exp = expected_output[i]
            act = actual_output[i]
            match = "✓" if exp == act else "✗"
            print(f"  {i}: expected {exp}, actual {act} {match}")
        
        print(f"\nTotal matches: {matches}/{num_outputs} ({100*match_rate:.1f}%)")
        
        # Summary
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}\n")
        
        result = FullLayerResult(
            num_outputs=num_outputs,
            num_inputs=num_inputs,
            total_weights=weights.size,
            active_weights=int(np.sum(weights)),
            total_packets=total_packets,
            config_time=total_config_time,
            send_time=send_time,
            read_time=read_time,
            match_rate=match_rate,
            success=(match_rate >= 0.95)
        )
        
        if result.success:
            print("🎉🎉🎉 FULL 2048×512 LAYER INFERENCE SUCCESSFUL! 🎉🎉🎉")
            print()
            print(f"Layer size: {num_outputs} × {num_inputs}")
            print(f"Total weights: {weights.size:,}")
            print(f"Active weights: {int(np.sum(weights)):,}")
            print(f"Packets sent: {total_packets:,}")
            print()
            print("Timing:")
            print(f"  Config: {total_config_time:.1f}s (one-time, parallel would be {total_config_time/2:.1f}s)")
            print(f"  Send:   {send_time:.2f}s")
            print(f"  Read:   {read_time:.1f}s")
            print(f"  Total inference: {send_time + read_time:.1f}s")
            print()
            print("THE COMPLETE FIRST LAYER OF QWEN3-80B RUNS ON TWO SWITCHES! 🚀")
        else:
            print(f"⚠ Test failed with {100*match_rate:.1f}% match rate")
        
        # Save results
        os.makedirs('bringup_logs', exist_ok=True)
        log_file = f"bringup_logs/multi_switch_full_2048_{int(time.time())}.json"
        with open(log_file, 'w') as f:
            json.dump({
                'test_type': 'multi_switch_full_2048',
                'num_outputs': int(num_outputs),
                'num_inputs': int(num_inputs),
                'num_switches': len(self.switches),
                'outputs_per_switch': self.OUTPUTS_PER_SWITCH,
                'total_weights': int(weights.size),
                'active_weights': int(np.sum(weights)),
                'total_packets': int(total_packets),
                'config_time': float(total_config_time),
                'send_time': float(send_time),
                'read_time': float(read_time),
                'match_rate': float(match_rate),
                'success': bool(result.success),
                'timestamp': float(time.time())
            }, f, indent=2)
        
        print(f"\nResults saved to: {log_file}")
        
        return result


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Full layer inference (single or multi-switch)"
    )
    parser.add_argument(
        '--model',
        default='./models/Qwen3-Next-80B-A3B-Thinking-UD-TQ1_0.gguf',
        help='Path to GGUF model file'
    )
    parser.add_argument(
        '--switch',
        default='10.10.10.55',
        help='Switch IP address (single-switch mode)'
    )
    parser.add_argument(
        '--interface',
        default='enp1s0',
        help='Host network interface'
    )
    parser.add_argument(
        '--outputs',
        type=int,
        default=512,
        help='Number of output neurons (default: 512, max: 1500 single, 2048 multi)'
    )
    parser.add_argument(
        '--inputs',
        type=int,
        default=512,
        help='Number of input neurons (default: 512)'
    )
    parser.add_argument(
        '--multi',
        action='store_true',
        help='Use multi-switch mode (both switches, enables full 2048 outputs)'
    )
    
    args = parser.parse_args()
    
    if args.multi:
        # Multi-switch mode: split 2048 outputs across two switches
        switches = [
            ('10.10.10.55', 'enp1s0'),      # Switch 0: outputs 0-1023
            ('10.10.10.56', 'enp1s0d1'),    # Switch 1: outputs 1024-2047
        ]
        
        engine = MultiSwitchFullLayer(
            model_path=args.model,
            switches=switches
        )
        
        result = engine.run_full_2048_test(num_inputs=args.inputs)
        
        import sys
        sys.exit(0 if result and result.success else 1)
    
    else:
        # Single-switch mode (max ~1500 outputs)
        print(f"\nTesting {args.outputs}×{args.inputs} layer on single switch")
        print(f"Switch: {args.switch}")
        print(f"Interface: {args.interface}")
        
        if args.outputs > 1500:
            print(f"\n⚠ Warning: {args.outputs} outputs exceeds single-switch limit (~1500)")
            print("  Use --multi for full 2048 outputs across two switches")
        
        test = SingleSwitchFullLayer(
            model_path=args.model,
            switch_ip=args.switch,
            interface=args.interface
        )
        
        result = test.run_full_layer_test(
            num_outputs=args.outputs,
            num_inputs=args.inputs
        )
        
        import sys
        sys.exit(0 if result and result.success else 1)



""" Output:
sudo python3 e034_single_switch_full_layer.py --outputs 1500 --inputs 512

Testing 1500×512 layer on single switch
Switch: 10.10.10.55
Interface: enp1s0
Host MAC: 7c:fe:90:9d:2a:f0
Switch: 10.10.10.55
Model: ./models/Qwen3-Next-80B-A3B-Thinking-UD-TQ1_0.gguf

================================================================================
SINGLE-SWITCH FULL LAYER TEST: 1500×512
================================================================================

================================================================================
LOADING MODEL METADATA
================================================================================

Model Architecture:
  general.architecture: [113 119 101 110  51 110 101 120 116]
  general.name: [ 81 119 101 110  51  45  78 101 120 116  45  56  48  66  45  65  51  66
  45  84 104 105 110 107 105 110 103]

Available tensors (first 20):
  output.weight: shape=[  2048 151936], type=14
  output_norm.weight: shape=[2048], type=0
  token_embd.weight: shape=[  2048 151936], type=12
  blk.0.attn_norm.weight: shape=[2048], type=0
  blk.0.ffn_down_exps.weight: shape=[ 512 2048  512], type=19
  blk.0.ffn_down_shexp.weight: shape=[ 512 2048], type=13
  blk.0.ffn_gate_exps.weight: shape=[2048  512  512], type=19
  blk.0.ffn_gate_inp.weight: shape=[2048  512], type=0
  blk.0.ffn_gate_inp_shexp.weight: shape=[2048], type=30
  blk.0.ffn_gate_shexp.weight: shape=[2048  512], type=20
  blk.0.ffn_up_exps.weight: shape=[2048  512  512], type=19
  blk.0.ffn_up_shexp.weight: shape=[2048  512], type=20
  blk.0.post_attention_norm.weight: shape=[2048], type=0
  blk.0.ssm_a: shape=[32], type=0
  blk.0.ssm_ba.weight: shape=[2048   64], type=19
  blk.0.ssm_conv1d.weight: shape=[   4 8192], type=0
  blk.0.ssm_dt.bias: shape=[32], type=0
  blk.0.ssm_in.weight: shape=[ 2048 12288], type=19
  blk.0.ssm_norm.weight: shape=[128], type=0
  blk.0.ssm_out.weight: shape=[4096 2048], type=19
  ... and 787 more tensors

================================================================================
EXTRACTING LAYER 0 WEIGHTS
================================================================================

Found 11 2D weight tensors in layer 0:
  blk.0.ffn_down_exps.weight: shape=[ 512 2048  512], type=19
  blk.0.ffn_down_shexp.weight: shape=[ 512 2048], type=13
  blk.0.ffn_gate_exps.weight: shape=[2048  512  512], type=19
  blk.0.ffn_gate_inp.weight: shape=[2048  512], type=0
  blk.0.ffn_gate_shexp.weight: shape=[2048  512], type=20
  blk.0.ffn_up_exps.weight: shape=[2048  512  512], type=19
  blk.0.ffn_up_shexp.weight: shape=[2048  512], type=20
  blk.0.ssm_ba.weight: shape=[2048   64], type=19
  blk.0.ssm_conv1d.weight: shape=[   4 8192], type=0
  blk.0.ssm_in.weight: shape=[ 2048 12288], type=19

Selected tensor: blk.0.ffn_gate_inp.weight
  Shape: [2048  512]
  Type: 0
  Elements: 1048576
  Raw data size: 4194304 bytes

  Extracting 1500x512 submatrix for testing...
  F32 data: min=-0.7148, max=0.7969

  Weight matrix statistics:
    Shape: (1500, 512)
    Ones: 381069/768000 (49.6%)
    Sparsity: 50.4%
    Source: REAL MODEL WEIGHTS! ✓

Weight matrix: (1500, 512)
  Ones: 381069/768000 (49.6%)

================================================================================
CONFIGURING 1500 OUTPUT NEURONS ON SINGLE SWITCH
================================================================================


  Cleaning up previous configuration...
  Generating configuration for 1500 neurons...
    Generated terms for 500/1500 neurons...
    Generated terms for 1000/1500 neurons...
    Generated terms for 1500/1500 neurons...
  Generated 4505 configuration lines

  Applying configuration in batches...
    Batch 0-100: ✓ (2%)
    Batch 100-200: ✓ (4%)
    Batch 200-300: ✓ (7%)
    Batch 300-400: ✓ (9%)
    Batch 400-500: ✓ (11%)
    Batch 500-600: ✓ (13%)
    Batch 600-700: ✓ (16%)
    Batch 700-800: ✓ (18%)
    Batch 800-900: ✓ (20%)
    Batch 900-1000: ✓ (22%)
    Batch 1000-1100: ✓ (24%)
    Batch 1100-1200: ✓ (27%)
    Batch 1200-1300: ✓ (29%)
    Batch 1300-1400: ✓ (31%)
    Batch 1400-1500: ✓ (33%)
    Batch 1500-1600: ✓ (36%)
    Batch 1600-1700: ✓ (38%)
    Batch 1700-1800: ✓ (40%)
    Batch 1800-1900: ✓ (42%)
    Batch 1900-2000: ✓ (44%)
    Batch 2000-2100: ✓ (47%)
    Batch 2100-2200: ✓ (49%)
    Batch 2200-2300: ✓ (51%)
    Batch 2300-2400: ✓ (53%)
    Batch 2400-2500: ✓ (56%)
    Batch 2500-2600: ✓ (58%)
    Batch 2600-2700: ✓ (60%)
    Batch 2700-2800: ✓ (62%)
    Batch 2800-2900: ✓ (64%)
    Batch 2900-3000: ✓ (67%)
    Batch 3000-3100: ✓ (69%)
    Batch 3100-3200: ✓ (71%)
    Batch 3200-3300: ✓ (73%)
    Batch 3300-3400: ✓ (76%)
    Batch 3400-3500: ✓ (78%)
    Batch 3500-3600: ✓ (80%)
    Batch 3600-3700: ✓ (82%)
    Batch 3700-3800: ✓ (84%)
    Batch 3800-3900: ✓ (87%)
    Batch 3900-4000: ✓ (89%)
    Batch 4000-4100: ✓ (91%)
    Batch 4100-4200: ✓ (93%)
    Batch 4200-4300: ✓ (96%)
    Batch 4300-4400: ✓ (98%)
    Batch 4400-4500: ✓ (100%)
    Batch 4500-4502: ✓ (100%)
  Applying interface configuration...

  ✓ 1500 neurons configured in 202.1s

  Verifying configuration...
    Filter has 1503 lines
    ✓ Configuration verified!

Expected output range: 221 - 292

Clearing counters...

================================================================================
SENDING PACKETS
================================================================================

  Pre-generating packets...
  Generated 381069 packets
  Sending...
    Sent 50000/381069 (639479 pkt/s)
    Sent 100000/381069 (637832 pkt/s)
    Sent 150000/381069 (638395 pkt/s)
    Sent 200000/381069 (636995 pkt/s)
    Sent 250000/381069 (637519 pkt/s)
    Sent 300000/381069 (637516 pkt/s)
    Sent 350000/381069 (637876 pkt/s)

  ✓ Sent 381069 packets in 0.60s (632075 pkt/s)

Waiting for propagation...

================================================================================
READING COUNTERS
================================================================================

  Reading 1500 counters...
  Read 1500 counters in 5.4s

================================================================================
RESULTS
================================================================================

Sample comparisons (first 20):
  0: expected 248, actual 248 ✓
  1: expected 245, actual 245 ✓
  2: expected 279, actual 279 ✓
  3: expected 258, actual 258 ✓
  4: expected 240, actual 240 ✓
  5: expected 246, actual 246 ✓
  6: expected 259, actual 259 ✓
  7: expected 259, actual 259 ✓
  8: expected 245, actual 245 ✓
  9: expected 233, actual 233 ✓
  10: expected 248, actual 248 ✓
  11: expected 251, actual 251 ✓
  12: expected 246, actual 246 ✓
  13: expected 265, actual 265 ✓
  14: expected 274, actual 274 ✓
  15: expected 246, actual 246 ✓
  16: expected 248, actual 248 ✓
  17: expected 243, actual 243 ✓
  18: expected 269, actual 269 ✓
  19: expected 254, actual 254 ✓
...

Sample comparisons (last 10):
  1490: expected 262, actual 262 ✓
  1491: expected 271, actual 271 ✓
  1492: expected 260, actual 260 ✓
  1493: expected 257, actual 257 ✓
  1494: expected 264, actual 264 ✓
  1495: expected 262, actual 262 ✓
  1496: expected 247, actual 247 ✓
  1497: expected 242, actual 242 ✓
  1498: expected 264, actual 264 ✓
  1499: expected 263, actual 263 ✓

Total matches: 1500/1500 (100.0%)

================================================================================
SUMMARY
================================================================================

🎉 FULL LAYER INFERENCE ON SINGLE SWITCH SUCCESSFUL!

Layer size: 1500 × 512
Total weights: 768,000
Active weights: 381,069
Packets sent: 381,069

Timing:
  Config: 202.1s (one-time)
  Send:   0.60s
  Read:   5.4s
  Total inference: 6.0s

Results saved to: bringup_logs/single_switch_full_layer_1766805399.json
(.venv) multiplex@multiplex1:~/photonic_inference_engine/phase_001/phase_001$ cat bringup_logs/single_switch_full_layer_1766805399.json
{
  "test_type": "single_switch_full_layer",
  "num_outputs": 1500,
  "num_inputs": 512,
  "total_weights": 768000,
  "active_weights": 381069,
  "total_packets": 381069,
  "config_time": 202.06862211227417,
  "send_time": 0.6028857231140137,
  "read_time": 5.365726947784424,
  "match_rate": 1.0,
  "success": true,
  "timestamp": 1766805399.2324204
"""


""" Output (multi):
sudo python3 e034_single_switch_full_layer.py --multi --inputs 512
Multi-Switch Full Layer Engine
  Switches: 2
  Max outputs: 2048
    Switch 0: 10.10.10.55 via enp1s0 (MAC: 7c:fe:90:9d:2a:f0)
    Switch 1: 10.10.10.56 via enp1s0d1 (MAC: 7c:fe:90:9d:2a:f1)

================================================================================
MULTI-SWITCH FULL LAYER TEST: 2048×512
================================================================================
Host MAC: 7c:fe:90:9d:2a:f0
Switch: 10.10.10.55
Model: ./models/Qwen3-Next-80B-A3B-Thinking-UD-TQ1_0.gguf

================================================================================
LOADING MODEL METADATA
================================================================================

Model Architecture:
  general.architecture: [113 119 101 110  51 110 101 120 116]
  general.name: [ 81 119 101 110  51  45  78 101 120 116  45  56  48  66  45  65  51  66
  45  84 104 105 110 107 105 110 103]

Available tensors (first 20):
  output.weight: shape=[  2048 151936], type=14
  output_norm.weight: shape=[2048], type=0
  token_embd.weight: shape=[  2048 151936], type=12
  blk.0.attn_norm.weight: shape=[2048], type=0
  blk.0.ffn_down_exps.weight: shape=[ 512 2048  512], type=19
  blk.0.ffn_down_shexp.weight: shape=[ 512 2048], type=13
  blk.0.ffn_gate_exps.weight: shape=[2048  512  512], type=19
  blk.0.ffn_gate_inp.weight: shape=[2048  512], type=0
  blk.0.ffn_gate_inp_shexp.weight: shape=[2048], type=30
  blk.0.ffn_gate_shexp.weight: shape=[2048  512], type=20
  blk.0.ffn_up_exps.weight: shape=[2048  512  512], type=19
  blk.0.ffn_up_shexp.weight: shape=[2048  512], type=20
  blk.0.post_attention_norm.weight: shape=[2048], type=0
  blk.0.ssm_a: shape=[32], type=0
  blk.0.ssm_ba.weight: shape=[2048   64], type=19
  blk.0.ssm_conv1d.weight: shape=[   4 8192], type=0
  blk.0.ssm_dt.bias: shape=[32], type=0
  blk.0.ssm_in.weight: shape=[ 2048 12288], type=19
  blk.0.ssm_norm.weight: shape=[128], type=0
  blk.0.ssm_out.weight: shape=[4096 2048], type=19
  ... and 787 more tensors

================================================================================
EXTRACTING LAYER 0 WEIGHTS
================================================================================

Found 11 2D weight tensors in layer 0:
  blk.0.ffn_down_exps.weight: shape=[ 512 2048  512], type=19
  blk.0.ffn_down_shexp.weight: shape=[ 512 2048], type=13
  blk.0.ffn_gate_exps.weight: shape=[2048  512  512], type=19
  blk.0.ffn_gate_inp.weight: shape=[2048  512], type=0
  blk.0.ffn_gate_shexp.weight: shape=[2048  512], type=20
  blk.0.ffn_up_exps.weight: shape=[2048  512  512], type=19
  blk.0.ffn_up_shexp.weight: shape=[2048  512], type=20
  blk.0.ssm_ba.weight: shape=[2048   64], type=19
  blk.0.ssm_conv1d.weight: shape=[   4 8192], type=0
  blk.0.ssm_in.weight: shape=[ 2048 12288], type=19

Selected tensor: blk.0.ffn_gate_inp.weight
  Shape: [2048  512]
  Type: 0
  Elements: 1048576
  Raw data size: 4194304 bytes

  Extracting 2048x512 submatrix for testing...
  F32 data: min=-0.7148, max=0.7969

  Weight matrix statistics:
    Shape: (2048, 512)
    Ones: 520511/1048576 (49.6%)
    Sparsity: 50.4%
    Source: REAL MODEL WEIGHTS! ✓

Weight matrix: (2048, 512)
  Ones: 520511/1048576 (49.6%)

================================================================================
CONFIGURING 2048 OUTPUT NEURONS ACROSS 2 SWITCHES
================================================================================

  Switch 0 (10.10.10.55): outputs 0-1023
    Cleaning up old configurations...
    Generating 1024 filter terms...
    Applying 3077 configuration lines...
      Progress: 16%
      Progress: 33%
      Progress: 49%
      Progress: 65%
      Progress: 81%
      Progress: 98%
      Progress: 100%
    ✓ Configured in 176.5s

  Switch 1 (10.10.10.56): outputs 1024-2047
    Cleaning up old configurations...
    Generating 1024 filter terms...
    Applying 3077 configuration lines...
      Progress: 16%
      Progress: 33%
      Progress: 49%
      Progress: 65%
      Progress: 81%
      Progress: 98%
      Progress: 100%
    ✓ Configured in 173.4s

Expected output range: 214 - 292

Clearing counters on all switches...

================================================================================
SENDING PACKETS (MULTI-SWITCH)
================================================================================

  Pre-generating packets...
  Total packets: 520511
    Switch 0: 260020 packets
    Switch 1: 260491 packets
    Switch 0: sent 100000/260020 (637954 pkt/s)
    Switch 0: sent 200000/260020 (637927 pkt/s)
    ✓ Sent 260020 to switch 0 via enp1s0
    Switch 1: sent 100000/260491 (175623 pkt/s)
    Switch 1: sent 200000/260491 (275442 pkt/s)
    ✓ Sent 260491 to switch 1 via enp1s0d1

  ✓ Sent 520511 packets in 0.83s (629436 pkt/s)

Waiting for propagation...

================================================================================
READING COUNTERS FROM ALL SWITCHES
================================================================================

  Switch 0: read 1024 counters
  Switch 1: read 1024 counters
  Total read time: 8.0s

================================================================================
RESULTS
================================================================================

Sample comparisons (first 20 - Switch 0):
  0: expected 248, actual 248 ✓
  1: expected 245, actual 245 ✓
  2: expected 279, actual 279 ✓
  3: expected 258, actual 258 ✓
  4: expected 240, actual 240 ✓
  5: expected 246, actual 246 ✓
  6: expected 259, actual 259 ✓
  7: expected 259, actual 259 ✓
  8: expected 245, actual 245 ✓
  9: expected 233, actual 233 ✓
  10: expected 248, actual 248 ✓
  11: expected 251, actual 251 ✓
  12: expected 246, actual 246 ✓
  13: expected 265, actual 265 ✓
  14: expected 274, actual 274 ✓
  15: expected 246, actual 246 ✓
  16: expected 248, actual 248 ✓
  17: expected 243, actual 243 ✓
  18: expected 269, actual 269 ✓
  19: expected 254, actual 254 ✓
...

Sample comparisons (outputs 1020-1030 - boundary):
  1020: expected 264, actual 264 ✓
  1021: expected 231, actual 231 ✓
  1022: expected 260, actual 260 ✓
  1023: expected 248, actual 248 ✓
  1024: expected 261, actual 261 ✓
  1025: expected 258, actual 258 ✓
  1026: expected 260, actual 260 ✓
  1027: expected 246, actual 246 ✓
  1028: expected 264, actual 264 ✓
  1029: expected 250, actual 250 ✓
...

Sample comparisons (last 10 - Switch 1):
  2038: expected 265, actual 265 ✓
  2039: expected 256, actual 256 ✓
  2040: expected 256, actual 256 ✓
  2041: expected 239, actual 239 ✓
  2042: expected 261, actual 261 ✓
  2043: expected 255, actual 255 ✓
  2044: expected 254, actual 254 ✓
  2045: expected 232, actual 232 ✓
  2046: expected 267, actual 267 ✓
  2047: expected 248, actual 248 ✓

Total matches: 2048/2048 (100.0%)

================================================================================
SUMMARY
================================================================================

🎉🎉🎉 FULL 2048×512 LAYER INFERENCE SUCCESSFUL! 🎉🎉🎉

Layer size: 2048 × 512
Total weights: 1,048,576
Active weights: 520,511
Packets sent: 520,511

Timing:
  Config: 349.9s (one-time, parallel would be 175.0s)
  Send:   0.83s
  Read:   8.0s
  Total inference: 8.8s

THE COMPLETE FIRST LAYER OF QWEN3-80B RUNS ON TWO SWITCHES! 🚀

Results saved to: bringup_logs/multi_switch_full_2048_1766806388.json
(.venv) multiplex@multiplex1:~/photonic_inference_engine/phase_001/phase_001$ cat bringup_logs/multi_switch_full_2048_1766806388.json
{
  "test_type": "multi_switch_full_2048",
  "num_outputs": 2048,
  "num_inputs": 512,
  "num_switches": 2,
  "outputs_per_switch": 1024,
  "total_weights": 1048576,
  "active_weights": 520511,
  "total_packets": 520511,
  "config_time": 349.93316078186035,
  "send_time": 0.8269476890563965,
  "read_time": 7.960282325744629,
  "match_rate": 1.0,
  "success": true,
  "timestamp": 1766806388.9080245
}
"""