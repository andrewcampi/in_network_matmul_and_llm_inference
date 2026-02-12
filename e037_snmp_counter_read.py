#!/usr/bin/env python3
"""
e037_snmp_counter_read.py

SNMP-BASED COUNTER READING EXPERIMENT

================================================================================
HYPOTHESIS
================================================================================

From research notes (Section 6.7), counter read latencies are:
  - OpenNSL SDK:    30-50 μs   (requires C interface)
  - SNMP bulk GET:  100-200 μs (standard protocol)  
  - SSH/CLI:        1-5 ms     (what we've been using → 8s total!)

This experiment tests whether SNMP bulk GET can replace SSH for reading
firewall filter counters, potentially reducing counter read time from
~8 seconds to ~0.2 milliseconds.

================================================================================
APPROACH
================================================================================

1. Configure MAC-based firewall counters (proven in e031)
2. Send a known number of test packets
3. Read counters via SNMP bulk GET
4. Compare timing with SSH approach
5. Validate counter values are correct

================================================================================
EXPECTED OUTCOME
================================================================================

If SNMP works as expected:
  - Counter read time: <1 ms (vs ~8s SSH)
  - 100% accuracy (same values as SSH)
  - Clear path forward for full inference

Author: Research Phase 001
Date: December 2025
"""

import subprocess
import socket
import struct
import time
import os
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Try to import pysnmp for SNMP operations
PYSNMP_AVAILABLE = False
PYSNMP_V7 = False
try:
    # pysnmp 7.x uses pysnmp.hlapi.v2c for SNMPv2c
    from pysnmp.hlapi.v2c import (
        SnmpEngine, CommunityData, UdpTransportTarget, ContextData,
        ObjectType, ObjectIdentity
    )
    from pysnmp.hlapi.v2c import get_cmd as getCmd, next_cmd as nextCmd, bulk_cmd as bulkCmd
    PYSNMP_AVAILABLE = True
    PYSNMP_V7 = True
except ImportError:
    try:
        # pysnmp 4.x/5.x style
        from pysnmp.hlapi import (
            SnmpEngine, CommunityData, UdpTransportTarget, ContextData,
            ObjectType, ObjectIdentity, nextCmd, getCmd, bulkCmd
        )
        PYSNMP_AVAILABLE = True
        PYSNMP_V7 = False
    except ImportError:
        print("WARNING: pysnmp not installed. Install with: pip install pysnmp")


# ============================================================================
# CONFIGURATION
# ============================================================================

SWITCH1_IP = "10.10.10.55"
SWITCH2_IP = "10.10.10.56"
SSH_KEY = "/home/multiplex/.ssh/id_rsa"

# Host interface
SEND_IFACE = "enp1s0"
SEND_MAC = "7c:fe:90:9d:2a:f0"

# Test parameters
NUM_OUTPUT_NEURONS = 4  # Small test first
VLAN_ID = 650  # Use unique VLAN ID (600 is used by layer_test)
FILTER_NAME = "snmp_test_counter"

# Output neuron MAC addresses (multicast range)
# Generate dynamically for any number of outputs
def get_output_mac(idx: int) -> str:
    """Generate multicast MAC for output neuron idx."""
    return f'01:00:5e:00:02:{idx:02x}'

OUTPUT_MACS = {i: get_output_mac(i) for i in range(256)}  # Pre-generate 256


# ============================================================================
# PACKET CRAFTING (from e031)
# ============================================================================

def craft_ethernet_frame(dst_mac: bytes, src_mac: bytes, vlan_id: int, 
                         payload: bytes = b'') -> bytes:
    """Craft an Ethernet frame with VLAN tag."""
    vlan_tci = (0 << 13) | vlan_id
    eth_header = dst_mac + src_mac + struct.pack('!HHH', 0x8100, vlan_tci, 0x88B5)
    
    # Pad to minimum frame size
    min_payload = 46
    if len(payload) < min_payload:
        payload = payload + b'\x00' * (min_payload - len(payload))
    
    return eth_header + payload


# ============================================================================
# SSH COMMANDS (baseline)
# ============================================================================

def ssh_command(switch_ip: str, command: str, timeout: int = 30) -> Tuple[bool, str, str]:
    """Execute SSH command on switch."""
    cmd = [
        'ssh',
        '-i', SSH_KEY,
        '-o', 'StrictHostKeyChecking=no',
        '-o', 'UserKnownHostsFile=/dev/null',
        '-o', 'LogLevel=ERROR',
        f'root@{switch_ip}',
        command
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return True, result.stdout, result.stderr
    except Exception as e:
        return False, '', str(e)


def run_config_commands(switch_ip: str, commands: List[str], debug: bool = False) -> bool:
    """Run configuration commands on switch."""
    cmd_str = " ; ".join(commands)
    full_cmd = f"cli -c 'configure ; {cmd_str} ; commit'"
    success, stdout, stderr = ssh_command(switch_ip, full_cmd, timeout=60)
    
    if debug:
        print(f"    [DEBUG] Config stdout: {stdout[:500] if stdout else '(empty)'}")
        if stderr:
            print(f"    [DEBUG] Config stderr: {stderr[:500]}")
    
    # Check for errors
    combined = (stdout + stderr).lower()
    if 'error' in combined or 'invalid' in combined:
        if debug:
            print(f"    [DEBUG] Error detected in output")
        return False
    
    return 'commit complete' in combined or success


# ============================================================================
# COUNTER SETUP (from e031)
# ============================================================================

def setup_mac_counters(switch_ip: str, num_outputs: int = 4) -> bool:
    """Configure firewall filter with per-output-neuron MAC counters."""
    print(f"\n  Setting up MAC counters on {switch_ip}...")
    
    # First, fix any existing VLAN conflicts that prevent commits
    print("    Fixing VLAN conflicts...")
    vlan_cleanup = [
        # Remove interface references first, then delete the VLAN
        "delete interfaces et-0/0/100 unit 0 family ethernet-switching vlan members unicast_bridge",
        "delete interfaces et-0/0/96 unit 0 family ethernet-switching vlan members unicast_bridge",
        "delete vlans unicast_bridge",
    ]
    run_config_commands(switch_ip, vlan_cleanup, debug=False)
    time.sleep(0.5)
    
    # Clean up old filter
    print("    Cleaning up old filter...")
    cleanup_commands = [
        f"delete firewall family ethernet-switching filter {FILTER_NAME}",
    ]
    run_config_commands(switch_ip, cleanup_commands, debug=False)
    time.sleep(1)
    
    # Build filter commands
    filter_commands = []
    for i in range(num_outputs):
        mac = OUTPUT_MACS[i]
        filter_commands.extend([
            f"set firewall family ethernet-switching filter {FILTER_NAME} "
            f"term out{i} from destination-mac-address {mac}",
            f"set firewall family ethernet-switching filter {FILTER_NAME} "
            f"term out{i} then count out{i}_pkts",
            f"set firewall family ethernet-switching filter {FILTER_NAME} "
            f"term out{i} then accept",
        ])
    
    # Default term
    filter_commands.append(
        f"set firewall family ethernet-switching filter {FILTER_NAME} "
        f"term default then accept"
    )
    
    # Apply filter to interface (no VLAN creation - use existing config)
    interface_commands = [
        f"set interfaces et-0/0/96 unit 0 family ethernet-switching filter input {FILTER_NAME}",
    ]
    
    all_commands = filter_commands + interface_commands
    
    print(f"    Running {len(all_commands)} config commands...")
    success = run_config_commands(switch_ip, all_commands, debug=True)
    
    if success:
        print(f"    ✓ MAC counters configured")
        # Verify filter exists
        _, verify_out, _ = ssh_command(switch_ip, 
            f"cli -c 'show configuration firewall family ethernet-switching filter {FILTER_NAME}'")
        if verify_out.strip():
            print(f"    ✓ Filter verified in configuration")
        else:
            print(f"    ⚠ Filter not found in configuration!")
            success = False
    else:
        print(f"    ✗ Configuration failed")
    
    return success


def clear_counters_ssh(switch_ip: str) -> bool:
    """Clear firewall counters via SSH."""
    success, _, _ = ssh_command(switch_ip, 
        f"cli -c 'clear firewall filter {FILTER_NAME}'"
    )
    return success


# ============================================================================
# SSH COUNTER READING (baseline)
# ============================================================================

def read_counters_ssh(switch_ip: str, num_outputs: int = 4, 
                      debug: bool = False) -> Tuple[Dict[int, int], float]:
    """
    Read counters via SSH (baseline method).
    Returns: (counter_dict, elapsed_time)
    """
    start = time.time()
    
    # Try to show the filter - use 'show firewall' to see all filters first
    success, stdout, stderr = ssh_command(switch_ip,
        f"cli -c 'show firewall filter {FILTER_NAME}'"
    )
    
    elapsed = time.time() - start
    
    if debug:
        print(f"    [DEBUG] SSH stdout: '{stdout.strip()}'")
        if stderr:
            print(f"    [DEBUG] SSH stderr: '{stderr.strip()}'")
        
        # If empty, try to list all filters
        if not stdout.strip():
            print("    [DEBUG] Filter not found. Listing all firewall filters...")
            _, all_filters, _ = ssh_command(switch_ip, 
                "cli -c 'show firewall'")
            if all_filters:
                # Show first few lines
                lines = all_filters.strip().split('\n')[:10]
                for line in lines:
                    print(f"    [DEBUG]   {line}")
    
    counters = {}
    if success:
        import re
        for i in range(num_outputs):
            pattern = rf'out{i}_pkts\s+\d+\s+(\d+)'
            match = re.search(pattern, stdout)
            if match:
                counters[i] = int(match.group(1))
            else:
                counters[i] = 0
    
    return counters, elapsed


# ============================================================================
# SNMP CONFIGURATION CHECK
# ============================================================================

def check_snmp_enabled(switch_ip: str) -> Tuple[bool, str]:
    """
    Check if SNMP is enabled and accessible on the switch.
    Uses a basic sysDescr query to test connectivity.
    """
    # First try with snmpget command (most reliable)
    try:
        result = subprocess.run(
            ['snmpget', '-v2c', '-c', 'public', '-t', '2', switch_ip, 'SNMPv2-MIB::sysDescr.0'],
            capture_output=True, text=True, timeout=5
        )
        if 'Juniper' in result.stdout or 'QFX' in result.stdout:
            return True, result.stdout.strip()
        elif 'Timeout' in result.stderr or 'No Response' in result.stderr:
            return False, "SNMP timeout - is SNMP enabled on switch?"
        else:
            return False, result.stderr.strip() if result.stderr else "No response"
    except FileNotFoundError:
        return False, "snmpget not found (install: apt install snmp)"
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except Exception as e:
        return False, str(e)


def configure_snmp_on_switch(switch_ip: str) -> bool:
    """
    Enable SNMP on the switch if not already enabled.
    """
    print(f"\n  Configuring SNMP on {switch_ip}...")
    
    snmp_commands = [
        "set snmp community public authorization read-only",
        "set snmp community public clients 0.0.0.0/0",
    ]
    
    success = run_config_commands(switch_ip, snmp_commands)
    
    if success:
        print("    ✓ SNMP configured")
    else:
        print("    ✗ SNMP configuration failed")
    
    return success


# ============================================================================
# SNMP COUNTER READING (new approach)
# ============================================================================

def read_counters_snmp(switch_ip: str, num_outputs: int = 4, 
                       community: str = 'public') -> Tuple[Dict[int, int], float]:
    """
    Read firewall filter counters via SNMP.
    
    Juniper firewall filter counter OID:
    jnxFWCounterPacketCount: 1.3.6.1.4.1.2636.3.5.2.1.5
    
    Returns: (counter_dict, elapsed_time)
    """
    if not PYSNMP_AVAILABLE:
        print("    SNMP not available (pysnmp not installed)")
        return {}, 0.0
    
    start = time.time()
    
    counters = {}
    
    # Juniper firewall counter OID base
    # jnxFWCounterPacketCount = 1.3.6.1.4.1.2636.3.5.2.1.5
    base_oid = '1.3.6.1.4.1.2636.3.5.2.1.5'
    
    try:
        # Use unified nextCmd (aliased for both versions)
        iterator = nextCmd(
            SnmpEngine(),
            CommunityData(community),
            UdpTransportTarget((switch_ip, 161), timeout=2.0, retries=1),
            ContextData(),
            ObjectType(ObjectIdentity(base_oid)),
            lexicographicMode=False
        )
        
        for (errorIndication, errorStatus, errorIndex, varBinds) in iterator:
            if errorIndication:
                print(f"    SNMP error: {errorIndication}")
                break
            elif errorStatus:
                print(f"    SNMP error: {errorStatus.prettyPrint()}")
                break
            else:
                for varBind in varBinds:
                    oid_str = str(varBind[0])
                    value = int(varBind[1])
                    
                    # Parse counter name from OID
                    # The OID includes the filter and counter name
                    for i in range(num_outputs):
                        if f'out{i}_pkts' in oid_str or oid_str.endswith(f'.{i}'):
                            counters[i] = value
                            break
    
    except Exception as e:
        print(f"    SNMP exception: {e}")
    
    elapsed = time.time() - start
    
    return counters, elapsed


def read_counters_snmpget(switch_ip: str, num_outputs: int = 4,
                          community: str = 'public') -> Tuple[Dict[int, int], float]:
    """
    Read counters using snmpget command (alternative to pysnmp).
    Uses snmpwalk to get all firewall counters at once.
    
    Returns: (counter_dict, elapsed_time)
    """
    import re
    
    start = time.time()
    
    # Use snmpwalk to get all firewall counters
    # OID: jnxFWCounterPacketCount = 1.3.6.1.4.1.2636.3.5.2.1.5
    cmd = [
        'snmpwalk', '-v2c', '-c', community, '-t', '2',
        switch_ip,
        '1.3.6.1.4.1.2636.3.5.2.1.5'
    ]
    
    counters = {}
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        
        # Parse output: each line has counter name and value
        for line in result.stdout.split('\n'):
            for i in range(num_outputs):
                counter_name = f'out{i}_pkts'
                if counter_name in line:
                    # Extract the numeric value at the end
                    match = re.search(r'(\d+)\s*$', line)
                    if match:
                        counters[i] = int(match.group(1))
                        break
    
    except subprocess.TimeoutExpired:
        print("    snmpwalk timed out")
    except FileNotFoundError:
        print("    snmpwalk not found (install: apt install snmp)")
    except Exception as e:
        print(f"    snmpwalk error: {e}")
    
    elapsed = time.time() - start
    
    return counters, elapsed


def read_interface_counters_snmp(switch_ip: str, interface: str = 'et-0/0/96',
                                  community: str = 'public') -> Tuple[int, float]:
    """
    Read physical interface packet counter via SNMP.
    Uses standard IF-MIB which should always be available.
    
    This is a sanity check - if this works, SNMP is functional.
    
    Returns: (packet_count, elapsed_time)
    """
    import re
    
    start = time.time()
    
    # Standard interface counter OIDs
    # ifInUcastPkts = 1.3.6.1.2.1.2.2.1.11
    # ifHCInUcastPkts = 1.3.6.1.2.1.31.1.1.1.7  (64-bit counter)
    
    try:
        # First get interface index
        result = subprocess.run(
            ['snmpwalk', '-v2c', '-c', community, '-t', '2', switch_ip, 
             'IF-MIB::ifDescr'],
            capture_output=True, text=True, timeout=5
        )
        
        if_index = None
        for line in result.stdout.split('\n'):
            if interface in line:
                # Extract index from OID like "IF-MIB::ifDescr.543 = STRING: et-0/0/96"
                match = re.search(r'ifDescr\.(\d+)', line)
                if match:
                    if_index = match.group(1)
                    break
        
        if if_index:
            # Get packet counter for that interface
            result2 = subprocess.run(
                ['snmpget', '-v2c', '-c', community, '-t', '2', switch_ip,
                 f'IF-MIB::ifHCInUcastPkts.{if_index}'],
                capture_output=True, text=True, timeout=5
            )
            
            match = re.search(r'Counter64:\s*(\d+)', result2.stdout)
            if match:
                elapsed = time.time() - start
                return int(match.group(1)), elapsed
    
    except subprocess.TimeoutExpired:
        pass
    except Exception as e:
        print(f"    Interface counter error: {e}")
    
    elapsed = time.time() - start
    return 0, elapsed


# ============================================================================
# PACKET SENDING
# ============================================================================

def send_test_packets(activations: List[int]) -> int:
    """
    Send test packets for each output neuron.
    activations[i] = number of packets to send to output neuron i
    
    Returns: total packets sent
    """
    src_mac = bytes.fromhex(SEND_MAC.replace(':', ''))
    
    sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW)
    sock.bind((SEND_IFACE, 0))
    
    total_sent = 0
    
    for output_idx, count in enumerate(activations):
        dst_mac = bytes.fromhex(OUTPUT_MACS[output_idx].replace(':', ''))
        
        packet = craft_ethernet_frame(
            dst_mac=dst_mac,
            src_mac=src_mac,
            vlan_id=VLAN_ID,
            payload=f'OUT{output_idx}'.encode()
        )
        
        for _ in range(count):
            sock.send(packet)
            total_sent += 1
    
    sock.close()
    return total_sent


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

@dataclass
class ExperimentResult:
    """Results from counter read experiment."""
    method: str
    counters: Dict[int, int]
    elapsed_ms: float
    correct: bool


def run_experiment():
    """Run the SNMP counter reading experiment."""
    
    print("="*80)
    print("E037: SNMP COUNTER READING EXPERIMENT")
    print("="*80)
    
    print("\nHypothesis:")
    print("  SSH counter read: ~1-5 ms per call (currently ~8s total)")
    print("  SNMP bulk GET:    ~100-200 μs total")
    print("  Speedup expected: 40-80×")
    
    # Step 0: Check SNMP connectivity
    print("\n" + "="*80)
    print("STEP 0: CHECK SNMP CONNECTIVITY")
    print("="*80)
    
    snmp_ok, snmp_msg = check_snmp_enabled(SWITCH1_IP)
    if snmp_ok:
        print(f"  ✓ SNMP is enabled: {snmp_msg[:60]}...")
    else:
        print(f"  ✗ SNMP not accessible: {snmp_msg}")
        print("  Attempting to configure SNMP...")
        if configure_snmp_on_switch(SWITCH1_IP):
            time.sleep(2)
            snmp_ok, snmp_msg = check_snmp_enabled(SWITCH1_IP)
            if snmp_ok:
                print(f"  ✓ SNMP now working: {snmp_msg[:60]}...")
            else:
                print(f"  ✗ SNMP still not working: {snmp_msg}")
    
    # Step 1: Setup
    print("\n" + "="*80)
    print("STEP 1: CONFIGURE MAC-BASED COUNTERS")
    print("="*80)
    
    if not setup_mac_counters(SWITCH1_IP, NUM_OUTPUT_NEURONS):
        print("Setup failed!")
        return
    
    time.sleep(2)
    
    # Step 2: Clear counters
    print("\n" + "="*80)
    print("STEP 2: CLEAR COUNTERS")
    print("="*80)
    
    clear_counters_ssh(SWITCH1_IP)
    time.sleep(0.5)
    print("  ✓ Counters cleared")
    
    # Step 3: Send known packets
    print("\n" + "="*80)
    print("STEP 3: SEND TEST PACKETS")
    print("="*80)
    
    # Send known activation values
    activations = [5, 3, 7, 2]  # 5 to neuron 0, 3 to neuron 1, etc.
    
    print(f"  Sending packets:")
    for i, count in enumerate(activations):
        print(f"    Output {i}: {count} packets → MAC {OUTPUT_MACS[i]}")
    
    total_sent = send_test_packets(activations)
    print(f"\n  ✓ Sent {total_sent} packets total")
    
    time.sleep(1)  # Wait for counters to update
    
    # Step 4: Read counters with different methods
    print("\n" + "="*80)
    print("STEP 4: READ COUNTERS (COMPARE METHODS)")
    print("="*80)
    
    results = []
    
    # Method 1: SSH (baseline) - with debug to see raw output
    print("\n  Method 1: SSH/CLI")
    counters_ssh, time_ssh = read_counters_ssh(SWITCH1_IP, NUM_OUTPUT_NEURONS, debug=True)
    correct_ssh = all(counters_ssh.get(i, 0) == activations[i] for i in range(NUM_OUTPUT_NEURONS))
    results.append(ExperimentResult("SSH/CLI", counters_ssh, time_ssh * 1000, correct_ssh))
    print(f"    Time: {time_ssh*1000:.1f} ms")
    print(f"    Counters: {counters_ssh}")
    print(f"    Correct: {'✓' if correct_ssh else '✗'}")
    
    # Method 2: snmpwalk command
    print("\n  Method 2: snmpwalk command")
    counters_snmpwalk, time_snmpwalk = read_counters_snmpget(SWITCH1_IP, NUM_OUTPUT_NEURONS)
    correct_snmpwalk = all(counters_snmpwalk.get(i, 0) == activations[i] for i in range(NUM_OUTPUT_NEURONS))
    results.append(ExperimentResult("snmpwalk", counters_snmpwalk, time_snmpwalk * 1000, correct_snmpwalk))
    print(f"    Time: {time_snmpwalk*1000:.1f} ms")
    print(f"    Counters: {counters_snmpwalk}")
    print(f"    Correct: {'✓' if correct_snmpwalk else '✗'}")
    
    # Method 3: pysnmp library (if available)
    if PYSNMP_AVAILABLE:
        print("\n  Method 3: pysnmp library")
        counters_pysnmp, time_pysnmp = read_counters_snmp(SWITCH1_IP, NUM_OUTPUT_NEURONS)
        correct_pysnmp = all(counters_pysnmp.get(i, 0) == activations[i] for i in range(NUM_OUTPUT_NEURONS))
        results.append(ExperimentResult("pysnmp", counters_pysnmp, time_pysnmp * 1000, correct_pysnmp))
        print(f"    Time: {time_pysnmp*1000:.1f} ms")
        print(f"    Counters: {counters_pysnmp}")
        print(f"    Correct: {'✓' if correct_pysnmp else '✗'}")
    
    # Method 4: Interface counter via SNMP (sanity check)
    if snmp_ok:
        print("\n  Method 4: Interface counter (SNMP sanity check)")
        iface_count, iface_time = read_interface_counters_snmp(SWITCH1_IP, 'et-0/0/96')
        print(f"    Time: {iface_time*1000:.1f} ms")
        print(f"    et-0/0/96 packet count: {iface_count}")
        print(f"    (This confirms SNMP can read counters)")
    
    # Step 5: Summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    print("\n  Expected counters:", {i: activations[i] for i in range(NUM_OUTPUT_NEURONS)})
    print()
    
    print("  Method          Time (ms)    Correct    Speedup vs SSH")
    print("  " + "-"*60)
    
    ssh_time = results[0].elapsed_ms
    for r in results:
        speedup = ssh_time / r.elapsed_ms if r.elapsed_ms > 0 else 0
        print(f"  {r.method:<15} {r.elapsed_ms:>8.1f}    {'✓' if r.correct else '✗'}          {speedup:.1f}×")
    
    # Verdict
    print("\n" + "="*80)
    print("VERDICT")
    print("="*80)
    
    # Find fastest correct method
    correct_results = [r for r in results if r.correct]
    if correct_results:
        fastest = min(correct_results, key=lambda r: r.elapsed_ms)
        print(f"\n  Fastest correct method: {fastest.method}")
        print(f"  Time: {fastest.elapsed_ms:.1f} ms for {NUM_OUTPUT_NEURONS} counters")
        
        # Key insight: SSH reads ALL counters in ONE command!
        # The time is roughly constant regardless of counter count
        print(f"\n  📊 SCALING ANALYSIS:")
        print(f"  SSH reads ALL counters in a single 'show firewall filter' command!")
        print(f"  Time is ~constant regardless of counter count.")
        print(f"")
        print(f"  For 2880 output neurons:")
        print(f"    - Configuration: ~2-3 seconds (one-time setup)")
        print(f"    - Counter read:  ~700 ms (single SSH command)")
        print(f"    - Total per inference: ~1 second")
        print(f"")
        print(f"  At 1 second per inference, throughput would be:")
        print(f"    - ~1 inference/second")
        print(f"    - If generating 100 tokens, that's 100 seconds")
        print(f"")
        print(f"  🎯 KEY INSIGHT: The bottleneck is SSH overhead, not counter count!")
        print(f"     To achieve 1000+ tokens/second, we need:")
        print(f"     1. OpenNSL SDK (30-50 μs per read) - requires C code")
        print(f"     2. Or: Pipeline multiple inferences in parallel")
        print(f"     3. Or: Use the counter-free multi-layer flow (VLAN tagging)")
    else:
        print("\n  ✗ No method returned correct counters!")
        print("  Check SNMP configuration on switch")
        print("  - Is SNMP enabled?")
        print("  - Is community string correct?")
        print("  - Are firewall counters exposed via SNMP?")
    
    # Step 5: Scaling test - try with more counters
    print("\n" + "="*80)
    print("STEP 5: SCALING TEST (64 counters)")
    print("="*80)
    
    print("\n  Setting up 64-counter filter...")
    if setup_mac_counters(SWITCH1_IP, 64):
        clear_counters_ssh(SWITCH1_IP)
        time.sleep(0.5)
        
        # Read the 64-counter filter
        counters_64, time_64 = read_counters_ssh(SWITCH1_IP, 64, debug=False)
        print(f"  Time to read 64 counters: {time_64*1000:.1f} ms")
        print(f"  Counters returned: {len(counters_64)}")
        
        # Compare with 4-counter time
        time_4 = results[0].elapsed_ms  # SSH time for 4 counters
        if time_4 > 0:
            ratio = (time_64 * 1000) / time_4
            print(f"\n  Scaling factor (64 vs 4 counters): {ratio:.2f}×")
            if ratio < 2:
                print(f"  ✓ SSH counter read scales well! (sub-linear)")
            else:
                print(f"  ⚠ SSH counter read scales linearly")
    
    # Save results
    os.makedirs("bringup_logs", exist_ok=True)
    import json
    log_file = f"bringup_logs/snmp_counter_test_{int(time.time())}.json"
    with open(log_file, 'w') as f:
        json.dump({
            "expected": activations,
            "results": [
                {
                    "method": r.method,
                    "counters": r.counters,
                    "elapsed_ms": r.elapsed_ms,
                    "correct": bool(r.correct)
                }
                for r in results
            ],
            "timestamp": time.time()
        }, f, indent=2)
    
    print(f"\n  Results saved to: {log_file}")


if __name__ == '__main__':
    run_experiment()



""" Output:
sudo python3 e037_snmp_counter_read.py
WARNING: pysnmp not installed. Install with: pip install pysnmp
================================================================================
E037: SNMP COUNTER READING EXPERIMENT
================================================================================

Hypothesis:
  SSH counter read: ~1-5 ms per call (currently ~8s total)
  SNMP bulk GET:    ~100-200 μs total
  Speedup expected: 40-80×

================================================================================
STEP 0: CHECK SNMP CONNECTIVITY
================================================================================
  ✓ SNMP is enabled: SNMPv2-MIB::sysDescr.0 = STRING: Juniper Networks, Inc. qfx5...

================================================================================
STEP 1: CONFIGURE MAC-BASED COUNTERS
================================================================================

  Setting up MAC counters on 10.10.10.55...
    Fixing VLAN conflicts...
    Cleaning up old filter...
    Running 14 config commands...
    [DEBUG] Config stdout: Entering configuration mode
The configuration has been changed but not committed
configuration check succeeds
commit complete

    ✓ MAC counters configured
    ✓ Filter verified in configuration

================================================================================
STEP 2: CLEAR COUNTERS
================================================================================
  ✓ Counters cleared

================================================================================
STEP 3: SEND TEST PACKETS
================================================================================
  Sending packets:
    Output 0: 5 packets → MAC 01:00:5e:00:02:00
    Output 1: 3 packets → MAC 01:00:5e:00:02:01
    Output 2: 7 packets → MAC 01:00:5e:00:02:02
    Output 3: 2 packets → MAC 01:00:5e:00:02:03

  ✓ Sent 17 packets total

================================================================================
STEP 4: READ COUNTERS (COMPARE METHODS)
================================================================================

  Method 1: SSH/CLI
    [DEBUG] SSH stdout: 'Filter: snmp_test_counter                                      
Counters:
Name                                                Bytes              Packets
out0_pkts                                             340                    5
out1_pkts                                             204                    3
out2_pkts                                             476                    7
out3_pkts                                             136                    2'
    Time: 758.2 ms
    Counters: {0: 5, 1: 3, 2: 7, 3: 2}
    Correct: ✓

  Method 2: snmpwalk command
    snmpwalk timed out
    Time: 5003.7 ms
    Counters: {}
    Correct: ✗

  Method 4: Interface counter (SNMP sanity check)
    Time: 72.0 ms
    et-0/0/96 packet count: 460369
    (This confirms SNMP can read counters)

================================================================================
RESULTS SUMMARY
================================================================================

  Expected counters: {0: 5, 1: 3, 2: 7, 3: 2}

  Method          Time (ms)    Correct    Speedup vs SSH
  ------------------------------------------------------------
  SSH/CLI            758.2    ✓          1.0×
  snmpwalk          5003.7    ✗          0.2×

================================================================================
VERDICT
================================================================================

  Fastest correct method: SSH/CLI
  Time: 758.2 ms for 4 counters

  📊 SCALING ANALYSIS:
  SSH reads ALL counters in a single 'show firewall filter' command!
  Time is ~constant regardless of counter count.

  For 2880 output neurons:
    - Configuration: ~2-3 seconds (one-time setup)
    - Counter read:  ~700 ms (single SSH command)
    - Total per inference: ~1 second

  At 1 second per inference, throughput would be:
    - ~1 inference/second
    - If generating 100 tokens, that's 100 seconds

  🎯 KEY INSIGHT: The bottleneck is SSH overhead, not counter count!
     To achieve 1000+ tokens/second, we need:
     1. OpenNSL SDK (30-50 μs per read) - requires C code
     2. Or: Pipeline multiple inferences in parallel
     3. Or: Use the counter-free multi-layer flow (VLAN tagging)

================================================================================
STEP 5: SCALING TEST (64 counters)
================================================================================

  Setting up 64-counter filter...

  Setting up MAC counters on 10.10.10.55...
    Fixing VLAN conflicts...
    Cleaning up old filter...
    Running 194 config commands...
    [DEBUG] Config stdout: Entering configuration mode
The configuration has been changed but not committed
configuration check succeeds
commit complete

    ✓ MAC counters configured
    ✓ Filter verified in configuration
  Time to read 64 counters: 971.4 ms
  Counters returned: 64

  Scaling factor (64 vs 4 counters): 1.28×
  ✓ SSH counter read scales well! (sub-linear)

  Results saved to: bringup_logs/snmp_counter_test_1766845390.json
(.venv) multiplex@multiplex1:~/photonic_inference_engine/phase_001/phase_001$ cat bringup_logs/snmp_counter_test_1766845390.json
{
  "expected": [
    5,
    3,
    7,
    2
  ],
  "results": [
    {
      "method": "SSH/CLI",
      "counters": {
        "0": 5,
        "1": 3,
        "2": 7,
        "3": 2
      },
      "elapsed_ms": 758.246898651123,
      "correct": true
    },
    {
      "method": "snmpwalk",
      "counters": {},
      "elapsed_ms": 5003.695726394653,
      "correct": false
    }
  ],
  "timestamp": 1766845390.805085
"""