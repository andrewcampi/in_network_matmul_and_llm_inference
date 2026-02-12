#!/usr/bin/env python3
"""
e134_dual_counter_signed_encoding.py

PROOF: DUAL COUNTERS (PACKETS + BYTES) CAN REPLACE POS/NEG DUAL-TERM COUNTERS
============================================================================

Goal
----
Show that a SINGLE firewall term / counter per neuron is sufficient to recover
both positive and negative contributions, by encoding sign using *frame length*.

Hardware fact (Junos firewall counters)
--------------------------------------
For each `then count <name>` counter, Junos reports TWO accumulated values:
  - Bytes
  - Packets

Key idea
--------
Send two different frame sizes to the SAME neuron MAC:
  - Positive contributions: frames of effective size L_pos bytes
  - Negative contributions: frames of effective size L_neg bytes

For one neuron term, the switch reports:
  P = packets = p + n
  B = bytes   = p*L_pos + n*L_neg

Solve exactly (when L_pos != L_neg):
  p = (B - P*L_neg) / (L_pos - L_neg)
  n = P - p

This can eliminate the 2× TCAM term multiplier from "dual counters" (pos/neg),
doubling neuron capacity per filter under the ~1152-term limit.

Important nuance
----------------
"Bytes" counted by Junos may include L2/VLAN headers and may differ from the
payload length you think you're sending. So we *calibrate* effective L_pos/L_neg
on hardware by sending only one length at a time and measuring bytes/packet.

Usage
-----
  sudo python3 e134_dual_counter_signed_encoding.py

Notes
-----
- Uses SSH to configure + read `show firewall filter ...` counters.
- Uses raw AF_PACKET socket send (same infra as earlier experiments).
- Uses a single VLAN + input filter attachment (minimal hardware resource use).

"""

from __future__ import annotations

import os
import random
import re
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from e038_counter_free_layers import (  # proven infra
    SWITCH1_IP,
    SEND_IFACE,
    craft_vlan_packet,
    get_mac_address,
    run_config_commands,
    send_packets,
    ssh_command,
)
from e045_real_weights_inference import mac_str_to_bytes
from e053_mac_encoded_layers import get_layer_neuron_mac


# =============================================================================
# CONFIG
# =============================================================================

TEST_VLAN = 451
VLAN_NAME = f"vlan{TEST_VLAN}"
FILTER_NAME = "e134_dual_counter_filter"
INPUT_IFACE = "et-0/0/96"

LAYER_ID = 0
NUM_NEURONS = 16  # keep small for fast proof; scale later

# Choose payload sizes well above the minimum padding threshold so frames differ.
# craft_vlan_packet pads payload to at least 42 bytes (since header is 18 bytes).
POS_PAYLOAD_BYTES = 200
NEG_PAYLOAD_BYTES = 80

# Calibration and test sizes
CALIB_PACKETS = 20
RANDOM_SEED = 134
MAX_P = 20
MAX_N = 20


# =============================================================================
# PARSING + MATH
# =============================================================================

@dataclass(frozen=True)
class CounterValue:
    bytes: int
    packets: int


def _parse_filter_counters(show_output: str) -> Dict[str, CounterValue]:
    """
    Parse Junos `show firewall filter <name>` output.

    We rely on the known table header used throughout this repo:
      Name                                                Bytes              Packets
      <counter_name>                                      <bytes>            <packets>
    """
    counters: Dict[str, CounterValue] = {}
    lines = show_output.splitlines()
    # Heuristic: lines containing counters typically have at least 3 columns
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("name") and "bytes" in line.lower() and "packets" in line.lower():
            continue
        # Split on whitespace, but counter names can be long; we keep first token as name.
        parts = re.split(r"\s+", line)
        if len(parts) < 3:
            continue
        name = parts[0]
        # Some Junos outputs include extra columns; we expect the last two are Bytes and Packets,
        # but in our repo the order is: Bytes then Packets.
        # We'll try to parse (bytes, packets) as the last two numeric fields on the line.
        nums = [p for p in parts[1:] if p.isdigit()]
        if len(nums) < 2:
            continue
        b = int(nums[-2])
        p = int(nums[-1])
        counters[name] = CounterValue(bytes=b, packets=p)
    return counters


def _read_counters_ssh(switch_ip: str) -> Dict[str, CounterValue]:
    ok, out, err = ssh_command(switch_ip, f"cli -c 'show firewall filter {FILTER_NAME}'", timeout=15)
    if not ok:
        raise RuntimeError(f"SSH read failed: {err[:300]}")
    parsed = _parse_filter_counters(out)
    return parsed


def _clear_counters(switch_ip: str) -> None:
    ssh_command(switch_ip, f"cli -c 'clear firewall filter {FILTER_NAME}'", timeout=10)


def _solve_pn(P: int, B: int, L_pos: int, L_neg: int) -> Tuple[int, int]:
    """
    Recover (p, n) from (P, B) given effective per-packet byte sizes.
    """
    if P == 0:
        return 0, 0
    if L_pos == L_neg:
        raise ValueError("L_pos must differ from L_neg")
    denom = (L_pos - L_neg)
    num = (B - P * L_neg)
    # Expect exact integer solution. Use integer check to avoid float drift.
    if num % denom != 0:
        raise ValueError(f"Non-integer solution: num={num}, denom={denom}, P={P}, B={B}, L+={L_pos}, L-={L_neg}")
    p = num // denom
    n = P - p
    return p, n


# =============================================================================
# SWITCH CONFIG
# =============================================================================

def _run_config_commands_verbose(
    switch_ip: str,
    commands: List[str],
    *,
    label: str,
    timeout: int = 90,
    debug: bool = True,
) -> Tuple[bool, str, str]:
    """
    Run a Junos config transaction and return (ok, stdout, stderr).

    We *do not* rely on the `success` boolean from `e038_counter_free_layers.ssh_command`,
    because that wrapper does not check return codes. Instead, we detect success via
    'commit complete' and detect failure via common error strings.
    """
    cmd_str = " ; ".join(commands)
    full_cmd = f"cli -c 'configure ; {cmd_str} ; commit'"
    _, stdout, stderr = ssh_command(switch_ip, full_cmd, timeout=timeout)
    combined = (stdout + "\n" + stderr).lower()

    ok = ("commit complete" in combined) and ("error" not in combined) and ("failed" not in combined)

    if debug:
        print(f"    [{label}] ok={ok}")
        if stdout:
            print(f"    [{label}] stdout (first 800 chars):\n{stdout[:800]}")
        if stderr:
            print(f"    [{label}] stderr (first 800 chars):\n{stderr[:800]}")

    return ok, stdout, stderr


def _discover_ethernet_switching_units(switch_ip: str) -> List[Tuple[str, str]]:
    """
    Return a list of (iface, unit) for any interface units that have `family ethernet-switching`
    configured. This is a *surgical* cleanup helper to avoid orphaned VLAN references.

    We intentionally only touch interface units that already have ethernet-switching,
    so we don't disturb management / inet interfaces.
    """
    ok, out, err = ssh_command(
        switch_ip,
        "cli -c 'show configuration interfaces | display set | match \"family ethernet-switching\"'",
        timeout=20,
    )
    if not ok:
        # Best-effort: if discovery fails, fall back to static cleanup in full_cleanup.
        return []

    units: List[Tuple[str, str]] = []
    for line in out.splitlines():
        line = line.strip()
        # Example:
        # set interfaces xe-0/0/0 unit 0 family ethernet-switching interface-mode trunk
        m = re.match(r"set interfaces (\S+) unit (\d+) family ethernet-switching\b", line)
        if not m:
            continue
        iface, unit = m.group(1), m.group(2)
        # Avoid duplicates
        units.append((iface, unit))

    # Dedup while preserving order
    seen = set()
    deduped: List[Tuple[str, str]] = []
    for iface, unit in units:
        key = (iface, unit)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((iface, unit))
    return deduped


def full_cleanup(switch_ip: str) -> None:
    """
    Minimal cleanup for this experiment.

    Note: hardware_limits.md says cleanup order matters; the simplest safe cleanup
    here is: delete VLANs (removes attachments), delete filter, delete iface family.
    """
    # Critical: avoid orphaned "vlan members <name>" references under interfaces.
    # If a prior experiment deleted VLAN definitions but left interface membership,
    # Junos will fail all commits with "vlan X configured under interface Y does not exist".
    #
    # So we delete *all existing ethernet-switching stanzas* first, then delete VLANs, then filters.
    cleanup_cmds: List[str] = []

    # 1) Delete ethernet-switching from any interface units that currently have it.
    units = _discover_ethernet_switching_units(switch_ip)
    for iface, unit in units:
        cleanup_cmds.append(f"delete interfaces {iface} unit {unit} family ethernet-switching")

    # 2) Remove VLAN filter attachments by deleting VLANs (after removing interface refs).
    cleanup_cmds.append("delete vlans")

    # 3) Remove our filter (and any others under ethernet-switching family).
    cleanup_cmds.append("delete firewall family ethernet-switching")

    # If discovery failed and nothing was found, keep a small static safety net for common ports.
    if not units:
        for iface in [
            "et-0/0/96", "et-0/0/97", "et-0/0/98", "et-0/0/99", "et-0/0/100", "et-0/0/103",
            "xe-0/0/0", "xe-0/0/1", "xe-0/0/2", "xe-0/0/3",
            "xe-0/0/40", "xe-0/0/41", "xe-0/0/42", "xe-0/0/43",
        ]:
            cleanup_cmds.insert(0, f"delete interfaces {iface} unit 0 family ethernet-switching")

    ok, _, _ = _run_config_commands_verbose(
        switch_ip,
        cleanup_cmds,
        label="cleanup",
        timeout=120,
        debug=False,  # keep cleanup quiet unless it fails
    )
    if not ok:
        # Re-run once with debug output so the operator sees the exact Junos complaint.
        _run_config_commands_verbose(switch_ip, cleanup_cmds, label="cleanup(debug)", timeout=120, debug=True)
    time.sleep(0.5)


def configure_filter(switch_ip: str, num_neurons: int) -> None:
    """
    Configure a single VLAN + a single filter with 1 term per neuron MAC.
    """
    # Break into stages so we can see exactly what Junos rejects.
    # This is critical because Junos can fail silently if we don't print commit output.

    # Stage A: VLAN + interface
    stage_a = [
        f"set vlans {VLAN_NAME} vlan-id {TEST_VLAN}",
        f"delete interfaces {INPUT_IFACE} unit 0 family ethernet-switching",
        f"set interfaces {INPUT_IFACE} unit 0 family ethernet-switching interface-mode trunk",
        f"set interfaces {INPUT_IFACE} unit 0 family ethernet-switching vlan members {VLAN_NAME}",
    ]
    ok, _, _ = _run_config_commands_verbose(switch_ip, stage_a, label="cfg:stage_a(vlan+iface)", timeout=60, debug=True)
    if not ok:
        raise RuntimeError("Stage A failed: VLAN/interface configuration rejected by Junos (see output above).")

    # Stage B: filter terms
    stage_b: List[str] = [f"delete firewall family ethernet-switching filter {FILTER_NAME}"]
    for neuron in range(num_neurons):
        mac = get_layer_neuron_mac(LAYER_ID, neuron)
        term = f"n{neuron}"
        stage_b.append(
            f"set firewall family ethernet-switching filter {FILTER_NAME} term {term} "
            f"from destination-mac-address {mac}/48"
        )
        stage_b.append(f"set firewall family ethernet-switching filter {FILTER_NAME} term {term} then count {term}")
        stage_b.append(f"set firewall family ethernet-switching filter {FILTER_NAME} term {term} then accept")
    stage_b.append(f"set firewall family ethernet-switching filter {FILTER_NAME} term default then accept")

    ok, _, _ = _run_config_commands_verbose(switch_ip, stage_b, label="cfg:stage_b(filter_terms)", timeout=90, debug=True)
    if not ok:
        raise RuntimeError("Stage B failed: filter/term configuration rejected by Junos (see output above).")

    # Stage C: bind filter to VLAN input
    stage_c = [f"set vlans {VLAN_NAME} forwarding-options filter input {FILTER_NAME}"]
    ok, _, _ = _run_config_commands_verbose(switch_ip, stage_c, label="cfg:stage_c(bind_vlan_input_filter)", timeout=60, debug=True)
    if not ok:
        raise RuntimeError("Stage C failed: VLAN filter attachment rejected by Junos (see output above).")

    time.sleep(0.5)


# =============================================================================
# PACKET TX
# =============================================================================

def _mk_payload(length: int, tag: bytes) -> bytes:
    """
    Make a payload with a fixed total length. If tag is shorter, pad with zeros.
    """
    if len(tag) > length:
        return tag[:length]
    return tag + b"\x00" * (length - len(tag))


def send_to_neuron(neuron: int, count: int, payload_len: int, label: str) -> int:
    dst_mac = mac_str_to_bytes(get_layer_neuron_mac(LAYER_ID, neuron))
    src_mac = mac_str_to_bytes(get_mac_address(SEND_IFACE))
    payload = _mk_payload(payload_len, f"E134:{label}:N{neuron}".encode())
    pkts = [
        craft_vlan_packet(dst_mac=dst_mac, src_mac=src_mac, vlan_id=TEST_VLAN, payload=payload)
        for _ in range(int(count))
    ]
    return send_packets(SEND_IFACE, pkts)


# =============================================================================
# EXPERIMENT
# =============================================================================

def calibrate_effective_lengths(switch_ip: str, calib_neuron: int = 0) -> Tuple[int, int]:
    """
    Measure effective byte count per packet for POS and NEG sizes.
    Returns (L_pos_effective, L_neg_effective).
    """
    print("\n[1/3] Calibrating effective bytes-per-packet as counted by Junos...")

    def measure(payload_len: int, label: str) -> int:
        _clear_counters(switch_ip)
        time.sleep(0.2)
        sent = send_to_neuron(calib_neuron, CALIB_PACKETS, payload_len, label=label)
        if sent != CALIB_PACKETS:
            raise RuntimeError(f"send mismatch: sent={sent}, expected={CALIB_PACKETS}")
        time.sleep(0.5)
        counters = _read_counters_ssh(switch_ip)
        cv = counters.get(f"n{calib_neuron}", CounterValue(bytes=0, packets=0))
        if cv.packets == 0:
            raise RuntimeError("Calibration saw 0 packets in counter (filter not matching / VLAN path issue).")
        if cv.bytes % cv.packets != 0:
            raise RuntimeError(f"Bytes not divisible by packets in calibration: bytes={cv.bytes}, packets={cv.packets}")
        return cv.bytes // cv.packets

    L_pos = measure(POS_PAYLOAD_BYTES, label="POS_CAL")
    L_neg = measure(NEG_PAYLOAD_BYTES, label="NEG_CAL")

    if L_pos == L_neg:
        raise RuntimeError(f"Calibration failed: L_pos == L_neg == {L_pos}. Pick more-separated payload sizes.")

    print(f"  ✓ Effective sizes: L_pos={L_pos} bytes/packet, L_neg={L_neg} bytes/packet")
    return L_pos, L_neg


def run_trial(switch_ip: str, L_pos: int, L_neg: int) -> None:
    """
    Randomized multi-neuron test: send mixed POS/NEG lengths and recover (p,n).
    """
    print("\n[2/3] Running randomized signed-count recovery test across neurons...")
    random.seed(RANDOM_SEED)

    # Generate expected (p,n) per neuron
    expected: Dict[int, Tuple[int, int]] = {}
    for n in range(NUM_NEURONS):
        p = random.randint(0, MAX_P)
        neg = random.randint(0, MAX_N)
        expected[n] = (p, neg)

    _clear_counters(switch_ip)
    time.sleep(0.2)

    total_sent = 0
    for neuron, (p, neg) in expected.items():
        if p:
            total_sent += send_to_neuron(neuron, p, POS_PAYLOAD_BYTES, label="POS")
        if neg:
            total_sent += send_to_neuron(neuron, neg, NEG_PAYLOAD_BYTES, label="NEG")

    time.sleep(0.8)
    counters = _read_counters_ssh(switch_ip)

    # Verify decode
    failures: List[str] = []
    for neuron, (p_exp, n_exp) in expected.items():
        cv = counters.get(f"n{neuron}", CounterValue(bytes=0, packets=0))
        try:
            p_hat, n_hat = _solve_pn(cv.packets, cv.bytes, L_pos=L_pos, L_neg=L_neg)
        except Exception as e:
            failures.append(f"neuron {neuron}: decode error: {e} (P={cv.packets}, B={cv.bytes})")
            continue
        if (p_hat, n_hat) != (p_exp, n_exp):
            failures.append(
                f"neuron {neuron}: expected (p,n)=({p_exp},{n_exp}), got ({p_hat},{n_hat}) "
                f"(P={cv.packets}, B={cv.bytes})"
            )

    if failures:
        print("  ✗ FAIL: Some neurons did not decode correctly:")
        for msg in failures[:20]:
            print(f"    - {msg}")
        if len(failures) > 20:
            print(f"    ... and {len(failures) - 20} more")
        raise SystemExit(1)

    print(f"  ✓ PASS: All {NUM_NEURONS} neurons decoded correctly")
    print(f"     Total packets sent: {total_sent}")


def main() -> int:
    print("=" * 80)
    print("E134: DUAL COUNTER SIGNED ENCODING (PACKETS + BYTES)")
    print("=" * 80)
    print(f"Switch: {SWITCH1_IP}")
    print(f"VLAN:   {TEST_VLAN} ({VLAN_NAME})")
    print(f"Filter: {FILTER_NAME}")
    print(f"Neurons: {NUM_NEURONS} (1 term each)")
    print(f"Payload sizes: POS={POS_PAYLOAD_BYTES} bytes, NEG={NEG_PAYLOAD_BYTES} bytes")

    print("\n[0/3] Configuring switch...")
    full_cleanup(SWITCH1_IP)
    configure_filter(SWITCH1_IP, NUM_NEURONS)

    L_pos, L_neg = calibrate_effective_lengths(SWITCH1_IP, calib_neuron=0)
    run_trial(SWITCH1_IP, L_pos=L_pos, L_neg=L_neg)

    print("\n[3/3] Conclusion:")
    print("  ✓ A single Junos counter provides two independent observables (Bytes + Packets).")
    print("  ✓ With two frame sizes, we can recover (p,n) exactly → signed accumulation without 2 terms/neuron.")
    print("  ✓ This can double neuron capacity per filter under the TCAM term limit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


