#!/usr/bin/env python3
"""
e002_theoretical_speed_calculator.py

Theoretical performance modeling for photonic inference engine.

This script calculates expected speeds based on:
- Hardware specifications (not Python interpreter overhead)
- Memory bandwidth limits
- DMA transfer rates
- NIC capabilities
- Physical limits (speed of light, PCIe bandwidth, etc.)

Author: Andrew Campi
Date: December 19, 2025
"""

from dataclasses import dataclass
from typing import Dict


# =============================================================================
# HARDWARE SPECIFICATIONS
# =============================================================================

@dataclass
class SystemSpecs:
    """Hardware specifications for the inference engine."""
    
    # CPU & Memory
    cpu_freq_ghz: float = 3.4              # i5-7500 base clock
    cpu_cores: int = 4                      # Physical cores
    ram_bandwidth_gbps: float = 37.5        # DDR4-2400 dual-channel
    l3_cache_mb: float = 6.0                # SmartCache
    
    # NIC (Mellanox ConnectX-3)
    nic_bandwidth_gbps: float = 40          # Per port
    nic_max_pps: float = 59.5e6             # Million packets/sec
    nic_dma_latency_us: float = 0.5         # DMA setup
    
    # PCIe
    pcie_gen: int = 3                       # PCIe 3.0
    pcie_lanes: int = 8                     # x8 electrical
    pcie_bandwidth_gbps: float = 64         # 8 GT/s × 8 lanes
    
    # Switch
    switch_latency_ns: float = 550          # Cut-through per hop
    switch_bandwidth_gbps: float = 1920     # Total fabric
    
    # Model
    num_neurons: int = 2880
    num_layers: int = 32
    avg_activation: float = 1.0             # Binary/sparse activations (0 or 1)
    packet_size_bytes: int = 64


# =============================================================================
# THEORETICAL SPEED CALCULATIONS
# =============================================================================

def calculate_memcpy_speed(specs: SystemSpecs) -> Dict:
    """
    Calculate packet buffer preparation speed using memcpy.
    
    Modern CPUs can memcpy at ~80-90% of RAM bandwidth with:
    - AVX2 SIMD instructions
    - Non-temporal stores (bypass cache)
    - Aligned memory access
    """
    # Theoretical max: RAM bandwidth
    max_bandwidth_gbps = specs.ram_bandwidth_gbps
    
    # Realistic achievable: 85% of theoretical (modern memcpy)
    achievable_bandwidth_gbps = max_bandwidth_gbps * 0.85
    achievable_bandwidth_bps = achievable_bandwidth_gbps * 1e9 / 8  # bytes/sec
    
    # Avg packets per token
    avg_packets = specs.num_neurons * specs.avg_activation
    total_bytes = avg_packets * specs.packet_size_bytes
    
    # Time to prepare packets (memcpy to NIC ring buffer)
    memcpy_time_us = (total_bytes / achievable_bandwidth_bps) * 1e6
    
    return {
        'max_bandwidth_gbps': max_bandwidth_gbps,
        'achievable_bandwidth_gbps': achievable_bandwidth_gbps,
        'total_bytes': total_bytes,
        'avg_packets': avg_packets,
        'memcpy_time_us': memcpy_time_us
    }


def calculate_dma_transfer_speed(specs: SystemSpecs) -> Dict:
    """
    Calculate DMA transfer time from host RAM to NIC.
    
    PCIe DMA uses scatter-gather with zero-copy:
    - Packets stay in host memory
    - NIC reads via PCIe DMA
    - Achieves near line-rate
    """
    # Average packets
    avg_packets = specs.num_neurons * specs.avg_activation
    total_bytes = avg_packets * specs.packet_size_bytes
    
    # PCIe bandwidth (bidirectional, but TX is what matters)
    pcie_bandwidth_bps = specs.pcie_bandwidth_gbps * 1e9 / 8
    
    # DMA setup overhead (per-batch, not per-packet)
    dma_setup_us = specs.nic_dma_latency_us
    
    # DMA transfer time
    dma_transfer_us = (total_bytes / pcie_bandwidth_bps) * 1e6
    
    # Total
    total_dma_us = dma_setup_us + dma_transfer_us
    
    return {
        'pcie_bandwidth_gbps': specs.pcie_bandwidth_gbps,
        'total_bytes': total_bytes,
        'dma_setup_us': dma_setup_us,
        'dma_transfer_us': dma_transfer_us,
        'total_dma_us': total_dma_us
    }


def calculate_nic_tx_speed(specs: SystemSpecs) -> Dict:
    """
    Calculate NIC transmission time (wire speed).
    
    40 Gbps Ethernet with:
    - 64-byte minimum frames
    - 12-byte inter-frame gap (IPG)
    - 8-byte preamble
    """
    # Average packets
    avg_packets = specs.num_neurons * specs.avg_activation
    
    # Wire protocol overhead
    frame_size = specs.packet_size_bytes
    ipg_bytes = 12
    preamble_bytes = 8
    total_bytes_per_packet = frame_size + ipg_bytes + preamble_bytes
    
    # Total wire bytes
    total_wire_bytes = avg_packets * total_bytes_per_packet
    
    # Wire time at 40 Gbps
    wire_bandwidth_bps = specs.nic_bandwidth_gbps * 1e9 / 8
    wire_time_us = (total_wire_bytes / wire_bandwidth_bps) * 1e6
    
    # Packet rate limit check
    max_pps = specs.nic_max_pps
    time_at_max_pps_us = (avg_packets / max_pps) * 1e6
    
    # Take the slower of the two (usually wire speed for 64B packets)
    actual_tx_time_us = max(wire_time_us, time_at_max_pps_us)
    
    return {
        'avg_packets': avg_packets,
        'wire_time_us': wire_time_us,
        'time_at_max_pps_us': time_at_max_pps_us,
        'actual_tx_time_us': actual_tx_time_us,
        'bottleneck': 'wire_speed' if wire_time_us > time_at_max_pps_us else 'pps_limit'
    }


def calculate_dpdk_packet_generation(specs: SystemSpecs) -> Dict:
    """
    Calculate DPDK packet generation with pre-allocated buffers.
    
    DPDK achieves near-zero overhead via:
    - Pre-allocated mbuf pool (zero malloc)
    - Bulk operations (batch processing)
    - Cache-line alignment
    - Prefetching
    """
    avg_packets = specs.num_neurons * specs.avg_activation
    
    # DPDK operations per packet (in CPU cycles)
    cycles_per_packet = {
        'mbuf_lookup': 10,          # Get pre-allocated mbuf from pool
        'header_copy': 30,          # Copy 64-byte header template
        'vlan_write': 5,            # Write VLAN ID (2 bytes)
        'metadata_update': 10,      # Update mbuf metadata
        'total': 55                 # Total cycles per packet
    }
    
    # CPU cycles available
    cpu_freq_hz = specs.cpu_freq_ghz * 1e9
    cycles_per_us = cpu_freq_hz / 1e6
    
    # Time to prepare all packets
    total_cycles = avg_packets * cycles_per_packet['total']
    generation_time_us = total_cycles / cycles_per_us
    
    # Add batch overhead (negligible)
    batch_overhead_us = 1.0
    
    total_time_us = generation_time_us + batch_overhead_us
    
    return {
        'avg_packets': avg_packets,
        'cycles_per_packet': cycles_per_packet['total'],
        'cpu_freq_ghz': specs.cpu_freq_ghz,
        'generation_time_us': generation_time_us,
        'batch_overhead_us': batch_overhead_us,
        'total_time_us': total_time_us
    }


def calculate_switch_processing(specs: SystemSpecs) -> Dict:
    """
    Calculate switch fabric processing time.
    
    Per-layer operations:
    - TCAM lookup: ~100 ns (hardware parallel)
    - Crossbar forwarding: 550 ns (datasheet)
    - VLAN rewrite: <10 ns (hardware register)
    """
    # Per-layer timing
    tcam_lookup_ns = 100
    crossbar_forward_ns = specs.switch_latency_ns
    vlan_rewrite_ns = 5
    
    per_layer_ns = tcam_lookup_ns + crossbar_forward_ns + vlan_rewrite_ns
    per_layer_us = per_layer_ns / 1000
    
    # Total for all layers
    total_us = per_layer_us * specs.num_layers
    
    return {
        'tcam_lookup_ns': tcam_lookup_ns,
        'crossbar_forward_ns': crossbar_forward_ns,
        'vlan_rewrite_ns': vlan_rewrite_ns,
        'per_layer_us': per_layer_us,
        'num_layers': specs.num_layers,
        'total_us': total_us
    }


def calculate_counter_read(specs: SystemSpecs) -> Dict:
    """
    Calculate counter read time via OpenNSL.
    
    Hardware counter read via DMA:
    - Snapshot command: ~5 μs
    - DMA transfer: 96 counters × 8 bytes = 768 bytes
    - PCIe latency: ~5 μs
    - Parsing overhead: ~10 μs
    """
    num_counters = 96
    bytes_per_counter = 8
    total_bytes = num_counters * bytes_per_counter
    
    # Breakdown
    snapshot_command_us = 5
    pcie_bandwidth_bps = specs.pcie_bandwidth_gbps * 1e9 / 8
    dma_transfer_us = (total_bytes / pcie_bandwidth_bps) * 1e6
    pcie_latency_us = 5
    parsing_us = 10
    
    total_us = snapshot_command_us + dma_transfer_us + pcie_latency_us + parsing_us
    
    return {
        'num_counters': num_counters,
        'total_bytes': total_bytes,
        'snapshot_command_us': snapshot_command_us,
        'dma_transfer_us': dma_transfer_us,
        'pcie_latency_us': pcie_latency_us,
        'parsing_us': parsing_us,
        'total_us': total_us
    }


# =============================================================================
# END-TO-END LATENCY MODELING
# =============================================================================

def calculate_end_to_end_latency(specs: SystemSpecs) -> Dict:
    """
    Calculate complete end-to-end latency per token.
    """
    # Phase breakdown
    dpdk_gen = calculate_dpdk_packet_generation(specs)
    nic_tx = calculate_nic_tx_speed(specs)
    switch_proc = calculate_switch_processing(specs)
    counter = calculate_counter_read(specs)
    
    # RX is same as TX (symmetric)
    nic_rx_us = nic_tx['actual_tx_time_us']
    
    # Token decoding (CPU-side)
    decoding_us = 10.0  # Argmax + softmax
    
    # Total
    total_us = (
        dpdk_gen['total_time_us'] +
        nic_tx['actual_tx_time_us'] +
        switch_proc['total_us'] +
        counter['total_us'] +
        nic_rx_us +
        decoding_us
    )
    
    return {
        'packet_generation_us': dpdk_gen['total_time_us'],
        'nic_tx_us': nic_tx['actual_tx_time_us'],
        'switch_processing_us': switch_proc['total_us'],
        'counter_read_us': counter['total_us'],
        'nic_rx_us': nic_rx_us,
        'decoding_us': decoding_us,
        'total_latency_us': total_us,
        'total_latency_ms': total_us / 1000,
        'throughput_tokens_per_sec': 1_000_000 / total_us if total_us > 0 else 0
    }


# =============================================================================
# COMPARISON WITH DOCUMENT PREDICTIONS
# =============================================================================

def compare_with_document_predictions(theoretical: Dict) -> None:
    """
    Compare theoretical calculations with research document predictions.
    """
    print(f"\n{'='*70}")
    print(f"COMPARISON: Theoretical vs. Document Predictions")
    print(f"{'='*70}\n")
    
    # Document predictions (from research_phase_001.md)
    doc_predictions = {
        'packet_generation_us': 20.0,
        'nic_tx_us': 147.46,
        'switch_processing_us': 20.83,
        'counter_read_us': 50.0,
        'nic_rx_us': 147.46,
        'decoding_us': 10.0,
        'total_latency_us': 395.74,
        'throughput_tokens_per_sec': 2527
    }
    
    print(f"{'Component':<25} {'Document':<15} {'Theoretical':<15} {'Diff':<10}")
    print(f"{'-'*70}")
    
    for key, doc_val in doc_predictions.items():
        if key in theoretical:
            theo_val = theoretical[key]
            
            if key == 'throughput_tokens_per_sec':
                diff_pct = ((theo_val - doc_val) / doc_val) * 100
                print(f"{key:<25} {doc_val:<15.0f} {theo_val:<15.0f} {diff_pct:>+7.1f}%")
            else:
                diff_pct = ((theo_val - doc_val) / doc_val) * 100
                print(f"{key:<25} {doc_val:<15.2f} {theo_val:<15.2f} {diff_pct:>+7.1f}%")
    
    print(f"{'-'*70}")
    
    # Overall assessment
    theo_throughput = theoretical['throughput_tokens_per_sec']
    doc_throughput = doc_predictions['throughput_tokens_per_sec']
    
    if abs(theo_throughput - doc_throughput) / doc_throughput < 0.2:
        print(f"\n✓ Theoretical model agrees with document (within 20%)")
    else:
        print(f"\n⚠ Theoretical model differs from document (>20% variance)")
    
    print(f"\nTheoretical throughput: {theo_throughput:.0f} tok/s")
    print(f"Document prediction:    {doc_throughput:.0f} tok/s")
    print(f"Difference:             {theo_throughput - doc_throughput:+.0f} tok/s")


# =============================================================================
# SENSITIVITY ANALYSIS
# =============================================================================

def sensitivity_analysis(base_specs: SystemSpecs) -> None:
    """
    Analyze how performance changes with key parameters.
    """
    print(f"\n{'='*70}")
    print(f"SENSITIVITY ANALYSIS")
    print(f"{'='*70}\n")
    
    # Baseline
    base_result = calculate_end_to_end_latency(base_specs)
    base_throughput = base_result['throughput_tokens_per_sec']
    
    print(f"Baseline: {base_throughput:.0f} tok/s\n")
    
    # Test variations
    variations = [
        ('Counter read: 30 μs (optimistic)', 'counter_latency', 30),
        ('Counter read: 100 μs (pessimistic)', 'counter_latency', 100),
        ('Switch latency: 1000 ns (conservative)', 'switch_latency_ns', 1000),
        ('Avg activation: 3.0 (sparse)', 'avg_activation', 3.0),
        ('Avg activation: 7.0 (dense)', 'avg_activation', 7.0),
    ]
    
    print(f"{'Scenario':<40} {'Throughput':<15} {'Change':<10}")
    print(f"{'-'*70}")
    
    for desc, param, value in variations:
        test_specs = SystemSpecs(
            cpu_freq_ghz=base_specs.cpu_freq_ghz,
            cpu_cores=base_specs.cpu_cores,
            ram_bandwidth_gbps=base_specs.ram_bandwidth_gbps,
            l3_cache_mb=base_specs.l3_cache_mb,
            nic_bandwidth_gbps=base_specs.nic_bandwidth_gbps,
            nic_max_pps=base_specs.nic_max_pps,
            nic_dma_latency_us=base_specs.nic_dma_latency_us,
            pcie_gen=base_specs.pcie_gen,
            pcie_lanes=base_specs.pcie_lanes,
            pcie_bandwidth_gbps=base_specs.pcie_bandwidth_gbps,
            switch_latency_ns=base_specs.switch_latency_ns,
            switch_bandwidth_gbps=base_specs.switch_bandwidth_gbps,
            num_neurons=base_specs.num_neurons,
            num_layers=base_specs.num_layers,
            avg_activation=base_specs.avg_activation,
            packet_size_bytes=base_specs.packet_size_bytes
        )
        
        # Apply variation
        if param == 'counter_latency':
            # Modify counter read calculation
            result = calculate_end_to_end_latency(test_specs)
            # Adjust for counter read time
            delta = value - 20  # Difference from baseline ~20μs
            adjusted_latency = result['total_latency_us'] + delta
            throughput = 1_000_000 / adjusted_latency
        elif param == 'switch_latency_ns':
            test_specs.switch_latency_ns = value
            result = calculate_end_to_end_latency(test_specs)
            throughput = result['throughput_tokens_per_sec']
        elif param == 'avg_activation':
            test_specs.avg_activation = value
            result = calculate_end_to_end_latency(test_specs)
            throughput = result['throughput_tokens_per_sec']
        
        change_pct = ((throughput - base_throughput) / base_throughput) * 100
        print(f"{desc:<40} {throughput:<15.0f} {change_pct:>+7.1f}%")


# =============================================================================
# MAIN REPORT
# =============================================================================

def generate_report():
    """
    Generate complete theoretical performance report.
    """
    print(f"\n{'#'*70}")
    print(f"#  THEORETICAL PERFORMANCE ANALYSIS")
    print(f"#  Photonic Inference Engine - Research Phase 001")
    print(f"{'#'*70}\n")
    
    # Initialize specs
    specs = SystemSpecs()
    
    print(f"SYSTEM SPECIFICATIONS:")
    print(f"{'─'*70}")
    print(f"  CPU:              {specs.cpu_cores}× cores @ {specs.cpu_freq_ghz} GHz")
    print(f"  RAM Bandwidth:    {specs.ram_bandwidth_gbps} GB/s")
    print(f"  NIC Bandwidth:    {specs.nic_bandwidth_gbps} Gbps")
    print(f"  NIC Max PPS:      {specs.nic_max_pps/1e6:.1f} Mpps")
    print(f"  PCIe:             Gen{specs.pcie_gen} x{specs.pcie_lanes} ({specs.pcie_bandwidth_gbps} Gbps)")
    print(f"  Switch Latency:   {specs.switch_latency_ns} ns")
    print(f"  Neurons:          {specs.num_neurons}")
    print(f"  Layers:           {specs.num_layers}")
    print(f"  Avg Activation:   {specs.avg_activation}")
    
    # Calculate each component
    print(f"\n{'='*70}")
    print(f"COMPONENT ANALYSIS")
    print(f"{'='*70}\n")
    
    # DPDK packet generation
    dpdk = calculate_dpdk_packet_generation(specs)
    print(f"1. DPDK Packet Generation:")
    print(f"   Packets:           {dpdk['avg_packets']:.0f}")
    print(f"   Cycles/packet:     {dpdk['cycles_per_packet']}")
    print(f"   Generation time:   {dpdk['total_time_us']:.2f} μs ✓")
    
    # NIC transmission
    nic_tx = calculate_nic_tx_speed(specs)
    print(f"\n2. NIC Transmission (40G Wire):")
    print(f"   Packets:           {nic_tx['avg_packets']:.0f}")
    print(f"   Wire time:         {nic_tx['wire_time_us']:.2f} μs")
    print(f"   Bottleneck:        {nic_tx['bottleneck']}")
    print(f"   Actual TX time:    {nic_tx['actual_tx_time_us']:.2f} μs")
    
    # Switch processing
    switch = calculate_switch_processing(specs)
    print(f"\n3. Switch Fabric Processing:")
    print(f"   Per-layer:         {switch['per_layer_us']:.3f} μs")
    print(f"   Total (32 layers): {switch['total_us']:.2f} μs")
    print(f"   TCAM lookup:       {switch['tcam_lookup_ns']} ns")
    print(f"   Crossbar forward:  {switch['crossbar_forward_ns']} ns")
    
    # Counter read
    counter = calculate_counter_read(specs)
    print(f"\n4. Counter Read (OpenNSL):")
    print(f"   Counters:          {counter['num_counters']}")
    print(f"   DMA transfer:      {counter['dma_transfer_us']:.3f} μs")
    print(f"   Total time:        {counter['total_us']:.2f} μs")
    
    # End-to-end
    e2e = calculate_end_to_end_latency(specs)
    print(f"\n{'='*70}")
    print(f"END-TO-END LATENCY BREAKDOWN")
    print(f"{'='*70}\n")
    print(f"  Packet generation:    {e2e['packet_generation_us']:>8.2f} μs  ({e2e['packet_generation_us']/e2e['total_latency_us']*100:>5.1f}%)")
    print(f"  NIC TX:               {e2e['nic_tx_us']:>8.2f} μs  ({e2e['nic_tx_us']/e2e['total_latency_us']*100:>5.1f}%)")
    print(f"  Switch processing:    {e2e['switch_processing_us']:>8.2f} μs  ({e2e['switch_processing_us']/e2e['total_latency_us']*100:>5.1f}%)")
    print(f"  Counter read:         {e2e['counter_read_us']:>8.2f} μs  ({e2e['counter_read_us']/e2e['total_latency_us']*100:>5.1f}%)")
    print(f"  NIC RX:               {e2e['nic_rx_us']:>8.2f} μs  ({e2e['nic_rx_us']/e2e['total_latency_us']*100:>5.1f}%)")
    print(f"  Decoding:             {e2e['decoding_us']:>8.2f} μs  ({e2e['decoding_us']/e2e['total_latency_us']*100:>5.1f}%)")
    print(f"  {'─'*70}")
    print(f"  TOTAL:                {e2e['total_latency_us']:>8.2f} μs")
    print(f"\n  Latency:              {e2e['total_latency_ms']:.3f} ms")
    print(f"  Throughput:           {e2e['throughput_tokens_per_sec']:.0f} tokens/sec")
    
    # Compare with document
    compare_with_document_predictions(e2e)
    
    # Sensitivity analysis
    sensitivity_analysis(specs)
    
    # Final assessment
    print(f"\n{'='*70}")
    print(f"ASSESSMENT")
    print(f"{'='*70}\n")
    
    if e2e['throughput_tokens_per_sec'] >= 2000:
        print(f"✓ Theoretical model predicts >2,000 tok/s (conservative target)")
    if e2e['throughput_tokens_per_sec'] >= 2500:
        print(f"✓ Theoretical model predicts >2,500 tok/s (baseline target)")
    
    # GPU comparison
    gpu_throughput = 40  # tok/s (real-world from Perplexity)
    speedup = e2e['throughput_tokens_per_sec'] / gpu_throughput
    
    print(f"\nGPU Comparison (RTX 3090 @ {gpu_throughput} tok/s):")
    print(f"  Speedup:              {speedup:.1f}×")
    print(f"  Latency improvement:  {(25000 / e2e['total_latency_us']):.1f}×")
    
    print(f"\n{'='*70}\n")


# =============================================================================
# RUN ANALYSIS
# =============================================================================

if __name__ == "__main__":
    generate_report()


""" Output:
python3 e002_theoretical_speed_calculator.py

######################################################################
#  THEORETICAL PERFORMANCE ANALYSIS
#  Photonic Inference Engine - Research Phase 001
######################################################################

SYSTEM SPECIFICATIONS:
──────────────────────────────────────────────────────────────────────
  CPU:              4× cores @ 3.4 GHz
  RAM Bandwidth:    37.5 GB/s
  NIC Bandwidth:    40 Gbps
  NIC Max PPS:      59.5 Mpps
  PCIe:             Gen3 x8 (64 Gbps)
  Switch Latency:   550 ns
  Neurons:          2880
  Layers:           32
  Avg Activation:   1.0

======================================================================
COMPONENT ANALYSIS
======================================================================

1. DPDK Packet Generation:
   Packets:           2880
   Cycles/packet:     55
   Generation time:   47.59 μs ✓

2. NIC Transmission (40G Wire):
   Packets:           2880
   Wire time:         48.38 μs
   Bottleneck:        pps_limit
   Actual TX time:    48.40 μs

3. Switch Fabric Processing:
   Per-layer:         0.655 μs
   Total (32 layers): 20.96 μs
   TCAM lookup:       100 ns
   Crossbar forward:  550 ns

4. Counter Read (OpenNSL):
   Counters:          96
   DMA transfer:      0.096 μs
   Total time:        20.10 μs

======================================================================
END-TO-END LATENCY BREAKDOWN
======================================================================

  Packet generation:       47.59 μs  ( 24.3%)
  NIC TX:                  48.40 μs  ( 24.8%)
  Switch processing:       20.96 μs  ( 10.7%)
  Counter read:            20.10 μs  ( 10.3%)
  NIC RX:                  48.40 μs  ( 24.8%)
  Decoding:                10.00 μs  (  5.1%)
  ──────────────────────────────────────────────────────────────────────
  TOTAL:                  195.45 μs

  Latency:              0.195 ms
  Throughput:           5116 tokens/sec

======================================================================
COMPARISON: Theoretical vs. Document Predictions
======================================================================

Component                 Document        Theoretical     Diff      
----------------------------------------------------------------------
packet_generation_us      20.00           47.59            +137.9%
nic_tx_us                 147.46          48.40             -67.2%
switch_processing_us      20.83           20.96              +0.6%
counter_read_us           50.00           20.10             -59.8%
nic_rx_us                 147.46          48.40             -67.2%
decoding_us               10.00           10.00              +0.0%
total_latency_us          395.74          195.45            -50.6%
throughput_tokens_per_sec 2527            5116             +102.5%
----------------------------------------------------------------------

⚠ Theoretical model differs from document (>20% variance)

Theoretical throughput: 5116 tok/s
Document prediction:    2527 tok/s
Difference:             +2589 tok/s

======================================================================
SENSITIVITY ANALYSIS
======================================================================

Baseline: 5116 tok/s

Scenario                                 Throughput      Change    
----------------------------------------------------------------------
Counter read: 30 μs (optimistic)         4867               -4.9%
Counter read: 100 μs (pessimistic)       3630              -29.0%
Switch latency: 1000 ns (conservative)   4765               -6.9%
Avg activation: 3.0 (sparse)             2074              -59.5%
Avg activation: 7.0 (dense)              947               -81.5%

======================================================================
ASSESSMENT
======================================================================

✓ Theoretical model predicts >2,000 tok/s (conservative target)
✓ Theoretical model predicts >2,500 tok/s (baseline target)

GPU Comparison (RTX 3090 @ 40 tok/s):
  Speedup:              127.9×
  Latency improvement:  127.9×

======================================================================
"""