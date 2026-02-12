#!/usr/bin/env python3
"""
e003_tcam_config_generator.py

TCAM Configuration Generator for Photonic Inference Engine

Converts neural network weight matrices into Junos switch configurations:
- TCAM forwarding rules (ACLs)
- Multicast groups for packet fan-out
- VLAN-based layer progression
- Port-to-neuron mappings

Generates ready-to-deploy Junos CLI commands for QFX5100 switches.

Author: Andrew Campi
Date: December 19, 2025
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from pathlib import Path


# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Hardware limits (Juniper QFX5100)
MAX_MULTICAST_GROUPS_PER_VLAN = 2048    # Hardware limit
MAX_PORTS_PER_SWITCH = 96               # 96× 10G SFP+ ports
MAX_VLANS = 4096                        # 802.1Q standard
MAX_TCAM_RULES_PER_VLAN = 16000         # Approximate per VLAN

# Architecture parameters
TOTAL_NEURONS = 2880
NUM_LAYERS = 32
VLANS_PER_LAYER = 2  # Shard to stay under multicast limit


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class MulticastGroup:
    """Represents a multicast group for packet fan-out."""
    group_id: str
    source_neuron: int
    destination_ports: List[int]
    layer_id: int
    vlan_id: int
    weight_multiplier: int = 1  # For INT2: replicate destinations


@dataclass
class TCamRule:
    """Represents a TCAM forwarding rule."""
    rule_name: str
    layer_id: int
    vlan_id: int
    source_port: int
    multicast_group_id: str
    next_vlan_id: int
    
    def to_junos_cli(self, filter_name: str) -> List[str]:
        """Convert to Junos CLI commands."""
        commands = [
            f"set firewall family ethernet-switching filter {filter_name} term {self.rule_name} from vlan-id {self.vlan_id}",
            f"set firewall family ethernet-switching filter {filter_name} term {self.rule_name} from source-port xe-0/0/{self.source_port}",
            f"set firewall family ethernet-switching filter {filter_name} term {self.rule_name} then vlan {self.next_vlan_id}",
            f"set firewall family ethernet-switching filter {filter_name} term {self.rule_name} then forwarding-class {self.multicast_group_id}",
            f"set firewall family ethernet-switching filter {filter_name} term {self.rule_name} then accept"
        ]
        return commands


@dataclass
class ConfigurationStats:
    """Statistics about generated configuration."""
    total_rules: int
    total_multicast_groups: int
    rules_per_layer: Dict[int, int]
    groups_per_vlan: Dict[int, int]
    avg_fanout: float
    max_fanout: int
    sparsity: float


# =============================================================================
# NEURON-TO-PORT MAPPING
# =============================================================================

def neuron_to_port(neuron_id: int, neurons_per_port: int = 30) -> Tuple[int, int]:
    """
    Map neuron ID to (switch_id, port_id).
    
    With 2,880 neurons and 96 ports per switch (2 switches = 192 total):
    - Use temporal multiplexing (30 neurons per port)
    - Split across 2 switches
    
    Args:
        neuron_id: Global neuron ID (0-2879)
        neurons_per_port: Neurons sharing each port (default 30)
        
    Returns:
        (switch_id, port_id) tuple
    """
    # Simple mapping: first 1440 neurons on switch 0, rest on switch 1
    switch_id = 0 if neuron_id < 1440 else 1
    local_neuron = neuron_id % 1440
    
    # Map to port (90 active ports per switch)
    port_id = local_neuron % 90
    
    return switch_id, port_id


def calculate_vlan_id(layer_id: int, shard_id: int) -> int:
    """
    Calculate VLAN ID for a given layer and shard.
    
    VLAN ID scheme:
    - Layer 0, Shard 0: VLAN 0
    - Layer 0, Shard 1: VLAN 1
    - Layer 1, Shard 0: VLAN 2
    - Layer 1, Shard 1: VLAN 3
    - ...
    
    Args:
        layer_id: Layer number (0-31)
        shard_id: Shard within layer (0 or 1)
        
    Returns:
        VLAN ID (0-63 for 32 layers × 2 shards)
    """
    return layer_id * VLANS_PER_LAYER + shard_id


# =============================================================================
# WEIGHT MATRIX PROCESSING
# =============================================================================

def load_test_matrix(matrix_type: str = "identity", size: int = 10, sparsity: float = 0.95) -> np.ndarray:
    """
    Generate test weight matrices for validation.
    
    Args:
        matrix_type: Type of matrix ("identity", "random_sparse", "random_dense", "all_ones")
        size: Matrix dimension
        sparsity: Fraction of zeros (for random matrices)
        
    Returns:
        Binary weight matrix (size × size)
    """
    if matrix_type == "identity":
        return np.eye(size, dtype=np.int8)
    
    elif matrix_type == "all_ones":
        return np.ones((size, size), dtype=np.int8)
    
    elif matrix_type == "random_sparse":
        W = np.random.binomial(1, 1-sparsity, (size, size)).astype(np.int8)
        return W
    
    elif matrix_type == "random_dense":
        W = np.random.binomial(1, 0.5, (size, size)).astype(np.int8)
        return W
    
    else:
        raise ValueError(f"Unknown matrix type: {matrix_type}")


def analyze_weight_matrix(W: np.ndarray) -> Dict:
    """
    Analyze weight matrix statistics.
    
    Returns:
        Dictionary with sparsity, fanout stats, etc.
    """
    size = W.shape[0]
    nonzero_count = np.count_nonzero(W)
    total_count = size * size
    
    # Per-neuron fanout (outgoing connections)
    fanouts = [np.count_nonzero(W[:, i]) for i in range(size)]
    
    stats = {
        'size': size,
        'nonzero_weights': nonzero_count,
        'total_weights': total_count,
        'sparsity': 1.0 - (nonzero_count / total_count),
        'avg_fanout': np.mean(fanouts),
        'max_fanout': np.max(fanouts),
        'min_fanout': np.min(fanouts),
        'std_fanout': np.std(fanouts)
    }
    
    return stats


# =============================================================================
# MULTICAST GROUP GENERATION
# =============================================================================

def generate_multicast_groups(
    W: np.ndarray,
    layer_id: int,
    weight_type: str = "INT1"
) -> List[MulticastGroup]:
    """
    Generate multicast groups from weight matrix.
    
    Each source neuron with non-zero connections gets a multicast group.
    
    Args:
        W: Weight matrix (neurons × neurons)
        layer_id: Layer number
        weight_type: "INT1" (binary) or "INT2" (2-bit)
        
    Returns:
        List of MulticastGroup objects
    """
    groups = []
    size = W.shape[0]
    
    for src_neuron in range(size):
        # Find destination neurons where weight > 0
        dest_neurons = np.where(W[:, src_neuron] > 0)[0]
        
        if len(dest_neurons) == 0:
            continue  # No outgoing connections
        
        # Determine shard based on neuron ID
        shard_id = 0 if src_neuron < (size // 2) else 1
        vlan_id = calculate_vlan_id(layer_id, shard_id)
        
        # Get port mapping
        switch_id, port_id = neuron_to_port(src_neuron)
        
        # Create multicast group
        group_id = f"L{layer_id}_N{src_neuron}_MCAST"
        
        # For INT2, replicate destinations based on weight value
        if weight_type == "INT2":
            dest_ports = []
            for dst in dest_neurons:
                weight = W[dst, src_neuron]
                _, dst_port = neuron_to_port(dst)
                # Replicate destination 'weight' times
                dest_ports.extend([dst_port] * weight)
        else:
            # INT1: Simple list of destination ports
            dest_ports = [neuron_to_port(dst)[1] for dst in dest_neurons]
        
        group = MulticastGroup(
            group_id=group_id,
            source_neuron=src_neuron,
            destination_ports=dest_ports,
            layer_id=layer_id,
            vlan_id=vlan_id
        )
        
        groups.append(group)
    
    return groups


def multicast_group_to_junos_cli(group: MulticastGroup) -> List[str]:
    """
    Convert multicast group to Junos CLI commands.
    
    Returns:
        List of CLI commands
    """
    commands = []
    
    # Create multicast group with destination ports
    for port in set(group.destination_ports):  # Use set to get unique ports
        cmd = (f"set vlans VLAN_{group.vlan_id} forwarding-options multicast "
               f"group {group.group_id} interface xe-0/0/{port}")
        commands.append(cmd)
    
    return commands


# =============================================================================
# TCAM RULE GENERATION
# =============================================================================

def generate_tcam_rules(
    W: np.ndarray,
    layer_id: int,
    multicast_groups: List[MulticastGroup]
) -> List[TCamRule]:
    """
    Generate TCAM forwarding rules from weight matrix and multicast groups.
    
    Args:
        W: Weight matrix
        layer_id: Layer number
        multicast_groups: Pre-generated multicast groups
        
    Returns:
        List of TCamRule objects
    """
    rules = []
    size = W.shape[0]
    
    # Create a mapping of source_neuron -> multicast_group
    neuron_to_group = {g.source_neuron: g for g in multicast_groups}
    
    for src_neuron in range(size):
        if src_neuron not in neuron_to_group:
            continue  # No outgoing connections
        
        group = neuron_to_group[src_neuron]
        
        # Determine next VLAN ID
        current_shard = 0 if src_neuron < (size // 2) else 1
        current_vlan = calculate_vlan_id(layer_id, current_shard)
        next_vlan = calculate_vlan_id(layer_id + 1, 0)  # Next layer, shard 0
        
        # Get port mapping
        switch_id, port_id = neuron_to_port(src_neuron)
        
        # Create TCAM rule
        rule = TCamRule(
            rule_name=f"N{src_neuron}",
            layer_id=layer_id,
            vlan_id=current_vlan,
            source_port=port_id,
            multicast_group_id=group.group_id,
            next_vlan_id=next_vlan
        )
        
        rules.append(rule)
    
    return rules


# =============================================================================
# CONFIGURATION FILE GENERATION
# =============================================================================

def generate_layer_config(
    W: np.ndarray,
    layer_id: int,
    output_dir: Path,
    weight_type: str = "INT1"
) -> Tuple[List[str], ConfigurationStats]:
    """
    Generate complete configuration for one layer.
    
    Args:
        W: Weight matrix for this layer
        layer_id: Layer number
        output_dir: Directory to save config files
        weight_type: Weight quantization type
        
    Returns:
        (list_of_commands, statistics)
    """
    print(f"\nGenerating configuration for Layer {layer_id}...")
    
    # Generate multicast groups
    multicast_groups = generate_multicast_groups(W, layer_id, weight_type)
    print(f"  Multicast groups: {len(multicast_groups)}")
    
    # Generate TCAM rules
    tcam_rules = generate_tcam_rules(W, layer_id, multicast_groups)
    print(f"  TCAM rules: {len(tcam_rules)}")
    
    # Combine all commands
    all_commands = []
    
    # 1. VLAN configuration
    for shard in range(VLANS_PER_LAYER):
        vlan_id = calculate_vlan_id(layer_id, shard)
        all_commands.append(f"set vlans VLAN_{vlan_id} vlan-id {vlan_id}")
    
    # 2. Multicast groups
    for group in multicast_groups:
        all_commands.extend(multicast_group_to_junos_cli(group))
    
    # 3. Create filter for this layer
    filter_name = f"LAYER_{layer_id}_FILTER"
    all_commands.append(f"set firewall family ethernet-switching filter {filter_name}")
    
    # 4. TCAM rules
    for rule in tcam_rules:
        all_commands.extend(rule.to_junos_cli(filter_name))
    
    # 5. Apply filter to interfaces (simplified - apply to all ports)
    # In practice, you'd apply based on neuron-to-port mapping
    all_commands.append(f"# Apply filter to interfaces (configure per-port in production)")
    
    # Calculate statistics
    stats = ConfigurationStats(
        total_rules=len(tcam_rules),
        total_multicast_groups=len(multicast_groups),
        rules_per_layer={layer_id: len(tcam_rules)},
        groups_per_vlan={},
        avg_fanout=np.mean([len(g.destination_ports) for g in multicast_groups]) if multicast_groups else 0,
        max_fanout=max([len(g.destination_ports) for g in multicast_groups]) if multicast_groups else 0,
        sparsity=analyze_weight_matrix(W)['sparsity']
    )
    
    # Count groups per VLAN
    for group in multicast_groups:
        vlan = group.vlan_id
        stats.groups_per_vlan[vlan] = stats.groups_per_vlan.get(vlan, 0) + 1
    
    # Write to file
    output_file = output_dir / f"layer_{layer_id:02d}_config.txt"
    with open(output_file, 'w') as f:
        f.write(f"# Layer {layer_id} Configuration\n")
        f.write(f"# Generated by e003_tcam_config_generator.py\n")
        f.write(f"# Neurons: {W.shape[0]}\n")
        f.write(f"# Sparsity: {stats.sparsity:.2%}\n")
        f.write(f"# Rules: {len(tcam_rules)}\n")
        f.write(f"# Multicast groups: {len(multicast_groups)}\n\n")
        
        for cmd in all_commands:
            f.write(cmd + "\n")
    
    print(f"  Saved to: {output_file}")
    
    return all_commands, stats


# =============================================================================
# VALIDATION
# =============================================================================

def validate_configuration(stats: ConfigurationStats) -> Tuple[bool, List[str]]:
    """
    Validate generated configuration against hardware limits.
    
    Returns:
        (is_valid, list_of_warnings)
    """
    warnings = []
    is_valid = True
    
    # Check multicast groups per VLAN
    for vlan_id, group_count in stats.groups_per_vlan.items():
        if group_count > MAX_MULTICAST_GROUPS_PER_VLAN:
            warnings.append(
                f"VLAN {vlan_id}: {group_count} groups exceeds limit of {MAX_MULTICAST_GROUPS_PER_VLAN}"
            )
            is_valid = False
    
    # Check TCAM rules per layer
    if stats.total_rules > MAX_TCAM_RULES_PER_VLAN:
        warnings.append(
            f"Total rules {stats.total_rules} exceeds recommended limit of {MAX_TCAM_RULES_PER_VLAN}"
        )
    
    # Check average fanout (high fanout = potential bottleneck)
    if stats.avg_fanout > 200:
        warnings.append(
            f"High average fanout ({stats.avg_fanout:.0f}) may cause packet replication overhead"
        )
    
    return is_valid, warnings


# =============================================================================
# TEST SUITE
# =============================================================================

def test_small_network():
    """
    Test with small 10×10 network for validation.
    """
    print("\n" + "="*70)
    print("TEST: Small 10×10 Network")
    print("="*70)
    
    output_dir = Path("./configs_test_10x10")
    output_dir.mkdir(exist_ok=True)
    
    # Generate identity matrix (simple test)
    W = load_test_matrix("identity", size=10)
    
    print(f"\nWeight matrix: 10×10 identity")
    print(f"Sparsity: {analyze_weight_matrix(W)['sparsity']:.1%}")
    
    commands, stats = generate_layer_config(W, layer_id=0, output_dir=output_dir)
    
    print(f"\nGenerated {len(commands)} commands")
    print(f"TCAM rules: {stats.total_rules}")
    print(f"Multicast groups: {stats.total_multicast_groups}")
    
    # Validate
    is_valid, warnings = validate_configuration(stats)
    if is_valid:
        print("\n✓ Configuration is valid")
    else:
        print("\n✗ Configuration has warnings:")
        for w in warnings:
            print(f"  - {w}")


def test_sparse_network():
    """
    Test with 100×100 sparse network (95% sparsity).
    """
    print("\n" + "="*70)
    print("TEST: Sparse 100×100 Network (95% sparsity)")
    print("="*70)
    
    output_dir = Path("./configs_test_100x100_sparse")
    output_dir.mkdir(exist_ok=True)
    
    # Generate random sparse matrix
    W = load_test_matrix("random_sparse", size=100, sparsity=0.95)
    
    stats = analyze_weight_matrix(W)
    print(f"\nWeight matrix: 100×100 random sparse")
    print(f"Sparsity: {stats['sparsity']:.1%}")
    print(f"Avg fanout: {stats['avg_fanout']:.1f}")
    print(f"Max fanout: {stats['max_fanout']}")
    
    commands, config_stats = generate_layer_config(W, layer_id=0, output_dir=output_dir)
    
    print(f"\nGenerated {len(commands)} commands")
    print(f"TCAM rules: {config_stats.total_rules}")
    print(f"Multicast groups: {config_stats.total_multicast_groups}")
    
    # Validate
    is_valid, warnings = validate_configuration(config_stats)
    if is_valid:
        print("\n✓ Configuration is valid")
    else:
        print("\n✗ Configuration has warnings:")
        for w in warnings:
            print(f"  - {w}")


def test_full_scale_single_layer():
    """
    Test with full 2,880×2,880 network (single layer).
    """
    print("\n" + "="*70)
    print("TEST: Full-Scale 2,880×2,880 Network (1 layer)")
    print("="*70)
    
    output_dir = Path("./configs_test_2880x2880")
    output_dir.mkdir(exist_ok=True)
    
    print("\nGenerating 2,880×2,880 sparse matrix (95% sparsity)...")
    W = load_test_matrix("random_sparse", size=2880, sparsity=0.95)
    
    stats = analyze_weight_matrix(W)
    print(f"\nWeight matrix: 2,880×2,880 random sparse")
    print(f"Sparsity: {stats['sparsity']:.1%}")
    print(f"Avg fanout: {stats['avg_fanout']:.1f}")
    print(f"Max fanout: {stats['max_fanout']}")
    print(f"Total non-zero weights: {stats['nonzero_weights']:,}")
    
    commands, config_stats = generate_layer_config(W, layer_id=0, output_dir=output_dir)
    
    print(f"\nGenerated {len(commands)} commands")
    print(f"TCAM rules: {config_stats.total_rules}")
    print(f"Multicast groups: {config_stats.total_multicast_groups}")
    
    # Validate
    is_valid, warnings = validate_configuration(config_stats)
    if is_valid:
        print("\n✓ Configuration is valid")
    else:
        print("\n✗ Configuration has warnings:")
        for w in warnings:
            print(f"  - {w}")
    
    # Check groups per VLAN
    print(f"\nMulticast groups per VLAN:")
    for vlan_id, count in sorted(config_stats.groups_per_vlan.items()):
        status = "✓" if count <= MAX_MULTICAST_GROUPS_PER_VLAN else "✗"
        print(f"  VLAN {vlan_id}: {count:4d} groups {status}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    Run all tests and generate configurations.
    """
    print("\n" + "#"*70)
    print("#  TCAM CONFIGURATION GENERATOR")
    print("#  Photonic Inference Engine - Research Phase 001")
    print("#"*70)
    
    # Run tests in order of increasing complexity
    test_small_network()
    test_sparse_network()
    test_full_scale_single_layer()
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETE")
    print("="*70)
    print("\nConfiguration files generated in:")
    print("  ./configs_test_10x10/")
    print("  ./configs_test_100x100_sparse/")
    print("  ./configs_test_2880x2880/")
    print("\nTo deploy on switch:")
    print("  switch# configure")
    print("  switch# load merge terminal < layer_00_config.txt")
    print("  switch# commit")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()


""" Output:
python3 e003_tcam_config_generator.py 

######################################################################
#  TCAM CONFIGURATION GENERATOR
#  Photonic Inference Engine - Research Phase 001
######################################################################

======================================================================
TEST: Small 10×10 Network
======================================================================

Weight matrix: 10×10 identity
Sparsity: 90.0%

Generating configuration for Layer 0...
  Multicast groups: 10
  TCAM rules: 10
  Saved to: configs_test_10x10/layer_00_config.txt

Generated 64 commands
TCAM rules: 10
Multicast groups: 10

✓ Configuration is valid

======================================================================
TEST: Sparse 100×100 Network (95% sparsity)
======================================================================

Weight matrix: 100×100 random sparse
Sparsity: 95.2%
Avg fanout: 4.8
Max fanout: 10

Generating configuration for Layer 0...
  Multicast groups: 97
  TCAM rules: 97
  Saved to: configs_test_100x100_sparse/layer_00_config.txt

Generated 964 commands
TCAM rules: 97
Multicast groups: 97

✓ Configuration is valid

======================================================================
TEST: Full-Scale 2,880×2,880 Network (1 layer)
======================================================================

Generating 2,880×2,880 sparse matrix (95% sparsity)...

Weight matrix: 2,880×2,880 random sparse
Sparsity: 95.0%
Avg fanout: 144.3
Max fanout: 194
Total non-zero weights: 415,518

Generating configuration for Layer 0...
  Multicast groups: 2880
  TCAM rules: 2880
  Saved to: configs_test_2880x2880/layer_00_config.txt

Generated 223527 commands
TCAM rules: 2880
Multicast groups: 2880

✓ Configuration is valid

Multicast groups per VLAN:
  VLAN 0: 1440 groups ✓
  VLAN 1: 1440 groups ✓

======================================================================
ALL TESTS COMPLETE
======================================================================

Configuration files generated in:
  ./configs_test_10x10/
  ./configs_test_100x100_sparse/
  ./configs_test_2880x2880/

To deploy on switch:
  switch# configure
  switch# load merge terminal < layer_00_config.txt
  switch# commit

======================================================================
"""

""" Output:
ls -lh configs_test_2880x2880/
total 37928
-rw-r--r--@ 1 andrewcampi  staff    19M Dec 19 08:05 layer_00_config.txt
"""