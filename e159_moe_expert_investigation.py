"""
e159_moe_expert_investigation.py

UNDERSTANDING GPT-OSS-20B MoE ARCHITECTURE
==========================================

GPT-OSS-20B uses Mixture-of-Experts (MoE) with 32 experts per FFN layer.
Standard approach: Route each token to top-k experts (typically k=4-8).
Current e158 approach: Average all 32 experts (simplified but loses MoE benefit).

This experiment investigates:
1. What is the actual GGUF tensor structure for MoE experts?
2. Can we load expert weights individually?
3. What does proper MoE routing look like?
4. Can we implement MoE routing on the switch?

MoE Architecture:
- 32 experts per layer
- Each expert is a small FFN: up_proj, down_proj, gate_proj
- Router network selects top-k experts per token
- Final output = weighted sum of selected experts

Goals:
1. Understand the GGUF MoE tensor format
2. Load individual expert weights
3. Implement CPU-based MoE routing (baseline)
4. Design switch-based MoE routing (future)
"""

import numpy as np
import gguf
import os
from typing import List, Tuple, Optional

from e158_gpt_oss_weight_loader import GPTOSSWeightLoader


def inspect_moe_tensor_structure(model_path: str, layer_idx: int = 0):
    """
    Deep dive into the MoE tensor structure in GGUF.
    
    Args:
        model_path: Path to GGUF file
        layer_idx: Which layer to inspect
    """
    print("="*80)
    print(f"INSPECTING MoE TENSOR STRUCTURE - Layer {layer_idx}")
    print("="*80)
    
    reader = gguf.GGUFReader(model_path)
    
    # Find all MoE-related tensors for this layer
    moe_tensors = []
    for tensor in reader.tensors:
        if f"blk.{layer_idx}.ffn" in tensor.name and "exps" in tensor.name:
            moe_tensors.append(tensor)
    
    print(f"\nFound {len(moe_tensors)} MoE tensors for layer {layer_idx}:")
    
    for tensor in moe_tensors:
        print(f"\n  Tensor: {tensor.name}")
        print(f"    Shape: {tensor.shape}")
        print(f"    Type: {tensor.tensor_type}")
        print(f"    Size: {tensor.n_bytes / (1024**2):.1f} MB")
        
        # Try to understand the structure
        if len(tensor.shape) == 3:
            print(f"    → 3D tensor: [{tensor.shape[0]}, {tensor.shape[1]}, {tensor.shape[2]}]")
            print(f"    → Likely: [out_dim, in_dim, num_experts]")
        elif len(tensor.shape) == 2:
            print(f"    → 2D tensor: [{tensor.shape[0]}, {tensor.shape[1]}]")
        else:
            print(f"    → Unexpected {len(tensor.shape)}D tensor!")
    
    # Also look for router/gate tensors
    print("\n" + "="*80)
    print("Looking for router/gate tensors:")
    print("="*80)
    
    router_tensors = []
    for tensor in reader.tensors:
        if f"blk.{layer_idx}" in tensor.name and ("router" in tensor.name or "gate" in tensor.name.lower()):
            router_tensors.append(tensor)
    
    if router_tensors:
        for tensor in router_tensors:
            print(f"\n  Tensor: {tensor.name}")
            print(f"    Shape: {tensor.shape}")
            print(f"    Type: {tensor.tensor_type}")
    else:
        print("\n  No explicit router tensors found!")
        print("  (Router might be implicit or named differently)")


def test_individual_expert_loading(model_path: str, layer_idx: int = 0):
    """
    Try to load individual expert weights instead of averaging.
    
    Args:
        model_path: Path to GGUF file
        layer_idx: Which layer to load
    """
    print("\n" + "="*80)
    print(f"ATTEMPTING INDIVIDUAL EXPERT LOADING - Layer {layer_idx}")
    print("="*80)
    
    reader = gguf.GGUFReader(model_path)
    
    # Get the MoE expert tensor
    up_exps_tensor = None
    for tensor in reader.tensors:
        if tensor.name == f"blk.{layer_idx}.ffn_up_exps.weight":
            up_exps_tensor = tensor
            break
    
    if not up_exps_tensor:
        print("  ✗ ffn_up_exps.weight not found!")
        return
    
    print(f"\nTensor: {up_exps_tensor.name}")
    print(f"  Shape: {up_exps_tensor.shape}")
    print(f"  Type: {up_exps_tensor.tensor_type}")
    
    # Try to dequantize
    try:
        if hasattr(up_exps_tensor, 'data'):
            data = up_exps_tensor.data
        else:
            data = up_exps_tensor
        
        result = gguf.dequantize(data, up_exps_tensor.tensor_type)
        result = result.astype(np.float32)
        
        print(f"\n✓ Dequantization successful!")
        print(f"  Result shape: {result.shape}")
        print(f"  Result dtype: {result.dtype}")
        print(f"  Memory: {result.nbytes / (1024**2):.1f} MB")
        
        # If 3D, try to extract individual experts
        if result.ndim == 3:
            num_experts = result.shape[2]
            print(f"\n✓ 3D tensor confirmed with {num_experts} experts")
            
            # Show stats for first 3 experts
            for expert_idx in range(min(3, num_experts)):
                expert_weight = result[:, :, expert_idx]
                print(f"\n  Expert {expert_idx}:")
                print(f"    Shape: {expert_weight.shape}")
                print(f"    Mean: {expert_weight.mean():.6f}")
                print(f"    Std: {expert_weight.std():.6f}")
                print(f"    Min/Max: [{expert_weight.min():.6f}, {expert_weight.max():.6f}]")
        
        return result
        
    except Exception as e:
        print(f"\n✗ Dequantization failed!")
        print(f"  Error: {e}")
        print(f"  Error type: {type(e).__name__}")
        
        # Try to understand why
        print("\n  Debugging info:")
        print(f"    Tensor shape: {up_exps_tensor.shape}")
        print(f"    Data type: {type(data)}")
        if hasattr(data, 'shape'):
            print(f"    Data shape: {data.shape}")
        if hasattr(data, 'dtype'):
            print(f"    Data dtype: {data.dtype}")


def design_moe_routing_cpu(hidden_state: np.ndarray, 
                          expert_weights: List[np.ndarray],
                          router_logits: np.ndarray,
                          top_k: int = 4) -> np.ndarray:
    """
    Implement MoE routing on CPU (baseline for understanding).
    
    Args:
        hidden_state: Input [hidden_dim]
        expert_weights: List of expert weight matrices
        router_logits: Router output [num_experts]
        top_k: Number of experts to select
    
    Returns:
        Output [hidden_dim]
    """
    print("\n" + "="*80)
    print("CPU MoE ROUTING BASELINE")
    print("="*80)
    
    num_experts = len(expert_weights)
    
    # Step 1: Router selects top-k experts
    top_k_indices = np.argsort(router_logits)[-top_k:]
    top_k_weights = router_logits[top_k_indices]
    
    # Normalize weights (softmax over top-k)
    top_k_weights = np.exp(top_k_weights)
    top_k_weights = top_k_weights / top_k_weights.sum()
    
    print(f"\nRouter selection:")
    print(f"  Top-{top_k} experts: {top_k_indices}")
    print(f"  Weights: {top_k_weights}")
    
    # Step 2: Compute output from each selected expert
    outputs = []
    for idx, expert_idx in enumerate(top_k_indices):
        expert_out = expert_weights[expert_idx] @ hidden_state
        outputs.append(expert_out * top_k_weights[idx])
        print(f"  Expert {expert_idx}: weight={top_k_weights[idx]:.3f}, out_mean={expert_out.mean():.6f}")
    
    # Step 3: Weighted sum
    final_output = np.sum(outputs, axis=0)
    
    print(f"\nFinal output:")
    print(f"  Mean: {final_output.mean():.6f}")
    print(f"  Std: {final_output.std():.6f}")
    
    return final_output


def design_moe_routing_switch():
    """
    Design how MoE routing would work on a switch.
    
    This is conceptual - we need to figure out:
    1. How to implement router network on switch
    2. How to conditionally apply different expert weights
    3. How to do weighted sum of expert outputs
    """
    print("\n" + "="*80)
    print("SWITCH-BASED MoE ROUTING DESIGN (CONCEPTUAL)")
    print("="*80)
    
    print("""
MoE on Switch - Design Options:

OPTION 1: Full MoE (Complex)
  - Implement router network on switch (matmul + argmax)
  - Load all expert weights, conditionally apply based on routing
  - Challenge: Switch counters can't do conditional logic easily
  - Challenge: 32 experts × 6 projections × 24 layers = huge config
  
OPTION 2: Pre-computed Routing (Hybrid)
  - CPU: Run router network, decide which experts
  - Switch: Only send packets for selected top-k experts
  - Switch: Still need separate counters for each expert
  - Pro: Reduces switch complexity
  - Con: CPU overhead for routing
  
OPTION 3: Expert Averaging (Current - Simple)
  - Average all 32 experts into one weight matrix
  - Treat as standard FFN (no routing needed)
  - Pro: Works with existing switch pipeline
  - Con: Loses MoE sparsity benefit (always uses all experts)
  
OPTION 4: Top-K Averaging (Hybrid)
  - CPU: Route to top-k experts
  - CPU: Average only those k expert weights
  - Switch: Process averaged weight matrix
  - Pro: Some sparsity benefit, simple switch logic
  - Con: Still need to load all expert weights to CPU
  
RECOMMENDATION: Start with Option 3 (current), prove full pipeline works.
                Then try Option 2 or 4 for performance gains.
    """)


def run_investigation():
    """Main investigation function."""
    print("="*80)
    print("E159: MoE EXPERT INVESTIGATION")
    print("="*80)
    
    model_path = "models/gpt-oss-20b-F16.gguf"
    
    if not os.path.exists(model_path):
        print(f"\n✗ Model file not found: {model_path}")
        return
    
    # Investigation 1: Tensor structure
    inspect_moe_tensor_structure(model_path, layer_idx=0)
    
    # Investigation 2: Individual expert loading
    expert_weights = test_individual_expert_loading(model_path, layer_idx=0)
    
    # Investigation 3: CPU routing baseline
    if expert_weights is not None and expert_weights.ndim == 3:
        print("\n" + "="*80)
        print("TESTING CPU MoE ROUTING")
        print("="*80)
        
        # Extract individual experts
        num_experts = expert_weights.shape[2]
        experts_list = [expert_weights[:64, :64, i] for i in range(num_experts)]
        
        # Simulate router output (random for testing)
        router_logits = np.random.randn(num_experts)
        
        # Test input
        test_input = np.random.randn(64).astype(np.float32)
        
        # Run MoE routing
        output = design_moe_routing_cpu(test_input, experts_list, router_logits, top_k=4)
        
        print(f"\n✓ MoE routing test complete")
    
    # Investigation 4: Switch design
    design_moe_routing_switch()
    
    print("\n" + "="*80)
    print("INVESTIGATION COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("  1. Verify expert weight loading works")
    print("  2. Decide on MoE routing strategy (Option 3 recommended)")
    print("  3. Integrate e158 loader into full pipeline")
    print("  4. Test full 2880d × 24 layer inference")


if __name__ == "__main__":
    run_investigation()

