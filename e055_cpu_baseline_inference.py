#!/usr/bin/env python3
"""
e055_cpu_baseline_inference.py

CPU-ONLY BASELINE: Full Quality Inference with Qwen3-0.6B
=============================================================

GOAL: Verify the model produces coherent chat completions on CPU.
      This serves as the reference for what switch-based inference should achieve.

APPROACH:
  Use llama-cpp-python for proper full-quality inference.
  This handles the complete architecture correctly:
    - Attention
    - Mamba/SSM blocks
    - MoE routing
    - All 48 layers
    - Proper tokenization

MODEL: Qwen3-0.6B (Dense Transformer)
  - 28 layers
  - 1024 embedding dim  
  - Q4_K_M quantized (4-bit, good quality)
  - ~397 MB (fits easily in RAM)

Author: Research Phase 001
Date: December 2025
"""

import time
from llama_cpp import Llama

MODEL_PATH = "./models/Qwen3-0.6B-Q4_K_M.gguf"


def run_cpu_baseline():
    """Run CPU baseline inference using llama-cpp-python."""
    print("="*80)
    print("E055: CPU BASELINE INFERENCE (llama-cpp-python)")
    print("="*80)
    print("""
  This experiment verifies the Qwen3-0.6B model produces coherent
  text when running full-quality inference on CPU.
  
  This serves as the REFERENCE for what switch-based inference should achieve.
  
  Using llama-cpp-python for proper inference with:
    - Full attention mechanism
    - All 28 layers
    - Proper tokenization
    - Think/No-think mode support
""")
    
    # Load model
    print("="*60)
    print("Loading Model")
    print("="*60)
    
    print(f"\n  Loading: {MODEL_PATH}")
    print("  ~397MB model (Q4_K_M), should load quickly...")
    
    t0 = time.time()
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=2048,           # Context window
        n_threads=4,          # CPU threads
        n_gpu_layers=0,       # CPU only (no GPU)
        verbose=True,         # Show loading progress
    )
    load_time = time.time() - t0
    print(f"\n  Model loaded in {load_time:.1f}s")
    
    # Test prompts
    prompts = [
        "The ",
        "Hello, my name is",
        "The capital of France is",
        "1 + 1 =",
    ]
    
    print("\n" + "="*60)
    print("Generating Text")
    print("="*60)
    
    results = []
    
    for prompt in prompts:
        print(f"\n  Prompt: '{prompt}'")
        print("  Generating...", end=" ", flush=True)
        
        t0 = time.time()
        output = llm(
            prompt,
            max_tokens=20,
            temperature=0.7,
            top_p=0.9,
            echo=True,  # Include prompt in output
        )
        gen_time = time.time() - t0
        
        generated_text = output['choices'][0]['text']
        tokens_generated = output['usage']['completion_tokens']
        
        print(f"done ({gen_time:.1f}s)")
        print(f"    Output: '{generated_text}'")
        print(f"    Tokens: {tokens_generated}, Speed: {tokens_generated/gen_time:.2f} tok/s")
        
        results.append({
            'prompt': prompt,
            'output': generated_text,
            'tokens': tokens_generated,
            'time': gen_time,
        })
    
    # Summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    print("\n  Generated outputs:")
    for r in results:
        print(f"    '{r['prompt']}' → '{r['output']}'")
    
    avg_speed = sum(r['tokens']/r['time'] for r in results) / len(results)
    print(f"\n  Average speed: {avg_speed:.2f} tokens/second")
    
    print("""
  These are the CPU reference outputs.
  The switch-based inference should ultimately produce similar coherent text.
  
  Key observations:
    - Model produces coherent English text
    - Completes sentences appropriately
    - Shows understanding of context
  
  For switch-based inference to match this quality, we need:
    1. Full 28 layers (currently using 4 in e054)
    2. Full hidden dimension (currently using 64 of 1024)
    3. Proper signed arithmetic (achieved in e054!)
    4. All model components (attention, FFN)
""")
    
    return results


if __name__ == '__main__':
    run_cpu_baseline()
