#!/usr/bin/env python3
"""Test script to check memory usage during gradient computation."""

import torch
import gc
import subprocess

def get_gpu_memory():
    """Get current GPU memory usage."""
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        used, total = result.stdout.strip().split(', ')
        return int(used), int(total)
    return 0, 0

def test_sequence_gradient_memory():
    """Test memory usage with gradient computation on sequences."""
    
    print("🔍 Testing gradient computation memory usage...")
    
    # Simulate the same setup as importance sampling
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab_size = 151936  # Qwen tokenizer size
    seq_len = 400  # Typical sequence length
    batch_size = 1  # importance_microbatch_size
    
    print(f"Device: {device}")
    print(f"Simulating: batch_size={batch_size}, seq_len={seq_len}, vocab_size={vocab_size}")
    
    # Clear cache
    torch.cuda.empty_cache()
    gc.collect()
    
    used_before, total = get_gpu_memory()
    print(f"Memory before: {used_before}/{total} MB")
    
    # Create a dummy model similar to lm_head
    dummy_model = torch.nn.Linear(1536, vocab_size, bias=False).to(device).half()
    dummy_input = torch.randn(batch_size, seq_len, 1536, device=device, requires_grad=True).half()
    
    used_after_model, _ = get_gpu_memory()
    print(f"Memory after model: {used_after_model}/{total} MB (+{used_after_model-used_before} MB)")
    
    # Test forward pass with gradients
    print("Testing forward pass with gradients...")
    try:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = dummy_model(dummy_input)  # [batch_size, seq_len, vocab_size]
            
        used_after_forward, _ = get_gpu_memory()
        print(f"Memory after forward: {used_after_forward}/{total} MB (+{used_after_forward-used_after_model} MB)")
        
        # Test backward pass
        print("Testing backward pass...")
        dummy_loss = logits.sum()
        dummy_loss.backward()
        
        used_after_backward, _ = get_gpu_memory()
        print(f"Memory after backward: {used_after_backward}/{total} MB (+{used_after_backward-used_after_forward} MB)")
        
        print("✅ Gradient computation successful!")
        
    except torch.cuda.OutOfMemoryError as e:
        used_oom, _ = get_gpu_memory()
        print(f"❌ OOM at memory: {used_oom}/{total} MB")
        print(f"Error: {e}")
        
        # Try with gradient checkpointing
        print("\nTrying with torch.no_grad()...")
        torch.cuda.empty_cache()
        try:
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits_nograd = dummy_model(dummy_input)
            used_nograd, _ = get_gpu_memory()
            print(f"✅ torch.no_grad() successful: {used_nograd}/{total} MB")
        except torch.cuda.OutOfMemoryError:
            print("❌ Even torch.no_grad() failed")
    
    finally:
        # Cleanup
        del dummy_model, dummy_input
        if 'logits' in locals():
            del logits
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    test_sequence_gradient_memory()