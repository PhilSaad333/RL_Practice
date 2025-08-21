#!/usr/bin/env python3
"""
Test the specific entropy probe computation to see why gradients are zero.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

def test_entropy_probe_computation():
    print("Testing entropy probe specific computation...")
    
    # Load model (same as training)
    model_name = "Qwen/Qwen2.5-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Add LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    model = get_peft_model(model, lora_config)
    model.train()
    
    # Get trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Number of trainable parameters: {len(trainable_params)}")
    
    # Create a batch of sequences (simulating our training setup)
    texts = [
        "What is 2+2? The answer is 4.",
        "What is 3+3? The answer is 6.", 
        "What is 4+4? The answer is 8."
    ]
    
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # Forward pass
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    
    # Compute token-level log probabilities (same as training)
    logits = logits / 1.0  # temperature
    logp_all = F.log_softmax(logits.float(), dim=-1)
    
    # Simulate gathering log probs for generated tokens (like training does)
    # For simplicity, use the actual input tokens as "generated" tokens
    token_log_probs = logp_all.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)  # (B, seq_len)
    
    # Create generation mask (exclude padding)
    gen_mask = attention_mask.float()
    
    # Compute sequence log probabilities (same as training: sum over tokens)
    seq_log_probs = (token_log_probs * gen_mask).sum(dim=-1)  # (B,)
    
    print(f"Sequence log probs: {seq_log_probs}")
    print(f"Mean seq log prob: {seq_log_probs.mean()}")
    
    # Test different entropy loss formulations
    
    # Test 1: Our current formulation (with detached centering)
    print("\nTest 1: Current entropy probe formulation (detached)")
    mean_log_prob = seq_log_probs.mean()
    centered_log_probs = (seq_log_probs - mean_log_prob).detach()
    entropy_loss1 = torch.sum(centered_log_probs * seq_log_probs)
    
    print(f"  Centered log probs: {centered_log_probs}")
    print(f"  Entropy loss: {entropy_loss1}")
    
    try:
        grads1 = torch.autograd.grad(
            outputs=entropy_loss1,
            inputs=trainable_params,
            retain_graph=True,
            allow_unused=True
        )
        grad_count1 = sum(1 for g in grads1 if g is not None and torch.any(g != 0))
        print(f"  Non-zero gradients: {grad_count1}/{len(trainable_params)}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test 2: Simplified version (just sum of seq_log_probs)
    print("\nTest 2: Simple sum of sequence log probs")
    entropy_loss2 = seq_log_probs.sum()
    print(f"  Loss: {entropy_loss2}")
    
    try:
        grads2 = torch.autograd.grad(
            outputs=entropy_loss2,
            inputs=trainable_params,
            retain_graph=True,
            allow_unused=True
        )
        grad_count2 = sum(1 for g in grads2 if g is not None and torch.any(g != 0))
        print(f"  Non-zero gradients: {grad_count2}/{len(trainable_params)}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test 3: Mathematical decomposition to see what's happening
    print("\nTest 3: Mathematical analysis")
    print("  Our loss: sum((S_i - S_mean) * S_i)")
    print("  Expanded: sum(S_i^2 - S_mean * S_i)")
    print("  = sum(S_i^2) - S_mean * sum(S_i)")
    print("  = sum(S_i^2) - (sum(S_i)/N) * sum(S_i)")
    print("  = sum(S_i^2) - (sum(S_i)^2)/N")
    
    sum_sq = (seq_log_probs ** 2).sum()
    sum_s = seq_log_probs.sum()
    N = seq_log_probs.shape[0]
    
    analytical_loss = sum_sq - (sum_s ** 2) / N
    print(f"  Analytical result: {analytical_loss}")
    print(f"  Our computed result: {entropy_loss1}")
    print(f"  Difference: {abs(analytical_loss - entropy_loss1)}")
    
    # Test 4: The analytical form
    print("\nTest 4: Using analytical form directly")
    entropy_loss4 = sum_sq - (sum_s ** 2) / N
    
    try:
        grads4 = torch.autograd.grad(
            outputs=entropy_loss4,
            inputs=trainable_params,
            retain_graph=True,
            allow_unused=True
        )
        grad_count4 = sum(1 for g in grads4 if g is not None and torch.any(g != 0))
        print(f"  Non-zero gradients: {grad_count4}/{len(trainable_params)}")
    except Exception as e:
        print(f"  Error: {e}")

if __name__ == "__main__":
    test_entropy_probe_computation()