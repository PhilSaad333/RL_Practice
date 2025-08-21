#!/usr/bin/env python3
"""
Test script to verify gradient connection from new_logp to LoRA parameters.
This will help debug why torch.autograd.grad() returns all None values.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

def test_gradient_connection():
    print("Testing gradient connection from logp to LoRA parameters...")
    
    # Load a small model with LoRA (same setup as training)
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
    
    # Create a simple input
    text = "What is 2+2? The answer is"
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # Forward pass
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    
    # Compute log probabilities (same way as training)
    logits = logits / 1.0  # temperature
    logp_all = F.log_softmax(logits.float(), dim=-1)
    
    # Take the last token log prob (simulating generated token)
    next_token_logp = logp_all[0, -1, :]  # (vocab_size,)
    
    # Create a simple loss (sum of log probs for a few tokens)
    test_tokens = torch.tensor([1, 2, 3, 4, 5], device=logits.device)  # some token ids
    loss = next_token_logp[test_tokens].sum()
    
    print(f"Test loss: {loss.item()}")
    
    # Test 1: Use .backward()
    print("\nTest 1: Using .backward()")
    loss.backward(retain_graph=True)
    
    backward_grad_count = 0
    for i, param in enumerate(trainable_params):
        if param.grad is not None and torch.any(param.grad != 0):
            backward_grad_count += 1
    
    print(f"  Non-zero gradients with .backward(): {backward_grad_count}/{len(trainable_params)}")
    
    # Clear gradients
    for param in trainable_params:
        if param.grad is not None:
            param.grad.zero_()
    
    # Test 2: Use torch.autograd.grad()
    print("\nTest 2: Using torch.autograd.grad()")
    try:
        grads = torch.autograd.grad(
            outputs=loss,
            inputs=trainable_params,
            retain_graph=True,
            create_graph=False,
            only_inputs=True,
            allow_unused=True
        )
        
        autograd_grad_count = 0
        for i, (param, grad) in enumerate(zip(trainable_params, grads)):
            if grad is not None and torch.any(grad != 0):
                autograd_grad_count += 1
        
        print(f"  Non-zero gradients with autograd.grad(): {autograd_grad_count}/{len(trainable_params)}")
        
    except Exception as e:
        print(f"  autograd.grad() failed: {e}")
    
    # Test 3: Create sequence-level loss (like our entropy probe)
    print("\nTest 3: Sequence-level loss (like entropy probe)")
    
    # Simulate sequence log probabilities
    seq_len = input_ids.size(1)
    # Get log probs for the actual tokens in sequence
    target_logp = logp_all.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)  # (batch, seq_len)
    seq_logp = target_logp.sum(dim=-1)  # (batch,) - sequence level
    
    # Simple entropy-like loss
    seq_loss = seq_logp.sum()
    print(f"  Sequence loss: {seq_loss.item()}")
    
    try:
        seq_grads = torch.autograd.grad(
            outputs=seq_loss,
            inputs=trainable_params,
            retain_graph=True,
            create_graph=False,
            only_inputs=True,
            allow_unused=True
        )
        
        seq_grad_count = 0
        for i, (param, grad) in enumerate(zip(trainable_params, seq_grads)):
            if grad is not None and torch.any(grad != 0):
                seq_grad_count += 1
        
        print(f"  Non-zero gradients with sequence loss: {seq_grad_count}/{len(trainable_params)}")
        
    except Exception as e:
        print(f"  Sequence loss autograd.grad() failed: {e}")

if __name__ == "__main__":
    test_gradient_connection()