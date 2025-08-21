#!/usr/bin/env python3
"""
Minimal test to verify torch.autograd.grad() works with DDP parameters.
This will help isolate whether the issue is DDP-specific or something else.
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
import os
import sys

def setup_ddp():
    """Setup DDP for testing"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Get rank and world size from torchrun
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    if world_size > 1:
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank

def test_ddp_autograd():
    rank, world_size, local_rank = setup_ddp()
    
    print(f"[Rank {rank}] Starting DDP autograd test...")
    
    # Load model with LoRA (same as training)
    model_name = "Qwen/Qwen2.5-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=f"cuda:{local_rank}" if world_size > 1 else "auto"
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
    
    # Wrap with DDP if multi-GPU
    if world_size > 1:
        ddp_model = DDP(model, device_ids=[local_rank])
    else:
        ddp_model = model
    
    # Test 1: Get parameters both ways
    if world_size > 1:
        ddp_params = [p for p in ddp_model.parameters() if p.requires_grad]
        unwrapped_params = [p for p in ddp_model.module.parameters() if p.requires_grad]
    else:
        ddp_params = [p for p in ddp_model.parameters() if p.requires_grad]
        unwrapped_params = ddp_params
    
    print(f"[Rank {rank}] DDP params: {len(ddp_params)}")
    print(f"[Rank {rank}] Unwrapped params: {len(unwrapped_params)}")
    
    # Create simple input
    text = "What is 2+2? The answer is"
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    if world_size > 1:
        input_ids = inputs["input_ids"].to(f"cuda:{local_rank}")
        attention_mask = inputs["attention_mask"].to(f"cuda:{local_rank}")
    else:
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
    
    # Forward pass through DDP model (same as training)
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        outputs = ddp_model(input_ids, attention_mask=attention_mask)
    
    logits = outputs.logits / 1.0  # temperature
    logp_all = F.log_softmax(logits.float(), dim=-1)
    
    # Simulate token-level log probs (like training)
    B, seq_len = input_ids.shape
    T_g = 5  # Last 5 tokens as "generated"
    token_log_probs = logp_all[:, -T_g:, :].gather(-1, input_ids[:, -T_g:].unsqueeze(-1)).squeeze(-1)
    
    # Simulate gen_mask
    gen_mask = torch.ones_like(token_log_probs)
    
    # Compute sequence log probs (same as entropy probe)
    seq_log_probs = (token_log_probs * gen_mask).sum(dim=-1)  # (B,)
    
    # Entropy probe computation (exact same as our code)
    mean_log_prob = seq_log_probs.mean()
    centered_log_probs = (seq_log_probs - mean_log_prob).detach()
    entropy_loss = torch.sum(centered_log_probs * seq_log_probs)
    
    print(f"[Rank {rank}] Entropy loss: {entropy_loss.item():.6f}")
    print(f"[Rank {rank}] Seq logp grad_fn: {seq_log_probs.grad_fn}")
    
    # Test autograd.grad with DDP parameters
    try:
        entropy_grads = torch.autograd.grad(
            outputs=entropy_loss,
            inputs=ddp_params,
            retain_graph=True,
            create_graph=False,
            only_inputs=True,
            allow_unused=True
        )
        
        non_zero_count = sum(1 for g in entropy_grads if g is not None and torch.any(g != 0))
        none_count = sum(1 for g in entropy_grads if g is None)
        
        print(f"[Rank {rank}] DDP autograd.grad:")
        print(f"[Rank {rank}]   Non-zero gradients: {non_zero_count}/{len(ddp_params)}")
        print(f"[Rank {rank}]   None gradients: {none_count}/{len(ddp_params)}")
        
    except Exception as e:
        print(f"[Rank {rank}] DDP autograd.grad failed: {e}")
    
    # Test autograd.grad with unwrapped parameters  
    if world_size > 1:
        try:
            unwrapped_grads = torch.autograd.grad(
                outputs=entropy_loss,
                inputs=unwrapped_params,
                retain_graph=True,
                create_graph=False,
                only_inputs=True,
                allow_unused=True
            )
            
            non_zero_count2 = sum(1 for g in unwrapped_grads if g is not None and torch.any(g != 0))
            none_count2 = sum(1 for g in unwrapped_grads if g is None)
            
            print(f"[Rank {rank}] Unwrapped autograd.grad:")
            print(f"[Rank {rank}]   Non-zero gradients: {non_zero_count2}/{len(unwrapped_params)}")
            print(f"[Rank {rank}]   None gradients: {none_count2}/{len(unwrapped_params)}")
            
        except Exception as e:
            print(f"[Rank {rank}] Unwrapped autograd.grad failed: {e}")
    
    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()
    
    print(f"[Rank {rank}] Test complete!")

if __name__ == "__main__":
    test_ddp_autograd()