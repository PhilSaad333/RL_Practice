#!/usr/bin/env python3
"""
Debug exact token-level slicing to find where extraction goes wrong.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
from sequence_processing import SequenceProcessor, GenerationConfig
from entropy_experiments.offline_entropy_probe import load_peft_for_probe
from transformers import AutoTokenizer

def debug_token_slicing():
    """Debug exactly where token slicing goes wrong."""
    
    print("🔬 DEBUGGING TOKEN SLICING")
    print("=" * 80)
    
    # Load minimal setup
    checkpoint_path = "/content/drive/MyDrive/RL_Practice_Files/new_rl_checkpoint/step_60/model"
    base_model = "Qwen/Qwen2.5-1.5B"
    
    model = load_peft_for_probe(base_id=base_model, adapter_path=checkpoint_path, 
                               mode="lora_simple", dtype="bf16", device_map="cuda")
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    
    # Get one sample prompt
    from rlp_datasets.gsm8k_r1_template import build_gsm8k
    dataset = build_gsm8k(split="train")
    prompt = dataset[0].question  # First example for reproducible debug
    
    print(f"ORIGINAL PROMPT (last 100 chars):")
    print(f"'{prompt[-100:]}'")
    print()
    
    # Step 1: Tokenize single prompt
    tokenized = tokenizer(prompt, return_tensors="pt", truncation=False)
    prompt_input_ids = tokenized['input_ids'].to(model.device)  # [1, prompt_len]
    original_prompt_len = prompt_input_ids.shape[1]
    
    print(f"Original tokenized prompt shape: {prompt_input_ids.shape}")
    print(f"Original prompt length: {original_prompt_len}")
    
    # Decode to verify
    decoded_prompt = tokenizer.decode(prompt_input_ids[0], skip_special_tokens=True)
    print(f"Decoded prompt ends with: '{decoded_prompt[-50:]}'")
    print()
    
    # Step 2: Expand for G=2 (simulate batch expansion)
    G = 2
    expanded_input_ids = prompt_input_ids.repeat_interleave(G, dim=0)  # [2, prompt_len]
    print(f"Expanded input shape: {expanded_input_ids.shape}")
    
    # Step 3: Generate
    print("Generating...")
    with torch.no_grad():
        full_sequences = model.generate(
            input_ids=expanded_input_ids,
            attention_mask=torch.ones_like(expanded_input_ids),  # Add attention mask
            do_sample=True,
            num_return_sequences=1,  # Fixed value
            max_new_tokens=50,
            temperature=1.0,
            top_p=0.99,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )  # This returns tensor directly, not dict
    print(f"Generated sequences shape: {full_sequences.shape}")
    print()
    
    # Step 4: Analyze each generated sequence
    for g in range(G):
        print(f"--- GENERATION {g+1} ---")
        full_seq = full_sequences[g]  # [total_len]
        
        print(f"Full sequence length: {len(full_seq)}")
        
        # Decode full sequence
        full_text = tokenizer.decode(full_seq, skip_special_tokens=True)
        print(f"Full text: '{full_text}'")
        print()
        
        # Extract using our current approach
        gen_tokens_our_way = full_seq[original_prompt_len:]
        # Remove padding
        non_pad_mask = gen_tokens_our_way != tokenizer.pad_token_id
        if non_pad_mask.any():
            last_valid_idx = non_pad_mask.nonzero()[-1].item() + 1
            gen_tokens_our_way = gen_tokens_our_way[:last_valid_idx]
        
        our_response = tokenizer.decode(gen_tokens_our_way, skip_special_tokens=True)
        print(f"OUR EXTRACTION: '{our_response}'")
        
        # Extract using collect_rollouts approach (direct slicing)
        # In collect_rollouts: gen_ids = full_ids[:, :, prompt_ids.size(1):]
        collect_rollouts_slice = full_seq[original_prompt_len:]
        collect_rollouts_response = tokenizer.decode(collect_rollouts_slice, skip_special_tokens=True)
        print(f"COLLECT_ROLLOUTS APPROACH: '{collect_rollouts_response}'")
        
        # Show token-by-token around the boundary
        print("\nTOKEN-BY-TOKEN ANALYSIS:")
        boundary_start = max(0, original_prompt_len - 5)
        boundary_end = min(len(full_seq), original_prompt_len + 10)
        
        for i in range(boundary_start, boundary_end):
            token_id = full_seq[i].item()
            token_text = tokenizer.decode([token_id], skip_special_tokens=True)
            marker = " <- BOUNDARY" if i == original_prompt_len else ""
            print(f"  Token {i}: {token_id} -> '{token_text}'{marker}")
        
        print()
        
        # Check what the prompt portion looks like
        prompt_portion = full_seq[:original_prompt_len]
        prompt_text = tokenizer.decode(prompt_portion, skip_special_tokens=True)
        print(f"PROMPT PORTION (last 50 chars): '{prompt_text[-50:]}'")
        
        print("-" * 50)

if __name__ == "__main__":
    debug_token_slicing()