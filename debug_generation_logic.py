#!/usr/bin/env python3
"""
Deep debug of generation logic to find mid-sentence response issue.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
from transformers import AutoTokenizer
from sequence_processing import SequenceProcessor, GenerationConfig
from entropy_experiments.offline_entropy_probe import load_peft_for_probe

def debug_generation_step_by_step():
    """Debug each step of the generation process in detail."""
    
    print("=" * 80)
    print("🔬 DEEP GENERATION LOGIC DEBUG")
    print("=" * 80)
    
    # Load model and tokenizer (use smaller checkpoint for debugging)
    checkpoint_path = "/content/drive/MyDrive/RL_Practice_Files/new_rl_checkpoint/step_60/model"
    base_model = "Qwen/Qwen2.5-1.5B"
    
    model = load_peft_for_probe(
        base_id=base_model,
        adapter_path=checkpoint_path,
        mode="lora_simple", 
        dtype="bf16",
        device_map="cuda"
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    
    # Sample one prompt from dataset
    from rlp_datasets.gsm8k_r1_template import build_gsm8k
    dataset = build_gsm8k(split="train")
    example = dataset[0]  # Use first example for reproducible debug
    
    prompt = example.question
    print(f"ORIGINAL PROMPT:")
    print(f"'{prompt}'")
    print(f"Prompt length: {len(prompt)} chars")
    print()
    
    # Step 1: Tokenize the prompt
    print("STEP 1: TOKENIZATION")
    tokenized = tokenizer(prompt, return_tensors="pt", truncation=False)
    prompt_input_ids = tokenized['input_ids'].to(model.device)
    prompt_attention_mask = tokenized['attention_mask'].to(model.device)
    
    # Calculate prompt length
    prompt_len = prompt_attention_mask.sum(dim=1).item()
    print(f"Tokenized prompt shape: {prompt_input_ids.shape}")
    print(f"Calculated prompt_len: {prompt_len}")
    
    # Decode back to check tokenization
    decoded_prompt = tokenizer.decode(prompt_input_ids[0], skip_special_tokens=True)
    print(f"Decoded prompt: '{decoded_prompt[-100:]}'")  # Show last 100 chars
    print(f"Tokenization roundtrip OK: {decoded_prompt.strip() == prompt.strip()}")
    print()
    
    # Step 2: Generate one sequence
    print("STEP 2: GENERATION")
    G = 1  # Single generation for debugging
    expanded_input_ids = prompt_input_ids.repeat_interleave(G, dim=0)
    expanded_attention_mask = prompt_attention_mask.repeat_interleave(G, dim=0)
    
    print(f"Expanded input shape: {expanded_input_ids.shape}")
    
    # Generate with exact same settings as sequence processor
    with torch.no_grad():
        gen_output = model.generate(
            input_ids=expanded_input_ids,
            attention_mask=expanded_attention_mask,
            do_sample=True,
            num_return_sequences=1,  # Should be 1 as we fixed
            max_new_tokens=50,  # Short for debugging
            temperature=1.0,
            top_p=0.99,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
            synced_gpus=False
        )
    
    sequences = gen_output.sequences  # [1, total_len]
    print(f"Generated sequences shape: {sequences.shape}")
    print()
    
    # Step 3: Analyze the full generated sequence
    print("STEP 3: FULL SEQUENCE ANALYSIS")
    full_seq = sequences[0]  # [total_len]
    full_text = tokenizer.decode(full_seq, skip_special_tokens=True)
    
    print(f"Full generated sequence length: {len(full_seq)} tokens")
    print(f"Full generated text: '{full_text}'")
    print()
    
    # Step 4: Extract response using sequence processor logic
    print("STEP 4: RESPONSE EXTRACTION")
    
    # Calculate generation length (same logic as sequence processor)
    non_pad_mask = full_seq != tokenizer.pad_token_id
    total_len = non_pad_mask.sum().item()
    gen_len = total_len - prompt_len
    
    print(f"Total sequence length (non-pad): {total_len}")  
    print(f"Prompt length: {prompt_len}")
    print(f"Calculated gen_len: {gen_len}")
    
    if gen_len > 0:
        # Extract generation tokens (same logic as sequence processor)
        gen_tokens = full_seq[prompt_len:prompt_len + gen_len]
        response_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        
        print(f"Generated tokens: {gen_tokens}")
        print(f"EXTRACTED RESPONSE: '{response_text}'")
        print()
        
        # Step 5: Show where extraction comes from in full text
        print("STEP 5: EXTRACTION BOUNDARY ANALYSIS")
        
        # Decode prompt part
        prompt_tokens = full_seq[:prompt_len] 
        prompt_part = tokenizer.decode(prompt_tokens, skip_special_tokens=True)
        
        print(f"Prompt part (first {prompt_len} tokens): '{prompt_part[-100:]}'")
        print(f"Generated part (tokens {prompt_len} to {prompt_len + gen_len}): '{response_text}'")
        
        # Check if boundary makes sense
        expected_boundary = "Response: <think>"
        if prompt_part.endswith(expected_boundary):
            print("✅ Boundary looks correct - prompt ends with 'Response: <think>'")
        else:
            print("❌ BOUNDARY ISSUE - prompt doesn't end with expected text")
            print(f"   Prompt actually ends with: '{prompt_part[-50:]}'")
    
    print("=" * 80)

if __name__ == "__main__":
    debug_generation_step_by_step()