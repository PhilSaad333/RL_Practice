#!/usr/bin/env python3
"""
Test script for SequenceProcessor on Lambda H100 instance.

This script:
1. Loads a LoRA checkpoint using the proven pattern from offline_entropy_probe.py
2. Tests SequenceProcessor with dataset sampling (option 2)
3. Validates that generation and logprob computation work correctly

Usage on Lambda:
    cd ~/RL_Practice
    python sequence_processing/test_sequence_processor.py
"""

import sys
import os
import torch
from transformers import AutoTokenizer

# Add project root to path
sys.path.append('/home/ubuntu/RL_Practice')

def load_peft_for_test(
    base_id: str,
    adapter_path: str,
    *,
    mode: str = "lora_simple",
    dtype: str = "bf16",
    device_map: str = "cuda",
):
    """Load LoRA model - simplified from offline_entropy_probe.py"""
    from transformers import AutoModelForCausalLM
    from peft import PeftModel

    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[dtype]

    print(f"Loading base model: {base_id}")
    base = AutoModelForCausalLM.from_pretrained(
        base_id, 
        device_map=device_map, 
        torch_dtype=torch_dtype,
        attn_implementation="eager"
    )
    
    if hasattr(base, "gradient_checkpointing_disable"):
        base.gradient_checkpointing_disable()
    if hasattr(base.config, "use_cache"):
        base.config.use_cache = True

    print(f"Loading LoRA adapter: {adapter_path}")
    peft = PeftModel.from_pretrained(base, adapter_path, is_trainable=True)
    
    if hasattr(peft, "enable_input_require_grads"):
        peft.enable_input_require_grads()

    return peft


def main():
    print("=== SequenceProcessor Test on Lambda H100 ===")
    
    # Checkpoint path on Lambda - the LoRA adapter is in the model/ subdirectory
    checkpoint_path = "/home/ubuntu/localfs/training_runs/run_2025-08-24_22-13-22/training_state/step_40/model"
    base_model_id = "Qwen/Qwen2.5-1.5B"
    
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Base model: {base_model_id}")
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint path does not exist: {checkpoint_path}")
        print("Available paths in /home/ubuntu/localfs/:")
        if os.path.exists("/home/ubuntu/localfs"):
            for item in os.listdir("/home/ubuntu/localfs"):
                print(f"  {item}")
        return
    
    # Load model and tokenizer
    print("\n1. Loading model and tokenizer...")
    try:
        model = load_peft_for_test(base_model_id, checkpoint_path)
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        
        # Set pad token if not set and fix padding side for decoder-only models
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'  # Important for decoder-only models
        
        print(f"Model loaded successfully!")
        print(f"Model device: {next(model.parameters()).device}")
        print(f"Model dtype: {next(model.parameters()).dtype}")
        
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return
    
    # Initialize SequenceProcessor
    print("\n2. Initializing SequenceProcessor...")
    try:
        from sequence_processing import SequenceProcessor, GenerationConfig
        
        config = GenerationConfig(
            temperature=0.8,
            max_new_tokens=256,
            gen_batch_size=4,  # Small batch for testing
            tf_batch_size=8
        )
        
        processor = SequenceProcessor(model, tokenizer, config)
        print("SequenceProcessor initialized successfully!")
        
    except Exception as e:
        print(f"ERROR initializing SequenceProcessor: {e}")
        return
    
    # Test dataset sampling and generation
    print("\n3. Testing dataset sampling and generation...")
    try:
        # Test with small numbers first
        sequences, logprob_results = processor.generate_with_logprobs(
            dataset_name="gsm8k_r1_template",
            split="train",
            num_prompts=2,  # Start small
            G=3,            # 3 responses per prompt
            seed=42,        # Reproducible
            with_grad=False # No gradients for initial test
        )
        
        print(f"Generation successful!")
        print(f"Sequences shape: {sequences.sequences.shape}")
        print(f"Number of prompts: {len(sequences.prompt_lens)}")
        print(f"Prompt lengths: {sequences.prompt_lens}")
        
        # Show detailed sample outputs
        print(f"\n=== DETAILED SAMPLE OUTPUTS ===")
        for b in range(len(sequences.responses_text)):
            print(f"\n--- PROMPT {b} (Length: {sequences.prompt_lens[b]} tokens) ---")
            
            # Try to reconstruct the original prompt
            if len(sequences.sequences[b]) > 0:
                prompt_tokens = sequences.sequences[b][0][:sequences.prompt_lens[b]]
                original_prompt = tokenizer.decode(prompt_tokens, skip_special_tokens=True)
                print(f"ORIGINAL PROMPT: {original_prompt}")
            
            for g in range(len(sequences.responses_text[b])):
                response_text = sequences.responses_text[b][g]
                gen_len = sequences.gen_lens[b][g] if b < len(sequences.gen_lens) and g < len(sequences.gen_lens[b]) else 0
                
                print(f"\n  Response {g} (Generated {gen_len} tokens):")
                print(f"  TEXT: {response_text}")
                
                # Show logprob stats if available
                if (b < len(logprob_results.logprobs) and 
                    g < len(logprob_results.logprobs[b]) and 
                    len(logprob_results.logprobs[b][g]) > 0):
                    
                    token_logprobs = logprob_results.logprobs[b][g]
                    seq_logprob = logprob_results.sequence_logprobs[b][g]
                    entropies = logprob_results.entropies[b][g]
                    
                    print(f"  LOGPROBS: mean={token_logprobs.mean():.3f}, "
                          f"total_seq={seq_logprob:.3f}, "
                          f"entropy_mean={entropies.mean():.3f}")
                else:
                    print(f"  LOGPROBS: No logprob data available")
            
            print(f"--- END PROMPT {b} ---")
        
        print(f"\n=== SUMMARY STATS ===")
        print(f"Total prompts: {len(sequences.responses_text)}")
        print(f"Responses per prompt: {len(sequences.responses_text[0]) if sequences.responses_text else 0}")
        print(f"Sequence tensor shape: {sequences.sequences.shape}")
        print(f"Attention mask shape: {sequences.attention_masks.shape}")
        print(f"Prompt lengths: {sequences.prompt_lens}")
        print(f"Generation lengths: {sequences.gen_lens}")
        
    except Exception as e:
        print(f"ERROR in generation test: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n✅ SequenceProcessor test completed successfully!")
    print("The implementation is working on Lambda H100!")


if __name__ == "__main__":
    main()