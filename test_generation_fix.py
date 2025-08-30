#!/usr/bin/env python3
"""
Simple test script to verify generation extraction fix.
Just generates a few responses and checks they start correctly.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
from sequence_processing import SequenceProcessor, GenerationConfig
from entropy_experiments.offline_entropy_probe import load_peft_for_probe

def test_generation_extraction():
    """Test that responses start with proper reasoning, not mid-sentence."""
    
    print("🧪 TESTING GENERATION EXTRACTION FIX")
    print("=" * 60)
    
    # Load model and tokenizer
    checkpoint_path = "/content/drive/MyDrive/RL_Practice_Files/new_rl_checkpoint/step_60/model"
    base_model = "Qwen/Qwen2.5-1.5B"
    
    print("Loading model...")
    model = load_peft_for_probe(
        base_id=base_model,
        adapter_path=checkpoint_path,
        mode="lora_simple", 
        dtype="bf16",
        device_map="cuda"
    )
    model.eval()
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.padding_side = "left"  
    tokenizer.pad_token = tokenizer.eos_token
    
    # Setup simple config for testing
    config = GenerationConfig(
        temperature=1.0,
        top_p=0.99,
        max_new_tokens=100,  # Enough to see if response starts correctly
        do_sample=True,
        gen_batch_size=4,    # Small batch for testing
        tf_batch_size=4
    )
    
    processor = SequenceProcessor(model, tokenizer, config)
    print("✅ Model loaded")
    print()
    
    # Test with multiple prompts and G=2 to test batch handling
    print("🔄 Generating responses...")
    print("Dataset: gsm8k_r1_template, 3 prompts, G=2 generations each")
    
    sequences, logprob_results, _ = processor.generate_with_logprobs(
        prompts=None,  # Sample from dataset
        G=2,  # Test multiple generations
        dataset_name="gsm8k_r1_template", 
        split="train",
        num_prompts=3,
        seed=42,  # Reproducible
        with_grad=False,
        compute_rb=False  # Skip RB entropy for speed
    )
    
    print("✅ Generation complete")
    print()
    
    # Check each response
    print("📋 RESPONSE ANALYSIS:")
    print("=" * 60)
    
    all_good = True
    
    for b in range(len(sequences.responses_text)):
        print(f"\n--- PROMPT {b+1} ---")
        
        for g in range(len(sequences.responses_text[b])):
            response = sequences.responses_text[b][g]
            reward = logprob_results.rewards[b][g] if logprob_results.rewards else 0.0
            
            print(f"Response {g+1}: '{response[:100]}{'...' if len(response) > 100 else ''}'")
            print(f"Reward: {reward:.3f}")
            
            # Check if response starts properly (not mid-sentence)
            starts_properly = (
                response.strip().startswith((' ', 'First', 'Let', 'I', 'We', 'To', 'The', 'In', 'Since', 'Given', 'Step')) or
                any(response.strip().startswith(name) for name in ['Natalia', 'John', 'Mary', 'Anna', 'Sam', 'Tom', 'Bob', 'Alice']) or
                response.strip()[0].isupper()  # Starts with capital letter
            )
            
            problematic_starts = [
                '> answer here', '.\nQuestion:', 'Response: <think>', 'Question:', 
                'nswer>', 'hink>', ' show up', ' people', ' clips', ' sold'
            ]
            is_problematic = any(response.startswith(bad) for bad in problematic_starts)
            
            if is_problematic or not starts_properly:
                print("❌ PROBLEMATIC: Response appears to start mid-sentence!")
                all_good = False
            else:
                print("✅ GOOD: Response starts properly")
            
            print()
    
    print("=" * 60)
    if all_good:
        print("🎉 SUCCESS: All responses start properly!")
        print("✅ Generation extraction fix is working")
    else:
        print("⚠️  ISSUE: Some responses still start mid-sentence")
        print("❌ Generation extraction needs more work")
    
    print("=" * 60)
    return all_good

if __name__ == "__main__":
    test_generation_extraction()