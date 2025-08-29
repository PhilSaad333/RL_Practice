#!/usr/bin/env python3
"""
Debug prompt handling differences between SequenceProcessor and collect_rollouts.py

This script will help identify why SequenceProcessor is generating incorrect responses
that don't match the input prompts.
"""

import torch
from transformers import AutoTokenizer
import sys
sys.path.append('/home/ubuntu/RL_Practice')

def debug_tokenization(prompt_text, tokenizer):
    """Debug how a prompt gets tokenized."""
    print(f"=== TOKENIZATION DEBUG ===")
    print(f"Original prompt: {repr(prompt_text)}")
    print(f"Length: {len(prompt_text)} characters")
    print()
    
    # Tokenize the prompt
    tokens = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True)
    input_ids = tokens.input_ids[0]
    
    print(f"Token IDs: {input_ids.tolist()}")
    print(f"Number of tokens: {len(input_ids)}")
    print()
    
    # Decode back to verify
    decoded = tokenizer.decode(input_ids, skip_special_tokens=False)
    decoded_clean = tokenizer.decode(input_ids, skip_special_tokens=True)
    
    print(f"Decoded (with special): {repr(decoded)}")
    print(f"Decoded (clean): {repr(decoded_clean)}")
    print()
    
    # Check if reconstruction matches original
    if decoded_clean.strip() == prompt_text.strip():
        print("✅ Tokenization round-trip successful")
    else:
        print("❌ Tokenization round-trip FAILED!")
        print(f"Difference: {repr(decoded_clean)} vs {repr(prompt_text)}")
    
    return input_ids

def test_dataset_sampling():
    """Test how dataset sampling works."""
    print(f"=== DATASET SAMPLING DEBUG ===")
    
    try:
        import rlp_datasets
        ds_builder = rlp_datasets.DATASET_REGISTRY["gsm8k_r1_template"]
        ds = ds_builder("test")
        
        print(f"Dataset length: {len(ds)}")
        
        # Sample the first few items
        for i in range(3):
            item = ds[i]
            print(f"\n--- Item {i} ---")
            print(f"Question: {repr(item.question[:100])}...")
            if hasattr(item, 'answer'):
                print(f"Answer: {repr(str(item.answer)[:50])}...")
                
        return ds
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def compare_generation_approaches(tokenizer):
    """Compare how collect_rollouts vs SequenceProcessor would handle the same prompt."""
    
    # Use the problematic prompt from the logs
    test_prompt = """You are solving math problems. Respond by reasoning through the problem then providing a final answer. Enclose the reasoning process within <think> </think> and the answer within <answer> </answer> tags, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
Question: A whirligig spins at five times the speed of a thingamabob. A whatchamacallit spins eleven times faster than a thingamabob. A whatchamacallit spins at 121 meters per second. How fast does a whirligig spin?
Response: """
    
    print(f"=== GENERATION COMPARISON ===")
    print("Test prompt (first 200 chars):")
    print(repr(test_prompt[:200]))
    print()
    
    # Debug tokenization
    input_ids = debug_tokenization(test_prompt, tokenizer)
    
    # Check specific numbers in the prompt
    print(f"=== NUMBER DETECTION ===")
    if "eleven" in test_prompt:
        print("✅ Found 'eleven' in prompt")
    else:
        print("❌ Missing 'eleven' in prompt")
        
    if "five times" in test_prompt:
        print("✅ Found 'five times' in prompt")
    else:
        print("❌ Missing 'five times' in prompt")
        
    if "121" in test_prompt:
        print("✅ Found '121' in prompt")
    else:
        print("❌ Missing '121' in prompt")
    
    return test_prompt, input_ids

def main():
    """Main debug function."""
    print("=== PROMPT HANDLING DEBUG ===")
    print("Investigating SequenceProcessor generation issues\n")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    
    print(f"Tokenizer: {tokenizer.name_or_path}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Pad token: {repr(tokenizer.pad_token)}")
    print(f"EOS token: {repr(tokenizer.eos_token)}")
    print()
    
    # Test dataset sampling
    ds = test_dataset_sampling()
    print()
    
    # Compare generation approaches
    test_prompt, input_ids = compare_generation_approaches(tokenizer)
    print()
    
    # Test SequenceProcessor import
    print(f"=== SEQUENCEPROCESSOR DEBUG ===")
    try:
        from sequence_processing import SequenceProcessor, GenerationConfig
        print("✅ SequenceProcessor import successful")
        
        # Test configuration
        config = GenerationConfig(
            temperature=0.8,
            max_new_tokens=50,
            num_return_sequences=2
        )
        print(f"✅ GenerationConfig created: {config}")
        
    except Exception as e:
        print(f"❌ SequenceProcessor import failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n=== SUMMARY ===")
    print("This debug script helps identify prompt handling issues.")
    print("Run this on Lambda to compare actual behavior.")

if __name__ == "__main__":
    main()