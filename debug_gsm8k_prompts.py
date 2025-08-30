#!/usr/bin/env python3
"""
Debug script to check GSM8K prompt generation and model responses.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from rlp_datasets.gsm8k_r1_template import build_gsm8k, prompt_template
from transformers import AutoTokenizer
import random

def debug_prompts():
    """Debug GSM8K prompt generation and check for truncation issues."""
    print("=" * 80)
    print("🔍 DEBUGGING GSM8K PROMPTS")
    print("=" * 80)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    dataset = build_gsm8k(split="train")
    print(f"Dataset loaded: {len(dataset)} examples")
    
    # Check a few random examples
    random.seed(42)
    samples = random.sample(dataset, 3)
    
    for i, example in enumerate(samples):
        print(f"\n--- EXAMPLE {i+1} ---")
        print(f"Original text: {example.text[:200]}...")
        print(f"Generated question: {example.question}")
        
        # Tokenize the question to check length
        tokens = tokenizer(example.question, add_special_tokens=True, return_tensors="pt")
        token_count = tokens.input_ids.shape[1]
        
        print(f"Question token count: {token_count}")
        print(f"Question ends with: '{example.question[-50:]}'")
        
        # Check if prompt is too long (Qwen2.5-1.5B has 32k context)
        if token_count > 2048:  # Conservative check
            print("⚠️  WARNING: Prompt is very long, may cause truncation!")
        
        # Decode back to check tokenization issues
        decoded = tokenizer.decode(tokens.input_ids[0], skip_special_tokens=True)
        if decoded.strip() != example.question.strip():
            print("⚠️  WARNING: Tokenization roundtrip failed!")
            print(f"Original:  '{example.question[-100:]}'")
            print(f"Decoded:   '{decoded[-100:]}'")
        
        print("-" * 40)
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    debug_prompts()