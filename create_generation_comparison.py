#!/usr/bin/env python3
"""
Create detailed generation comparison JSONL file.

This script:
1. Picks a specific prompt from gsm8k_r1_template dataset
2. Generates responses using both collect_rollouts style and SequenceProcessor
3. Saves everything to a JSONL file for detailed inspection
"""

import json
import torch
from transformers import AutoTokenizer
import sys
sys.path.append('/home/ubuntu/RL_Practice')

def load_model_for_test():
    """Load the same model both methods would use."""
    from entropy_experiments.offline_entropy_probe import load_peft_for_probe
    
    checkpoint_path = "/home/ubuntu/localfs/training_runs/run_2025-08-24_22-13-22/training_state/step_40/model"
    base_model_id = "Qwen/Qwen2.5-1.5B"
    
    model = load_peft_for_probe(
        base_id=base_model_id,
        adapter_path=checkpoint_path,
        mode="lora_simple",
        dtype="bf16",
        device_map="cuda",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    
    return model, tokenizer

def get_test_prompt():
    """Get a specific test prompt from the dataset."""
    import rlp_datasets
    ds = rlp_datasets.DATASET_REGISTRY["gsm8k_r1_template"]("test")
    
    # Use a specific index for reproducibility
    test_idx = 10
    item = ds[test_idx]
    
    return {
        "dataset_index": test_idx,
        "prompt": item.question,
        "expected_answer": str(item.answer) if hasattr(item, 'answer') else None
    }

def collect_rollouts_generation(model, tokenizer, prompt, num_sequences=3, seed=42):
    """Generate using collect_rollouts style with detailed logging."""
    from sequence_processing.sequence_processor import StopAfterAnswer
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    
    # Tokenize prompt
    tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    prompt_ids = tokens.input_ids.to(model.device)
    attn = tokens.attention_mask.to(model.device)
    
    # Create logits processor
    stop_processor = StopAfterAnswer(tokenizer)
    
    # Generate exactly like collect_rollouts.py
    old_cache = getattr(model.config, "use_cache", False)
    model.config.use_cache = True
    
    try:
        gen_out = model.generate(
            prompt_ids,
            attention_mask=attn,
            do_sample=True,
            num_return_sequences=num_sequences,
            max_new_tokens=250,
            temperature=0.8,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            logits_processor=[stop_processor],
            return_dict_in_generate=True,
            output_scores=True,
            synced_gpus=False,
        )
        
        # Process results
        generated_sequences = gen_out.sequences
        responses = []
        
        for i in range(num_sequences):
            full_sequence = generated_sequences[i]
            # Extract only the generated part (after prompt)
            gen_part = full_sequence[prompt_ids.shape[1]:]
            response_text = tokenizer.decode(gen_part, skip_special_tokens=True)
            
            responses.append({
                "response_index": i,
                "generated_tokens": len(gen_part),
                "full_generated_sequence": gen_part.tolist(),
                "response_text": response_text
            })
        
        return {
            "method": "collect_rollouts_style",
            "generation_config": {
                "max_new_tokens": 250,
                "temperature": 0.8,
                "do_sample": True,
                "num_return_sequences": num_sequences,
                "seed": seed
            },
            "prompt_tokens": prompt_ids.shape[1],
            "responses": responses
        }
        
    finally:
        model.config.use_cache = old_cache

def sequence_processor_generation(model, tokenizer, prompt, num_sequences=3, seed=42):
    """Generate using SequenceProcessor with detailed logging."""
    from sequence_processing import SequenceProcessor, GenerationConfig
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    
    # Create config
    config = GenerationConfig(
        temperature=0.8,
        max_new_tokens=250,
        gen_batch_size=4,
        tf_batch_size=8
    )
    
    # Initialize processor
    processor = SequenceProcessor(model, tokenizer, config)
    
    try:
        # Generate
        sequences, logprob_results = processor.generate_with_logprobs(
            prompts=[prompt],
            G=num_sequences,
            with_grad=False
        )
        
        responses = []
        if len(sequences.responses_text) > 0:
            prompt_responses = sequences.responses_text[0]  # First (and only) prompt's responses
            
            for i, response_text in enumerate(prompt_responses):
                # Try to get additional info if available
                gen_len = sequences.gen_lens[0][i] if (0 < len(sequences.gen_lens) and i < len(sequences.gen_lens[0])) else 0
                
                responses.append({
                    "response_index": i,
                    "generated_tokens": gen_len,
                    "response_text": response_text
                })
        
        return {
            "method": "sequence_processor",
            "generation_config": {
                "max_new_tokens": 250,
                "temperature": 0.8,
                "do_sample": True,
                "num_return_sequences": num_sequences,
                "seed": seed
            },
            "prompt_tokens": sequences.prompt_lens[0] if len(sequences.prompt_lens) > 0 else 0,
            "responses": responses
        }
        
    except Exception as e:
        return {
            "method": "sequence_processor", 
            "error": str(e),
            "generation_config": {
                "max_new_tokens": 250,
                "temperature": 0.8,
                "seed": seed
            }
        }

def main():
    """Create complete comparison JSONL."""
    print("=== Creating Generation Comparison JSONL ===")
    
    # Load model
    print("Loading model...")
    model, tokenizer = load_model_for_test()
    model.eval()
    
    # Get test prompt
    print("Getting test prompt...")
    prompt_data = get_test_prompt()
    print(f"Selected prompt from index {prompt_data['dataset_index']}")
    print(f"Prompt length: {len(prompt_data['prompt'])} characters")
    print(f"Expected answer: {prompt_data['expected_answer']}")
    
    # Generate with both methods
    print("\nGenerating with collect_rollouts style...")
    collect_results = collect_rollouts_generation(model, tokenizer, prompt_data['prompt'])
    
    print("Generating with SequenceProcessor...")
    sp_results = sequence_processor_generation(model, tokenizer, prompt_data['prompt'])
    
    # Combine everything
    comparison_data = {
        "dataset": "gsm8k_r1_template",
        "split": "test", 
        "checkpoint": "/home/ubuntu/localfs/training_runs/run_2025-08-24_22-13-22/training_state/step_40/model",
        "base_model": "Qwen/Qwen2.5-1.5B",
        "prompt_data": prompt_data,
        "generation_results": {
            "collect_rollouts_style": collect_results,
            "sequence_processor": sp_results
        }
    }
    
    # Save to JSONL
    output_file = "/home/ubuntu/RL_Practice/generation_comparison.jsonl"
    with open(output_file, "w") as f:
        f.write(json.dumps(comparison_data, indent=2) + "\n")
    
    print(f"\n✅ Saved complete comparison to: {output_file}")
    
    # Print summary
    print(f"\n=== SUMMARY ===")
    print(f"Prompt: {prompt_data['prompt'][:100]}...")
    print(f"collect_rollouts responses: {len(collect_results.get('responses', []))}")
    print(f"SequenceProcessor responses: {len(sp_results.get('responses', []))}")
    
    if 'responses' in collect_results and len(collect_results['responses']) > 0:
        print(f"\nFirst collect_rollouts response:")
        print(f"  Text: {collect_results['responses'][0]['response_text'][:150]}...")
    
    if 'responses' in sp_results and len(sp_results['responses']) > 0:
        print(f"\nFirst SequenceProcessor response:")
        print(f"  Text: {sp_results['responses'][0]['response_text'][:150]}...")

if __name__ == "__main__":
    main()