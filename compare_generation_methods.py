#!/usr/bin/env python3
"""
Compare SequenceProcessor vs collect_rollouts generation side-by-side
to debug why SequenceProcessor generates incorrect responses.
"""

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

def collect_rollouts_style_generation(model, tokenizer, prompt, num_sequences=2):
    """Generate using the exact same parameters as collect_rollouts.py"""
    from sequence_processing.sequence_processor import StopAfterAnswer
    
    # Tokenize prompt
    tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    prompt_ids = tokens.input_ids.to(model.device)
    attn = tokens.attention_mask.to(model.device)
    
    # Create logits processor
    stop_processor = StopAfterAnswer(tokenizer)
    
    print(f"=== COLLECT_ROLLOUTS STYLE ===")
    print(f"Input prompt: {repr(prompt[:100])}...")
    print(f"Input tokens: {prompt_ids.shape}")
    print(f"Device: {prompt_ids.device}")
    
    # Generate exactly like collect_rollouts.py
    old_cache = getattr(model.config, "use_cache", False)
    model.config.use_cache = True
    
    try:
        gen_out = model.generate(
            prompt_ids,
            attention_mask=attn,
            do_sample=True,
            num_return_sequences=num_sequences,
            max_new_tokens=100,
            temperature=0.8,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            logits_processor=[stop_processor],
            return_dict_in_generate=True,
            output_scores=True,
            synced_gpus=False,
        )
        
        # Decode results
        generated_sequences = gen_out.sequences
        responses = []
        
        for i in range(num_sequences):
            full_sequence = generated_sequences[i]
            # Extract only the generated part (after prompt)
            gen_part = full_sequence[prompt_ids.shape[1]:]
            response_text = tokenizer.decode(gen_part, skip_special_tokens=True)
            responses.append(response_text)
            
            print(f"\nResponse {i}:")
            print(f"Generated tokens: {len(gen_part)}")
            print(f"Text: {repr(response_text[:200])}...")
        
        return responses
        
    finally:
        model.config.use_cache = old_cache

def sequence_processor_generation(model, tokenizer, prompt, num_sequences=2):
    """Generate using SequenceProcessor"""
    from sequence_processing import SequenceProcessor, GenerationConfig
    
    print(f"\n=== SEQUENCE PROCESSOR ===")
    print(f"Input prompt: {repr(prompt[:100])}...")
    
    # Create config (without num_return_sequences for now)
    config = GenerationConfig(
        temperature=0.8,
        max_new_tokens=100,
        gen_batch_size=4,
        tf_batch_size=8
    )
    
    # Initialize processor
    processor = SequenceProcessor(model, tokenizer, config)
    
    try:
        # Generate with manual G parameter
        sequences, logprob_results = processor.generate_with_logprobs(
            prompts=[prompt],
            G=num_sequences,
            with_grad=False
        )
        
        responses = []
        if len(sequences.responses_text) > 0:
            responses = sequences.responses_text[0]  # First (and only) prompt's responses
            
            for i, response_text in enumerate(responses):
                print(f"\nResponse {i}:")
                print(f"Text: {repr(response_text[:200])}...")
        
        return responses
        
    except Exception as e:
        print(f"SequenceProcessor failed: {e}")
        import traceback
        traceback.print_exc()
        return []

def main():
    """Compare both generation methods on the same prompt."""
    print("=== GENERATION METHOD COMPARISON ===")
    
    # Load model
    print("Loading model...")
    model, tokenizer = load_model_for_test()
    model.eval()
    
    # Test prompt - use a real dataset sample
    import rlp_datasets
    ds = rlp_datasets.DATASET_REGISTRY["gsm8k_r1_template"]("test")
    test_prompt = ds[5].question  # Use a specific sample
    
    print(f"\nTest prompt: {repr(test_prompt[:150])}...")
    
    # Test both methods
    print(f"\n{'='*60}")
    responses_collect = collect_rollouts_style_generation(model, tokenizer, test_prompt, num_sequences=2)
    
    print(f"\n{'='*60}")
    responses_sp = sequence_processor_generation(model, tokenizer, test_prompt, num_sequences=2)
    
    # Compare results
    print(f"\n{'='*60}")
    print("=== COMPARISON ===")
    print(f"collect_rollouts responses: {len(responses_collect)}")
    print(f"SequenceProcessor responses: {len(responses_sp)}")
    
    for i in range(min(len(responses_collect), len(responses_sp))):
        print(f"\n--- Response {i} ---")
        print(f"collect_rollouts: {repr(responses_collect[i][:100])}")
        print(f"SequenceProcessor: {repr(responses_sp[i][:100])}")
        
        # Check if they match
        if responses_collect[i].strip() == responses_sp[i].strip():
            print("✅ MATCH")
        else:
            print("❌ DIFFERENT")

if __name__ == "__main__":
    main()