"""
Test script to reproduce the entropy study generation and identify reward computation issues.

This simulates the process used to generate the 32k entropy study datasets.
Run this in Colab with the checkpoint to see where rewards become zero.
"""

import sys
import os
import json
import torch
from pathlib import Path

# Add project root to path  
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_mini_entropy_study():
    """Generate a mini version of the entropy study to debug reward computation."""
    print("=" * 60)
    print("MINI ENTROPY STUDY - REWARD DEBUG")
    print("=" * 60)
    
    try:
        # Import SequenceProcessor and related components
        from sequence_processing.sequence_processor import SequenceProcessor, GenerationConfig
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        print("✅ Imports successful")
        
        # Set up model and tokenizer (you'll need to load your actual checkpoint in Colab)
        model_name = "Qwen/Qwen2.5-1.5B"  # Base model for testing
        print(f"Loading tokenizer from {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # You'll need to replace this with your actual checkpoint in Colab:
        # model = AutoModelForCausalLM.from_pretrained(colab_checkpoint, torch_dtype=torch.bfloat16)
        print("⚠️ Note: In Colab, load your actual checkpoint here")
        print("⚠️ For now, we'll test just the reward computation logic")
        
        # Test dataset loading and reward computation WITHOUT model generation
        print("\n--- Testing Dataset and Reward Computation ---")
        
        # Create processor config
        config = GenerationConfig(
            temperature=0.7,
            top_p=0.995,
            max_new_tokens=200,
            do_sample=True,
            gen_batch_size=4,
            tf_batch_size=8
        )
        
        # Mock processor (we'll test without actual model)
        print("Testing dataset sampling...")
        
        import rlp_datasets
        from rlp_datasets.registry import DATASET_REGISTRY
        
        # Sample a few prompts like the entropy study does
        dataset_name = "gsm8k_r1_template"
        split = "train"
        
        dataset = DATASET_REGISTRY[dataset_name](split=split)
        dataset_list = list(dataset)
        
        # Take first few examples for testing
        num_test = 3
        test_examples = dataset_list[:num_test]
        test_prompts = [ex.question for ex in test_examples]
        
        print(f"✅ Loaded {num_test} test examples")
        for i, ex in enumerate(test_examples):
            print(f"Example {i}: gold answer = '{ex.answer}'")
        
        # Test reward computation with mock responses
        print("\n--- Testing Reward Computation with Mock Responses ---")
        
        from rl_training.rewards.tag_pref import reward_fn, PROMPT2GOLD
        
        # Create mock responses that should get rewards
        mock_responses = [
            f"Let me solve this step by step. <answer>{ex.answer}</answer>",  # Correct answer
            f"I think the answer is <answer>999</answer>",  # Wrong answer
        ]
        
        # Test reward computation for each example
        original_mapping = PROMPT2GOLD.copy()
        
        try:
            all_rewards = []
            for b, example in enumerate(test_examples):
                # Set up PROMPT2GOLD like _compute_rewards does
                prompt_id = b
                PROMPT2GOLD[prompt_id] = example.answer
                
                print(f"\nPrompt {b}:")
                print(f"  Gold answer: '{example.answer}'")
                print(f"  Set PROMPT2GOLD[{prompt_id}] = '{example.answer}'")
                
                # Compute rewards
                try:
                    reward_tensor = reward_fn(prompt_id, mock_responses)
                    rewards = reward_tensor.tolist()
                    all_rewards.append(rewards)
                    
                    print(f"  Mock responses rewards: {rewards}")
                    print(f"  Response 0 (correct): '{mock_responses[0][:50]}...' → {rewards[0]}")
                    print(f"  Response 1 (wrong): '{mock_responses[1][:50]}...' → {rewards[1]}")
                    
                except Exception as e:
                    print(f"  ❌ Reward computation failed: {e}")
                    all_rewards.append([0.0, 0.0])
        
        finally:
            # Restore mapping
            PROMPT2GOLD.clear()
            PROMPT2GOLD.update(original_mapping)
        
        # Summary
        total_rewards = sum(sum(batch) for batch in all_rewards)
        print(f"\n--- SUMMARY ---")
        print(f"Total rewards across all tests: {total_rewards}")
        print(f"All rewards: {all_rewards}")
        
        if total_rewards > 0:
            print("✅ SUCCESS: Reward computation works!")
            print("🔍 The issue is likely in the actual entropy study generation process:")
            print("   1. Check if examples are passed to _compute_rewards")
            print("   2. Check if generate_with_replacement_sampling preserves examples")
            print("   3. Check if the actual generated responses have proper <answer> tags")
            return True
        else:
            print("❌ ISSUE: All rewards are zero even with mock data!")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_actual_entropy_data():
    """Analyze the actual 32k entropy study data to understand the structure."""
    print("\n" + "=" * 60)
    print("ANALYZING ACTUAL ENTROPY STUDY DATA")
    print("=" * 60)
    
    data_files = [
        "entropy_experiments/estimation_experiments/data/entropy_study_T07_32k.json",
        "entropy_experiments/estimation_experiments/data/entropy_study_T1_32k.json"
    ]
    
    for file_path in data_files:
        full_path = project_root / file_path
        print(f"\n📁 Analyzing: {file_path}")
        
        if not full_path.exists():
            print(f"❌ File not found")
            continue
            
        try:
            with open(full_path, 'r') as f:
                data = json.load(f)
            
            print(f"✅ Loaded successfully")
            
            # Check per_sequence_data
            if 'per_sequence_data' in data:
                per_seq = data['per_sequence_data']
                print(f"📊 Sequences: {len(per_seq)}")
                
                if len(per_seq) > 0:
                    first_seq = per_seq[0]
                    print(f"🔑 Fields in first sequence: {list(first_seq.keys())}")
                    
                    # Check rewards specifically
                    if 'reward' in first_seq:
                        all_rewards = [seq.get('reward', 0) for seq in per_seq]
                        unique_rewards = set(all_rewards)
                        non_zero_count = sum(1 for r in all_rewards if r != 0)
                        
                        print(f"🎯 Reward analysis:")
                        print(f"   Total sequences: {len(all_rewards)}")
                        print(f"   Non-zero rewards: {non_zero_count}")
                        print(f"   Unique reward values: {sorted(unique_rewards)}")
                        
                        if non_zero_count == 0:
                            print("❌ CONFIRMED: All rewards are zero!")
                        else:
                            print(f"✅ Found {non_zero_count} non-zero rewards")
                    
                    # Check response format
                    if 'response_text' in first_seq:
                        sample_responses = [per_seq[i].get('response_text', '') for i in range(min(5, len(per_seq)))]
                        print(f"📝 Sample responses:")
                        for i, resp in enumerate(sample_responses):
                            has_answer_tag = '<answer>' in resp and '</answer>' in resp
                            print(f"   Response {i}: Has answer tags: {has_answer_tag}")
                            if has_answer_tag:
                                # Extract answer
                                start = resp.find('<answer>') + 8
                                end = resp.find('</answer>')
                                if start < end:
                                    answer_text = resp[start:end].strip()
                                    print(f"     Extracted answer: '{answer_text}'")
                            print(f"     Response preview: '{resp[:100]}...'")
                    
                    # Check prompt information
                    if 'prompt_text' in first_seq:
                        print("✅ Prompt text available")
                    else:
                        print("❌ No prompt_text field")
                        
            # Check experiment info
            if 'experiment_info' in data:
                exp_info = data['experiment_info']
                print(f"🔬 Experiment info available: {list(exp_info.keys())}")
                if 'generation_params' in exp_info:
                    gen_params = exp_info['generation_params']
                    print(f"   Temperature: {gen_params.get('temperature', 'N/A')}")
                    print(f"   Top-p: {gen_params.get('top_p', 'N/A')}")
            
        except Exception as e:
            print(f"❌ Failed to analyze {file_path}: {e}")

def main():
    """Run the entropy study debug tests."""
    print("🔍 DEBUGGING ENTROPY STUDY REWARD COMPUTATION")
    print("This will help identify why all 32k samples have zero reward")
    print()
    
    # Test 1: Mini entropy study with mock data
    mock_test_passed = test_mini_entropy_study()
    
    # Test 2: Analyze actual entropy study data
    analyze_actual_entropy_data()
    
    print("\n" + "=" * 60)
    print("🎯 DIAGNOSTIC RECOMMENDATIONS")
    print("=" * 60)
    
    if mock_test_passed:
        print("✅ Reward computation logic works correctly")
        print("🔍 Likely causes of zero rewards in entropy study:")
        print("   1. Examples not passed to _compute_rewards in entropy study")
        print("   2. generate_with_replacement_sampling doesn't preserve examples")
        print("   3. Generated responses don't contain <answer>...</answer> tags")
        print("   4. PROMPT2GOLD mapping setup fails in batch processing")
        print("\n💡 Next steps:")
        print("   1. Check how entropy study calls generate_with_replacement_sampling")
        print("   2. Verify examples are passed through the call chain")
        print("   3. Examine actual generated response format")
    else:
        print("❌ Reward computation logic is broken")
        print("💡 Next steps:")
        print("   1. Check math_verify import and dependency")
        print("   2. Verify PROMPT2GOLD mapping setup")
        print("   3. Test tag_pref reward function in isolation")

if __name__ == "__main__":
    main()