"""
Debug script to isolate the reward computation issue.

This script tests each step of the reward computation pipeline:
1. Dataset loading and gold answer extraction
2. PROMPT2GOLD mapping setup
3. Sample response generation simulation
4. reward_fn computation

Run in Colab with the checkpoint to debug the zero rewards issue.
"""

import sys
import os
import json
import torch
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_dataset_loading():
    """Test 1: Verify dataset loads correctly and has gold answers."""
    print("=" * 60)
    print("TEST 1: Dataset Loading and Gold Answer Extraction")
    print("=" * 60)
    
    try:
        import rlp_datasets
        from rlp_datasets.registry import DATASET_REGISTRY
        
        # Load dataset like sequence_processor does
        dataset_name = "gsm8k_r1_template"
        split = "train"
        
        print(f"Loading dataset: {dataset_name}:{split}")
        dataset = DATASET_REGISTRY[dataset_name](split=split)
        dataset_list = list(dataset)
        
        print(f"✅ Loaded {len(dataset_list)} examples")
        
        # Check first few examples
        for i in range(min(3, len(dataset_list))):
            example = dataset_list[i]
            print(f"\n--- Example {i} ---")
            print(f"Type: {type(example)}")
            print(f"Has .answer attribute: {hasattr(example, 'answer')}")
            print(f"Has .question attribute: {hasattr(example, 'question')}")
            print(f"Has .text attribute: {hasattr(example, 'text')}")
            print(f"Has .meta attribute: {hasattr(example, 'meta')}")
            
            if hasattr(example, 'answer'):
                answer = example.answer
                print(f"Gold answer: '{answer}' (type: {type(answer)}, len: {len(answer) if answer else 0})")
                print(f"Is None: {answer is None}")
                print(f"Is empty: {answer == '' if answer is not None else 'N/A'}")
            
            if hasattr(example, 'question'):
                question = example.question
                print(f"Question (first 100 chars): '{question[:100]}...' (len: {len(question)})")
        
        return dataset_list[:5]  # Return first 5 for testing
        
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_reward_function_setup():
    """Test 2: Verify reward function works with manual setup."""
    print("=" * 60)
    print("TEST 2: Reward Function Setup and PROMPT2GOLD")
    print("=" * 60)
    
    try:
        from rl_training.rewards.tag_pref import reward_fn, PROMPT2GOLD, set_prompt2gold
        
        # Clear existing mapping
        PROMPT2GOLD.clear()
        print(f"Cleared PROMPT2GOLD. Current size: {len(PROMPT2GOLD)}")
        
        # Test manual setup
        test_gold = "42"
        test_mapping = {0: test_gold}
        set_prompt2gold(test_mapping)
        
        print(f"✅ Set PROMPT2GOLD: {dict(PROMPT2GOLD)}")
        
        # Test responses with different formats
        test_responses = [
            "The answer is <answer>42</answer>",  # Correct format and answer
            "The answer is <answer>43</answer>",  # Correct format, wrong answer  
            "The answer is 42",  # Missing tags
            "<answer>42</answer>",  # Just the answer
            "<answer></answer>",  # Empty answer
            "No answer provided",  # No answer at all
        ]
        
        print("\nTesting reward computation:")
        for i, response in enumerate(test_responses):
            try:
                reward_tensor = reward_fn(0, [response])  # prompt_id=0
                reward = reward_tensor.item()
                print(f"Response {i}: '{response}' → reward: {reward}")
            except Exception as e:
                print(f"Response {i}: '{response}' → ERROR: {e}")
        
        # Test with missing prompt_id
        try:
            reward_tensor = reward_fn(999, ["<answer>42</answer>"])  # Non-existent prompt_id
            reward = reward_tensor.item()
            print(f"Missing prompt_id test: reward = {reward} (should be 0)")
        except Exception as e:
            print(f"Missing prompt_id test: ERROR: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed reward function test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sequence_processor_integration(examples: List):
    """Test 3: Test SequenceProcessor._compute_rewards with real examples."""
    print("=" * 60)
    print("TEST 3: SequenceProcessor._compute_rewards Integration")
    print("=" * 60)
    
    if not examples:
        print("❌ No examples provided, skipping integration test")
        return False
    
    try:
        # Simulate what SequenceProcessor does
        from rl_training.rewards.tag_pref import reward_fn, PROMPT2GOLD
        
        # Mock prompt and sequences data
        B = min(2, len(examples))  # Test with 2 examples
        G = 2  # 2 generations per prompt
        
        prompts = [examples[i].question for i in range(B)]
        
        # Mock BatchedSequences.responses_text structure: [B][G] list of strings
        mock_responses = [
            [
                f"I think the answer is <answer>{examples[i].answer}</answer>",  # Correct response
                f"Let me try <answer>999</answer>",  # Wrong response
            ]
            for i in range(B)
        ]
        
        print(f"Testing with {B} prompts, {G} responses each")
        for i in range(B):
            print(f"Prompt {i} gold answer: '{examples[i].answer}'")
            print(f"Prompt {i} responses: {mock_responses[i]}")
        
        # Store original mapping
        original_mapping = PROMPT2GOLD.copy()
        
        try:
            # Simulate _compute_rewards logic
            all_rewards = []
            
            for b in range(B):
                # Get gold answer (like sequence_processor does)
                gold_answer = None
                if examples and b < len(examples):
                    gold_answer = examples[b].answer
                
                print(f"\nBatch {b}: gold_answer = '{gold_answer}'")
                
                if gold_answer is None:
                    print(f"Batch {b}: No gold answer → zero rewards")
                    all_rewards.append([0.0] * G)
                    continue
                
                # Set up PROMPT2GOLD mapping
                prompt_id = b
                PROMPT2GOLD[prompt_id] = gold_answer
                print(f"Batch {b}: Set PROMPT2GOLD[{prompt_id}] = '{gold_answer}'")
                
                # Get responses for this prompt
                responses = mock_responses[b]
                
                # Compute rewards using tag_pref function
                try:
                    reward_tensor = reward_fn(prompt_id, responses)
                    rewards = reward_tensor.tolist()
                    print(f"Batch {b}: rewards = {rewards}")
                except Exception as e:
                    rewards = [0.0] * G
                    print(f"Batch {b}: reward computation failed: {e}")
                
                all_rewards.append(rewards)
        
        finally:
            # Restore original mapping
            PROMPT2GOLD.clear()
            PROMPT2GOLD.update(original_mapping)
        
        print(f"\n✅ Final rewards structure: {all_rewards}")
        
        # Check if we got any non-zero rewards
        total_rewards = sum(sum(batch_rewards) for batch_rewards in all_rewards)
        print(f"Total rewards across all sequences: {total_rewards}")
        
        if total_rewards > 0:
            print("✅ SUCCESS: Got some non-zero rewards!")
            return True
        else:
            print("❌ ISSUE: All rewards are zero!")
            return False
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_entropy_study_data_structure():
    """Test 4: Check the actual entropy study data structure."""
    print("=" * 60)
    print("TEST 4: Entropy Study Data Structure Analysis")
    print("=" * 60)
    
    # Check if entropy study data files exist and examine structure
    data_files = [
        "entropy_experiments/estimation_experiments/data/entropy_study_T07_32k.json",
        "entropy_experiments/estimation_experiments/data/entropy_study_T1_32k.json"
    ]
    
    for file_path in data_files:
        full_path = project_root / file_path
        print(f"\nChecking: {file_path}")
        
        if not full_path.exists():
            print(f"❌ File not found: {full_path}")
            continue
        
        try:
            with open(full_path, 'r') as f:
                data = json.load(f)
            
            print(f"✅ File loaded successfully")
            print(f"Top-level keys: {list(data.keys())}")
            
            if 'per_sequence_data' in data:
                per_seq = data['per_sequence_data']
                print(f"Number of sequences: {len(per_seq)}")
                
                if len(per_seq) > 0:
                    first_seq = per_seq[0]
                    print(f"First sequence keys: {list(first_seq.keys())}")
                    
                    # Check for reward field
                    if 'reward' in first_seq:
                        rewards = [seq.get('reward', 0) for seq in per_seq[:100]]  # First 100
                        non_zero_rewards = sum(1 for r in rewards if r != 0)
                        print(f"Rewards in first 100 sequences:")
                        print(f"  Non-zero rewards: {non_zero_rewards}/100")
                        print(f"  Sample rewards: {rewards[:10]}")
                    else:
                        print("❌ No 'reward' field found in sequence data")
                        
                    # Check for other relevant fields
                    relevant_fields = ['prompt', 'response', 'prompt_text', 'response_text', 'basic_entropy_sum', 'diag_rb_entropy_sum']
                    for field in relevant_fields:
                        if field in first_seq:
                            print(f"✅ Found field: '{field}'")
                        else:
                            print(f"❌ Missing field: '{field}'")
            
            if 'experiment_info' in data:
                exp_info = data['experiment_info']
                print(f"Experiment info keys: {list(exp_info.keys())}")
                
        except Exception as e:
            print(f"❌ Failed to analyze {file_path}: {e}")

def main():
    """Run all debug tests."""
    print("🔍 REWARD COMPUTATION DEBUG ANALYSIS")
    print("This script will test each step of the reward computation pipeline")
    print()
    
    # Test 1: Dataset loading
    examples = test_dataset_loading()
    
    # Test 2: Reward function
    reward_fn_works = test_reward_function_setup()
    
    # Test 3: Integration test
    if examples and reward_fn_works:
        integration_works = test_sequence_processor_integration(examples)
    else:
        print("⏭️ Skipping integration test due to previous failures")
        integration_works = False
    
    # Test 4: Check actual entropy study data
    test_entropy_study_data_structure()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Dataset loading: {'✅ PASS' if examples else '❌ FAIL'}")
    print(f"Reward function: {'✅ PASS' if reward_fn_works else '❌ FAIL'}")
    print(f"Integration test: {'✅ PASS' if integration_works else '❌ FAIL'}")
    
    if examples and reward_fn_works and integration_works:
        print("\n🎯 DIAGNOSIS: Reward computation pipeline works correctly!")
        print("   The issue is likely in how the entropy study was generated.")
        print("   Check if examples were properly passed to _compute_rewards.")
    elif not examples:
        print("\n🚨 DIAGNOSIS: Dataset loading failed!")
        print("   Check if rlp_datasets is properly installed and gsm8k_r1_template exists.")
    elif not reward_fn_works:
        print("\n🚨 DIAGNOSIS: Reward function is broken!")
        print("   Check imports and math_verify dependency.")
    else:
        print("\n🚨 DIAGNOSIS: Integration issues!")
        print("   The individual parts work but integration fails.")

if __name__ == "__main__":
    main()