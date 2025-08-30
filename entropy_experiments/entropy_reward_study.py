#!/usr/bin/env python3
"""
Comprehensive Entropy and Reward Study

Analyzes entropy estimation with reduced variance (RB), rewards, and distribution diagnostics.
Uses G=1 (one generation per prompt) to focus on per-sequence analysis.

Features measured per sequence:
- Basic entropy (ordinary entropies) 
- RB entropy (Rao-Blackwell reduced variance estimate)
- Response length in tokens
- Rewards using tag_pref reward function
- Distribution diagnostics for control variates

Study Goals:
- Compare basic vs RB entropy estimates
- Analyze relationship between entropy, rewards, and response characteristics
- Collect features for variance reduction via control variates
- Understand distribution properties (tail behavior, effective support, etc.)
"""

import os
import sys
import json
import gc
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from sequence_processing import SequenceProcessor, GenerationConfig
from entropy_experiments.offline_entropy_probe import load_peft_for_probe


def clear_gpu_memory():
    """Clear GPU memory in Colab - run this if you get OOM errors."""
    import gc
    import torch
    
    print("🧹 Clearing GPU memory...")
    
    # Aggressive multi-round cleanup for stubborn A100 OOM
    for i in range(3):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    
    # Print final memory status
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        print(f"   GPU Memory: {allocated:.1f}GB allocated, {cached:.1f}GB cached")
    
    print("✅ GPU memory cleared!")


def load_model_and_tokenizer(checkpoint_path: str, base_model_id: str = "Qwen/Qwen2.5-1.5B"):
    """Load LoRA model and tokenizer for entropy analysis."""
    print(f"Loading model from checkpoint: {checkpoint_path}")
    print(f"Base model: {base_model_id}")
    
    model = load_peft_for_probe(
        base_id=base_model_id,
        adapter_path=checkpoint_path,
        mode="lora_simple",
        dtype="bf16",
        device_map="cuda",
    )
    
    if hasattr(model, "set_adapter"):
        model.set_adapter("default")
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    
    print("✅ Model and tokenizer loaded successfully")
    return model, tokenizer


def run_entropy_study(
    checkpoint_path: str,
    num_prompts: int = 100,
    dataset_name: str = "gsm8k_r1_template",
    split: str = "train",
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_p: float = 1.0,
    seed: Optional[int] = None,
    output_dir: str = "entropy_study_results"
) -> Dict[str, Any]:
    """
    Run comprehensive entropy and reward study.
    
    Args:
        checkpoint_path: Path to LoRA checkpoint
        num_prompts: Number of prompts to analyze  
        dataset_name: Dataset to sample from
        split: Dataset split
        max_new_tokens: Maximum tokens to generate
        temperature: Generation temperature
        top_p: Nucleus sampling parameter
        seed: Random seed for reproducibility
        output_dir: Directory to save results
        
    Returns:
        Dictionary with comprehensive results
    """
    print("=" * 80)
    print("🧪 ENTROPY AND REWARD STUDY")
    print("=" * 80)
    print(f"Dataset: {dataset_name}:{split}")
    print(f"Prompts: {num_prompts}")
    print(f"G=1 (one generation per prompt)")
    print(f"Max tokens: {max_new_tokens}")
    print(f"Temperature: {temperature}, Top-p: 0.99")
    print(f"Seed: {seed}")
    print("=" * 80)
    
    start_time = datetime.now()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(checkpoint_path)
    model.eval()
    
    # Setup generation config optimized for A100 with G=1
    # A100 has half the RAM of H100, but G=1 vs G=8 gives us 8x memory savings
    # With padding fix, we can increase batch sizes 4x for better A100 utilization
    # Using top_p=0.99 to reduce RB entropy computation cost
    config = GenerationConfig(
        temperature=temperature,
        top_p=0.995,          # Slightly increased from 0.99 for better sampling
        max_new_tokens=max_new_tokens,
        do_sample=True,
        gen_batch_size=64,    # Increased 4x: 16 -> 64 for better A100 utilization
        tf_batch_size=64      # Increased 4x: 16 -> 64 for better A100 utilization
    )
    
    # Initialize sequence processor
    processor = SequenceProcessor(model, tokenizer, config)
    
    # Check GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gpu_memory_used = torch.cuda.memory_allocated() / 1024**3
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory: {gpu_memory_used:.1f}GB / {gpu_memory_total:.1f}GB")
        print()
    
    # Generate sequences with comprehensive metrics (G=1)
    print("🔄 Generating sequences and computing metrics...")
    generation_start = datetime.now()
    
    sequences, logprob_results, diagnostics_results = processor.generate_with_logprobs(
        prompts=None,  # Sample from dataset
        G=1,  # One generation per prompt
        dataset_name=dataset_name,
        split=split,
        num_prompts=num_prompts,
        seed=seed,
        with_grad=False,  # No gradients needed
        compute_rb=True   # Compute RB entropies
    )
    
    # Aggressive memory cleanup for A100
    torch.cuda.empty_cache()
    gc.collect()
    
    generation_duration = (datetime.now() - generation_start).total_seconds()
    print(f"✅ Generation completed in {generation_duration:.1f}s ({generation_duration/60:.1f} min)")
    print(f"Throughput: {num_prompts/generation_duration:.1f} sequences/second")
    print()
    
    # Extract and analyze results
    print("📊 Analyzing results...")
    analysis_start = datetime.now()
    
    results = analyze_entropy_results(
        sequences, logprob_results, diagnostics_results,
        checkpoint_path, num_prompts, generation_duration
    )
    
    analysis_duration = (datetime.now() - analysis_start).total_seconds()
    total_duration = (datetime.now() - start_time).total_seconds()
    
    # Add timing information
    results["timing"] = {
        "total_duration_seconds": total_duration,
        "generation_duration_seconds": generation_duration,
        "analysis_duration_seconds": analysis_duration,
        "sequences_per_second": num_prompts / generation_duration,
        "start_time": start_time.isoformat(),
        "end_time": datetime.now().isoformat()
    }
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Main results file
    results_file = os.path.join(output_dir, f"entropy_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"💾 Results saved to: {results_file}")
    
    # Summary statistics
    print_summary(results)
    
    return results


def analyze_entropy_results(
    sequences, logprob_results, diagnostics_results,
    checkpoint_path: str, num_prompts: int, generation_duration: float
) -> Dict[str, Any]:
    """Analyze and summarize all collected metrics."""
    
    B = len(sequences.responses_text)  # Number of prompts
    G = 1  # One generation per prompt
    
    # Extract per-sequence metrics
    per_sequence_data = []
    
    for b in range(B):
        for g in range(G):  # G=1, so just one iteration
            # Basic sequence info
            seq_data = {
                "prompt_idx": b,
                "generation_idx": g,
                "response_text": sequences.responses_text[b][g],
                "response_length_tokens": sequences.gen_lens[b][g],
                "prompt_length_tokens": sequences.prompt_lens[b]
            }
            
            # Entropy metrics
            if len(logprob_results.entropies[b]) > g:
                basic_entropies = logprob_results.entropies[b][g]  # Per-token entropies
                rb_entropies = logprob_results.rb_entropies[b][g]  # Per-token RB entropies
                
                seq_data.update({
                    "basic_entropy_mean": float(np.mean(basic_entropies)) if len(basic_entropies) > 0 else 0.0,
                    "basic_entropy_sum": float(np.sum(basic_entropies)) if len(basic_entropies) > 0 else 0.0,
                    "rb_entropy_mean": float(np.mean(rb_entropies)) if len(rb_entropies) > 0 else 0.0,
                    "rb_entropy_sum": float(np.sum(rb_entropies)) if len(rb_entropies) > 0 else 0.0,
                    "entropy_reduction": float(np.mean(basic_entropies) - np.mean(rb_entropies)) if len(basic_entropies) > 0 and len(rb_entropies) > 0 else 0.0,
                    "sequence_logprob": logprob_results.sequence_logprobs[b][g]
                })
            else:
                seq_data.update({
                    "basic_entropy_mean": 0.0,
                    "basic_entropy_sum": 0.0,
                    "rb_entropy_mean": 0.0,
                    "rb_entropy_sum": 0.0,
                    "entropy_reduction": 0.0,
                    "sequence_logprob": 0.0
                })
            
            # Rewards
            seq_data["reward"] = logprob_results.rewards[b][g] if len(logprob_results.rewards) > b and len(logprob_results.rewards[b]) > g else 0.0
            
            # Diagnostics (comprehensive)
            if len(diagnostics_results.diagnostics[b]) > g:
                diag_pack = diagnostics_results.diagnostics[b][g]
                
                # Sequence-level diagnostics
                seq_diag = diag_pack.seq
                seq_data.update({
                    "diag_T": seq_diag.T,
                    "diag_rb_entropy_sum": seq_diag.rb_entropy_sum,
                    "diag_rb_entropy_mean": seq_diag.rb_entropy_mean,
                    "diag_rb_entropy_max": seq_diag.rb_entropy_max,
                    "diag_rb_entropy_min": seq_diag.rb_entropy_min,
                    "diag_early_rb_entropy_mean": seq_diag.early_rb_entropy_mean,
                    "diag_late_rb_entropy_mean": seq_diag.late_rb_entropy_mean,
                    "diag_naive_surprisal_sum": seq_diag.naive_surprisal_sum,
                    "diag_naive_surprisal_mean": seq_diag.naive_surprisal_mean,
                    "diag_margin_mean": seq_diag.margin_mean,
                    "diag_margin_sum": seq_diag.margin_sum,
                    "diag_top1_prob_mean": seq_diag.top1_prob_mean,
                    "diag_collision_mean": seq_diag.collision_mean,
                    "diag_renyi2_mean": seq_diag.renyi2_mean,
                    "diag_eff_support_mean": seq_diag.eff_support_mean,
                    "diag_eos_prob_mean": seq_diag.eos_prob_mean
                })
            else:
                # Empty diagnostics
                seq_data.update({
                    "diag_T": 0, "diag_rb_entropy_sum": 0.0, "diag_rb_entropy_mean": 0.0,
                    "diag_rb_entropy_max": 0.0, "diag_rb_entropy_min": 0.0,
                    "diag_early_rb_entropy_mean": 0.0, "diag_late_rb_entropy_mean": 0.0,
                    "diag_naive_surprisal_sum": 0.0, "diag_naive_surprisal_mean": 0.0,
                    "diag_margin_mean": 0.0, "diag_margin_sum": 0.0, "diag_top1_prob_mean": 0.0,
                    "diag_collision_mean": 0.0, "diag_renyi2_mean": 0.0, "diag_eff_support_mean": 0.0,
                    "diag_eos_prob_mean": None
                })
            
            per_sequence_data.append(seq_data)
    
    # Compute summary statistics
    summary_stats = compute_summary_statistics(per_sequence_data)
    
    # Compile comprehensive results
    results = {
        "experiment_info": {
            "experiment_type": "entropy_reward_study",
            "checkpoint_path": checkpoint_path,
            "num_prompts": num_prompts,
            "G": G,
            "dataset": "gsm8k_r1_template",
            "split": "train"
        },
        "per_sequence_data": per_sequence_data,
        "summary_statistics": summary_stats,
        "data_shapes": {
            "num_sequences": len(per_sequence_data),
            "avg_response_length": np.mean([seq["response_length_tokens"] for seq in per_sequence_data]),
            "avg_prompt_length": np.mean([seq["prompt_length_tokens"] for seq in per_sequence_data])
        }
    }
    
    return results


def compute_summary_statistics(per_sequence_data: List[Dict]) -> Dict[str, Any]:
    """Compute summary statistics across all sequences."""
    
    if not per_sequence_data:
        return {}
    
    # Extract arrays for analysis
    basic_entropies = [seq["basic_entropy_mean"] for seq in per_sequence_data]
    rb_entropies = [seq["rb_entropy_mean"] for seq in per_sequence_data]
    entropy_reductions = [seq["entropy_reduction"] for seq in per_sequence_data]
    rewards = [seq["reward"] for seq in per_sequence_data]
    response_lengths = [seq["response_length_tokens"] for seq in per_sequence_data]
    
    # Summary statistics
    stats = {
        "basic_entropy": {
            "mean": float(np.mean(basic_entropies)),
            "std": float(np.std(basic_entropies)),
            "min": float(np.min(basic_entropies)),
            "max": float(np.max(basic_entropies)),
            "median": float(np.median(basic_entropies))
        },
        "rb_entropy": {
            "mean": float(np.mean(rb_entropies)),
            "std": float(np.std(rb_entropies)),
            "min": float(np.min(rb_entropies)),
            "max": float(np.max(rb_entropies)),
            "median": float(np.median(rb_entropies))
        },
        "entropy_reduction": {
            "mean": float(np.mean(entropy_reductions)),
            "std": float(np.std(entropy_reductions)),
            "min": float(np.min(entropy_reductions)),
            "max": float(np.max(entropy_reductions)),
            "median": float(np.median(entropy_reductions))
        },
        "rewards": {
            "mean": float(np.mean(rewards)),
            "std": float(np.std(rewards)),
            "min": float(np.min(rewards)),
            "max": float(np.max(rewards)),
            "success_rate": float(np.mean([r > 0 for r in rewards])),
            "perfect_rate": float(np.mean([r >= 1.0 for r in rewards]))
        },
        "response_lengths": {
            "mean": float(np.mean(response_lengths)),
            "std": float(np.std(response_lengths)),
            "min": float(np.min(response_lengths)),
            "max": float(np.max(response_lengths)),
            "median": float(np.median(response_lengths))
        }
    }
    
    # Correlation analysis
    stats["correlations"] = {
        "entropy_vs_reward": float(np.corrcoef(basic_entropies, rewards)[0,1]) if len(basic_entropies) > 1 else 0.0,
        "rb_entropy_vs_reward": float(np.corrcoef(rb_entropies, rewards)[0,1]) if len(rb_entropies) > 1 else 0.0,
        "entropy_vs_length": float(np.corrcoef(basic_entropies, response_lengths)[0,1]) if len(basic_entropies) > 1 else 0.0,
        "reward_vs_length": float(np.corrcoef(rewards, response_lengths)[0,1]) if len(rewards) > 1 else 0.0
    }
    
    return stats


def print_summary(results: Dict[str, Any]):
    """Print a summary of the study results."""
    
    stats = results["summary_statistics"]
    timing = results["timing"]
    
    print("=" * 80)
    print("📈 STUDY RESULTS SUMMARY")
    print("=" * 80)
    
    print(f"🕒 Duration: {timing['total_duration_seconds']:.1f}s ({timing['total_duration_seconds']/60:.1f} min)")
    print(f"⚡ Throughput: {timing['sequences_per_second']:.1f} sequences/sec")
    print()
    
    print("🎯 ENTROPY ANALYSIS:")
    print(f"  Basic Entropy:     {stats['basic_entropy']['mean']:.4f} ± {stats['basic_entropy']['std']:.4f}")
    print(f"  RB Entropy:        {stats['rb_entropy']['mean']:.4f} ± {stats['rb_entropy']['std']:.4f}")
    print(f"  Variance Reduction: {stats['entropy_reduction']['mean']:.4f} ± {stats['entropy_reduction']['std']:.4f}")
    print()
    
    print("🎁 REWARD ANALYSIS:")
    print(f"  Mean Reward:       {stats['rewards']['mean']:.4f} ± {stats['rewards']['std']:.4f}")
    print(f"  Success Rate:      {stats['rewards']['success_rate']:.1%} (reward > 0)")
    print(f"  Perfect Rate:      {stats['rewards']['perfect_rate']:.1%} (reward = 1.0)")
    print()
    
    print("📝 RESPONSE CHARACTERISTICS:")
    print(f"  Avg Length:        {stats['response_lengths']['mean']:.1f} ± {stats['response_lengths']['std']:.1f} tokens")
    print(f"  Length Range:      {stats['response_lengths']['min']:.0f} - {stats['response_lengths']['max']:.0f} tokens")
    print()
    
    print("🔗 CORRELATIONS:")
    print(f"  Entropy vs Reward:   {stats['correlations']['entropy_vs_reward']:.3f}")
    print(f"  RB Entropy vs Reward: {stats['correlations']['rb_entropy_vs_reward']:.3f}")
    print(f"  Entropy vs Length:   {stats['correlations']['entropy_vs_length']:.3f}")
    print(f"  Reward vs Length:    {stats['correlations']['reward_vs_length']:.3f}")
    
    print("=" * 80)


def main():
    """Main entry point optimized for Google Colab."""
    
    # Colab checkpoint path (mounted Google Drive)
    colab_checkpoint = "/content/drive/MyDrive/RL_Practice_Files/new_rl_checkpoint/step_60/model"
    
    # Run study with Colab-optimized parameters
    results = run_entropy_study(
        checkpoint_path=colab_checkpoint,
        num_prompts=500,  # Increased for A100 capability
        dataset_name="gsm8k_r1_template",
        split="train",
        max_new_tokens=200,
        temperature=1.0,
        top_p=0.99,       # Reduced to make RB entropy computation cheaper
        seed=None,  # No seed for natural sampling
        output_dir="/content/entropy_study_results"  # Colab-accessible path
    )
    
    print("🎉 Entropy and reward study completed successfully!")
    return results


def run_colab_study(num_prompts: int = 500, max_new_tokens: int = 200):
    """Convenience function for Colab execution with common parameters."""
    
    checkpoint_path = "/content/drive/MyDrive/RL_Practice_Files/new_rl_checkpoint/step_60/model"
    
    print(f"🚀 Running entropy study on Google Colab A100")
    print(f"📊 {num_prompts} prompts, {max_new_tokens} max tokens")
    print(f"🔧 Ultra-conservative A100: gen=16, tf=16, top_p=0.99")
    
    return run_entropy_study(
        checkpoint_path=checkpoint_path,
        num_prompts=num_prompts,
        max_new_tokens=max_new_tokens,
        top_p=0.99,  # A100 optimization
        output_dir="/content/entropy_study_results"
    )


if __name__ == "__main__":
    main()