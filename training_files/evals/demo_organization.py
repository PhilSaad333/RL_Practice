#!/usr/bin/env python3
"""
Demo script to show the new evaluation result organization structure.
"""

import json
from pathlib import Path
from evals.result_organizer import EvaluationResultOrganizer

def demo_new_structure():
    """Demonstrate the new evaluation result structure."""
    
    print("🏗️  Demonstration: New Evaluation Result Organization")
    print("=" * 60)
    
    # Create a demo organizer
    organizer = EvaluationResultOrganizer(
        eval_run_name="run_2025-08-20_03-31-43",
        eval_dataset="gsm8k_r1_template",
        runs_root="demo_eval_runs"
    )
    
    print(f"📁 Created evaluation structure at: {organizer.run_dir}")
    print()
    
    # Show the directory structure that gets created
    print("📂 New Directory Structure:")
    print(f"{organizer.run_dir}/")
    print(f"├── config.yaml              # Evaluation configuration")
    print(f"├── metadata.json            # Run metadata & timing")
    print(f"├── results/                 # Organized results")
    print(f"│   ├── step_10/")
    print(f"│   │   ├── metrics.csv      # Clean tabular metrics")
    print(f"│   │   ├── predictions.jsonl # Human-readable predictions")
    print(f"│   │   ├── summary.json     # Quick stats for this step")
    print(f"│   │   └── records.jsonl.gz # Compressed raw data")
    print(f"│   ├── step_20/")
    print(f"│   ├── step_final/")
    print(f"│   ├── consolidated_metrics.csv  # All steps together")
    print(f"│   └── consolidated_summary.json # Cross-step analysis")
    print(f"└── logs/                    # Evaluation logs")
    print(f"    ├── step_10.log")
    print(f"    └── evaluation.log")
    print()
    
    # Show example configuration
    demo_config = {
        "training_run": "run_2025-08-20_03-31-43",
        "backbone": "qwen2_5_15",
        "eval_dataset": "gsm8k_r1_template",
        "subset_frac": 1.0,
        "batch_size": "auto",
        "auto_detected_batch_size": 32,
        "tf_micro_batch": 32,
        "temperature": 0.7,
        "num_return_sequences": 8,
        "max_new_tokens": 200,
        "checkpoints_evaluated": ["10", "20", "30", "40", "50", "60", "final"],
        "gpu_info": {
            "device_name": "NVIDIA H100 PCIe",
            "memory_gb": 80
        }
    }
    
    organizer.save_evaluation_config(demo_config)
    
    print("📄 Example config.yaml content:")
    print("```yaml")
    import yaml
    print(yaml.dump(demo_config, default_flow_style=False, sort_keys=False))
    print("```")
    print()
    
    # Show example summary
    demo_summary = {
        "step": "10",
        "timestamp": "2025-08-20T10:30:00Z",
        "num_samples": 1319,
        "pass_rate": 0.742,
        "evaluation_time_seconds": 245.3,
        "gpu_memory_used_gb": 45.2,
        "metrics_summary": {
            "entropy_mean": {"mean": 2.34, "std": 0.89, "min": 0.12, "max": 5.67},
            "response_len_mean": {"mean": 87.3, "std": 34.2, "min": 12, "max": 189}
        }
    }
    
    print("📊 Example step summary.json:")
    print("```json")
    print(json.dumps(demo_summary, indent=2))
    print("```")
    print()
    
    # Show benefits
    print("✨ Benefits of New Organization:")
    print("• 🔍 Easy to find specific step results")
    print("• 📊 Consolidated view across all steps")  
    print("• 🔧 Human-readable predictions for debugging")
    print("• 💾 Compressed raw data for completeness")
    print("• 📈 Quick summaries for monitoring")
    print("• 🚀 Scales to 100+ steps efficiently")
    print("• 🔗 Integration with VS Code + Lambda workflow")
    print()
    
    # Show usage examples
    print("🚀 Usage Examples:")
    print()
    print("# Use enhanced batch evaluation:")
    print("python rl_training/runners/enhanced_eval_batch.py \\")
    print("  --training-run run_2025-08-20_03-31-43 \\")
    print("  --profile full_evaluation")
    print()
    print("# Quick test with auto-batch sizing:")
    print("python rl_training/runners/enhanced_eval_batch.py \\")
    print("  --training-run run_2025-08-20_03-31-43 \\")
    print("  --profile quick_test \\")
    print("  --steps 10,20,final")
    print()
    print("# Check available profiles:")
    print("python rl_training/runners/enhanced_eval_batch.py --list-profiles")
    print()
    
    print("🎯 Next Steps:")
    print("• Test auto-batch sizing on Lambda H100")
    print("• Add progress monitoring with real-time ETA")
    print("• Implement parallel checkpoint evaluation")
    print("• Create visualization dashboard for results")
    
    # Clean up demo directory
    import shutil
    if organizer.run_dir.exists():
        shutil.rmtree(organizer.run_dir.parent)
    
    print(f"\n✅ Demo complete! (cleaned up {organizer.run_dir.parent})")


if __name__ == "__main__":
    demo_new_structure()