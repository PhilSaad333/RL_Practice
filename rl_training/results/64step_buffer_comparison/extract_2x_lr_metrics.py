#!/usr/bin/env python3
"""
Simple script to extract 2x LR metrics and add to comparison.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def extract_2x_lr_metrics():
    """Extract metrics from 2x LR run."""
    base_dir = Path(__file__).parent.parent.parent
    results_dir = base_dir / "new_64step_2x_lr_results"
    
    print(f"Looking in: {results_dir}")
    print(f"Exists: {results_dir.exists()}")
    
    if not results_dir.exists():
        print("Results directory not found!")
        return
    
    # Step mapping
    steps = {
        'step_10_gsm8k_r1_template': 10,
        'step_20_gsm8k_r1_template': 20,
        'step_30_gsm8k_r1_template': 30,
        'step_40_gsm8k_r1_template': 40,
        'step_50_gsm8k_r1_template': 50,
        'step_60_gsm8k_r1_template': 60,
        'step_final_gsm8k_r1_template': 64
    }
    
    results = []
    
    for step_name, step_num in steps.items():
        metrics_file = results_dir / step_name / "temp0.7_p1.0_r8" / "metrics.csv"
        
        if metrics_file.exists():
            print(f"Loading {step_name}...")
            df = pd.read_csv(metrics_file)
            print(f"  {len(df)} questions loaded")
            
            # Calculate averages
            result = {
                'experiment_name': '2x_lr_run',
                'config_type': '2x_learning_rate',
                'buffer_size': 256,
                'batch_size': 32,
                'step': step_num,
                'pass_rate': df['pass_rate'].mean(),
                'pass_at_1': df['pass@1'].mean(),
                'pass_at_2': df['pass@2'].mean(),
                'pass_at_4': df['pass@4'].mean(),
                'pass_at_8': df['pass@8'].mean(),
            }
            
            results.append(result)
            print(f"  Pass rate: {result['pass_rate']:.4f}")
        else:
            print(f"Missing: {metrics_file}")
    
    # Create DataFrame and save
    if results:
        new_df = pd.DataFrame(results)
        print(f"\nNew results summary:")
        print(new_df[['step', 'pass_rate', 'pass_at_1', 'pass_at_8']].to_string(index=False))
        
        # Load existing and combine
        existing_file = Path(__file__).parent / "summary_metrics.csv"
        if existing_file.exists():
            existing_df = pd.read_csv(existing_file)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            
            # Save updated file
            output_file = Path(__file__).parent / "updated_summary_metrics.csv"
            combined_df.to_csv(output_file, index=False)
            print(f"\nSaved combined metrics to: {output_file}")
            
            # Show comparison
            print("\nFinal step comparison:")
            final_comparison = combined_df.groupby('config_type').apply(lambda x: x.loc[x['step'].idxmax()]).reset_index(drop=True)
            print(final_comparison[['config_type', 'step', 'pass_rate', 'pass_at_8']].to_string(index=False))
            
            # Calculate improvement
            if len(final_comparison) > 0:
                buffer256_result = final_comparison[final_comparison['config_type'] == 'buffer_size_256']
                lr2x_result = final_comparison[final_comparison['config_type'] == '2x_learning_rate']
                
                if len(buffer256_result) > 0 and len(lr2x_result) > 0:
                    improvement = lr2x_result.iloc[0]['pass_rate'] - buffer256_result.iloc[0]['pass_rate']
                    print(f"\n2x LR improvement over buffer_size_256: {improvement:+.4f}")
        
    else:
        print("No results found!")

if __name__ == "__main__":
    extract_2x_lr_metrics()