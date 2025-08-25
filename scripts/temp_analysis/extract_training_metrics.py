#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorboard.backend.event_processing import event_accumulator
import json

def extract_scalars_from_tensorboard(tb_file_path, run_name):
    """Extract scalar metrics from tensorboard event files"""
    print(f"Extracting metrics from {run_name}...")
    
    # Create event accumulator
    ea = event_accumulator.EventAccumulator(tb_file_path)
    ea.Reload()
    
    # Get available scalar tags
    scalar_tags = ea.Tags()['scalars']
    print(f"Available scalar tags for {run_name}: {scalar_tags[:10]}...")  # Show first 10
    
    # Extract relevant metrics
    metrics = {}
    
    # GNS metrics
    gns_tags = [tag for tag in scalar_tags if 'gns' in tag.lower()]
    for tag in gns_tags:
        scalar_events = ea.Scalars(tag)
        steps = [event.step for event in scalar_events]
        values = [event.value for event in scalar_events]
        metrics[tag] = {'steps': steps, 'values': values}
    
    # Training loss
    loss_tags = [tag for tag in scalar_tags if 'loss' in tag.lower() or 'Loss' in tag]
    for tag in loss_tags:
        scalar_events = ea.Scalars(tag)
        steps = [event.step for event in scalar_events]
        values = [event.value for event in scalar_events]
        metrics[tag] = {'steps': steps, 'values': values}
    
    # Learning rate
    lr_tags = [tag for tag in scalar_tags if 'lr' in tag.lower() or 'learning_rate' in tag.lower()]
    for tag in lr_tags:
        scalar_events = ea.Scalars(tag)
        steps = [event.step for event in scalar_events]
        values = [event.value for event in scalar_events]
        metrics[tag] = {'steps': steps, 'values': values}
    
    # KL divergence
    kl_tags = [tag for tag in scalar_tags if 'kl' in tag.lower()]
    for tag in kl_tags:
        scalar_events = ea.Scalars(tag)
        steps = [event.step for event in scalar_events]
        values = [event.value for event in scalar_events]
        metrics[tag] = {'steps': steps, 'values': values}
    
    # Entropy metrics
    entropy_tags = [tag for tag in scalar_tags if 'entropy' in tag.lower()]
    for tag in entropy_tags:
        scalar_events = ea.Scalars(tag)
        steps = [event.step for event in scalar_events]
        values = [event.value for event in scalar_events]
        metrics[tag] = {'steps': steps, 'values': values}
    
    return metrics, scalar_tags

def save_metrics_to_csv(metrics, run_name, output_dir):
    """Save extracted metrics to CSV files"""
    
    # Create a comprehensive dataframe
    all_data = []
    
    for metric_name, data in metrics.items():
        for step, value in zip(data['steps'], data['values']):
            all_data.append({
                'run_name': run_name,
                'metric': metric_name,
                'step': step,
                'value': value
            })
    
    df = pd.DataFrame(all_data)
    csv_path = os.path.join(output_dir, f"{run_name}_training_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(all_data)} metric points to {csv_path}")
    
    return df

def plot_gns_comparison(all_metrics, output_dir):
    """Create comparison plots for GNS metrics across runs"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    colors = {'batch64': 'blue', 'batch128': 'red', 'buffer256': 'green'}
    labels = {'batch64': 'Batch Size 64', 'batch128': 'Batch Size 128', 'buffer256': 'Buffer Size 256'}
    
    # Find common GNS metrics
    gns_metrics = set()
    for run_name, metrics in all_metrics.items():
        gns_metrics.update([tag for tag in metrics.keys() if 'gns' in tag.lower()])
    
    gns_metrics = sorted(list(gns_metrics))
    
    for i, metric in enumerate(gns_metrics[:4]):  # Plot first 4 GNS metrics
        ax = axes[i]
        
        for run_name, metrics in all_metrics.items():
            if metric in metrics:
                steps = metrics[metric]['steps']
                values = metrics[metric]['values']
                ax.plot(steps, values, color=colors[run_name], label=labels[run_name], 
                       linewidth=2, marker='o', markersize=4, alpha=0.7)
        
        ax.set_title(f'{metric}')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gns_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    return gns_metrics

def main():
    # Define paths
    tensorboard_dir = "experimental_results/64step_buffer_comparison/tensorboard_data"
    output_dir = "experimental_results/64step_buffer_comparison"
    
    # File mapping (use directories instead of individual files)
    tb_files = {
        'batch64': 'experimental_results/64step_buffer_comparison/tensorboard_logs/batch64',
        'batch128': 'experimental_results/64step_buffer_comparison/tensorboard_logs/batch128',
        'buffer256': 'experimental_results/64step_buffer_comparison/tensorboard_logs/buffer256'
    }
    
    # Extract metrics from all runs
    all_metrics = {}
    all_dataframes = []
    
    for run_name, tb_file in tb_files.items():
        if os.path.exists(tb_file):
            try:
                metrics, all_tags = extract_scalars_from_tensorboard(tb_file, run_name)
                all_metrics[run_name] = metrics
                
                # Save individual run metrics
                df = save_metrics_to_csv(metrics, run_name, output_dir)
                all_dataframes.append(df)
                
                # Save available tags for reference
                with open(os.path.join(output_dir, f"{run_name}_available_tags.json"), 'w') as f:
                    json.dump(all_tags, f, indent=2)
                    
            except Exception as e:
                print(f"Error processing {run_name}: {e}")
        else:
            print(f"File not found: {tb_file}")
    
    # Combine all metrics
    if all_dataframes:
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        combined_df.to_csv(os.path.join(output_dir, 'combined_training_metrics.csv'), index=False)
        print(f"Saved combined metrics with {len(combined_df)} total points")
    
    # Create comparison plots
    if all_metrics:
        gns_metrics = plot_gns_comparison(all_metrics, output_dir)
        print(f"Created comparison plots for GNS metrics: {gns_metrics}")
    
    # Summary statistics
    print("\n=== SUMMARY ===")
    for run_name, metrics in all_metrics.items():
        print(f"\n{run_name}:")
        print(f"  Total metrics tracked: {len(metrics)}")
        gns_count = len([m for m in metrics.keys() if 'gns' in m.lower()])
        print(f"  GNS metrics: {gns_count}")
        
        # Show final GNS values if available
        for metric_name, data in metrics.items():
            if 'gns' in metric_name.lower() and 'noise_to_signal' in metric_name.lower():
                final_value = data['values'][-1] if data['values'] else None
                print(f"    Final {metric_name}: {final_value:.6f}" if final_value else f"    {metric_name}: No data")

if __name__ == "__main__":
    main()