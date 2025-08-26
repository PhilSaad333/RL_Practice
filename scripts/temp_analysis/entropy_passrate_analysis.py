#!/usr/bin/env python3
"""
Analysis of covariance between mean_entropy and pass_rate across training steps
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

def load_metrics_from_checkpoints(results_dir):
    """Load metrics.csv from each checkpoint step"""
    results_path = Path(results_dir)
    
    all_data = []
    step_pattern = re.compile(r'step_(\d+|final)_')
    
    for step_dir in sorted(results_path.glob('step_*_gsm8k_r1_template')):
        # Extract step number or 'final'
        match = step_pattern.search(step_dir.name)
        if not match:
            continue
        step = match.group(1)
        
        metrics_file = step_dir / 'temp0.7_p1.0_r8' / 'metrics.csv'
        if metrics_file.exists():
            df = pd.read_csv(metrics_file)
            df['step'] = step
            all_data.append(df)
            print(f"Loaded {len(df)} samples from {step}")
    
    return pd.concat(all_data, ignore_index=True)

def calculate_step_level_metrics(df):
    """Calculate covariance and correlation at each step"""
    step_metrics = []
    
    for step in df['step'].unique():
        step_data = df[df['step'] == step]
        
        # Calculate covariance and correlation
        cov_matrix = np.cov(step_data['entropy_mean'], step_data['pass_rate'])
        covariance = cov_matrix[0, 1]
        correlation = np.corrcoef(step_data['entropy_mean'], step_data['pass_rate'])[0, 1]
        
        # Calculate means
        mean_entropy = step_data['entropy_mean'].mean()
        mean_pass_rate = step_data['pass_rate'].mean()
        
        step_metrics.append({
            'step': step,
            'covariance': covariance,
            'correlation': correlation,
            'mean_entropy': mean_entropy,
            'mean_pass_rate': mean_pass_rate,
            'n_samples': len(step_data)
        })
    
    return pd.DataFrame(step_metrics)

def create_visualizations(df, step_metrics):
    """Create comprehensive visualizations"""
    # Convert step to numeric for plotting (handle 'final' as last step)
    numeric_steps = []
    for step in step_metrics['step']:
        if step == 'final':
            numeric_steps.append(64)  # Assuming final is step 64
        else:
            numeric_steps.append(int(step))
    step_metrics['step_numeric'] = numeric_steps
    
    # Sort by step for plotting
    step_metrics = step_metrics.sort_values('step_numeric')
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Entropy-Pass Rate Analysis Across Training Steps', fontsize=16)
    
    # 1. Covariance across steps
    axes[0, 0].plot(step_metrics['step_numeric'], step_metrics['covariance'], 'o-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Training Step')
    axes[0, 0].set_ylabel('Covariance(entropy_mean, pass_rate)')
    axes[0, 0].set_title('Covariance Between Mean Entropy and Pass Rate')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # 2. Correlation across steps
    axes[0, 1].plot(step_metrics['step_numeric'], step_metrics['correlation'], 'o-', color='orange', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Training Step')
    axes[0, 1].set_ylabel('Correlation(entropy_mean, pass_rate)')
    axes[0, 1].set_title('Correlation Between Mean Entropy and Pass Rate')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[0, 1].set_ylim(-1, 1)
    
    # 3. Mean entropy and pass rate over steps
    ax3 = axes[1, 0]
    ax3_twin = ax3.twinx()
    
    line1 = ax3.plot(step_metrics['step_numeric'], step_metrics['mean_entropy'], 'o-', color='blue', label='Mean Entropy', linewidth=2, markersize=8)
    line2 = ax3_twin.plot(step_metrics['step_numeric'], step_metrics['mean_pass_rate'], 'o-', color='green', label='Mean Pass Rate', linewidth=2, markersize=8)
    
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('Mean Entropy', color='blue')
    ax3_twin.set_ylabel('Mean Pass Rate', color='green')
    ax3.set_title('Mean Entropy vs Mean Pass Rate Over Training')
    ax3.grid(True, alpha=0.3)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper left')
    
    # 4. Scatter plot of all data points colored by step
    scatter_data = df.copy()
    scatter_data['step_numeric'] = scatter_data['step'].apply(lambda x: 64 if x == 'final' else int(x))
    
    scatter = axes[1, 1].scatter(scatter_data['entropy_mean'], scatter_data['pass_rate'], 
                               c=scatter_data['step_numeric'], cmap='viridis', alpha=0.6, s=30)
    axes[1, 1].set_xlabel('Mean Entropy')
    axes[1, 1].set_ylabel('Pass Rate')
    axes[1, 1].set_title('Entropy vs Pass Rate (All Data Points)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=axes[1, 1])
    cbar.set_label('Training Step')
    
    plt.tight_layout()
    plt.savefig('entropy_passrate_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def print_summary_statistics(step_metrics):
    """Print summary statistics"""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    print(f"Number of training steps analyzed: {len(step_metrics)}")
    print(f"Steps: {sorted(step_metrics['step'].tolist())}")
    
    print(f"\nCovariance Statistics:")
    print(f"  Mean covariance: {step_metrics['covariance'].mean():.6f}")
    print(f"  Std covariance: {step_metrics['covariance'].std():.6f}")
    print(f"  Min covariance: {step_metrics['covariance'].min():.6f} (step {step_metrics.loc[step_metrics['covariance'].idxmin(), 'step']})")
    print(f"  Max covariance: {step_metrics['covariance'].max():.6f} (step {step_metrics.loc[step_metrics['covariance'].idxmax(), 'step']})")
    
    print(f"\nCorrelation Statistics:")
    print(f"  Mean correlation: {step_metrics['correlation'].mean():.6f}")
    print(f"  Std correlation: {step_metrics['correlation'].std():.6f}")
    print(f"  Min correlation: {step_metrics['correlation'].min():.6f} (step {step_metrics.loc[step_metrics['correlation'].idxmin(), 'step']})")
    print(f"  Max correlation: {step_metrics['correlation'].max():.6f} (step {step_metrics.loc[step_metrics['correlation'].idxmax(), 'step']})")
    
    print(f"\nStep-by-step breakdown:")
    for _, row in step_metrics.iterrows():
        print(f"  Step {row['step']:>5}: cov={row['covariance']:>8.5f}, corr={row['correlation']:>8.5f}, "
              f"entropy={row['mean_entropy']:.5f}, pass_rate={row['mean_pass_rate']:.3f}")

def main():
    results_dir = "C:/Users/phils/OneDrive/Documents/GitHub/RL_Practice/new_64step_2x_lr_results"
    
    print("Loading metrics from all checkpoints...")
    df = load_metrics_from_checkpoints(results_dir)
    
    print(f"\nLoaded {len(df)} total samples across {df['step'].nunique()} steps")
    
    print("\nCalculating step-level metrics...")
    step_metrics = calculate_step_level_metrics(df)
    
    print("\nCreating visualizations...")
    fig = create_visualizations(df, step_metrics)
    
    print_summary_statistics(step_metrics)
    
    return df, step_metrics, fig

if __name__ == "__main__":
    df, step_metrics, fig = main()