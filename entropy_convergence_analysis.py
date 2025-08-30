#!/usr/bin/env python3
"""
Analyze convergence of basic entropy vs RB entropy estimators with batch size.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict

def analyze_entropy_convergence(results_file: str = "/content/entropy_study_results/entropy_study_20250830_074051.json"):
    """
    Compare convergence of basic_entropy_sum vs diag_rb_entropy_sum 
    as batch size increases.
    """
    
    # Load results
    print("Loading results...")
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Extract per-sequence data
    per_seq_data = results['per_sequence_data']
    N = len(per_seq_data)
    print(f"Total sequences: {N}")
    
    # Extract the two entropy measures
    basic_entropies = [seq['basic_entropy_sum'] for seq in per_seq_data]
    rb_entropies = [seq['diag_rb_entropy_sum'] for seq in per_seq_data]
    
    print(f"Basic entropy range: {np.min(basic_entropies):.3f} - {np.max(basic_entropies):.3f}")
    print(f"RB entropy range: {np.min(rb_entropies):.3f} - {np.max(rb_entropies):.3f}")
    print()
    
    # Batch sizes to test
    batch_sizes = [128, 256, 512, 1024]
    
    # Storage for results
    batch_means_basic = []
    batch_means_rb = []
    batch_stds_basic = []
    batch_stds_rb = []
    
    print("Computing batch statistics...")
    
    for B in batch_sizes:
        print(f"Batch size B={B}:")
        
        # How many complete batches can we make?
        n_batches = N // B
        print(f"  Number of complete batches: {n_batches}")
        
        # Compute mean for each batch
        basic_batch_means = []
        rb_batch_means = []
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * B
            end_idx = start_idx + B
            
            # Batch means
            batch_basic_mean = np.mean(basic_entropies[start_idx:end_idx])
            batch_rb_mean = np.mean(rb_entropies[start_idx:end_idx])
            
            basic_batch_means.append(batch_basic_mean)
            rb_batch_means.append(batch_rb_mean)
        
        # Statistics across batches
        basic_mean = np.mean(basic_batch_means)
        basic_std = np.std(basic_batch_means, ddof=1) if len(basic_batch_means) > 1 else 0
        
        rb_mean = np.mean(rb_batch_means) 
        rb_std = np.std(rb_batch_means, ddof=1) if len(rb_batch_means) > 1 else 0
        
        print(f"  Basic entropy - Mean: {basic_mean:.4f}, Std: {basic_std:.4f}")
        print(f"  RB entropy    - Mean: {rb_mean:.4f}, Std: {rb_std:.4f}")
        print(f"  Variance reduction: {(basic_std**2 - rb_std**2) / basic_std**2 * 100:.1f}%")
        print()
        
        batch_means_basic.append(basic_mean)
        batch_means_rb.append(rb_mean)
        batch_stds_basic.append(basic_std)
        batch_stds_rb.append(rb_std)
    
    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Convergence of means
    ax1.plot(batch_sizes, batch_means_basic, 'o-', label='Basic Entropy', linewidth=2, markersize=8)
    ax1.plot(batch_sizes, batch_means_rb, 's-', label='RB Entropy', linewidth=2, markersize=8)
    ax1.set_xlabel('Batch Size B')
    ax1.set_ylabel('Mean Entropy (averaged over batches)')
    ax1.set_title('Convergence of Mean Estimates')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)
    
    # Plot 2: Standard deviation (convergence rate)  
    ax2.plot(batch_sizes, batch_stds_basic, 'o-', label='Basic Entropy', linewidth=2, markersize=8)
    ax2.plot(batch_sizes, batch_stds_rb, 's-', label='RB Entropy', linewidth=2, markersize=8)
    ax2.set_xlabel('Batch Size B')
    ax2.set_ylabel('Standard Deviation (across batches)')
    ax2.set_title('Convergence Rate: Lower is Better')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log')
    
    # Plot 3: Distribution of basic entropies
    ax3.hist(basic_entropies, bins=50, alpha=0.7, density=True, color='blue', label='Basic Entropy')
    ax3.set_xlabel('Basic Entropy Sum')
    ax3.set_ylabel('Density') 
    ax3.set_title('Distribution of Basic Entropy Sum')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add statistics to the plot
    mean_basic = np.mean(basic_entropies)
    std_basic = np.std(basic_entropies)
    ax3.axvline(mean_basic, color='red', linestyle='--', label=f'Mean: {mean_basic:.3f}')
    ax3.axvline(mean_basic + std_basic, color='red', linestyle=':', alpha=0.7, label=f'±1σ: {std_basic:.3f}')
    ax3.axvline(mean_basic - std_basic, color='red', linestyle=':', alpha=0.7)
    ax3.legend()
    
    # Plot 4: Distribution of RB entropies
    ax4.hist(rb_entropies, bins=50, alpha=0.7, density=True, color='green', label='RB Entropy')
    ax4.set_xlabel('RB Entropy Sum')
    ax4.set_ylabel('Density')
    ax4.set_title('Distribution of RB Entropy Sum')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add statistics to the plot
    mean_rb = np.mean(rb_entropies)
    std_rb = np.std(rb_entropies)
    ax4.axvline(mean_rb, color='red', linestyle='--', label=f'Mean: {mean_rb:.3f}')
    ax4.axvline(mean_rb + std_rb, color='red', linestyle=':', alpha=0.7, label=f'±1σ: {std_rb:.3f}')
    ax4.axvline(mean_rb - std_rb, color='red', linestyle=':', alpha=0.7)
    ax4.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Summary table
    print("=" * 80)
    print("CONVERGENCE SUMMARY")
    print("=" * 80)
    print(f"{'Batch Size':<12} {'Basic Mean':<12} {'Basic Std':<12} {'RB Mean':<12} {'RB Std':<12} {'Var Reduction':<15}")
    print("-" * 80)
    
    for i, B in enumerate(batch_sizes):
        var_reduction = (batch_stds_basic[i]**2 - batch_stds_rb[i]**2) / batch_stds_basic[i]**2 * 100
        print(f"{B:<12} {batch_means_basic[i]:<12.4f} {batch_stds_basic[i]:<12.4f} "
              f"{batch_means_rb[i]:<12.4f} {batch_stds_rb[i]:<12.4f} {var_reduction:<15.1f}%")
    
    print("=" * 80)
    print("\nOVERALL STATISTICS:")
    print(f"Basic Entropy - Population mean: {mean_basic:.4f}, std: {std_basic:.4f}")
    print(f"RB Entropy    - Population mean: {mean_rb:.4f}, std: {std_rb:.4f}")
    print(f"Correlation between basic and RB: {np.corrcoef(basic_entropies, rb_entropies)[0,1]:.4f}")
    
    # Test for heavy tails
    print(f"\nHEAVY TAIL ANALYSIS:")
    basic_q99 = np.percentile(basic_entropies, 99)
    basic_q95 = np.percentile(basic_entropies, 95)
    rb_q99 = np.percentile(rb_entropies, 99) 
    rb_q95 = np.percentile(rb_entropies, 95)
    
    print(f"Basic entropy - 95th percentile: {basic_q95:.3f}, 99th percentile: {basic_q99:.3f}")
    print(f"RB entropy    - 95th percentile: {rb_q95:.3f}, 99th percentile: {rb_q99:.3f}")
    print(f"Basic entropy tail ratio (99th/95th): {basic_q99/basic_q95:.3f}")
    print(f"RB entropy tail ratio (99th/95th): {rb_q99/rb_q95:.3f}")

# For Colab usage
def run_analysis():
    """Simple function to run the analysis in Colab."""
    analyze_entropy_convergence()

if __name__ == "__main__":
    analyze_entropy_convergence()