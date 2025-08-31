#!/usr/bin/env python3
"""
Create SVG graph of entropy variance results with error bars.
"""

import json
import math

def create_entropy_variance_svg():
    """Create SVG visualization of jackknife entropy variance results."""
    
    # Load data
    with open('extensive_jackknife_results.jsonl', 'r') as f:
        data = json.load(f)
    
    batch_sizes = data['summary']['batch_sizes']
    mean_entropies = data['summary']['mean_entropies'] 
    std_devs = data['summary']['std_devs']
    
    print("Data loaded:")
    for i, (B, mean, std) in enumerate(zip(batch_sizes, mean_entropies, std_devs)):
        print(f"B={B}: {mean:.4f} ± {std:.4f}")
    
    # SVG dimensions and margins
    width = 800
    height = 600
    margin = {"top": 80, "right": 100, "bottom": 100, "left": 100}
    plot_width = width - margin["left"] - margin["right"]
    plot_height = height - margin["top"] - margin["bottom"]
    
    # Data ranges
    x_min, x_max = min(batch_sizes), max(batch_sizes)
    y_min = min(mean_entropies) - max(std_devs) - 0.1
    y_max = max(mean_entropies) + max(std_devs) + 0.1
    
    # Scaling functions
    def x_scale(x):
        return margin["left"] + (math.log10(x) - math.log10(x_min)) / (math.log10(x_max) - math.log10(x_min)) * plot_width
    
    def y_scale(y):
        return height - margin["bottom"] - (y - y_min) / (y_max - y_min) * plot_height
    
    # Start SVG
    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <!-- Gradients and filters -->
    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#f8f9fa"/>
      <stop offset="100%" stop-color="#e9ecef"/>
    </linearGradient>
    
    <filter id="shadow">
      <feDropShadow dx="2" dy="2" stdDeviation="1" flood-opacity="0.3"/>
    </filter>
    
    <filter id="glow">
      <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
      <feMerge>
        <feMergeNode in="coloredBlur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
  </defs>
  
  <!-- Background -->
  <rect width="{width}" height="{height}" fill="url(#bgGrad)"/>
  
  <!-- Plot area background -->
  <rect x="{margin['left']}" y="{margin['top']}" width="{plot_width}" height="{plot_height}" 
        fill="#ffffff" stroke="#dee2e6" stroke-width="1" filter="url(#shadow)"/>
  
  <!-- Grid lines -->
  <g stroke="#e9ecef" stroke-width="1" opacity="0.7">'''
    
    # Y-axis grid lines
    for i in range(5):
        y_val = y_min + (y_max - y_min) * i / 4
        y_pos = y_scale(y_val)
        svg += f'''
    <line x1="{margin['left']}" y1="{y_pos}" x2="{margin['left'] + plot_width}" y2="{y_pos}"/>'''
    
    # X-axis grid lines (log scale)
    for B in batch_sizes:
        x_pos = x_scale(B)
        svg += f'''
    <line x1="{x_pos}" y1="{margin['top']}" x2="{x_pos}" y2="{margin['top'] + plot_height}"/>'''
    
    svg += '''
  </g>
  
  <!-- Axes -->
  <g stroke="#495057" stroke-width="2" fill="none">'''
    
    # X and Y axes
    x_axis_y = margin['top'] + plot_height
    y_axis_x = margin['left']
    svg += f'''
    <line x1="{y_axis_x}" y1="{margin['top']}" x2="{y_axis_x}" y2="{x_axis_y}"/>
    <line x1="{y_axis_x}" y1="{x_axis_y}" x2="{margin['left'] + plot_width}" y2="{x_axis_y}"/>'''
    
    svg += '''
  </g>
  
  <!-- Error bars -->
  <g stroke="#6c757d" stroke-width="2" fill="none">'''
    
    for B, mean, std in zip(batch_sizes, mean_entropies, std_devs):
        x_pos = x_scale(B)
        y_mean = y_scale(mean)
        y_upper = y_scale(mean + std)
        y_lower = y_scale(mean - std)
        
        # Vertical error bar
        svg += f'''
    <line x1="{x_pos}" y1="{y_lower}" x2="{x_pos}" y2="{y_upper}"/>
    <!-- Error bar caps -->
    <line x1="{x_pos-5}" y1="{y_upper}" x2="{x_pos+5}" y2="{y_upper}"/>
    <line x1="{x_pos-5}" y1="{y_lower}" x2="{x_pos+5}" y2="{y_lower}"/>'''
    
    svg += '''
  </g>
  
  <!-- Data points -->'''
    
    # Draw line connecting points
    svg += '''
  <polyline points="'''
    for B, mean in zip(batch_sizes, mean_entropies):
        x_pos = x_scale(B)
        y_pos = y_scale(mean)
        svg += f'{x_pos},{y_pos} '
    svg += '''" fill="none" stroke="#0066cc" stroke-width="3" opacity="0.7"/>
  
  <!-- Data point circles -->
  <g>'''
    
    for B, mean, std in zip(batch_sizes, mean_entropies, std_devs):
        x_pos = x_scale(B)
        y_pos = y_scale(mean)
        svg += f'''
    <circle cx="{x_pos}" cy="{y_pos}" r="6" fill="#0066cc" stroke="#ffffff" stroke-width="2" filter="url(#shadow)"/>'''
    
    svg += '''
  </g>
  
  <!-- Axis labels -->
  <g font-family="Arial, sans-serif" font-size="12" fill="#495057" text-anchor="middle">'''
    
    # Y-axis labels
    for i in range(5):
        y_val = y_min + (y_max - y_min) * i / 4
        y_pos = y_scale(y_val)
        svg += f'''
    <text x="{margin['left'] - 15}" y="{y_pos + 4}" text-anchor="end">{y_val:.2f}</text>'''
    
    # X-axis labels
    for B in batch_sizes:
        x_pos = x_scale(B)
        svg += f'''
    <text x="{x_pos}" y="{margin['top'] + plot_height + 25}">{B}</text>'''
    
    svg += '''
  </g>
  
  <!-- Axis titles -->
  <g font-family="Arial, sans-serif" font-size="16" font-weight="bold" fill="#212529">
    <!-- X-axis title -->
    <text x="400" y="550" text-anchor="middle">Batch Size (B)</text>
    
    <!-- Y-axis title (rotated) -->
    <text x="30" y="320" text-anchor="middle" transform="rotate(-90 30 320)">Mean Per-Sequence Entropy</text>
  </g>
  
  <!-- Title -->
  <g font-family="Arial, sans-serif" text-anchor="middle">
    <text x="400" y="35" font-size="20" font-weight="bold" fill="#212529" filter="url(#shadow)">
      Jackknife Entropy Variance: Batch Size Analysis
    </text>
    <text x="400" y="55" font-size="14" fill="#6c757d">
      GSM8K R1 Template • Qwen2.5-1.5B • Step 40 Checkpoint • G=8
    </text>
  </g>
  
  <!-- Legend/Stats box -->
  <g>
    <rect x="620" y="120" width="160" height="140" fill="#ffffff" stroke="#dee2e6" stroke-width="1" rx="5" filter="url(#shadow)"/>
    <text x="700" y="140" font-family="Arial, sans-serif" font-size="14" font-weight="bold" fill="#212529" text-anchor="middle">
      Statistics
    </text>'''
    
    # Add min/max/range stats
    entropy_min = min(mean_entropies)
    entropy_max = max(mean_entropies) 
    entropy_range = entropy_max - entropy_min
    avg_std = sum(std_devs) / len(std_devs)
    
    stats_y = 160
    svg += f'''
    <text x="630" y="{stats_y}" font-family="Arial, sans-serif" font-size="11" fill="#495057">
      Min: {entropy_min:.3f}
    </text>
    <text x="630" y="{stats_y + 15}" font-family="Arial, sans-serif" font-size="11" fill="#495057">
      Max: {entropy_max:.3f}
    </text>
    <text x="630" y="{stats_y + 30}" font-family="Arial, sans-serif" font-size="11" fill="#495057">
      Range: {entropy_range:.3f}
    </text>
    <text x="630" y="{stats_y + 45}" font-family="Arial, sans-serif" font-size="11" fill="#495057">
      Avg Std: {avg_std:.4f}
    </text>
    <text x="630" y="{stats_y + 65}" font-family="Arial, sans-serif" font-size="11" fill="#495057">
      Total Sequences: {sum(B*8 for B in batch_sizes)}
    </text>
    <text x="630" y="{stats_y + 80}" font-family="Arial, sans-serif" font-size="11" fill="#495057">
      Duration: {data['total_duration_seconds']/60:.1f} min
    </text>'''
    
    svg += '''
  </g>
  
  <!-- Watermark -->
  <text x="750" y="580" font-family="Arial, sans-serif" font-size="10" fill="#adb5bd" text-anchor="end">
    Generated by Lord Krang's RL Laboratory
  </text>
</svg>'''
    
    return svg

if __name__ == "__main__":
    svg_content = create_entropy_variance_svg()
    
    output_file = "entropy_vs_batch_size_analysis.svg"
    with open(output_file, 'w') as f:
        f.write(svg_content)
    
    print(f"Created beautiful SVG graph: {output_file}")
    print("Graph shows entropy vs batch size with jackknife error bars")