#!/usr/bin/env python3
"""
Debug script to isolate the parsing issue with metrics extraction.

This script:
1. Runs a single probe test via subprocess (same as scaling study)
2. Captures and saves raw output
3. Tests parsing logic on captured output
4. Shows exactly what's happening
"""

import subprocess
import time
from pathlib import Path

def run_single_test_debug(config_path: str, checkpoint_path: str, output_file: str):
    """Run a single probe test and save all output for analysis."""
    cmd = [
        "torchrun", "--nproc_per_node=2", "--master_port=29530",
        "entropy_experiments/run_probe_sanity_check.py",
        "--config", config_path,
        "--checkpoint", checkpoint_path,
        "--runs", "1"
    ]
    
    print(f"🔍 Running command: {' '.join(cmd)}")
    print(f"📁 Output will be saved to: {output_file}")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
    end_time = time.time()
    
    print(f"⏱️  Runtime: {end_time - start_time:.1f}s")
    print(f"📤 Return code: {result.returncode}")
    
    # Save raw output
    with open(output_file, 'w') as f:
        f.write("=== STDOUT ===\n")
        f.write(result.stdout)
        f.write("\n\n=== STDERR ===\n")
        f.write(result.stderr)
    
    print(f"💾 Raw output saved to {output_file}")
    
    return result.stdout, result.stderr, end_time - start_time

def test_parsing_logic(stdout_text: str):
    """Test parsing logic on captured output."""
    lines = stdout_text.split('\n')
    
    metrics = {
        'deltaH1': None,
        'SE_conditional': None,
        'relative_se': None
    }
    
    print("\n🔍 PARSING ANALYSIS:")
    print("=" * 50)
    
    relevant_lines = []
    
    for i, line in enumerate(lines):
        # Look for relevant lines
        if any(pattern in line for pattern in ['δH₁:', 'SE_E(δH₁|U):', 'SE/δH₁ =']):
            relevant_lines.append((i+1, line))
            print(f"Line {i+1}: {repr(line)}")
    
    if not relevant_lines:
        print("❌ No relevant lines found!")
        print("\n🔍 Searching for partial matches:")
        for i, line in enumerate(lines):
            if 'δH' in line or 'SE_E' in line or 'SE/' in line:
                print(f"Line {i+1}: {repr(line)}")
        return metrics
    
    print(f"\n📊 Found {len(relevant_lines)} relevant lines")
    print("\n🎯 Testing parsing patterns:")
    
    for line_num, line in relevant_lines:
        # Parse δH₁ value
        if '   δH₁: ' in line:
            try:
                value = float(line.split('   δH₁: ')[1].strip())
                metrics['deltaH1'] = value
                print(f"✅ δH₁ parsed: {value} from line {line_num}")
            except (ValueError, IndexError) as e:
                print(f"❌ δH₁ parsing failed: {e} from line {line_num}")
        
        # Parse SE_E(δH₁|U) value
        elif '   SE_E(δH₁|U): ' in line:
            try:
                value = float(line.split('   SE_E(δH₁|U): ')[1].strip())
                metrics['SE_conditional'] = value
                print(f"✅ SE_E parsed: {value} from line {line_num}")
            except (ValueError, IndexError) as e:
                print(f"❌ SE_E parsing failed: {e} from line {line_num}")
        
        # Parse relative SE
        elif 'SE/δH₁ = ' in line:
            try:
                start = line.find('SE/δH₁ = ') + len('SE/δH₁ = ')
                end = line.find(')', start)
                if end == -1:
                    value_str = line[start:].split()[0]
                else:
                    value_str = line[start:end]
                value = float(value_str)
                metrics['relative_se'] = value
                print(f"✅ Relative SE parsed: {value} from line {line_num}")
            except (ValueError, IndexError) as e:
                print(f"❌ Relative SE parsing failed: {e} from line {line_num}")
    
    return metrics

def main():
    print("🚀 PARSING DEBUG SCRIPT")
    print("=" * 50)
    
    # Use debug config for fast testing
    config_path = "entropy_experiments/configs/debug_metrics_config.yaml"
    checkpoint_path = "/home/ubuntu/localfs/stage3_checkpoints/step_40"
    output_file = "debug_parsing_output.txt"
    
    # Run test
    stdout, stderr, runtime = run_single_test_debug(config_path, checkpoint_path, output_file)
    
    # Test parsing on stderr (where the actual output goes)
    print(f"📊 Stdout length: {len(stdout)} chars")
    print(f"📊 Stderr length: {len(stderr)} chars")
    print("🔍 Using stderr for parsing (where probe output actually goes)")
    metrics = test_parsing_logic(stderr)
    
    # Summary
    print("\n📋 FINAL RESULTS:")
    print("=" * 50)
    print(f"δH₁: {metrics['deltaH1']}")
    print(f"SE_E(δH₁|U): {metrics['SE_conditional']}")
    print(f"Relative SE: {metrics['relative_se']}")
    print(f"Runtime: {runtime:.1f}s")
    
    if all(v is not None for v in metrics.values()):
        print("🎉 SUCCESS: All metrics parsed correctly!")
    else:
        print("❌ FAILURE: Some metrics failed to parse")
        print(f"📁 Check {output_file} for detailed output analysis")

if __name__ == "__main__":
    main()