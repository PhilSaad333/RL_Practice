#!/usr/bin/env python3
"""
Debug VLLM memory usage and cleanup.
Run this to understand memory patterns and test cleanup strategies.
"""

import torch
import psutil
import os
import time
import subprocess
from pathlib import Path

def get_gpu_memory():
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return allocated, reserved
    return 0, 0

def get_vllm_processes():
    """Find all VLLM-related processes."""
    vllm_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'memory_info', 'cmdline']):
        try:
            if any('vllm' in str(arg).lower() for arg in proc.info['cmdline']):
                memory_gb = proc.info['memory_info'].rss / 1024**3
                vllm_processes.append({
                    'pid': proc.info['pid'],
                    'name': proc.info['name'],
                    'memory_gb': memory_gb,
                    'cmdline': ' '.join(proc.info['cmdline'][:3])  # First 3 args
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return vllm_processes

def monitor_memory_usage():
    """Monitor and report memory usage."""
    print("=" * 60)
    print("MEMORY USAGE REPORT")
    print("=" * 60)
    
    # GPU Memory
    allocated, reserved = get_gpu_memory()
    print(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    
    # VLLM Processes
    vllm_procs = get_vllm_processes()
    total_vllm_memory = sum(p['memory_gb'] for p in vllm_procs)
    
    print(f"\nVLLM Processes ({len(vllm_procs)} found):")
    for proc in vllm_procs:
        print(f"  PID {proc['pid']}: {proc['memory_gb']:.2f} GB - {proc['cmdline']}")
    print(f"Total VLLM Memory: {total_vllm_memory:.2f} GB")
    
    # System Memory
    system_mem = psutil.virtual_memory()
    print(f"\nSystem Memory: {system_mem.used / 1024**3:.2f} GB used of {system_mem.total / 1024**3:.2f} GB")
    
    return vllm_procs, allocated, reserved

def test_vllm_cleanup():
    """Test VLLM initialization and cleanup."""
    print("\n" + "=" * 60)
    print("TESTING VLLM CLEANUP")
    print("=" * 60)
    
    print("\n1. Before VLLM initialization:")
    before_procs, before_alloc, before_res = monitor_memory_usage()
    
    # Initialize VLLM
    print("\n2. Initializing VLLM...")
    try:
        from vllm import LLM, SamplingParams
        
        # Small VLLM instance for testing
        llm = LLM(
            model="Qwen/Qwen2.5-1.5B",
            gpu_memory_utilization=0.2,  # Very small for testing
            disable_log_stats=True,
            enforce_eager=True,
            trust_remote_code=True
        )
        
        print("VLLM initialized successfully")
        
        print("\n3. After VLLM initialization:")
        after_procs, after_alloc, after_res = monitor_memory_usage()
        
        # Test generation
        print("\n4. Testing generation...")
        sampling_params = SamplingParams(temperature=0.7, max_tokens=10)
        outputs = llm.generate(["Hello"], sampling_params)
        print(f"Generated: {outputs[0].outputs[0].text}")
        
        # Cleanup
        print("\n5. Cleaning up VLLM...")
        del llm
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        time.sleep(2)  # Wait for processes to cleanup
        
        print("\n6. After cleanup:")
        final_procs, final_alloc, final_res = monitor_memory_usage()
        
        # Analysis
        print("\n" + "=" * 60)
        print("CLEANUP ANALYSIS")
        print("=" * 60)
        print(f"GPU Memory Change: {final_alloc - before_alloc:.2f} GB")
        print(f"VLLM Processes Before: {len(before_procs)}")
        print(f"VLLM Processes After: {len(final_procs)}")
        
        if final_procs:
            print("⚠️  VLLM processes still running after cleanup!")
            for proc in final_procs:
                print(f"  Zombie PID {proc['pid']}: {proc['memory_gb']:.2f} GB")
        else:
            print("✅ All VLLM processes cleaned up successfully")
            
    except ImportError:
        print("VLLM not available for testing")
    except Exception as e:
        print(f"Error during VLLM test: {e}")

def suggest_fixes():
    """Suggest potential fixes based on observations."""
    print("\n" + "=" * 60)
    print("SUGGESTED FIXES")
    print("=" * 60)
    
    print("1. Force kill VLLM processes in cleanup:")
    print("   - Track VLLM PIDs during initialization")
    print("   - Use os.kill() to terminate worker processes")
    print("   - Add process cleanup to VLLMRolloutCollector")
    
    print("\n2. Reduce memory footprint:")
    print("   - Lower gpu_memory_utilization further (0.2 or 0.15)")
    print("   - Use smaller models for initial testing")
    print("   - Reduce max_tokens and context length")
    
    print("\n3. Alternative approaches:")
    print("   - Run VLLM in separate subprocess and kill entire process")
    print("   - Use VLLM server mode with HTTP API")
    print("   - Switch back to standard generation for A100 compatibility")

if __name__ == "__main__":
    print("VLLM Memory Debugging Tool")
    print("This will test VLLM initialization and cleanup patterns")
    input("Press Enter to start...")
    
    # Initial system state
    monitor_memory_usage()
    
    # Test VLLM cleanup
    test_vllm_cleanup()
    
    # Suggestions
    suggest_fixes()