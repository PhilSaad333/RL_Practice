#!/usr/bin/env python3
"""
Script to sync training and evaluation data from Lambda instances for local analysis.

Usage:
    python scripts/sync_lambda_data.py --ip 192.222.55.105 --run run_2025-08-19_12-34-56
    python scripts/sync_lambda_data.py --ip 192.222.55.105 --all
    python scripts/sync_lambda_data.py --ip 192.222.55.105 --metrics-only
"""

import argparse
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description=""):
    """Run a shell command and return success status"""
    print(f"[*] {description}")
    print(f"   Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"[+] Success")
            return True
        else:
            print(f"[-] Failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"[-] Error: {e}")
        return False

def sync_training_run(ip, run_name, data_dir):
    """Sync a specific training run"""
    local_path = data_dir / "runs" / run_name
    local_path.mkdir(parents=True, exist_ok=True)
    
    # Sync training run
    cmd = [
        "scp", "-r", "-i", "~/.ssh/lambda_new",
        "-o", "StrictHostKeyChecking=no", 
        "-o", "UserKnownHostsFile=/dev/null",
        f"ubuntu@{ip}:/home/ubuntu/localfs/training_runs/{run_name}/",
        str(local_path.parent)
    ]
    
    if not run_command(cmd, f"Syncing training run {run_name}"):
        return False
    
    # Sync corresponding evaluation run
    eval_local_path = data_dir / "eval_runs"
    eval_local_path.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "scp", "-r", "-i", "~/.ssh/lambda_new", 
        "-o", "StrictHostKeyChecking=no", 
        "-o", "UserKnownHostsFile=/dev/null",
        f"ubuntu@{ip}:/home/ubuntu/localfs/eval_runs/{run_name}_*",
        str(eval_local_path)
    ]
    
    run_command(cmd, f"Syncing evaluation results for {run_name}")
    return True

def sync_metrics_only(ip, data_dir):
    """Sync only key metrics files"""
    metrics_dir = data_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    # Sync consolidated metrics
    cmd = [
        "scp", "-i", "~/.ssh/lambda_new",
        "-o", "StrictHostKeyChecking=no", 
        "-o", "UserKnownHostsFile=/dev/null",
        f"ubuntu@{ip}:/home/ubuntu/localfs/eval_runs/*/consolidated_metrics.csv",
        str(metrics_dir)
    ]
    run_command(cmd, "Syncing consolidated metrics")
    
    # Sync training logs
    cmd = [
        "scp", "-i", "~/.ssh/lambda_new",
        "-o", "StrictHostKeyChecking=no", 
        "-o", "UserKnownHostsFile=/dev/null",
        f"ubuntu@{ip}:/home/ubuntu/localfs/training_runs/*/logs/train_log.jsonl",
        str(metrics_dir)
    ]
    run_command(cmd, "Syncing training logs")

def list_remote_runs(ip):
    """List available runs on Lambda instance"""
    print(f"[*] Available runs on {ip}:")
    
    # List training runs
    cmd = [
        "ssh", "-i", "~/.ssh/lambda_new", 
        "-o", "StrictHostKeyChecking=no", 
        "-o", "UserKnownHostsFile=/dev/null",
        f"ubuntu@{ip}",
        "ls -la /home/ubuntu/localfs/training_runs/ | grep run_"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            training_runs = [line.split()[-1] for line in lines if 'run_' in line]
            print(f"   Training runs: {training_runs}")
            return training_runs
        else:
            print(f"   Could not list runs: {result.stderr}")
            return []
    except Exception as e:
        print(f"   Error: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description="Sync data from Lambda instances")
    parser.add_argument("--ip", required=True, help="Lambda instance IP address")
    parser.add_argument("--run", help="Specific run name to sync")
    parser.add_argument("--all", action="store_true", help="Sync all available runs")
    parser.add_argument("--metrics-only", action="store_true", help="Sync only metrics files")
    parser.add_argument("--list", action="store_true", help="List available runs")
    
    args = parser.parse_args()
    
    # Setup paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    data_dir = project_dir / "data"
    
    print(f"[*] Syncing data from Lambda instance: {args.ip}")
    print(f"[*] Local data directory: {data_dir}")
    
    # List runs if requested
    if args.list:
        list_remote_runs(args.ip)
        return
    
    # Metrics only sync
    if args.metrics_only:
        sync_metrics_only(args.ip, data_dir)
        return
    
    # Get available runs
    available_runs = list_remote_runs(args.ip)
    
    if args.all:
        # Sync all runs
        success_count = 0
        for run_name in available_runs:
            if sync_training_run(args.ip, run_name, data_dir):
                success_count += 1
        
        print(f"\\n[+] Synced {success_count}/{len(available_runs)} runs successfully")
        
    elif args.run:
        # Sync specific run
        if args.run in available_runs:
            sync_training_run(args.ip, args.run, data_dir)
        else:
            print(f"[-] Run '{args.run}' not found on instance")
            print(f"   Available runs: {available_runs}")
    
    else:
        print("[-] Please specify --run, --all, --metrics-only, or --list")
        sys.exit(1)
    
    print(f"\\n[+] Data synced to {data_dir}")
    print("[*] Use the notebooks/training_analysis.ipynb notebook to analyze the data")

if __name__ == "__main__":
    main()