#!/usr/bin/env python3
"""
Lambda Cloud integration for remote evaluation.

This module provides tools to:
1. Submit evaluation jobs from VS Code to Lambda instances
2. Monitor remote evaluation progress
3. Auto-sync results when complete
4. Manage Lambda instances (start/stop/scale)

Usage from VS Code:
    python lambda/remote_eval.py submit --training-run run_2025-08-20_03-31-43 --profile full_evaluation
    python lambda/remote_eval.py monitor --job-id eval_123
    python lambda/remote_eval.py sync --job-id eval_123
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import yaml
import hashlib

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class LambdaEvaluationManager:
    """
    Manages remote evaluation jobs on Lambda Cloud instances.
    """
    
    def __init__(self, ssh_key_path: str = "~/.ssh/lambda_new", config_file: str = None):
        """
        Initialize Lambda evaluation manager.
        
        Args:
            ssh_key_path: Path to SSH private key for Lambda instances
            config_file: Optional config file for Lambda settings
        """
        self.ssh_key_path = os.path.expanduser(ssh_key_path)
        self.config = self._load_config(config_file)
        self.jobs_dir = Path.home() / ".lambda_eval_jobs"
        self.jobs_dir.mkdir(exist_ok=True)
        
        # Try to load current Lambda IP
        self.lambda_ip = self._get_lambda_ip()
    
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load configuration for Lambda integration."""
        default_config = {
            "remote_workspace": "/home/ubuntu/RL_Practice",
            "remote_localfs": "/home/ubuntu/localfs",
            "ssh_user": "ubuntu",
            "default_timeout": 3600,  # 1 hour default timeout
            "sync_results": True,
            "cleanup_remote": False,  # Keep results on Lambda by default
            "conda_env": "rl"
        }
        
        if config_file and Path(config_file).exists():
            with open(config_file) as f:
                user_config = yaml.safe_load(f) or {}
            default_config.update(user_config)
        
        return default_config
    
    def _get_lambda_ip(self) -> Optional[str]:
        """Try to get current Lambda IP from various sources."""
        # Try ~/.lambda_ip file
        lambda_ip_file = Path.home() / ".lambda_ip"
        if lambda_ip_file.exists():
            try:
                return lambda_ip_file.read_text().strip()
            except Exception:
                pass
        
        # Could add other methods (Lambda API, etc.)
        return None
    
    def set_lambda_ip(self, ip: str):
        """Set the current Lambda instance IP."""
        self.lambda_ip = ip
        
        # Save to file for persistence
        lambda_ip_file = Path.home() / ".lambda_ip"
        lambda_ip_file.write_text(ip)
        print(f"Lambda IP set to: {ip}")
    
    def test_connection(self) -> bool:
        """Test SSH connection to Lambda instance."""
        if not self.lambda_ip:
            print("[ERROR] No Lambda IP configured. Use --set-ip first.")
            return False
        
        try:
            result = subprocess.run([
                "ssh", "-i", self.ssh_key_path, "-o", "ConnectTimeout=10",
                f"{self.config['ssh_user']}@{self.lambda_ip}",
                "echo 'Connection successful'"
            ], capture_output=True, text=True, timeout=15)
            
            if result.returncode == 0:
                print(f"Successfully connected to Lambda instance: {self.lambda_ip}")
                return True
            else:
                print(f"Failed to connect to {self.lambda_ip}")
                print(f"   Error: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"[ERROR] Connection error: {e}")
            return False
    
    def sync_code(self) -> bool:
        """Sync latest code to Lambda instance."""
        if not self.lambda_ip:
            print("[ERROR] No Lambda IP configured")
            return False
        
        print("🔄 Syncing code to Lambda instance...")
        
        try:
            # Git pull on Lambda
            result = subprocess.run([
                "ssh", "-i", self.ssh_key_path,
                f"{self.config['ssh_user']}@{self.lambda_ip}",
                f"cd {self.config['remote_workspace']} && git pull"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print("[OK] Code sync successful")
                return True
            else:
                print(f"[ERROR] Code sync failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"[ERROR] Sync error: {e}")
            return False
    
    def submit_evaluation(
        self, 
        training_run: str,
        profile: str = "full_evaluation",
        eval_dataset: str = "gsm8k_r1_template",
        steps: Optional[List[str]] = None,
        **eval_kwargs
    ) -> Optional[str]:
        """
        Submit an evaluation job to Lambda.
        
        Args:
            training_run: Training run name (e.g., "run_2025-08-20_03-31-43")
            profile: Evaluation profile to use
            eval_dataset: Dataset to evaluate on
            steps: Specific steps to evaluate (None = all steps)
            **eval_kwargs: Additional evaluation parameters
        
        Returns:
            Job ID if successful, None otherwise
        """
        if not self.lambda_ip:
            print("[ERROR] No Lambda IP configured")
            return None
        
        # Generate unique job ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_hash = hashlib.md5(f"{training_run}_{profile}_{timestamp}".encode()).hexdigest()[:8]
        job_id = f"eval_{timestamp}_{job_hash}"
        
        # Create job configuration
        job_config = {
            "job_id": job_id,
            "created_at": datetime.now().isoformat(),
            "lambda_ip": self.lambda_ip,
            "training_run": training_run,
            "profile": profile,
            "eval_dataset": eval_dataset,
            "steps": steps,
            "eval_kwargs": eval_kwargs,
            "status": "submitted"
        }
        
        # Build remote command
        cmd_parts = [
            f"cd {self.config['remote_workspace']}",
            f"source ~/miniconda3/etc/profile.d/conda.sh && conda activate {self.config['conda_env']}",
            "PYTHONPATH=.",
            "nohup python rl_training/runners/enhanced_eval_batch.py",
            f"--training-run {training_run}",
            f"--profile {profile}",
            f"--eval-dataset {eval_dataset}"
        ]
        
        if steps:
            cmd_parts.append(f"--steps {','.join(steps)}")
        
        # Add any additional evaluation arguments
        for key, value in eval_kwargs.items():
            key_formatted = key.replace('_', '-')
            cmd_parts.append(f"--{key_formatted} {value}")
        
        # Add output redirection
        log_file = f"/tmp/eval_{job_id}.log"
        cmd_parts.extend([
            f"> {log_file} 2>&1 &",
            f"echo $! > /tmp/eval_{job_id}.pid"
        ])
        
        remote_cmd = " && ".join(cmd_parts)
        job_config["remote_command"] = remote_cmd
        job_config["remote_log_file"] = log_file
        job_config["remote_pid_file"] = f"/tmp/eval_{job_id}.pid"
        
        print(f"🚀 Submitting evaluation job: {job_id}")
        print(f"   Training run: {training_run}")
        print(f"   Profile: {profile}")
        print(f"   Dataset: {eval_dataset}")
        if steps:
            print(f"   Steps: {', '.join(steps)}")
        
        try:
            # Sync code first
            if not self.sync_code():
                print("[ERROR] Code sync failed - aborting job submission")
                return None
            
            # Submit the job
            result = subprocess.run([
                "ssh", "-i", self.ssh_key_path,
                f"{self.config['ssh_user']}@{self.lambda_ip}",
                remote_cmd
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                job_config["status"] = "running"
                job_config["submitted_at"] = datetime.now().isoformat()
                
                # Save job configuration locally
                job_file = self.jobs_dir / f"{job_id}.json"
                with open(job_file, 'w') as f:
                    json.dump(job_config, f, indent=2)
                
                print(f"[OK] Job submitted successfully!")
                print(f"   Job ID: {job_id}")
                print(f"   Monitor with: python lambda/remote_eval.py monitor --job-id {job_id}")
                print(f"   Sync results with: python lambda/remote_eval.py sync --job-id {job_id}")
                
                return job_id
            else:
                print(f"[ERROR] Job submission failed: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"[ERROR] Submission error: {e}")
            return None
    
    def monitor_job(self, job_id: str) -> Dict[str, Any]:
        """
        Monitor a remote evaluation job.
        
        Args:
            job_id: Job ID to monitor
        
        Returns:
            Job status information
        """
        job_file = self.jobs_dir / f"{job_id}.json"
        if not job_file.exists():
            print(f"[ERROR] Job {job_id} not found")
            return {}
        
        # Load job config
        with open(job_file) as f:
            job_config = json.load(f)
        
        if not self.lambda_ip:
            self.lambda_ip = job_config.get("lambda_ip")
        
        print(f"📊 Monitoring job: {job_id}")
        print(f"   Training run: {job_config['training_run']}")
        print(f"   Profile: {job_config['profile']}")
        print(f"   Submitted: {job_config.get('submitted_at', 'Unknown')}")
        
        try:
            # Check if process is still running
            pid_file = job_config["remote_pid_file"]
            pid_check = subprocess.run([
                "ssh", "-i", self.ssh_key_path,
                f"{self.config['ssh_user']}@{self.lambda_ip}",
                f"if [ -f {pid_file} ]; then pid=$(cat {pid_file}) && ps -p $pid > /dev/null && echo 'RUNNING' || echo 'FINISHED'; else echo 'NO_PID'; fi"
            ], capture_output=True, text=True, timeout=10)
            
            if pid_check.returncode == 0:
                status = pid_check.stdout.strip()
                
                # Get recent log output
                log_file = job_config["remote_log_file"]
                log_check = subprocess.run([
                    "ssh", "-i", self.ssh_key_path,
                    f"{self.config['ssh_user']}@{self.lambda_ip}",
                    f"if [ -f {log_file} ]; then tail -10 {log_file}; else echo 'No log file found'; fi"
                ], capture_output=True, text=True, timeout=10)
                
                job_status = {
                    "job_id": job_id,
                    "status": status,
                    "recent_logs": log_check.stdout if log_check.returncode == 0 else "Failed to get logs",
                    "checked_at": datetime.now().isoformat()
                }
                
                # Update job config
                job_config.update(job_status)
                with open(job_file, 'w') as f:
                    json.dump(job_config, f, indent=2)
                
                print(f"📊 Status: {status}")
                if status == "RUNNING":
                    print("🔄 Job is still running...")
                elif status == "FINISHED":
                    print("[OK] Job completed!")
                    print("💡 Use sync command to download results")
                
                print(f"\n📝 Recent logs:")
                print(job_status["recent_logs"])
                
                return job_status
            else:
                print(f"[ERROR] Failed to check job status: {pid_check.stderr}")
                return {}
                
        except Exception as e:
            print(f"[ERROR] Monitor error: {e}")
            return {}
    
    def sync_results(self, job_id: str, local_dir: Optional[str] = None) -> bool:
        """
        Sync evaluation results from Lambda to local machine.
        
        Args:
            job_id: Job ID to sync results for
            local_dir: Local directory to sync to (default: ./data/remote_eval_results/)
        
        Returns:
            True if sync successful, False otherwise
        """
        job_file = self.jobs_dir / f"{job_id}.json"
        if not job_file.exists():
            print(f"[ERROR] Job {job_id} not found")
            return False
        
        # Load job config
        with open(job_file) as f:
            job_config = json.load(f)
        
        if not self.lambda_ip:
            self.lambda_ip = job_config.get("lambda_ip")
        
        # Determine local sync directory
        if local_dir is None:
            local_dir = Path("data/remote_eval_results") / job_id
        else:
            local_dir = Path(local_dir)
        
        local_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📥 Syncing results for job: {job_id}")
        print(f"   From: {self.lambda_ip}")
        print(f"   To: {local_dir}")
        
        try:
            # Sync evaluation results
            training_run = job_config["training_run"]
            eval_dataset = job_config["eval_dataset"]
            
            remote_eval_dir = f"{self.config['remote_localfs']}/eval_runs"
            
            # Use rsync to sync results
            sync_cmd = [
                "rsync", "-avz", "--progress",
                "-e", f"ssh -i {self.ssh_key_path}",
                f"{self.config['ssh_user']}@{self.lambda_ip}:{remote_eval_dir}/",
                str(local_dir / "eval_runs/")
            ]
            
            print(f"🔄 Running sync command...")
            result = subprocess.run(sync_cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("[OK] Results sync successful!")
                
                # Also sync the job log
                log_file = job_config["remote_log_file"]
                log_sync = subprocess.run([
                    "scp", "-i", self.ssh_key_path,
                    f"{self.config['ssh_user']}@{self.lambda_ip}:{log_file}",
                    str(local_dir / f"job_{job_id}.log")
                ], capture_output=True, text=True, timeout=30)
                
                if log_sync.returncode == 0:
                    print(f"📝 Job log synced to: {local_dir / f'job_{job_id}.log'}")
                
                # Update job status
                job_config["synced_at"] = datetime.now().isoformat()
                job_config["local_results_dir"] = str(local_dir)
                with open(job_file, 'w') as f:
                    json.dump(job_config, f, indent=2)
                
                print(f"📁 Results available at: {local_dir}")
                return True
            else:
                print(f"[ERROR] Sync failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"[ERROR] Sync error: {e}")
            return False
    
    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all evaluation jobs."""
        jobs = []
        
        for job_file in self.jobs_dir.glob("*.json"):
            try:
                with open(job_file) as f:
                    job_config = json.load(f)
                jobs.append(job_config)
            except Exception as e:
                print(f"[WARNING]  Failed to load job file {job_file}: {e}")
        
        # Sort by creation time (newest first)
        jobs.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        return jobs
    
    def cleanup_job(self, job_id: str, remove_remote: bool = False):
        """
        Clean up job files.
        
        Args:
            job_id: Job ID to clean up
            remove_remote: Whether to remove remote files on Lambda
        """
        job_file = self.jobs_dir / f"{job_id}.json"
        
        if remove_remote and self.lambda_ip:
            # Load job config to get remote file paths
            try:
                with open(job_file) as f:
                    job_config = json.load(f)
                
                # Remove remote log and pid files
                cleanup_cmd = f"rm -f {job_config['remote_log_file']} {job_config['remote_pid_file']}"
                subprocess.run([
                    "ssh", "-i", self.ssh_key_path,
                    f"{self.config['ssh_user']}@{self.lambda_ip}",
                    cleanup_cmd
                ], capture_output=True, timeout=10)
                
                print(f"🧹 Cleaned up remote files for job {job_id}")
                
            except Exception as e:
                print(f"[WARNING]  Failed to cleanup remote files: {e}")
        
        # Remove local job file
        if job_file.exists():
            job_file.unlink()
            print(f"🧹 Removed local job file: {job_id}")


def main():
    parser = argparse.ArgumentParser(description="Lambda Cloud evaluation integration")
    parser.add_argument("--ssh-key", default="~/.ssh/lambda_new", help="SSH private key path")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Set IP command
    ip_parser = subparsers.add_parser("set-ip", help="Set Lambda instance IP")
    ip_parser.add_argument("ip", help="Lambda instance IP address")
    
    # Test connection
    test_parser = subparsers.add_parser("test", help="Test connection to Lambda")
    
    # Submit evaluation
    submit_parser = subparsers.add_parser("submit", help="Submit evaluation job")
    submit_parser.add_argument("--training-run", required=True, help="Training run name")
    submit_parser.add_argument("--profile", default="full_evaluation", help="Evaluation profile")
    submit_parser.add_argument("--eval-dataset", default="gsm8k_r1_template", help="Evaluation dataset")
    submit_parser.add_argument("--steps", help="Comma-separated list of steps to evaluate")
    submit_parser.add_argument("--subset-frac", type=float, help="Fraction of dataset to use")
    submit_parser.add_argument("--batch-size", help="Batch size (or 'auto')")
    
    # Monitor job
    monitor_parser = subparsers.add_parser("monitor", help="Monitor evaluation job")
    monitor_parser.add_argument("--job-id", required=True, help="Job ID to monitor")
    
    # Sync results
    sync_parser = subparsers.add_parser("sync", help="Sync evaluation results")
    sync_parser.add_argument("--job-id", required=True, help="Job ID to sync")
    sync_parser.add_argument("--local-dir", help="Local directory to sync to")
    
    # List jobs
    list_parser = subparsers.add_parser("list", help="List all jobs")
    
    # Cleanup
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up job files")
    cleanup_parser.add_argument("--job-id", required=True, help="Job ID to clean up")
    cleanup_parser.add_argument("--remove-remote", action="store_true", help="Remove remote files too")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize manager
    manager = LambdaEvaluationManager(ssh_key_path=args.ssh_key)
    
    if args.command == "set-ip":
        manager.set_lambda_ip(args.ip)
    
    elif args.command == "test":
        manager.test_connection()
    
    elif args.command == "submit":
        eval_kwargs = {}
        if args.subset_frac is not None:
            eval_kwargs["subset_frac"] = args.subset_frac
        if args.batch_size:
            eval_kwargs["batch_size"] = args.batch_size
        
        steps = None
        if args.steps:
            steps = [s.strip() for s in args.steps.split(",")]
        
        job_id = manager.submit_evaluation(
            training_run=args.training_run,
            profile=args.profile,
            eval_dataset=args.eval_dataset,
            steps=steps,
            **eval_kwargs
        )
        
        if job_id:
            print(f"\n💡 Next steps:")
            print(f"   Monitor: python lambda/remote_eval.py monitor --job-id {job_id}")
            print(f"   Sync: python lambda/remote_eval.py sync --job-id {job_id}")
    
    elif args.command == "monitor":
        manager.monitor_job(args.job_id)
    
    elif args.command == "sync":
        manager.sync_results(args.job_id, args.local_dir)
    
    elif args.command == "list":
        jobs = manager.list_jobs()
        if not jobs:
            print("📭 No jobs found")
        else:
            print(f"📋 Found {len(jobs)} evaluation jobs:")
            print("-" * 80)
            for job in jobs:
                status = job.get("status", "unknown")
                created = job.get("created_at", "unknown")[:16]  # Show date/time
                training_run = job.get("training_run", "unknown")
                profile = job.get("profile", "unknown")
                
                print(f"🔹 {job['job_id']}")
                print(f"   Status: {status} | Created: {created}")
                print(f"   Training: {training_run} | Profile: {profile}")
                
                if job.get("synced_at"):
                    print(f"   📥 Synced: {job['synced_at'][:16]}")
                print()
    
    elif args.command == "cleanup":
        manager.cleanup_job(args.job_id, args.remove_remote)


if __name__ == "__main__":
    main()