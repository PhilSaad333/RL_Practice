#!/usr/bin/env python3
"""
S3 sync utility for archiving completed RL training work from localfs.

Usage:
    python sync_to_s3.py --action backup --training-run run_2025-08-19_12-34-56
    python sync_to_s3.py --action backup-all
    python sync_to_s3.py --action list-backups
    python sync_to_s3.py --action restore --training-run run_2025-08-19_12-34-56
"""

import os
import sys
import argparse
import pathlib
import subprocess
from typing import List, Optional
import tempfile

# Add project root to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

LOCALFS_ROOT = pathlib.Path(os.environ.get("LOCALFS_ROOT", "/lambda/nfs/localfs"))
TRAINING_ROOT = LOCALFS_ROOT / "training_runs"
EVAL_ROOT = LOCALFS_ROOT / "eval_runs"
ARCHIVE_ROOT = LOCALFS_ROOT / "archive"

# S3 configuration
S3_BUCKET = "lambda_east3:9e733b11-9ff3-41c4-9328-29990fa02ade"
S3_BACKUP_PATH = f"{S3_BUCKET}/rl_backups"


def check_rclone():
    """Check if rclone is available and configured."""
    try:
        result = subprocess.run(["rclone", "version"], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ rclone not found or not working")
            return False
        
        # Test S3 connection
        result = subprocess.run(["rclone", "lsd", S3_BUCKET], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Cannot access S3 bucket: {S3_BUCKET}")
            print(f"Error: {result.stderr}")
            return False
        
        return True
    except FileNotFoundError:
        print("❌ rclone not found. Please install and configure rclone.")
        return False


def find_training_runs() -> List[pathlib.Path]:
    """Find all training runs in localfs."""
    if not TRAINING_ROOT.exists():
        return []
    
    runs = []
    for run_dir in TRAINING_ROOT.iterdir():
        if run_dir.is_dir() and run_dir.name.startswith("run_"):
            runs.append(run_dir)
    
    runs.sort(key=lambda x: x.name)
    return runs


def find_eval_runs() -> List[pathlib.Path]:
    """Find all eval runs in localfs."""
    if not EVAL_ROOT.exists():
        return []
    
    runs = []
    for run_dir in EVAL_ROOT.iterdir():
        if run_dir.is_dir():
            runs.append(run_dir)
    
    runs.sort(key=lambda x: x.name)
    return runs


def backup_training_run(training_run_dir: pathlib.Path, include_eval: bool = True) -> bool:
    """Backup a single training run to S3."""
    run_name = training_run_dir.name
    print(f"📦 Backing up training run: {run_name}")
    
    # Create backup structure in archive
    backup_dir = ARCHIVE_ROOT / "completed_runs" / run_name
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Sync training run
        print(f"   Syncing training data...")
        cmd = [
            "rclone", "sync", str(training_run_dir),
            f"{S3_BACKUP_PATH}/training_runs/{run_name}",
            "--progress", "--transfers=4", "--checkers=4"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Failed to backup training run: {result.stderr}")
            return False
        
        # Find and sync corresponding eval runs
        if include_eval:
            eval_runs = [d for d in find_eval_runs() if run_name in d.name]
            for eval_run_dir in eval_runs:
                print(f"   Syncing eval data: {eval_run_dir.name}")
                cmd = [
                    "rclone", "sync", str(eval_run_dir),
                    f"{S3_BACKUP_PATH}/eval_runs/{eval_run_dir.name}",
                    "--progress", "--transfers=4", "--checkers=4"
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"⚠️ Failed to backup eval run {eval_run_dir.name}: {result.stderr}")
        
        # Create backup marker
        backup_info = {
            "training_run": run_name,
            "backup_date": str(pathlib.Path()),
            "backed_up_to": f"{S3_BACKUP_PATH}/training_runs/{run_name}",
            "include_eval": include_eval
        }
        
        import json
        with open(backup_dir / "backup_info.json", 'w') as f:
            json.dump(backup_info, f, indent=2)
        
        print(f"✅ Successfully backed up {run_name}")
        return True
        
    except Exception as e:
        print(f"❌ Backup failed: {e}")
        return False


def list_s3_backups():
    """List available backups in S3."""
    print(f"📋 Listing backups in {S3_BACKUP_PATH}...")
    
    try:
        # List training runs
        cmd = ["rclone", "lsd", f"{S3_BACKUP_PATH}/training_runs/"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            training_runs = [line.split()[-1] for line in lines if line.strip()]
            
            print(f"\n🎯 Training runs backed up ({len(training_runs)}):")
            for run in sorted(training_runs):
                print(f"   {run}")
        else:
            print("❌ Failed to list training runs")
        
        # List eval runs
        cmd = ["rclone", "lsd", f"{S3_BACKUP_PATH}/eval_runs/"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            eval_runs = [line.split()[-1] for line in lines if line.strip()]
            
            print(f"\n📊 Eval runs backed up ({len(eval_runs)}):")
            for run in sorted(eval_runs):
                print(f"   {run}")
        else:
            print("❌ Failed to list eval runs")
            
    except Exception as e:
        print(f"❌ Failed to list backups: {e}")


def restore_from_s3(run_name: str, include_eval: bool = True) -> bool:
    """Restore a training run from S3."""
    print(f"📥 Restoring training run: {run_name}")
    
    try:
        # Restore training run
        restore_path = TRAINING_ROOT / run_name
        print(f"   Restoring to: {restore_path}")
        
        cmd = [
            "rclone", "sync",
            f"{S3_BACKUP_PATH}/training_runs/{run_name}",
            str(restore_path),
            "--progress", "--transfers=4", "--checkers=4"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Failed to restore training run: {result.stderr}")
            return False
        
        # Restore eval runs if requested
        if include_eval:
            # List eval runs that match this training run
            cmd = ["rclone", "lsd", f"{S3_BACKUP_PATH}/eval_runs/"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                eval_runs = [line.split()[-1] for line in lines if line.strip() and run_name in line]
                
                for eval_run in eval_runs:
                    print(f"   Restoring eval: {eval_run}")
                    eval_restore_path = EVAL_ROOT / eval_run
                    
                    cmd = [
                        "rclone", "sync",
                        f"{S3_BACKUP_PATH}/eval_runs/{eval_run}",
                        str(eval_restore_path),
                        "--progress", "--transfers=4", "--checkers=4"
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        print(f"⚠️ Failed to restore eval run {eval_run}: {result.stderr}")
        
        print(f"✅ Successfully restored {run_name}")
        return True
        
    except Exception as e:
        print(f"❌ Restore failed: {e}")
        return False


def cleanup_local_after_backup(run_name: str, keep_latest_checkpoint: bool = True):
    """Clean up local files after successful backup."""
    training_run_dir = TRAINING_ROOT / run_name
    
    if not training_run_dir.exists():
        return
    
    print(f"🧹 Cleaning up local files for {run_name}")
    
    if keep_latest_checkpoint:
        # Keep only the latest checkpoint and logs
        training_state_dir = training_run_dir / "training_state"
        if training_state_dir.exists():
            latest_link = training_state_dir / "step_latest"
            if latest_link.is_symlink():
                latest_checkpoint = latest_link.resolve()
                
                # Remove all checkpoints except latest
                for checkpoint_dir in training_state_dir.iterdir():
                    if (checkpoint_dir.is_dir() and 
                        checkpoint_dir != latest_checkpoint and 
                        checkpoint_dir.name != "step_latest"):
                        print(f"   Removing checkpoint: {checkpoint_dir.name}")
                        subprocess.run(["rm", "-rf", str(checkpoint_dir)])
        
        print(f"   Kept latest checkpoint and logs")
    else:
        # Move entire run to archive
        archive_dir = ARCHIVE_ROOT / "completed_runs" / run_name
        archive_dir.parent.mkdir(parents=True, exist_ok=True)
        
        if not archive_dir.exists():
            subprocess.run(["mv", str(training_run_dir), str(archive_dir)])
            print(f"   Moved to archive: {archive_dir}")


def main():
    parser = argparse.ArgumentParser(description="S3 sync utility for RL training data")
    parser.add_argument("--action", required=True, 
                       choices=["backup", "backup-all", "list-backups", "restore"],
                       help="Action to perform")
    parser.add_argument("--training-run", help="Specific training run name (for backup/restore)")
    parser.add_argument("--no-eval", action="store_true", help="Don't include eval runs")
    parser.add_argument("--cleanup", action="store_true", help="Clean up local files after backup")
    parser.add_argument("--keep-latest", action="store_true", help="Keep latest checkpoint when cleaning up")
    
    args = parser.parse_args()
    
    # Check rclone
    if not check_rclone():
        return 1
    
    if args.action == "list-backups":
        list_s3_backups()
        return 0
    
    elif args.action == "backup":
        if not args.training_run:
            print("❌ --training-run required for backup action")
            return 1
        
        training_run_dir = TRAINING_ROOT / args.training_run
        if not training_run_dir.exists():
            print(f"❌ Training run not found: {training_run_dir}")
            return 1
        
        success = backup_training_run(training_run_dir, include_eval=not args.no_eval)
        
        if success and args.cleanup:
            cleanup_local_after_backup(args.training_run, args.keep_latest)
        
        return 0 if success else 1
    
    elif args.action == "backup-all":
        training_runs = find_training_runs()
        if not training_runs:
            print("📭 No training runs found to backup")
            return 0
        
        print(f"📦 Found {len(training_runs)} training runs to backup")
        successful = 0
        
        for training_run_dir in training_runs:
            success = backup_training_run(training_run_dir, include_eval=not args.no_eval)
            if success:
                successful += 1
                if args.cleanup:
                    cleanup_local_after_backup(training_run_dir.name, args.keep_latest)
        
        print(f"\n🎉 Backup completed: {successful}/{len(training_runs)} successful")
        return 0 if successful == len(training_runs) else 1
    
    elif args.action == "restore":
        if not args.training_run:
            print("❌ --training-run required for restore action")
            return 1
        
        success = restore_from_s3(args.training_run, include_eval=not args.no_eval)
        return 0 if success else 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())