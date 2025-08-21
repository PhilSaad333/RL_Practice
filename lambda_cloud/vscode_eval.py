#!/usr/bin/env python3
"""
VS Code integration for Lambda evaluations.

Simple interface to submit and manage remote evaluations from VS Code.

Usage:
    python lambda/vscode_eval.py  # Interactive mode
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lambda_cloud.remote_eval import LambdaEvaluationManager


def interactive_submit():
    """Interactive evaluation submission."""
    print("🚀 Lambda Cloud Evaluation - Interactive Mode")
    print("=" * 50)
    
    manager = LambdaEvaluationManager()
    
    # Check connection
    if not manager.test_connection():
        ip = input("Enter Lambda instance IP: ").strip()
        if ip:
            manager.set_lambda_ip(ip)
            if not manager.test_connection():
                print("❌ Still can't connect. Please check your setup.")
                return
        else:
            print("❌ No IP provided. Exiting.")
            return
    
    print("\n📋 Available Evaluation Profiles:")
    profiles = [
        ("quick_test", "Fast testing with 2% of dataset"),
        ("full_evaluation", "Complete evaluation with full dataset"), 
        ("memory_optimized", "Full dataset with conservative memory"),
        ("high_throughput", "Maximum GPU utilization"),
        ("debug", "Minimal evaluation for debugging")
    ]
    
    for i, (profile, desc) in enumerate(profiles, 1):
        print(f"  {i}. {profile}: {desc}")
    
    print("\n🔧 Evaluation Setup:")
    
    # Get training run
    training_run = input("Training run name (e.g., run_2025-08-20_03-31-43): ").strip()
    if not training_run:
        print("❌ Training run required")
        return
    
    # Get profile
    profile_choice = input(f"Profile (1-{len(profiles)}, default: 2 for full_evaluation): ").strip()
    if not profile_choice:
        profile = "full_evaluation"
    else:
        try:
            profile = profiles[int(profile_choice) - 1][0]
        except (ValueError, IndexError):
            profile = "full_evaluation"
            print(f"Invalid choice, using default: {profile}")
    
    # Get dataset
    dataset = input("Evaluation dataset (default: gsm8k_r1_template): ").strip()
    if not dataset:
        dataset = "gsm8k_r1_template"
    
    # Get specific steps
    steps_input = input("Specific steps to evaluate (comma-separated, or Enter for all): ").strip()
    steps = None
    if steps_input:
        steps = [s.strip() for s in steps_input.split(",")]
    
    # Additional options
    eval_kwargs = {}
    
    batch_size = input("Batch size (default: auto): ").strip()
    if batch_size:
        eval_kwargs["batch_size"] = batch_size
    
    subset_frac = input("Dataset fraction (0.0-1.0, default: from profile): ").strip()
    if subset_frac:
        try:
            eval_kwargs["subset_frac"] = float(subset_frac)
        except ValueError:
            print("Invalid fraction, using profile default")
    
    print(f"\n🎯 Submitting Evaluation:")
    print(f"   Training run: {training_run}")
    print(f"   Profile: {profile}")
    print(f"   Dataset: {dataset}")
    if steps:
        print(f"   Steps: {', '.join(steps)}")
    if eval_kwargs:
        print(f"   Options: {eval_kwargs}")
    
    confirm = input("\nProceed? (y/N): ").strip().lower()
    if confirm != 'y':
        print("❌ Cancelled")
        return
    
    # Submit the job
    job_id = manager.submit_evaluation(
        training_run=training_run,
        profile=profile,
        eval_dataset=dataset,
        steps=steps,
        **eval_kwargs
    )
    
    if job_id:
        print(f"\n✅ Job submitted successfully: {job_id}")
        
        # Ask if user wants to monitor immediately
        monitor = input("Monitor job now? (y/N): ").strip().lower()
        if monitor == 'y':
            print("\n📊 Monitoring job (Ctrl+C to stop)...")
            try:
                while True:
                    status = manager.monitor_job(job_id)
                    if status.get("status") in ["FINISHED", "NO_PID"]:
                        break
                    
                    # Wait before next check
                    import time
                    time.sleep(30)  # Check every 30 seconds
                    
            except KeyboardInterrupt:
                print("\n⏹️  Monitoring stopped (job continues running)")
                print(f"   Resume monitoring with: python lambda/remote_eval.py monitor --job-id {job_id}")


def interactive_menu():
    """Main interactive menu."""
    manager = LambdaEvaluationManager()
    
    while True:
        print("\n🌟 Lambda Cloud Evaluation Manager")
        print("=" * 40)
        print("1. Submit new evaluation")
        print("2. List recent jobs")
        print("3. Monitor job")
        print("4. Sync results")
        print("5. Test connection")
        print("6. Set Lambda IP")
        print("0. Exit")
        
        choice = input("\nSelect option (0-6): ").strip()
        
        if choice == "0":
            print("👋 Goodbye!")
            break
        
        elif choice == "1":
            interactive_submit()
        
        elif choice == "2":
            jobs = manager.list_jobs()
            if not jobs:
                print("📭 No jobs found")
            else:
                print(f"\n📋 Recent Jobs ({len(jobs)}):")
                print("-" * 50)
                for i, job in enumerate(jobs[:5], 1):  # Show last 5
                    status = job.get("status", "unknown")
                    created = job.get("created_at", "unknown")[:16]
                    training_run = job.get("training_run", "unknown")
                    
                    print(f"{i}. {job['job_id']} ({status})")
                    print(f"   {training_run} - {created}")
                    if job.get("synced_at"):
                        print(f"   📥 Synced")
        
        elif choice == "3":
            job_id = input("Job ID to monitor: ").strip()
            if job_id:
                manager.monitor_job(job_id)
        
        elif choice == "4":
            job_id = input("Job ID to sync: ").strip()
            if job_id:
                manager.sync_results(job_id)
        
        elif choice == "5":
            manager.test_connection()
        
        elif choice == "6":
            ip = input("Lambda instance IP: ").strip()
            if ip:
                manager.set_lambda_ip(ip)
        
        else:
            print("❌ Invalid choice")


def main():
    """Main entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == "menu":
        interactive_menu()
    else:
        interactive_submit()


if __name__ == "__main__":
    main()