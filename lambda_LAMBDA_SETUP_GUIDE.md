# Lambda Cloud GPU Training Setup Guide

Complete guide for setting up and running RL training on Lambda Cloud GPU instances.

## Quick Start Commands

### 1. Launch New Instance
```powershell
# Start new instance and run full setup
.\ssh_workflow.ps1 -InstanceIP <NEW_IP> -Action full-setup -S3UUID "8c7f7fd3-ba01-40d8-b3dd-92090e4b3b0a"
```

**IMPORTANT:** Code syncing is done via git clone from GitHub, NOT file uploads. This ensures consistency and avoids file corruption issues.

### Quick Setup Checklist for New Instance
1. **Accept conda terms**: `~/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && ~/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r`
2. **Clone fresh code**: `rm -rf ~/RL_Practice && git clone https://github.com/PhilSaad333/RL_Practice.git`
3. **Install dependencies**: `~/miniconda3/bin/conda activate rl && pip install -r requirements.txt`
4. **Set up S3 credentials**: Create `~/.lambda_s3.env` with AWS credentials from Setup.txt
5. **Sync finetuned checkpoints**: Use correct S3 UUID `9e733b11-9ff3-41c4-9328-29990fa02ade`
6. **Verify tyro version**: Should be 0.9.28+ (auto-installed from requirements.txt)

### S3 Credentials Setup (Critical!)
```bash
cat > ~/.lambda_s3.env <<'EOF'
export AWS_ACCESS_KEY_ID=5EB1FPPIRA9S0BYBA8Q9
export AWS_SECRET_ACCESS_KEY=xSNViz9kasErzaa3l9iSh3RTa9ne6fMWOQETkJ8p
export AWS_REGION=us-east-3
export S3_ENDPOINT_URL=https://files.us-east-3.lambda.ai
export AWS_EC2_METADATA_DISABLED=true
EOF
chmod 600 ~/.lambda_s3.env

# Sync Lord Krang's finetuned checkpoints:
source ~/.lambda_s3.env && rclone copy lambda_east3:9e733b11-9ff3-41c4-9328-29990fa02ade/checkpoints/qwen2_5_15_finetuned/qwen2_5_15_gsm8k_lora/checkpoint-156 /tmp/checkpoints --ignore-checksum --size-only --transfers=4 --checkers=4 --progress
```

### 2. Run Training
```powershell
# Start overnight training (100 steps)
.\ssh_workflow.ps1 -InstanceIP <IP> -Action train -Config "rl_training/cfg/overnight_config.yaml"

# Start test training (4 steps)  
.\ssh_workflow.ps1 -InstanceIP <IP> -Action train -Config "rl_training/cfg/testconfig.yaml"
```

### 3. Monitor Training
```powershell
# Connect to training session
.\ssh_workflow.ps1 -InstanceIP <IP> -Action connect

# Start TensorBoard (runs on localhost:16006)
.\ssh_workflow.ps1 -InstanceIP <IP> -Action tensorboard
```

## Configuration Files

### testconfig.yaml (for testing)
- 4 steps total
- Eval every step
- Save every step  
- Good for debugging

### overnight_config.yaml (for production)
- 100 steps total
- Eval every 20 steps
- Save every 10 steps
- Optimized for overnight runs

## GPU Utilization Targets

**2x H100 80GB Setup:**
- Target: ~80% memory usage during rollout collection (~64GB per GPU)
- Target: ~50% memory usage during optimization (~40GB per GPU)
- Batch sizes: rollout_batch_size=40, buffer_size=64, microbatch_size=4

## Troubleshooting

### Common Issues
1. **Training stuck on step 0**: Usually evaluation phase issue, use overnight_config.yaml with less frequent eval
2. **CUDA OOM**: Reduce batch sizes in config file
3. **S3 access denied**: Check S3UUID and credentials in workflow script
4. **Conda terms of service error**: Run `~/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main`
5. **Conda command not found**: Use full path `~/miniconda3/bin/conda` instead of just `conda`
6. **eval_runner_fixed.py not found**: Fixed in codebase - was hardcoded wrong filename
7. **tyro argument parsing error**: Upgrade tyro to 0.9.28+ fixes "too many values to unpack" error
8. **eval_every=0 not respected**: Fixed in codebase - now properly skips evaluation when set to 0

### Log Locations
- Training logs: `/tmp/rl_runs/run_YYYY-MM-DD_HH-MM-SS/`
- Checkpoints: `/tmp/rl_runs/run_*/step_N/` 
- TensorBoard events: `/tmp/rl_runs/run_*/events.out.tfevents.*`

## Research Features

### Entropy Dynamics
- Configured in config under `entropy_probe:`
- Currently disabled (`sketch_r: 0`)
- Set `sketch_r > 0` to enable

### Gradient Noise Scale  
- Configured in config under `gns_probe:`
- Runs every step by default (`every: 1`)
- Measures gradient noise at different batch sizes
- Results logged to train_log.jsonl

## File Structure
```
lambda/
├── ssh_workflow.ps1           # Main automation script
├── LAMBDA_SETUP_GUIDE.md     # This guide
└── (auto-generated)
    ├── rclone_config.txt     # S3 configuration
    └── debug_outputs/        # Debug logs
```