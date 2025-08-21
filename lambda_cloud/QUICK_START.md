# Hybrid SSH + S3 Lambda Workflow

This workflow combines the best of both approaches: SSH for code sync and S3 for large checkpoint files.

## Prerequisites
- SSH key at: `C:\Users\phils\.ssh\lambda_new`
- Lambda GPU instance running with IP address
- Your S3 filesystem UUID (from Lambda console)
- S3 credentials set up on the remote instance

## One-Time Setup (per new instance)

First, you need to create the S3 credentials file on the remote instance:

```powershell
# SSH into the instance manually first
ssh -i "$env:USERPROFILE\.ssh\lambda_new" ubuntu@<YOUR_IP>

# Create the S3 credentials file (replace with your actual credentials)
cat > ~/.lambda_s3.env <<'EOF'
export AWS_ACCESS_KEY_ID=YOUR_ACCESS_KEY
export AWS_SECRET_ACCESS_KEY=YOUR_SECRET_KEY
export AWS_REGION=us-east-3
export S3_ENDPOINT_URL=https://files.us-east-3.lambda.ai
export AWS_EC2_METADATA_DISABLED=true
EOF
chmod 600 ~/.lambda_s3.env
exit
```

## Quick Start Commands

```powershell
# 1. Full setup (environment + code + checkpoints)
.\lambda\ssh_workflow.ps1 -InstanceIP <YOUR_IP> -Action full-setup -S3UUID <YOUR_S3_UUID>

# 2. Start training
.\lambda\ssh_workflow.ps1 -InstanceIP <YOUR_IP> -Action train

# 3. Start TensorBoard tunnel
.\lambda\ssh_workflow.ps1 -InstanceIP <YOUR_IP> -Action tensorboard

# 4. Connect to training session
.\lambda\ssh_workflow.ps1 -InstanceIP <YOUR_IP> -Action connect
```

## Individual Commands (for development iterations)

```powershell
# Sync only code changes (fast)
.\lambda\ssh_workflow.ps1 -InstanceIP <YOUR_IP> -Action sync-code

# Sync only checkpoints (when you have new model files)
.\lambda\ssh_workflow.ps1 -InstanceIP <YOUR_IP> -Action sync-checkpoints -S3UUID <YOUR_S3_UUID>

# Monitor GPU usage
.\lambda\ssh_workflow.ps1 -InstanceIP <YOUR_IP> -Action monitor
```

## Example Workflow

```powershell
# Replace with your actual values
$IP = "198.51.100.123"
$UUID = "9e733b11-9ff3-41c4-9328-29990fa02ade"

# Complete setup
.\lambda\ssh_workflow.ps1 -InstanceIP $IP -Action full-setup -S3UUID $UUID

# Start training and monitoring
.\lambda\ssh_workflow.ps1 -InstanceIP $IP -Action train
.\lambda\ssh_workflow.ps1 -InstanceIP $IP -Action tensorboard

# During development - just sync code changes
.\lambda\ssh_workflow.ps1 -InstanceIP $IP -Action sync-code
.\lambda\ssh_workflow.ps1 -InstanceIP $IP -Action train
```

## Advantages of This Approach

✅ **Fast code iteration**: SSH sync for small code files  
✅ **Efficient large files**: S3 rclone for multi-GB checkpoints  
✅ **All from VS Code**: No need to manually SSH for routine tasks  
✅ **Background training**: Uses tmux, survives disconnections  
✅ **Easy monitoring**: TensorBoard tunnel and GPU monitoring  
✅ **Modular**: Can run individual steps as needed  

## Tmux Session Management

```powershell
# Connect to training logs
.\lambda\ssh_workflow.ps1 -InstanceIP $IP -Action connect

# In the remote session, useful tmux commands:
# Ctrl+B then D = Detach (keep training running)
# Ctrl+C = Stop training
# tmux attach -t training = Reattach to training
# tmux attach -t tensorboard = Attach to tensorboard session
# tmux kill-session -t training = Kill training session
```

This approach gives you the simplicity of SSH operations for development while keeping the robust S3 transfer for large model files.