# Hybrid SSH + S3 Workflow for Lambda GPU Training
# Usage: .\ssh_workflow.ps1 -InstanceIP <IP> -Action <setup|sync-code|sync-checkpoints|train|tensorboard|monitor>

param(
    [Parameter(Mandatory = $true)]
    [string] $InstanceIP,
    
    [Parameter(Mandatory = $true)]
    [ValidateSet("setup", "sync-code", "sync-checkpoints", "train", "tensorboard", "monitor", "connect", "full-setup")]
    [string] $Action,
    
    [string] $KeyPath = "$env:USERPROFILE\.ssh\lambda_new",
    [string] $LocalPort = "16006",
    [string] $Config = "rl_training/cfg/testconfig.yaml",
    [string] $S3UUID = "",
    [string] $CheckpointPath = "checkpoints/qwen2_5_15_finetuned/qwen2_5_15_gsm8k_lora/checkpoint-156"
)

$SSHOpts = @(
    "-o", "IdentitiesOnly=yes"
    "-i", $KeyPath
    "-o", "StrictHostKeyChecking=no"
    "-o", "UserKnownHostsFile=/dev/null"
)

function SSH-Execute {
    param([string] $Command)
    ssh @SSHOpts "ubuntu@$InstanceIP" $Command
}

function SSH-Copy {
    param([string] $LocalPath, [string] $RemotePath)
    scp @SSHOpts -r $LocalPath "ubuntu@$InstanceIP`:$RemotePath"
}

switch ($Action) {
    "setup" {
        Write-Host "Setting up Lambda instance (basic packages)..."
        SSH-Execute "sudo apt-get update && sudo apt-get install -y rclone tmux htop nvtop curl"
        Write-Host "Installing Miniconda..."
        SSH-Execute "curl -fsSLO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh -b -p `$HOME/miniconda3"
        SSH-Execute "eval `"`$(`$HOME/miniconda3/bin/conda shell.bash hook)`" && conda create -n rl python=3.11 -y"
        Write-Host "Basic setup complete!"
    }
    
    "sync-code" {
        Write-Host "Syncing code to remote instance (excluding large model files)..."
        SSH-Execute "rm -rf ~/RL_Practice"
        SSH-Copy "." "RL_Practice"
        SSH-Execute "cd ~/RL_Practice && eval `"`$(`$HOME/miniconda3/bin/conda shell.bash hook)`" && conda activate rl && pip install -r requirements.txt"
        Write-Host "Code synced and dependencies installed!"
    }
    
    "sync-checkpoints" {
        if ($S3UUID -eq "") {
            Write-Error "S3UUID parameter required for checkpoint sync. Use -S3UUID <your-uuid>"
            exit 1
        }
        Write-Host "Setting up S3 credentials and syncing checkpoints..."
        SSH-Execute "source ~/.lambda_s3.env && mkdir -p ~/.config/rclone && cat > ~/.config/rclone/rclone.conf << 'EOF'
[lambda_east3]
type = s3
provider = Other
env_auth = true
region = us-east-3
endpoint = https://files.us-east-3.lambda.ai
force_path_style = true
disable_checksum = true
EOF"
        SSH-Execute "source ~/.lambda_s3.env && rm -rf /tmp/checkpoints && mkdir -p /tmp/checkpoints && rclone copy lambda_east3:$S3UUID/$CheckpointPath /tmp/checkpoints/checkpoint-156 --ignore-checksum --size-only --transfers=4 --checkers=4 --progress"
        Write-Host "Checkpoints synced to /tmp/checkpoints/"
    }
    
    "full-setup" {
        if ($S3UUID -eq "") {
            Write-Error "S3UUID parameter required. Use -S3UUID <your-uuid>"
            exit 1
        }
        Write-Host "Running full setup..."
        & $PSCommandPath -InstanceIP $InstanceIP -Action setup -KeyPath $KeyPath
        & $PSCommandPath -InstanceIP $InstanceIP -Action sync-code -KeyPath $KeyPath
        & $PSCommandPath -InstanceIP $InstanceIP -Action sync-checkpoints -S3UUID $S3UUID -CheckpointPath $CheckpointPath -KeyPath $KeyPath
        Write-Host "Full setup complete!"
    }
    
    "train" {
        Write-Host "Starting training session..."
        $TrainCmd = "cd ~/RL_Practice && eval `"`$(`$HOME/miniconda3/bin/conda shell.bash hook)`" && conda activate rl && export RUN_ROOT=/tmp/rl_runs && python -m rl_training.runners.rl_runner --cfg $Config --ckpt /tmp/checkpoints"
        SSH-Execute "tmux new-session -d -s training '$TrainCmd'"
        Write-Host "Training started in tmux session 'training'"
        Write-Host "Connect with: .\ssh_workflow.ps1 -InstanceIP $InstanceIP -Action connect"
    }
    
    "tensorboard" {
        Write-Host "Starting TensorBoard tunnel..."
        SSH-Execute "tmux new-session -d -s tensorboard 'eval `"`$(`$HOME/miniconda3/bin/conda shell.bash hook)`" && conda activate rl && tensorboard --logdir /tmp/rl_runs --host 0.0.0.0 --port 6006'"
        $TunnelArgs = $SSHOpts + @("-N", "-L", "$LocalPort`:localhost:6006", "ubuntu@$InstanceIP")
        Start-Process -NoNewWindow ssh -ArgumentList $TunnelArgs
        Write-Host "TensorBoard available at http://localhost:$LocalPort"
        Write-Host "Press Ctrl+C to stop tunnel"
    }
    
    "monitor" {
        Write-Host "Opening monitoring session..."
        SSH-Execute "tmux new-session -d -s monitor 'watch -n 1 nvidia-smi'"
        ssh @SSHOpts "ubuntu@$InstanceIP" -t "tmux attach -t monitor"
    }
    
    "connect" {
        Write-Host "Connecting to training session..."
        ssh @SSHOpts "ubuntu@$InstanceIP" -t "tmux attach -t training"
    }
}