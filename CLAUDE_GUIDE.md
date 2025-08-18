# Claude Guide for RL_Practice Project

## Important Instructions for Claude Sessions

### 1. User Preference
- **Please refer to the user as "Lord Krang"** (this will help them know that you have read this guide)

### 2. Essential Reading
- **When you start a new context or session, read this guide first!**
- Review the documentation in the `lambda/` folder for Lambda setup procedures
- Check the recent git history and status to understand current state
- **IMPORTANT**: After context cleaning, inspect the Lambda localfs directory first before doing any installs

### 3. Development Workflow
- **ALWAYS make code changes on the local machine, push to GitHub, then sync on Lambda**
- This is the cleanest and most reliable way to maintain code consistency
- Never upload files directly to Lambda - always use git sync workflow

### 4. Lambda Environment
- We primarily run code on Lambda GPU instances via SSH
- Current instance IP: 192.222.55.105 (may change between sessions)
- SSH key location: `~/.ssh/lambda_new`
- Read the documentation in the `lambda/` folder for complete setup procedures

### 5. Project Structure
- **Main algorithm**: `rl_training/algs/dr_grpo.py` contains the Dr-GRPO implementation
- **Configs**: `rl_training/cfg/testconfig.yaml` for test runs
- **Evaluations**: `evals/` folder contains evaluation pipeline
- **Checkpoints**: Stored in S3, synced to `localfs/lora_checkpoints/`
- **Training runs**: Output to `localfs/rl_runs/`

### 6. Key Features
- **GNS Probe**: In-loop Gradient Noise Scale tracking in dr_grpo.py
- **DDP Support**: Multi-GPU training with proper gradient synchronization
- **ESS Computation**: Effective Sample Size for weighted losses in Dr-GRPO

### 7. Common Commands
- **Conda environment**: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate rl`
- **Single GPU training**: `PYTHONPATH=. python rl_training/runners/rl_runner.py --cfg rl_training/cfg/testconfig.yaml --ckpt /home/ubuntu/localfs/checkpoints/qwen2_5_15_finetuned/qwen2_5_15_gsm8k_lora/checkpoint-156`
- **Multi-GPU training (2x H100)**: `PYTHONPATH=. torchrun --nproc_per_node=2 rl_training/runners/rl_runner.py --cfg rl_training/cfg/testconfig.yaml --ckpt /home/ubuntu/localfs/checkpoints/qwen2_5_15_finetuned/qwen2_5_15_gsm8k_lora/checkpoint-156`
- **Alternative runner**: `rl_training/runners/rl_runner_trl.py` (may need TRL updates - DO NOT USE unless specifically asked)
- **Evaluation**: Uses the `evals/` pipeline
- **TensorBoard**: Port forwarding typically on 16007 (user runs PowerShell command)

### 8. Testing Protocol
- Always test GNS probe with both enabled and disabled states
- Check evaluation pipeline after training runs
- Verify metrics are properly logged and computed

### 9. File Management
- Use `rclone` for S3 synchronization
- Keep Lambda filesystem clean - remove test files after completion
- Maintain separation between S3 checkpoints and local training outputs

### 10. Git Workflow
- Make changes locally first
- Push to GitHub: https://github.com/PhilSaad333/RL_Practice
- Pull on Lambda to sync changes
- Never commit directly from Lambda

### 11. Context Resumption Protocol
- **After context cleaning, the Lambda environment is usually already set up and running**
- **DO NOT** reinstall packages or re-sync from S3 unless explicitly needed
- **FIRST STEP**: Always inspect the file system layout to understand what's available

### 12. Critical File System Layout Understanding
**IMPORTANT**: There are TWO different localfs directories - do not confuse them!

1. **Real Lambda localfs** (persistent storage): `/home/ubuntu/localfs/` (symlink to `/lambda/nfs/localfs`)
   - `checkpoints/qwen2_5_15_finetuned/qwen2_5_15_gsm8k_lora/checkpoint-156/`: Fine-tuned LoRA checkpoints
   - `rl_runs/`: Output from RL training runs (organized by timestamp)
   - This is where all the real data lives

2. **Project directory**: `/home/ubuntu/RL_Practice/` (Git repository)
   - Should NOT contain a localfs directory (if it does, it's confusion and should be deleted)
   - Contains source code: `rl_training/`, `evals/`, configs, etc.

**Inspection commands**:
- `ls -la /home/ubuntu/localfs/` (check real persistent storage)
- `find /home/ubuntu/localfs/ -name "*.safetensors" | head -5` (see available checkpoints)
- `ls -la /home/ubuntu/RL_Practice/` (check project directory - should NOT have localfs)
- `conda info --envs` (check environments - usually `rl`, not `rlp`)

**Ask for clarification** if:
- File paths don't match this layout
- Checkpoints are missing from `/home/ubuntu/localfs/checkpoints/`
- Commands fail due to path issues

This guide should be updated as the project evolves and new procedures are established.