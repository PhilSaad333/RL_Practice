# New RL Training Workflow - Stop/Eval/Resume

This document describes the improved training workflow that separates training and evaluation for better stability and flexibility.

## Overview

**Old Problem**: Running evaluation during distributed training caused tensor size conflicts and hanging processes.

**New Solution**: Clean separation of training and evaluation phases with full state persistence.

## Workflow

```
1. Train for N steps → Save full state (model + optimizer + scheduler)
2. Exit training cleanly  
3. Run eval on saved checkpoints (separate process)
4. Resume training from saved state
5. Repeat until done
6. Archive important data to S3
```

## File Organization

### LocalFS Structure (`/home/ubuntu/localfs/`)

```
training_runs/
├── run_2025-08-19_12-34-56/          # Training session
│   ├── config.yaml                   # Original training config
│   ├── training_state/               # Full training state saves  
│   │   ├── step_10/
│   │   │   ├── model/                # LoRA weights
│   │   │   ├── optimizer.pt          # Optimizer state
│   │   │   ├── scheduler.pt          # LR scheduler state  
│   │   │   └── training_info.json    # Step, config, etc.
│   │   ├── step_20/
│   │   └── step_latest/             # Symlink to most recent
│   ├── logs/
│   │   ├── train_log.jsonl
│   │   ├── rollouts.jsonl
│   │   └── ratios.jsonl
│   └── tensorboard/
│       └── events.out.tfevents.*

eval_runs/
├── run_2025-08-19_12-34-56_gsm8k_r1_template/   # Eval session  
│   ├── eval_config.yaml             # Eval parameters used
│   ├── step_10_gsm8k_r1_template/   # Results per step
│   │   ├── metrics.csv
│   │   ├── completions.jsonl
│   │   └── summary.json
│   ├── step_20_gsm8k_r1_template/
│   └── consolidated_metrics.csv     # All steps together

archive/                             # For completed work
├── completed_runs/                  # Archived training runs
└── important_checkpoints/           # Key checkpoints to keep
```

## Scripts

### 1. `rl_runner.py` (Enhanced)

**New Features**:
- Saves full training state (model + optimizer + scheduler + metadata)
- Removed inline evaluation (no more hanging/tensor issues)
- Supports resumption from saved state
- Better file organization with logs/ and training_state/ subdirs

**Usage**:
```bash
# New training
PYTHONPATH=. torchrun --nproc_per_node=2 rl_training/runners/rl_runner.py \
  --cfg rl_training/cfg/testconfig.yaml \
  --ckpt /home/ubuntu/localfs/checkpoints/qwen2_5_15_finetuned/qwen2_5_15_gsm8k_lora/checkpoint-156

# Resume training
python rl_training/runners/resume_training.py \
  --training-run run_2025-08-19_12-34-56 \
  --from-step latest \
  --additional-steps 50
```

### 2. `eval_batch.py`

Evaluates all checkpoints from a training run in one go.

**Usage**:
```bash
# Evaluate all steps
python rl_training/runners/eval_batch.py \
  --training-run run_2025-08-19_12-34-56 \
  --eval-dataset gsm8k_r1_template \
  --subset-frac 0.01

# Evaluate specific steps
python rl_training/runners/eval_batch.py \
  --training-run run_2025-08-19_12-34-56 \
  --steps 1,5,10 \
  --subset-frac 0.1
```

**Features**:
- Evaluates multiple checkpoints automatically
- Creates consolidated metrics CSV
- Uses single GPU (no distributed conflicts)
- Proper error handling and timeouts

### 3. `resume_training.py`

Resumes training from any saved checkpoint with full state restoration.

**Usage**:
```bash
# Resume from latest checkpoint
python rl_training/runners/resume_training.py \
  --training-run run_2025-08-19_12-34-56 \
  --from-step latest

# Resume from specific step with additional training
python rl_training/runners/resume_training.py \
  --training-run run_2025-08-19_12-34-56 \
  --from-step 10 \
  --additional-steps 50

# Single GPU mode
python rl_training/runners/resume_training.py \
  --training-run run_2025-08-19_12-34-56 \
  --from-step latest \
  --no-torchrun
```

**Features**:
- Loads optimizer and scheduler states
- Continues from exact step number
- Validates checkpoint completeness
- Supports both single and multi-GPU

### 4. `sync_to_s3.py`

Archives completed work to S3 for persistence.

**Usage**:
```bash
# Backup specific run
python rl_training/runners/sync_to_s3.py \
  --action backup \
  --training-run run_2025-08-19_12-34-56

# Backup all runs and clean up
python rl_training/runners/sync_to_s3.py \
  --action backup-all \
  --cleanup \
  --keep-latest

# List S3 backups
python rl_training/runners/sync_to_s3.py \
  --action list-backups

# Restore from S3
python rl_training/runners/sync_to_s3.py \
  --action restore \
  --training-run run_2025-08-19_12-34-56
```

## Example Workflow

### 1. Start Training
```bash
cd RL_Practice
source ~/miniconda3/etc/profile.d/conda.sh && conda activate rl

# Train for a few steps
PYTHONPATH=. torchrun --nproc_per_node=2 rl_training/runners/rl_runner.py \
  --cfg rl_training/cfg/testconfig.yaml \
  --ckpt /home/ubuntu/localfs/checkpoints/qwen2_5_15_finetuned/qwen2_5_15_gsm8k_lora/checkpoint-156
```

### 2. Evaluate Checkpoints  
```bash
# After training completes, evaluate all steps
python rl_training/runners/eval_batch.py \
  --training-run run_2025-08-19_12-34-56 \
  --eval-dataset gsm8k_r1_template \
  --subset-frac 0.01

# Check results
cat /home/ubuntu/localfs/eval_runs/run_2025-08-19_12-34-56_gsm8k_r1_template/consolidated_metrics.csv
```

### 3. Resume for More Training
```bash
# Continue training for 50 more steps
python rl_training/runners/resume_training.py \
  --training-run run_2025-08-19_12-34-56 \
  --from-step latest \
  --additional-steps 50
```

### 4. Archive to S3
```bash
# When done, backup to S3 and clean up
python rl_training/runners/sync_to_s3.py \
  --action backup \
  --training-run run_2025-08-19_12-34-56 \
  --cleanup \
  --keep-latest
```

## Benefits

✅ **No more tensor size errors**: Evaluation runs in clean subprocess  
✅ **No more hanging**: Clean separation of training and eval  
✅ **Flexible evaluation**: Run eval on any checkpoint anytime  
✅ **Resumable training**: Never lose progress, start/stop anytime  
✅ **Lambda optimized**: Works with Lambda's ephemeral instances  
✅ **S3 integration**: Easy archival and restoration  
✅ **Better debugging**: Inspect any intermediate state  
✅ **Organized files**: Clear separation of logs, checkpoints, and results  

## Configuration

### Environment Variables
- `LOCALFS_ROOT`: Path to persistent storage (default: `/lambda/nfs/localfs`)

### Required Files
- Training config (e.g., `testconfig.yaml`)
- Base checkpoint (e.g., fine-tuned LoRA weights)
- S3 credentials in `~/.lambda_s3.env`

## Troubleshooting

### Training Issues
- **Checkpoint not found**: Check `training_state/` directory structure
- **Resume fails**: Verify all files exist (model/, optimizer.pt, scheduler.pt, training_info.json)
- **Permission errors**: Ensure localfs symlink is working

### Evaluation Issues
- **No metrics generated**: Check eval_config.yaml and error messages
- **Wrong dataset**: Ensure gsm8k_r1_template files exist in rlp_datasets/processed/
- **GPU memory**: Evaluation forces CUDA_VISIBLE_DEVICES=0

### S3 Issues
- **rclone errors**: Check ~/.lambda_s3.env and rclone configuration
- **Access denied**: Verify S3 bucket permissions
- **Slow transfers**: Adjust --transfers parameter

## Migration from Old Workflow

The new workflow is **completely backward compatible**:

1. Old training runs in `/home/ubuntu/localfs/rl_runs/` still work
2. Use `eval_batch.py` on old checkpoints (specify `--ckpt-path ./step_N/`)
3. New runs automatically use improved file structure
4. Old eval outputs remain in original locations

## Future Enhancements

- **Auto-resume**: Detect crashed training and auto-resume
- **Eval scheduling**: Run eval automatically when training completes  
- **Metric tracking**: Centralized metrics database across runs
- **Resource monitoring**: Track GPU usage and costs per run