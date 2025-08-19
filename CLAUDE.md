# Claude Memory - RL_Practice Project

## Essential Information
**User**: Lord Krang (always refer by this name)  
**Environment**: VS Code local + Lambda Cloud GPU instances via SSH  
**Project**: Reinforcement Learning training with Dr-GRPO algorithm  
**Repository**: https://github.com/PhilSaad333/RL_Practice  

## Current State (Aug 2025)
✅ **NEW STOP/EVAL/RESUME WORKFLOW IMPLEMENTED AND TESTED**  
✅ **All distributed training evaluation issues RESOLVED**  
✅ **GPU utilization optimized (32x batch size + teacher forcing)**  

## Critical Session Startup Protocol
1. **Always read** CLAUDE_GUIDE.md, NEW_TRAINING_WORKFLOW.md, and lambda/LAMBDA_SETUP_GUIDE.md
2. **Ask for current Lambda instance IP** (changes between sessions)
3. **SSH key location**: `~/.ssh/lambda_new`
4. **Never assume setup** - always inspect Lambda filesystem first
5. **Git workflow**: Make changes locally → push → pull on Lambda

## Core Workflow (WORKING PERFECTLY)

### File Structure on Lambda
```
/home/ubuntu/localfs/ (symlink to /lambda/nfs/localfs)
├── checkpoints/qwen2_5_15_finetuned/qwen2_5_15_gsm8k_lora/checkpoint-156/
├── training_runs/
│   └── run_YYYY-MM-DD_HH-MM-SS/
│       ├── config.yaml
│       ├── training_state/ (step_1/, step_2/, step_final/, step_latest→)
│       ├── logs/
│       └── tensorboard/
└── eval_runs/
    └── run_YYYY-MM-DD_HH-MM-SS_gsm8k_r1_template/
```

### 1. Training (Multi-GPU)
```bash
ssh -i ~/.ssh/lambda_new ubuntu@<IP>
cd ~/RL_Practice && source ~/miniconda3/etc/profile.d/conda.sh && conda activate rl

# Start training
PYTHONPATH=. torchrun --nproc_per_node=2 rl_training/runners/rl_runner.py \
  --cfg rl_training/cfg/testconfig.yaml \
  --ckpt /home/ubuntu/localfs/checkpoints/qwen2_5_15_finetuned/qwen2_5_15_gsm8k_lora/checkpoint-156
```

### 2. Evaluation (Single GPU, Subprocess)
```bash
# Evaluate all checkpoints from a training run
python rl_training/runners/eval_batch.py \
  --training-run run_YYYY-MM-DD_HH-MM-SS \
  --subset-frac 0.02
```

### 3. Resume Training
```bash
# Resume from latest checkpoint for additional steps
python rl_training/runners/resume_training.py \
  --training-run run_YYYY-MM-DD_HH-MM-SS \
  --from-step latest \
  --additional-steps 10
```

## Recent Critical Fixes (Aug 2025)

### ✅ Evaluation Issues RESOLVED
- **"final" step parsing**: Fixed eval_runner.py to handle non-integer step names
- **GPU utilization**: Increased batch_size from 8→32 AND tf_micro_batch from 8→32  
- **Distributed conflicts**: Complete separation via subprocess evaluation
- **File paths**: All evaluation now uses new training_state/ structure

### ✅ Training State Management
- **Full state persistence**: model/ + optimizer.pt + scheduler.pt + training_info.json
- **Resume capability**: Perfect continuation from any checkpoint
- **File organization**: New structure with training_state/, logs/, tensorboard/

## Configuration Files
- **testconfig.yaml**: 2 steps, save_every=1 (for testing)
- **overnight_config.yaml**: 100 steps, save_every=10 (for production)
- **Qwen2.5-1.5B model**: Always use `backbone: qwen2_5_15`
- **Dataset**: Always use `gsm8k_r1_template` for evaluation

## Key Parameters & Defaults
- **batch_size**: 32 (was 8, increased for GPU utilization)
- **tf_micro_batch**: 32 (was 8, this was the bottleneck!)
- **num_return_sequences**: 8 (keep at 8, don't increase)
- **grad_accum_steps**: Auto-calculated as buffer_size/(world_size×microbatch_size)
- **buffer_size**: 32, **microbatch_size**: 2 (for 2×GPU: grad_accum=8)

## Troubleshooting Quick Reference
- **Conda environment**: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate rl`
- **Git sync**: Always make changes locally first, then push/pull
- **S3 credentials**: Check ~/.lambda_s3.env if rclone fails
- **GPU memory**: Use nvidia-smi to check utilization
- **Training stuck**: Check logs in training_runs/*/logs/train_log.jsonl

## Research Features
- **Dr-GRPO**: Main RL algorithm in rl_training/algs/dr_grpo.py
- **GNS Probe**: Gradient Noise Scale tracking (configurable)
- **ESS**: Effective Sample Size computation for weighted losses
- **DDP**: Multi-GPU distributed training with PyTorch

## Testing Protocol
1. Run 2-step training test with testconfig.yaml
2. Test evaluation on all checkpoints (including "final")
3. Test resume training from latest checkpoint
4. Verify GPU utilization during eval (~80% memory usage)
5. Check file organization and symlinks

## Critical Notes
- **NEVER** reinstall packages unless explicitly needed
- **ALWAYS** inspect filesystem before assuming what's available
- **GPU utilization** was solved by fixing BOTH batch_size AND tf_micro_batch
- **Lambda instances** are ephemeral - always use S3 for persistence
- **IP address** changes between sessions - always ask Lord Krang

## Success Metrics
- ✅ Training completes without hanging
- ✅ Evaluation runs on all checkpoints including "final"
- ✅ Resume training works perfectly
- ✅ GPU memory usage >60% during evaluation
- ✅ File structure follows new training_runs/ organization