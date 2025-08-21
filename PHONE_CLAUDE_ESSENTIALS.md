# Phone Claude Essentials - RL_Practice Project

## Essential Information
**User**: Lord Krang (always refer by this name)  
**Environment**: VS Code local + Lambda Cloud GPU instances via SSH  
**Project**: Reinforcement Learning training with Dr-GRPO algorithm  
**Repository**: https://github.com/PhilSaad333/RL_Practice  

## Critical Authentication & Tokens
**SSH Key Location (on home laptop)**: `~/.ssh/lambda_new`
**GitHub Token**: `[LORD_KRANG_WILL_PROVIDE]`
- Token scope: `repo` (full repository access)  
- Usage: `curl -H "Authorization: token $TOKEN" https://api.github.com/...`

## Lambda Cloud Setup
**Current Lambda Instance IP**: ASK LORD KRANG (changes between sessions)
**Lambda File Structure**:
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
```

## Core Training Commands

### 1. Start Training (Multi-GPU)
```bash
ssh -i ~/.ssh/lambda_new ubuntu@<IP>
cd ~/RL_Practice && source ~/miniconda3/etc/profile.d/conda.sh && conda activate rl

# Start training
PYTHONPATH=. torchrun --nproc_per_node=2 rl_training/runners/rl_runner.py \
  --cfg rl_training/cfg/testconfig.yaml \
  --ckpt /home/ubuntu/localfs/checkpoints/qwen2_5_15_finetuned/qwen2_5_15_gsm8k_lora/checkpoint-156
```

**⚠️ CRITICAL**: The `--ckpt` argument is the ORIGINAL fine-tuned checkpoint for KL divergence reference

### 2. Check Training Status
```bash
ssh -i ~/.ssh/lambda_new ubuntu@<IP>
ps aux | grep -E "(rl_runner|torchrun)" | grep -v grep
nvidia-smi  # Check GPU usage
```

### 3. Evaluation
```bash
# Evaluate all checkpoints from a training run
python rl_training/runners/eval_batch.py \
  --training-run run_YYYY-MM-DD_HH-MM-SS \
  --subset-frac 0.02
```

### 4. Resume Training
```bash
python rl_training/runners/resume_training.py \
  --training-run run_YYYY-MM-DD_HH-MM-SS \
  --from-step latest \
  --additional-steps 10
```

## Key Configuration Files
- **testconfig.yaml**: 2 steps, save_every=1 (for testing)
- **gns_probe_64step_dual_h100.yaml**: 64 steps, GNS probe (for production runs)
- **Qwen2.5-1.5B model**: Always use `backbone: qwen2_5_15`

## Key Parameters
- **batch_size**: 32 (optimized for GPU utilization)
- **tf_micro_batch**: 32 (was the bottleneck!)
- **num_return_sequences**: 8 (keep at 8)
- **buffer_size**: 32, **microbatch_size**: 2 (for 2×GPU: grad_accum=8)

## Critical Session Protocol
1. **Always ask for current Lambda IP** (changes between sessions)
2. **SSH setup**: Use `~/.ssh/lambda_new` key (on home laptop)
3. **Inspect filesystem first**: `ls /home/ubuntu/localfs/training_runs/`
4. **Environment setup**: Always activate conda rl environment
5. **Git sync**: Pull latest changes before running

## Troubleshooting
- **Conda environment**: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate rl`
- **GPU memory**: Use `nvidia-smi` to check utilization
- **Training stuck**: Check logs in `training_runs/*/logs/train_log.jsonl`
- **File permissions**: Check `/home/ubuntu/localfs/` symlink

## Success Indicators
- ✅ Training completes without hanging
- ✅ GPU memory usage >60% during training/eval
- ✅ Resume training works from any checkpoint
- ✅ File structure follows new training_runs/ organization

## Common Issues
- **Lambda IP changes**: Always ask Lord Krang for current IP
- **Conda not found**: Run full conda activation command
- **Permission denied**: Check file ownership in localfs/
- **Git conflicts**: Make changes locally first, then push/pull