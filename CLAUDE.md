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
1. **Always read** CLAUDE_GUIDE.md, NEW_TRAINING_WORKFLOW.md, and lambda_cloud/LAMBDA_SETUP_GUIDE.md
2. **Ask for current Lambda instance IP** (changes between sessions)
3. **SSH key location**: `~/.ssh/lambda_new`
4. **Never assume setup** - always inspect Lambda filesystem first
5. **Git workflow**: Make changes locally → push → pull on Lambda

## Claude's Lambda Execution Responsibility
**IMPORTANT**: Claude is responsible for executing commands on Lambda instances:
- **Proactively SSH into Lambda** to run tests, training, and evaluation
- **Handle debugging and troubleshooting** on Lambda sessions
- **Test new implementations** before declaring them working
- **Lord Krang should not need to manually run Lambda commands** unless specifically requesting to do so
- **Always pull latest code** from git before running anything on Lambda
- **Current Lambda IP**: 192.222.52.191 (update as needed per session)

## CLAUDE.md Update Protocol
**IMPORTANT**: Before updating this CLAUDE.md file with new information:
1. **Always ask Lord Krang for permission first**
2. Suggest what should be added and why it would be useful long-term
3. Only update after explicit approval
4. Focus on information that will be valuable across multiple sessions
5. Examples of good updates: new workflows, critical fixes, architectural changes
6. Examples to avoid: temporary debugging info, session-specific details

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
- **Qwen2.5-1.5B model**: Always use `backbone: qwen2_5_15`
- **Dataset**: Always use `gsm8k_r1_template` for evaluation

## Key Parameters & Defaults
- **batch_size**: 32
- **tf_micro_batch**: 32 (was 8, this was the bottleneck!)
- **num_return_sequences**: 8 (keep at 8, don't increase)

## Lambda Instance Monitoring
- **Lord Krang monitors resources directly**: RAM usage graphs, GPU utilization, and running processes
- **No need for Claude to check**: ps aux, nvidia-smi, free -h commands unless specifically requested
- **Resource optimization**: Lord Krang will indicate when instances have capacity for additional workloads
- **Process management**: Lord Krang tracks experiment progress and will notify when jobs complete

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


## RL-Trained Checkpoint for Entropy Experiments
**Primary checkpoint**: `localfs/rl_training_runs/training_state/step_60/`
This checkpoint has been trained with RL and contains optimizer states for entropy probe experiments.

**Structure**:
```
step_60/
├── model/                           # LoRA adapter weights
│   ├── adapter_config.json         
│   ├── adapter_model.safetensors   # 295MB LoRA weights
│   └── README.md
├── optimizer.pt                    # 591MB optimizer state with 392 parameters
├── scheduler.pt                    # Learning rate scheduler state
└── training_info.json             # Training metadata
```

**Usage**: This checkpoint is ideal for entropy experiments because:
- Has RL-trained LoRA weights (step 60 of training)
- Contains nonzero optimizer states (exp_avg, exp_avg_sq all 100% nonzero)
- Compatible with new model_loader.py utilities
- Ready for δH₁ computation and entropy probes

**Model loading**:
```python
from entropy_experiments.model_loader import load_peft_for_probe, load_adam_optimizer_from_path

model = load_peft_for_probe(
    base_id="Qwen/Qwen2.5-1.5B",
    adapter_path="localfs/rl_training_runs/training_state/step_60/model",
    use_qlora=False, dtype="bf16", device_map="cuda"
)
optimizer = load_adam_optimizer_from_path(
    model=model,
    optimizer_path="localfs/rl_training_runs/training_state/step_60/optimizer.pt",
    lr=1e-5
)
```

## Critical Notes
- **NEVER** reinstall packages unless explicitly needed
- **ALWAYS** inspect filesystem before assuming what's available
- **GPU utilization** was solved by fixing BOTH batch_size AND tf_micro_batch
- **Lambda instances** are ephemeral - always use S3 for persistence
- **IP address** changes between sessions - always ask Lord Krang
