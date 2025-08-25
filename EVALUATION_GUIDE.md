# RL Training Evaluation Guide 

## Quick Reference: Parallel GPU Evaluation

### Working Setup (tested 2025-08-25)

**Checkpoint Structure:**
```
/training_runs/run_YYYY-MM-DD_HH-MM-SS/training_state/
├── step_10/model/    # ← LoRA adapter files are here
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── README.md
├── step_20/model/
└── ... (other steps)
```

**Working Command Template:**
```bash
CUDA_VISIBLE_DEVICES=X PYTHONPATH=. python evals/eval_runner.py \
  --backbone Qwen/Qwen2.5-1.5B \
  --ckpt-path /path/to/training_runs/run_YYYY-MM-DD_HH-MM-SS/training_state/step_N/model \
  --ckpt-step N \
  --eval-dataset gsm8k_r1_template \
  --batch-size auto \
  --max-new-tokens 200
```

### Parallel GPU Evaluation Recipe

1. **Check training run completion:**
   ```bash
   ssh -i ~/.ssh/lambda_new ubuntu@IP "ls -la ~/localfs/training_runs/run_*/training_state/"
   ```

2. **Start GPU 0 (first half of checkpoints):**
   ```bash
   ssh -i ~/.ssh/lambda_new ubuntu@IP "cd ~/RL_Practice && source ~/miniconda3/etc/profile.d/conda.sh && conda activate rl && 
   (
     CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python evals/eval_runner.py --backbone Qwen/Qwen2.5-1.5B --ckpt-path /home/ubuntu/localfs/training_runs/RUN_NAME/training_state/step_10/model --ckpt-step 10 --eval-dataset gsm8k_r1_template --batch-size auto --max-new-tokens 200
     CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python evals/eval_runner.py --backbone Qwen/Qwen2.5-1.5B --ckpt-path /home/ubuntu/localfs/training_runs/RUN_NAME/training_state/step_30/model --ckpt-step 30 --eval-dataset gsm8k_r1_template --batch-size auto --max-new-tokens 200  
     CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python evals/eval_runner.py --backbone Qwen/Qwen2.5-1.5B --ckpt-path /home/ubuntu/localfs/training_runs/RUN_NAME/training_state/step_50/model --ckpt-step 50 --eval-dataset gsm8k_r1_template --batch-size auto --max-new-tokens 200
     CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python evals/eval_runner.py --backbone Qwen/Qwen2.5-1.5B --ckpt-path /home/ubuntu/localfs/training_runs/RUN_NAME/training_state/step_final/model --ckpt-step final --eval-dataset gsm8k_r1_template --batch-size auto --max-new-tokens 200
   ) > gpu0_eval_full.log 2>&1 &"
   ```

3. **Start GPU 1 (second half of checkpoints):**
   ```bash
   ssh -i ~/.ssh/lambda_new ubuntu@IP "cd ~/RL_Practice && source ~/miniconda3/etc/profile.d/conda.sh && conda activate rl && 
   (
     CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python evals/eval_runner.py --backbone Qwen/Qwen2.5-1.5B --ckpt-path /home/ubuntu/localfs/training_runs/RUN_NAME/training_state/step_20/model --ckpt-step 20 --eval-dataset gsm8k_r1_template --batch-size auto --max-new-tokens 200
     CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python evals/eval_runner.py --backbone Qwen/Qwen2.5-1.5B --ckpt-path /home/ubuntu/localfs/training_runs/RUN_NAME/training_state/step_40/model --ckpt-step 40 --eval-dataset gsm8k_r1_template --batch-size auto --max-new-tokens 200
     CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python evals/eval_runner.py --backbone Qwen/Qwen2.5-1.5B --ckpt-path /home/ubuntu/localfs/training_runs/RUN_NAME/training_state/step_60/model --ckpt-step 60 --eval-dataset gsm8k_r1_template --batch-size auto --max-new-tokens 200
   ) > gpu1_eval_full.log 2>&1 &"
   ```

4. **Monitor progress:**
   ```bash
   # Check GPU utilization
   nvidia-smi
   
   # Check running processes  
   ps aux | grep python | grep eval_runner
   
   # Check for completed evaluations
   find eval_runs/ -name "metrics.csv" -newer gpu0_eval_full.log
   ```

### Key Points
- **CRITICAL**: Use `/step_N/model` path (not `/step_N` directly)
- **CRITICAL**: Specify correct `--backbone Qwen/Qwen2.5-1.5B` 
- **CRITICAL**: Include `--ckpt-step` parameter
- Use `--batch-size auto` for optimal GPU utilization
- Each evaluation takes ~5-10 minutes on H100
- Results saved to `eval_runs/` directory structure

### Troubleshooting
- **"Can't find adapter_config.json"**: Missing `/model` in path
- **"size mismatch"**: Wrong backbone (use `Qwen/Qwen2.5-1.5B`)  
- **"invalid literal for int()"**: Missing `--ckpt-step` parameter
- **No GPU utilization**: Check CUDA_VISIBLE_DEVICES and process status

### Expected Timeline
- ~3-5 minutes per checkpoint on H100
- 7 checkpoints total = ~15-25 minutes with 2 GPUs in parallel
- ~30-45 minutes if running on single GPU

## Checkpoint Storage & Backup

### Current Status (2025-08-25)
- **Best 64-step run**: `run_2025-08-24_22-13-22` 
- **Location**: `/home/ubuntu/localfs/training_runs/run_2025-08-24_22-13-22/`
- **Size**: ~5.8GB (models + optimizer states)
- **Storage**: Lambda NFS persistent storage (2.4P total, <1% used)
- **Includes**: All 7 checkpoints (step_10, step_20, step_30, step_40, step_50, step_60, step_final)

### S3 Backup (Optional)
To backup to S3 (requires AWS credentials setup):
```bash
# Configure AWS CLI first
aws configure

# Upload complete run
cd ~/localfs/training_runs
aws s3 sync run_2025-08-24_22-13-22/ s3://YOUR-BUCKET/rl_training/checkpoints/run_2025-08-24_22-13-22/ --storage-class STANDARD_IA

# Download later
aws s3 sync s3://YOUR-BUCKET/rl_training/checkpoints/run_2025-08-24_22-13-22/ ./run_2025-08-24_22-13-22/
```

### Direct SCP Download (if needed)
```bash
# Download complete run (5.8GB)
scp -r -i ~/.ssh/lambda_new ubuntu@IP:/home/ubuntu/localfs/training_runs/run_2025-08-24_22-13-22 ./

# Download just models (2GB)  
scp -r -i ~/.ssh/lambda_new ubuntu@IP:/home/ubuntu/localfs/training_runs/run_2025-08-24_22-13-22/training_state/*/model ./models/
```