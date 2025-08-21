# Data Organization Convention for Lambda Results

## Directory Structure

When downloading results from Lambda training runs, use this consistent structure:

```
data/downloaded_runs/
└── {experiment_name}_{date}_{model}_{hardware}_{steps}steps/
    ├── training_run/
    │   ├── config.yaml
    │   ├── logs/
    │   │   └── train_log.jsonl
    │   ├── tensorboard/
    │   └── training_state/
    │       ├── step_10/ (model/, optimizer.pt, scheduler.pt, training_info.json)
    │       ├── step_20/
    │       ├── ...
    │       ├── step_final/
    │       └── step_latest -> step_XX/
    ├── eval_results/
    │   └── run_{timestamp}_{dataset}/
    │       └── {model}_{dataset_name}/
    │           ├── step_10_{dataset}/
    │           │   └── temp0.7_p1.0_r8/
    │           │       ├── metrics.csv
    │           │       ├── records.jsonl.gz
    │           │       └── run_meta.json
    │           ├── step_20_{dataset}/
    │           └── ...
    ├── base_eval/  (if base checkpoint evaluated separately)
    │   └── temp0.7_p1.0_r8/
    │       ├── metrics.csv
    │       ├── records.jsonl.gz
    │       └── run_meta.json
    ├── analysis/
    │   ├── complete_pass_plot.py
    │   ├── detailed_passk_analysis.py
    │   ├── complete_gns_training_progress.png
    │   └── detailed_passk_progress.png
    └── README.md (experiment summary)
```

## Naming Conventions

### Experiment Directories
- Format: `{experiment_name}_{date}_{model}_{hardware}_{steps}steps`
- Example: `gns_test_run_2025-08-20_qwen2_5_1_5b_h100_64steps`

### Components:
- **experiment_name**: `gns_test`, `baseline_training`, `ablation_study`, etc.
- **date**: `YYYY-MM-DD` format
- **model**: `qwen2_5_1_5b`, `llama3_8b`, etc.
- **hardware**: `h100`, `a100`, `2x_h100`, `8x_h100`, etc.
- **steps**: Number of training steps

## Download Commands

### 1. Training Run Data (without intermediate checkpoints)
```bash
# Download training logs, config, tensorboard, and final checkpoint only
scp -i ~/.ssh/lambda_new -r ubuntu@{IP}:/home/ubuntu/localfs/training_runs/run_{timestamp}/ \
  "data/downloaded_runs/{experiment_name}/training_run/"

# Download only final checkpoint (not intermediate ones)
scp -i ~/.ssh/lambda_new -r ubuntu@{IP}:/home/ubuntu/localfs/training_runs/run_{timestamp}/training_state/step_final/ \
  "data/downloaded_runs/{experiment_name}/training_run/training_state/"
```

### 2. Evaluation Results
```bash
# Download all evaluation metrics and metadata
scp -i ~/.ssh/lambda_new -r ubuntu@{IP}:/home/ubuntu/localfs/eval_runs/run_{timestamp}_{dataset}/ \
  "data/downloaded_runs/{experiment_name}/eval_results/"
```

### 3. Base Checkpoint Evaluation (if separate)
```bash
# Download base checkpoint evaluation results
scp -i ~/.ssh/lambda_new -r ubuntu@{IP}:/home/ubuntu/localfs/eval_runs/{model}_{dataset}/step_base_full_{dataset}/ \
  "data/downloaded_runs/{experiment_name}/base_eval/"
```

## Analysis Scripts Location

Place all analysis scripts in the experiment directory root:
- `complete_pass_plot.py` - Overall pass rate progression
- `detailed_passk_analysis.py` - All pass@k metrics
- `training_analysis.py` - Training loss/metrics analysis
- `entropy_probe_analysis.py` - GNS probe analysis

## Lambda Integration Workflow

The Lambda Cloud integration (`lambda_cloud/remote_eval.py`) should automatically sync results using this structure:

```python
# Example sync command that follows this convention
manager.sync_results(job_id, local_dir="data/downloaded_runs/{experiment_name}")
```

## File Exclusions

**Always exclude from download:**
- `.ipynb_checkpoints/` directories
- Intermediate model checkpoints (keep only final)
- Large `records.jsonl.gz` files unless specifically needed
- Temporary files and logs not needed for analysis

**Include in download:**
- `metrics.csv` - Essential evaluation metrics
- `run_meta.json` - Run metadata
- `train_log.jsonl` - Training logs
- `config.yaml` - Configuration used
- Final model checkpoint only

## Example Implementation

```bash
# Complete download script for GNS experiment
EXPERIMENT="gns_test_run_2025-08-20_qwen2_5_1_5b_h100_64steps"
IP="192.222.52.195"
RUN_ID="run_2025-08-20_03-31-43"

# Create directory structure
mkdir -p "data/downloaded_runs/$EXPERIMENT"/{training_run,eval_results,base_eval,analysis}

# Download evaluation results
scp -i ~/.ssh/lambda_new -r ubuntu@$IP:/home/ubuntu/localfs/eval_runs/$RUN_ID_gsm8k_r1_template/ \
  "data/downloaded_runs/$EXPERIMENT/eval_results/"

# Download base evaluation
scp -i ~/.ssh/lambda_new -r ubuntu@$IP:/home/ubuntu/localfs/eval_runs/qwen2_5_15_gsm8k_finetuned/step_base_full_gsm8k_r1_template/ \
  "data/downloaded_runs/$EXPERIMENT/base_eval/"

# Download training metadata (no intermediate checkpoints)
scp -i ~/.ssh/lambda_new ubuntu@$IP:/home/ubuntu/localfs/training_runs/$RUN_ID/{config.yaml,logs/train_log.jsonl} \
  "data/downloaded_runs/$EXPERIMENT/training_run/"
```

This convention ensures:
1. **Consistency** across all experiments
2. **Clear separation** of training vs evaluation data
3. **Efficient storage** by excluding unnecessary files
4. **Easy analysis** with predictable paths
5. **Scalability** for future experiments