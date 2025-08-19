# Data Directory

This directory contains training and evaluation data downloaded from Lambda instances for local analysis.

## Structure

```
data/
├── runs/                    # Complete training runs
│   └── run_YYYY-MM-DD_HH-MM-SS/
│       ├── logs/
│       │   ├── train_log.jsonl
│       │   ├── rollouts.jsonl
│       │   └── ratios.jsonl
│       └── training_state/
├── eval_runs/              # Evaluation results
│   └── run_YYYY-MM-DD_HH-MM-SS_gsm8k_r1_template/
│       ├── consolidated_metrics.csv
│       └── step_*/
├── metrics/                # Quick access to key metrics files
│   ├── consolidated_metrics_*.csv
│   └── train_logs_*.jsonl
└── checkpoints/            # Selected checkpoints for local testing
    └── step_*/model/
```

## Sync Commands

### Full Run Sync
```bash
# Download complete training run
scp -r ubuntu@<LAMBDA_IP>:/lambda/nfs/localfs/training_runs/run_YYYY-MM-DD_HH-MM-SS \
  ./data/runs/

# Download evaluation results
scp -r ubuntu@<LAMBDA_IP>:/lambda/nfs/localfs/eval_runs/run_YYYY-MM-DD_HH-MM-SS_* \
  ./data/eval_runs/
```

### Quick Metrics Sync
```bash
# Just get the key metrics files
scp ubuntu@<LAMBDA_IP>:/lambda/nfs/localfs/eval_runs/*/consolidated_metrics.csv \
  ./data/metrics/

scp ubuntu@<LAMBDA_IP>:/lambda/nfs/localfs/training_runs/*/logs/train_log.jsonl \
  ./data/metrics/
```

## Git Ignore
This directory should be added to .gitignore to avoid committing large data files.