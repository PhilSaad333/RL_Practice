# RL Practice: Entropy Dynamics in Reinforcement Learning

A research project exploring entropy dynamics during reinforcement learning training, with a focus on understanding the theoretical foundations and empirical behavior of entropy changes in policy optimization.

## Project Overview

Sort of inspired by https://arxiv.org/abs/2505.22617. Their derivations are done under very unrealistic assumptions. A more proper analysis (as I attempt to do in the docs folder) is somewhat more involved, and points to a lot of interesting things to study. So while a "main goal" would be to see if a proper analysis suggests any further modifications to the algorithm changes suggested by Cui et al, I'm mostly trying to just get some experience with running experiments to study the things encountered in the high level theory stuff.

We can roughly split what I'm doing into:

- **1/3 Theoretical Investigation**: Understanding entropy dynamics during gradient steps in policy optimization (see [`docs/RL_studies (1).pdf`](docs/RL_studies%20(1).pdf))
- **1/3 Infrastructure & Tooling**: Building robust distributed training workflows, evaluation pipelines, and analysis tools
- **1/3 Empirical Exploration**: Investigating gradient noise scale, Fisher kernels, and other phenomena encountered along the way



### Key Components
```
rl_training/
├── algs/dr_grpo.py           # Main RL algorithm with entropy probes
├── runners/                  # Training orchestration
│   ├── rl_runner.py         # Main training loop
│   ├── eval_batch.py        # Batch evaluation of checkpoints
│   └── resume_training.py   # Resume from any checkpoint
└── utils/                   # Gradient noise scale, Fisher kernels, etc.

evals/                       # Evaluation pipeline
└── metrics/                 # Custom metrics for entropy analysis
```


### Models & Datasets
- **Models**: Qwen2.5-1.5B (LoRA fine-tuned)
- **Dataset**: GSM8K mathematical reasoning with R1 template format
- **Environment**: Lambda Cloud GPU instances (e.g 2x H100 80GB)

### Current Investigations
1. **Linear Order Entropy Changes**: Testing whether δH₁ ≈ actual entropy changes
2. **Fisher Kernel Structure**: Measuring K₁(t,t') between sequences
3. **Sequence Correlations**: Beyond Cui et al.'s token-level analysis

### File Organization
```
/lambda/nfs/localfs/
├── training_runs/              # Complete training sessions
│   └── run_YYYY-MM-DD_HH-MM-SS/
│       ├── training_state/     # Full checkpoints (model + optimizer)
│       ├── logs/              # Training metrics, rollouts, ratios
│       └── tensorboard/       # TensorBoard events
└── eval_runs/                 # Evaluation results
    └── *_gsm8k_r1_template/
        ├── consolidated_metrics.csv
        └── step_*/            # Per-checkpoint results
```


## Documentation

A lot of the documentation was written for claude code's usage. I started using claude code after switching from using colab's gpus to the lambda cloud. I found it pretty annoying to interact with the lambda cloud, especially having to set up every time, but now claude can run anything I'd like pretty seamlessly. Thanks Claude! 

- [`NEW_TRAINING_WORKFLOW.md`](NEW_TRAINING_WORKFLOW.md) - Complete workflow documentation
- [`CLAUDE_GUIDE.md`](CLAUDE_GUIDE.md) - Project context and commands
- [`lambda/LAMBDA_SETUP_GUIDE.md`](lambda/LAMBDA_SETUP_GUIDE.md) - Infrastructure setup
- [`docs/RL_studies (1).pdf`](docs/RL_studies%20(1).pdf) - Theoretical foundations

