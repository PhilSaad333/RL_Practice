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

### Research Features
- **In-Loop Gradient Noise Scale (GNS)**: Real-time estimation of critical batch sizes
- **Entropy Probes**: Custom metrics tracking entropy dynamics during training
- **Fisher Kernel Computation**: Experimental measurement of sequence-level correlations
- **Effective Sample Size (ESS)**: Analysis of weighted loss variance in Dr-GRPO

## 🧪 Experimental Setup

### Models & Datasets
- **Models**: Qwen2.5-1.5B (LoRA fine-tuned)
- **Dataset**: GSM8K mathematical reasoning with R1 template format
- **Environment**: Lambda Cloud GPU instances (2x H100 80GB)

### Current Investigations
1. **Linear Order Entropy Changes**: Testing whether δH₁ ≈ actual entropy changes
2. **Fisher Kernel Structure**: Measuring K₁(t,t') between sequences
3. **Sequence Correlations**: Beyond Cui et al.'s token-level analysis
4. **Gradient Conditioning**: Effects of Adam vs SGD on entropy dynamics

### Training Workflow
```bash
# 1. Start distributed training
PYTHONPATH=. torchrun --nproc_per_node=2 rl_training/runners/rl_runner.py \
  --cfg rl_training/cfg/testconfig.yaml \
  --ckpt /path/to/qwen2_5_15_gsm8k_lora/checkpoint-156

# 2. Evaluate all checkpoints
python rl_training/runners/eval_batch.py \
  --training-run run_2025-08-19_12-34-56 \
  --subset-frac 0.02

# 3. Resume training for more steps
python rl_training/runners/resume_training.py \
  --training-run run_2025-08-19_12-34-56 \
  --from-step latest \
  --additional-steps 50
```

## 📊 Data Analysis

Local analysis workflow for exploring training dynamics:

```bash
# Sync data from Lambda instances
python scripts/sync_lambda_data.py --ip <LAMBDA_IP> --run <RUN_NAME>

# Analyze in Jupyter notebook
# Open: notebooks/training_analysis.ipynb
```

The analysis notebook provides:
- Training progress visualization (loss, entropy, KL divergence)
- Evaluation metrics across checkpoints
- Multi-run comparisons
- Custom entropy dynamics analysis

## 🔧 Technical Implementation

### Distributed Training
- **Automatic grad_accum_steps**: `buffer_size / (world_size × microbatch_size)`
- **Memory-efficient evaluation**: Subprocess isolation to prevent tensor conflicts
- **Full state persistence**: Model + optimizer + scheduler for perfect resumption

### Research Probes
- **GNS Probe**: Measures gradient noise scale with zero extra forward/backward passes
- **Entropy Probe**: Tracks detailed entropy statistics during training
- **Fisher Kernel Estimation**: Custom matrix sketching for sequence correlations

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

## 🚀 Getting Started

1. **Setup Environment**:
   ```bash
   conda create -n rl python=3.10
   conda activate rl
   pip install -r requirements.txt
   ```

2. **Configure Lambda Instance** (see [`lambda/LAMBDA_SETUP_GUIDE.md`](lambda/LAMBDA_SETUP_GUIDE.md))

3. **Start Training**:
   ```bash
   # Test run (2 steps)
   PYTHONPATH=. torchrun --nproc_per_node=2 rl_training/runners/rl_runner.py \
     --cfg rl_training/cfg/testconfig.yaml \
     --ckpt <path_to_checkpoint>
   ```

4. **Analyze Results**:
   ```bash
   python scripts/sync_lambda_data.py --ip <IP> --run <RUN_NAME>
   # Open notebooks/training_analysis.ipynb
   ```

## 📖 Documentation

- [`NEW_TRAINING_WORKFLOW.md`](NEW_TRAINING_WORKFLOW.md) - Complete workflow documentation
- [`CLAUDE_GUIDE.md`](CLAUDE_GUIDE.md) - Project context and commands
- [`lambda/LAMBDA_SETUP_GUIDE.md`](lambda/LAMBDA_SETUP_GUIDE.md) - Infrastructure setup
- [`docs/RL_studies (1).pdf`](docs/RL_studies%20(1).pdf) - Theoretical foundations

## 🎪 The Journey

This project started as a simple question about entropy dynamics and grew into a deep dive spanning:

- **Mathematical Theory**: Deriving Fisher kernel relationships and second-order effects
- **Systems Engineering**: Building robust distributed training pipelines
- **Experimental Physics**: Measuring gradient noise scales and correlation matrices
- **Infrastructure**: Lambda Cloud, S3 sync, evaluation workflows
- **Data Science**: Analysis notebooks, visualization, multi-run comparisons

The meandering nature reflects the reality of research - each answer leads to new questions, and understanding often requires building the tools to explore properly.

## 🤝 Contributing

This is primarily a personal research project, but insights and discussions are welcome! Key areas of interest:

- Theoretical predictions vs empirical measurements of entropy dynamics
- Better matrix sketching techniques for Fisher kernel estimation
- Extensions to other RL algorithms beyond Dr-GRPO
- Connections to scaling laws and optimal batch size theory

---

*"The best way to understand entropy dynamics is to build the infrastructure to measure them properly."* - Lessons learned from this project