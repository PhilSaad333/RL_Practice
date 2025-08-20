# Entropy Experiments

**Two-tier approach for entropy analysis in RL training**

## Quick Start

### Tier 1: Regular Training with Simple δH
```bash
# Use existing rl_runner.py with entropy probe enabled
PYTHONPATH=. python rl_training/runners/rl_runner.py \
  --cfg rl_training/cfg/testconfig.yaml \
  --ckpt /path/to/checkpoint
```

### Tier 2: Detailed Fisher Analysis
```bash
# Run comprehensive Fisher kernel analysis on saved checkpoints
cd entropy_experiments
PYTHONPATH=.. python entropy_runner.py \
  --checkpoint /path/to/checkpoint \
  --config configs/fisher_analysis.yaml
```

## Scientific Approach

**Key insight**: Use **double-sampling** for theoretically correct entropy analysis:
- Sample **evaluation sequences** for E_t[...] computation
- Sample **training sequences** for gradient updates  
- Compute cross-Fisher kernel K₁(t_eval, t_train) between them

This enables studying how training updates affect logprobs of **any sequence**, not just training sequences.

### Critical Sampling Requirements

**⚠️ IMPORTANT: The two rollout buffers must have different sampling criteria:**

**Training Buffer (t')**: 
- **Filtered sampling** (current behavior)
- Reject sequences with all-correct or all-incorrect answers (zero advantages)
- Apply difficulty-based sampling weights from scheduler
- Only sequences that provide gradient signal

**Evaluation Buffer (t)**:
- **Unfiltered representative sampling** 
- **NO rejection** based on correctness
- **NO difficulty-based sampling bias**
- Pure random sampling from dataset
- Represents the true data distribution

**Rationale**: The Fisher kernel K(t,t') measures how gradient updates from filtered training sequences t' affect the log-probabilities of arbitrary evaluation sequences t. Using the same filtered sampling for both would bias the analysis and not reflect how training affects the broader data distribution.

## File Structure

- `entropy_runner.py` - Main specialized runner for detailed analysis
- `fisher_analyzer.py` - Fisher kernel computation and analysis
- `double_sampling.py` - Separate sampling for evaluation and training
- `checkpoint_loader.py` - Load training state from regular checkpoints
- `configs/` - Configuration files for different analysis types
- `results/` - Timestamped analysis results
- `notebooks/` - Analysis and visualization tools

## Documentation

See `ENTROPY_EXPERIMENTS_PLAN.md` for complete implementation details.